//#define TTG_USE_PARSEC 1

#ifdef TTG_USE_PARSEC
// Use the dynamic termination detection by default
#undef TTG_USE_USER_TERMDET
#endif  // TTG_USE_PARSEC

#include <ttg.h>
#include <ttg/serialization/std/tuple.h>

#include "plgsy.h"
#include "pmw.h"
#include "potrf.h"
#include "potri.h"
#include "result.h"

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char **argv)
{

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  int NB = 32;
  int N = 5*NB;
  int M = N;
  int nthreads = -1;
  int nruns = 3;
  const char* prof_filename = nullptr;
  char *opt = nullptr;
  int sequential=0;
  int ret = EXIT_SUCCESS;

  if( (opt = getCmdOption(argv+1, argv+argc, "-N")) != nullptr ) {
    N = M = atoi(opt);
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-t")) != nullptr ) {
    NB = atoi(opt);
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-c")) != nullptr ) {
    nthreads = atoi(opt);
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-dag")) != nullptr ) {
    prof_filename = opt;
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-nruns")) != nullptr ) {
    nruns = atoi(opt);
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-seq")) != nullptr ) {
    sequential = atoi(opt);
  }

  bool ttg_dags = cmdOptionExists(argv+1, argv+argc, "-ttg-dag");
  bool verbose = cmdOptionExists(argv+1, argv+argc, "-v");
  bool defer_cow_hint = !cmdOptionExists(argv+1, argv+argc, "-w");

  auto dashdash = std::find(argv, argv+argc, std::string("--"));
  char **ttg_argv;
  int ttg_argc;
  if(dashdash == argv+argc) {
    ttg_argv = new char *[2];
    ttg_argv[0] = argv[0];
    ttg_argv[1] = nullptr;
    ttg_argc = 1;
  } else {
    dashdash++;
    ttg_argv = new char *[argc];
    ttg_argv[0] = argv[0];
    int i = 1;
    for(; dashdash != argv+argc; i++, dashdash++) {
      ttg_argv[i] = *dashdash;
    }
    ttg_argv[i] = nullptr;
    ttg_argc = i;
  }

  ttg::initialize(ttg_argc, ttg_argv, nthreads);
  delete[] ttg_argv;

  /* set up TA to get the allocator */
  allocator_init(argc, argv);

  ttg::trace_on();

  auto world = ttg::default_execution_context();

  if(nullptr != prof_filename) {
    world.profile_on();
    world.dag_on(prof_filename);
  }

  int P = std::sqrt(world.size());
  int Q = (world.size() + P - 1)/P;
  while(P * Q != world.size()) {
    P--;
    Q = (world.size() + P - 1)/P;
  }

  if(verbose) {
    std::cout << "Creating 2D block cyclic matrix with NB " << NB << " N " << N << " M " << M << " P " << P << std::endl;
  }

  parsec_matrix_sym_block_cyclic_t dcA;
  parsec_matrix_sym_block_cyclic_init(&dcA, parsec_matrix_type_t::PARSEC_MATRIX_DOUBLE,
                                world.rank(),
                                NB, NB,
                                N, M,
                                0, 0,
                                N, M,
                                P, Q,
                                PARSEC_MATRIX_LOWER);
  dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                 (size_t)dcA.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
  parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, (char*)"Matrix A");

  /********************** Step 0: Generation  **********************/

  for(int t = 0; t <= nruns; t++) {
    ttg::Edge<Key2, void> startup("startup PLGSY");
    ttg::Edge<Key2, MatrixTile<double>> toresplgsy("store PLGSY");
    MatrixT<double> A{&dcA};
    int random_seed = 3872;

    auto init_tt =  ttg::make_tt<void>([&](std::tuple<ttg::Out<Key2, void>>& out) {
      for(int i = 0; i < A.rows(); i++) {
        for(int j = 0; j <= i && j < A.cols(); j++) {
          if(A.is_local(i, j)) {
            if(ttg::tracing()) ttg::print("init(", Key2{i, j}, ")");
            ttg::sendk<0>(Key2{i, j}, out);
          }
        }
      }
    }, ttg::edges(), ttg::edges(startup), "Startup Trigger for PLGSY", {}, {"startup"});
    init_tt->set_keymap([&]() {return world.rank();});
    init_tt->set_defer_writer(defer_cow_hint);
    auto plgsy_ttg = make_plgsy_ttg(A, N, random_seed, startup, toresplgsy, defer_cow_hint);
    auto store_plgsy_ttg = make_result_ttg(A, toresplgsy, defer_cow_hint);

    auto connected = make_graph_executable(init_tt.get());
    assert(connected);
    TTGUNUSED(connected);

    if(verbose) {
      std::cout << "Graph is connected: " << connected << std::endl;
    }

    if (world.rank() == 0) {
#if 1
      if(verbose) {
        std::cout << "==== begin dot ====\n";
        std::cout << ttg::Dot()(init_tt.get()) << std::endl;
        std::cout << "==== end dot ====\n";
      }
#endif // 0
    }
    init_tt->invoke();

    if(t == 0)
      ttg::execute(world);
    ttg::fence(world);

    if(sequential > 0) {
      ttg::Edge<Key2, MatrixTile<double>> topotrf("To POTRF");
      ttg::Edge<Key2, MatrixTile<double>> torespotrf("To Res POTRF");
      std::chrono::time_point<std::chrono::high_resolution_clock> begpotrf, endpotrf;

      /********************** First Step: POTRF  **********************/
      auto load_plgsy = make_load_tt(A, topotrf, defer_cow_hint);
      auto potrf_ttg = potrf::make_potrf_ttg(A, topotrf, torespotrf, defer_cow_hint);
      auto store_potrf_ttg = make_result_ttg(A, torespotrf, defer_cow_hint);

      connected = make_graph_executable(load_plgsy.get());
      assert(connected);
      TTGUNUSED(connected);

      if(verbose) {
        std::cout << "Graph is connected: " << connected << std::endl;
      }

      if (world.rank() == 0) {
    #if 1
        if(verbose) {
          std::cout << "==== begin dot ====\n";
          std::cout << ttg::Dot()(load_plgsy.get()) << std::endl;
          std::cout << "==== end dot ====\n";
        }
    #endif // 0
        beg = std::chrono::high_resolution_clock::now();
        begpotrf = beg;
      }
      load_plgsy->invoke();

      ttg::fence(world);
      if (world.rank() == 0) {
        endpotrf = std::chrono::high_resolution_clock::now();
      }

      if( 1 == sequential ) {
        ttg::Edge<Key2, MatrixTile<double>> topotri("To POTRI");
        ttg::Edge<Key2, MatrixTile<double>> toresult("To result");
        std::chrono::time_point<std::chrono::high_resolution_clock> begpotri, endpotri;

        /********************** Second step: POTRI  **********************/

        auto load_potrf = make_load_tt(A, topotri, defer_cow_hint);
        auto potri_ttg = potri::make_potri_ttg(A, topotri, toresult, defer_cow_hint);
        auto store_potri_ttg = make_result_ttg(A, toresult, defer_cow_hint);

        connected = make_graph_executable(load_potrf.get());
        assert(connected);
        TTGUNUSED(connected);
        if(verbose) {
          std::cout << "Graph is connected: " << connected << std::endl;
        }

        if (world.rank() == 0) {
      #if 1
          if(verbose) {
            std::cout << "==== begin dot ====\n";
            std::cout << ttg::Dot()(load_potrf.get()) << std::endl;
            std::cout << "==== end dot ====\n";
          }
      #endif // 0
          begpotri = std::chrono::high_resolution_clock::now();
        }
        load_potrf->invoke();

        ttg::fence(world);
        if (world.rank() == 0 && t > 0) {
          endpotri = std::chrono::high_resolution_clock::now();
          end = endpotri;

          auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(endpotrf - begpotrf).count());
          std::cout << "POINV (POTRF+POTRI) (" << (defer_cow_hint ? "with" : "without") << " defer writer) -- N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                    << " POTRF TTG Execution Time (milliseconds) : " << elapsed / 1E3 << " : Flops " << (potrf::FLOPS_DPOTRF(N)) << " " << (potrf::FLOPS_DPOTRF(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;

          elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(endpotri - begpotri).count());
          std::cout << "POINV (POTRF+POTRI) (" << (defer_cow_hint ? "with" : "without") << " defer writer) -- N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                    << " POTRI TTG Execution Time (milliseconds) : " << elapsed / 1E3 << " : Flops " << (potri::FLOPS_DPOTRI(N)) << " " << (potri::FLOPS_DPOTRI(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;

          elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
          std::cout << "POINV (POTRF+POTRI) (" << (defer_cow_hint ? "with" : "without") << " defer writer) N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                    << " TTG Execution Time (milliseconds) : " << elapsed / 1E3 << " : Flops " << (potrf::FLOPS_DPOTRF(N) + potri::FLOPS_DPOTRI(N)) << " " << ((potrf::FLOPS_DPOTRF(N)+potri::FLOPS_DPOTRI(N))/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
        }
      } else {
        ttg::Edge<Key2, MatrixTile<double>> totrtri("To TRTRI");
        ttg::Edge<Key2, MatrixTile<double>> torestrtri("To Res TRTRI");
        ttg::Edge<Key2, MatrixTile<double>> tolauum("To LAUUM");
        ttg::Edge<Key2, MatrixTile<double>> toresult("To result");
        std::chrono::time_point<std::chrono::high_resolution_clock> begtrtri, endtrtri;
        std::chrono::time_point<std::chrono::high_resolution_clock> beglauum, endlauum;

        /********************** Second step: TRTRI  **********************/

        auto load_potrf = make_load_tt(A, totrtri, defer_cow_hint);
        auto trtri_ttg = trtri_LOWER::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, torestrtri, defer_cow_hint);
        auto store_trtri_ttg = make_result_ttg(A, torestrtri, defer_cow_hint);

        connected = make_graph_executable(load_potrf.get());
        assert(connected);
        TTGUNUSED(connected);
        if(verbose) {
          std::cout << "Graph is connected: " << connected << std::endl;
        }

        if (world.rank() == 0) {
      #if 1
          if(verbose) {
            std::cout << "==== begin dot ====\n";
            std::cout << ttg::Dot()(load_potrf.get()) << std::endl;
            std::cout << "==== end dot ====\n";
          }
      #endif // 0
          begtrtri = std::chrono::high_resolution_clock::now();
        }
        load_potrf->invoke();

        ttg::fence(world);
        if (world.rank() == 0) {
          endtrtri = std::chrono::high_resolution_clock::now();
        }

        /********************** Last step: LAUUM  **********************/

        auto load_trtri = make_load_tt(A, tolauum, defer_cow_hint);
        auto lauum_ttg = lauum::make_lauum_ttg(A, tolauum, toresult, defer_cow_hint);
        auto result = make_result_ttg(A, toresult, defer_cow_hint);

        connected = make_graph_executable(load_trtri.get());
        assert(connected);
        TTGUNUSED(connected);
        if( verbose ) {
          std::cout << "Graph is connected: " << connected << std::endl;
        }

        if (world.rank() == 0) {
#if 1
          if(verbose) {
            std::cout << "==== begin dot ====\n";
            std::cout << ttg::Dot()(load_trtri.get()) << std::endl;
            std::cout << "==== end dot ====\n";
          }
#endif // 0
          beglauum = std::chrono::high_resolution_clock::now();
        }
        load_trtri->invoke();

        ttg::fence(world);
        if (world.rank() == 0 && t > 0) {
          endlauum = std::chrono::high_resolution_clock::now();
          end = endlauum;

          auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(endpotrf - begpotrf).count());
          std::cout << "POINV (POTRF+TRTRI+LAUUM) (" << (defer_cow_hint ? "with" : "without") << " defer writer) -- N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                    << " POTRF TTG Execution Time (milliseconds) : " << elapsed / 1E3 << " : Flops " << (potrf::FLOPS_DPOTRF(N)) << " " << (potrf::FLOPS_DPOTRF(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;

          elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(endtrtri - begtrtri).count());
          std::cout << "POINV (POTRF+TRTRI+LAUUM) (" << (defer_cow_hint ? "with" : "without") << " defer writer) -- N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                    << " TRTRI TTG Execution Time (milliseconds) : " << elapsed / 1E3 << " : Flops " << (trtri_LOWER::FLOPS_DTRTRI(N)) << " " << (trtri_LOWER::FLOPS_DTRTRI(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;

          elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(endlauum - beglauum).count());
          std::cout << "POINV (POTRF+TRTRI+LAUUM) (" << (defer_cow_hint ? "with" : "without") << " defer writer) -- N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                    << " LAUUM TTG Execution Time (milliseconds) : " << elapsed / 1E3 << " : Flops " << (lauum::FLOPS_DLAUUM(N)) << " " << (lauum::FLOPS_DLAUUM(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;

          elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
          std::cout << "POINV (POTRF+TRTRI+LAUUM) (" << (defer_cow_hint ? "with" : "without") << " defer writer) N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                    << " TTG Execution Time (milliseconds) : " << elapsed / 1E3
                    << " : Flops " << (potrf::FLOPS_DPOTRF(N) + potri::FLOPS_DPOTRI(N)) << " " << ((potrf::FLOPS_DPOTRF(N)+potri::FLOPS_DPOTRI(N))/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
        }
      }
    } else {
      ttg::Edge<Key2, MatrixTile<double>> topotrf("To POTRF");
      ttg::Edge<Key2, MatrixTile<double>> topotri("To POTRI");
      ttg::Edge<Key2, MatrixTile<double>> result("To result");

      auto load_plgsy = make_load_tt(A, topotrf, defer_cow_hint);
      auto potrf_ttg = potrf::make_potrf_ttg(A, topotrf, topotri, defer_cow_hint);
      auto potri_ttg = potri::make_potri_ttg(A, topotri, result, defer_cow_hint);
      auto result_ttg = make_result_ttg(A, result, defer_cow_hint);

      auto connected = make_graph_executable(load_plgsy.get());
      assert(connected);
      TTGUNUSED(connected);
      if(verbose) {
        std::cout << "Graph is connected: " << connected << std::endl;
      }

      if (world.rank() == 0) {
  #if 1
        if(verbose) {
          std::cout << "==== begin dot ====\n";
          std::cout << ttg::Dot()(load_plgsy.get()) << std::endl;
          std::cout << "==== end dot ====\n";
        }
  #endif // 0
        beg = std::chrono::high_resolution_clock::now();
      }
      load_plgsy->invoke();

      ttg::fence(world);
      if (world.rank() == 0 && t > 0) {
        end = std::chrono::high_resolution_clock::now();
        auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "POINV (POINV) (" << (defer_cow_hint ? "with" : "without") << " defer writer) N= " << N << " NB= " << NB <<  " P= " << P << " Q= " << Q << " nthreads= " << nthreads
                  << " TTG Execution Time (milliseconds) : " << elapsed / 1E3
                  << " : Flops " << (potrf::FLOPS_DPOTRF(N) + potri::FLOPS_DPOTRI(N)) << " " << ((potrf::FLOPS_DPOTRF(N)+potri::FLOPS_DPOTRI(N))/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
      }
    }
  }


  world.dag_off();
  world.profile_off();

  //delete A;
  /* cleanup allocated matrix before shutting down PaRSEC */
  parsec_data_free(dcA.mat); dcA.mat = NULL;
  parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcA);

  allocator_fini();
  ttg::finalize();
  return ret;
}
