//#define TTG_USE_PARSEC 1

#ifdef TTG_USE_PARSEC
// Use the dynamic termination detection by default
#undef TTG_USE_USER_TERMDET
#endif // TTG_USE_PARSEC

#include <ttg.h>

// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>

#include "pmw.h"
#include "plgsy.h"
#include "potrf.h"
#include "potri.h"
#include "result.h"

#ifdef USE_DPLASMA
#include <dplasma.h>
#endif

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
  int NB = 128;
  int N = 5*NB;
  int M = N;
  int check = 0;
  int nthreads = -1;
  const char* prof_filename = nullptr;
  char *opt = nullptr;

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

  ttg::initialize(argc, argv, nthreads);

  auto world = ttg::default_execution_context();
  
  if(nullptr != prof_filename) {
    ttg::profile_on();
    world.impl().start_tracing_dag_of_tasks(prof_filename);
  }

  int P = std::sqrt(world.size());
  int Q = (world.size() + P - 1)/P;

  static_assert(ttg::has_split_metadata<MatrixTile<double>>::value);

  std::cout << "Creating 2D block cyclic matrix with NB " << NB << " N " << N << " M " << M << " P " << P << std::endl;

  sym_two_dim_block_cyclic_t dcA;
  sym_two_dim_block_cyclic_init(&dcA, matrix_type::matrix_RealDouble,
                                world.size(), world.rank(), NB, NB, N, M,
                                0, 0, N, M, P, matrix_Lower);
  dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                 (size_t)dcA.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
  parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, (char*)"Matrix A");

  ttg::Edge<Key2, void> startup("startup");
  ttg::Edge<Key2, MatrixTile<double>> topotrf("To POTRF");
  ttg::Edge<Key2, MatrixTile<double>> topotri("To POTRI");
  ttg::Edge<Key2, MatrixTile<double>> result("To result");

  //Matrix<double>* A = new Matrix<double>(n_rows, n_cols, NB, NB);
  MatrixT<double> A{&dcA};
  /* TODO: initialize the matrix */
  /* This works only with the parsec backend! */
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
  }, ttg::edges(), ttg::edges(startup), "Startup Trigger", {}, {"startup"});
  init_tt->set_keymap([&]() {return world.rank();});

#if defined(USE_DPLASMA)
  dplasma_dplgsy( world.impl().context(), (double)(N), matrix_Lower,
                (parsec_tiled_matrix_dc_t *)&dcA, random_seed);
  auto init_tt  = make_matrix_reader_tt(A, startup, topotrf);
#else
  auto plgsy_ttg = make_plgsy_ttg(A, N, random_seed, startup, topotrf);
#endif // USE_DPLASMA

  auto potrf_ttg = potrf::make_potrf_ttg(A, topotrf, topotri);
  auto potri_ttg = potri::make_potri_ttg(A, topotri, result);
  auto result_ttg = make_result_ttg(A, result);

  auto connected = make_graph_executable(init_tt.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected: " << connected << std::endl;

  if (world.rank() == 0) {
#if 1
    std::cout << "==== begin dot ====\n";
    std::cout << ttg::Dot()(init_tt.get()) << std::endl;
    std::cout << "==== end dot ====\n";
#endif // 0
    beg = std::chrono::high_resolution_clock::now();
  }
  init_tt->invoke();

  ttg::execute(world);
  ttg::fence(world);
  if (world.rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << elapsed / 1E3 << " : Flops " << (potri::FLOPS_DPOTRI(N)) << " " << (potri::FLOPS_DPOTRI(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
  }

  world.impl().stop_tracing_dag_of_tasks();

  //delete A;
  /* cleanup allocated matrix before shutting down PaRSEC */
  parsec_data_free(dcA.mat); dcA.mat = NULL;
  parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

  ttg::finalize();
  return 0;
}

