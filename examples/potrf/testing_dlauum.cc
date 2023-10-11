//#define TTG_USE_PARSEC 1

#ifdef TTG_USE_PARSEC
// tell TTG/PARSEC that we need to rely on dynamic termination detection
#undef TTG_USE_USER_TERMDET
#endif  // TTG_USE_PARSEC

#include <ttg.h>
#include <ttg/serialization/std/tuple.h>

#include "lauum.h"
#include "plgsy.h"
#include "pmw.h"
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
  const char* prof_filename = nullptr;
  char *opt = nullptr;
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
  bool defer_cow_hint = !cmdOptionExists(argv+1, argv+argc, "-w");

  ttg::initialize(argc, argv, nthreads);

  auto world = ttg::default_execution_context();

  if( prof_filename != nullptr ) {
    world.profile_on();
    world.dag_on(prof_filename);
  }

  int P = std::sqrt(world.size());
  int Q = (world.size() + P - 1)/P;

  static_assert(ttg::has_split_metadata<MatrixTile<double>>::value);

  std::cout << "Creating 2D block cyclic matrix with NB " << NB << " N " << N << " M " << M << " P " << P << std::endl;

  parsec_matrix_sym_block_cyclic_t dcA;
  parsec_matrix_sym_block_cyclic_init(&dcA, parsec_matrix_type_t::PARSEC_MATRIX_DOUBLE,
                                world.rank(), NB, NB, N, M,
                                0, 0, N, M, P, Q, PARSEC_MATRIX_LOWER);
  dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                 (size_t)dcA.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
  parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, (char*)"Matrix A");

  ttg::Edge<Key2, void> startup("startup");
  ttg::Edge<Key2, MatrixTile<double>> tolauum("To LAUUM");
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

  auto plgsy_ttg = make_plgsy_ttg(A, N, random_seed, startup, tolauum, defer_cow_hint);
  auto lauum_ttg = lauum::make_lauum_ttg(A, tolauum, result, defer_cow_hint);
  auto result_ttg = make_result_ttg(A, result, defer_cow_hint);

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
              << elapsed / 1E3 << " : Flops " << (lauum::FLOPS_DLAUUM(N)) << " " << (lauum::FLOPS_DLAUUM(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
  }

  //delete A;
  /* cleanup allocated matrix before shutting down PaRSEC */
  parsec_data_free(dcA.mat); dcA.mat = NULL;
  parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcA);

  world.dag_off();
  world.profile_off();

  ttg::finalize();
  return ret;
}

static void
dplasma_dprint_tile( int m, int n,
                     const parsec_tiled_matrix_t* descA,
                     const double *M )
{
    int tempmm = ( m == descA->mt-1 ) ? descA->m - m*descA->mb : descA->mb;
    int tempnn = ( n == descA->nt-1 ) ? descA->n - n*descA->nb : descA->nb;
    int ldam = BLKLDD( descA, m );

    int ii, jj;

    fflush(stdout);
    for(ii=0; ii<tempmm; ii++) {
        if ( ii == 0 )
            fprintf(stdout, "(%2d, %2d) :", m, n);
        else
            fprintf(stdout, "          ");
        for(jj=0; jj<tempnn; jj++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
            fprintf(stdout, " (% e, % e)",
                    creal( M[jj*ldam + ii] ),
                    cimag( M[jj*ldam + ii] ));
#else
            fprintf(stdout, " % e", M[jj*ldam + ii]);
#endif
        }
        fprintf(stdout, "\n");
    }
    fflush(stdout);
    usleep(1000);
}
