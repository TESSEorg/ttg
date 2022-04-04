//#define TTG_USE_PARSEC 1

#ifdef TTG_USE_PARSEC
// tell TTG/PARSEC that we need to rely on dynamic termination detection
#undef TTG_USE_USER_TERMDET
#endif // TTG_USE_PARSEC

#include <ttg.h>

// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>

#include "pmw.h"
#include "plgsy.h"
#include "trtri.h"
#include "result.h"

#ifdef USE_DPLASMA
#include <dplasma.h>
#endif

int main(int argc, char **argv)
{
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  int NB = 128;
  int N = 5*NB;
  int M = N;
  int check = 0;
  int nthreads = 1;
  const char* prof_filename = nullptr;

  if (argc > 1) {
    N = M = atoi(argv[1]);
  }

  if (argc > 2) {
    NB = atoi(argv[2]);
  }

  if (argc > 3) {
    check = atoi(argv[3]);
  }

  if (argc > 4) {
    nthreads = atoi(argv[4]);
  }

  if (argc > 5) {
    prof_filename = argv[5];
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
  ttg::Edge<Key2, MatrixTile<double>> totrtri("To TRTRI");
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
          ttg::print("init(", Key2{i, j}, ")");
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
  auto plgsy_ttg = make_plgsy_ttg(A, N, random_seed, startup, totrtri);
#endif // USE_DPLASMA

  auto trtri_ttg = trtri::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, result);
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
              << elapsed / 1E3 << " : Flops " << (trtri::FLOPS_DTRTRI(N)) << " " << (trtri::FLOPS_DTRTRI(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
  }

  //delete A;
  /* cleanup allocated matrix before shutting down PaRSEC */
  parsec_data_free(dcA.mat); dcA.mat = NULL;
  parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

  world.impl().stop_tracing_dag_of_tasks();

  ttg::finalize();
  return 0;
}

static void
dplasma_dprint_tile( int m, int n,
                     const parsec_tiled_matrix_dc_t* descA,
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
