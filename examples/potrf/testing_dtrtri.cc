//#define TTG_USE_PARSEC 1

#ifdef TTG_USE_PARSEC
// tell TTG/PARSEC that we need to rely on dynamic termination detection
#undef TTG_USE_USER_TERMDET
#endif  // TTG_USE_PARSEC

#include <ttg.h>
#include <ttg/serialization/std/tuple.h>

#include "plgsy.h"
#include "pmw.h"
#include "result.h"
#include "trtri_L.h"
#include "trtri_U.h"

int check_dtrtri( lapack::Diag diag, lapack::Uplo uplo, double *A, double *Ainv, int N );

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

static bool inmatrix(int i, int j, lapack::Uplo uplo) {
  return (uplo == lapack::Uplo::Lower && j <= i) || (uplo == lapack::Uplo::Upper && i <= j);
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
  lapack::Uplo uplo;
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

  bool check = !cmdOptionExists(argv+1, argv+argc, "-x");
  bool cow_hint = !cmdOptionExists(argv+1, argv+argc, "-w");
  bool upper = cmdOptionExists(argv+1, argv+argc, "-U");

  if(upper)
    uplo = lapack::Uplo::Upper;
  else
    uplo = lapack::Uplo::Lower;

  ttg::initialize(argc, argv, nthreads);

  auto world = ttg::default_execution_context();

  if(nullptr != prof_filename) {
    world.profile_on();
    world.dag_on(prof_filename);
  }

  int P = std::sqrt(world.size());
  int Q = (world.size() + P - 1)/P;

   if(check && (P>1 || Q>1)) {
    std::cerr << "Check is disabled for distributed runs at this time" << std::endl;
    check = false;
  }

  static_assert(ttg::has_split_metadata<MatrixTile<double>>::value);

  std::cout << "Creating 2D block cyclic matrix with NB " << NB << " N " << N << " M " << M << " P " << P << std::endl;

  parsec_matrix_sym_block_cyclic_t dcA;
  parsec_matrix_sym_block_cyclic_init(&dcA, parsec_matrix_type_t::PARSEC_MATRIX_DOUBLE,
                                world.rank(), NB, NB, N, M,
                                0, 0, N, M, P, Q, uplo == lapack::Uplo::Lower ? PARSEC_MATRIX_LOWER : PARSEC_MATRIX_UPPER);
  dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                 (size_t)dcA.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
  parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, (char*)"Matrix A");

  if(!check) {

    ttg::Edge<Key2, void> startup("startup");
    ttg::Edge<Key2, MatrixTile<double>> totrtri("To TRTRI");
    ttg::Edge<Key2, MatrixTile<double>> result("To result");

    MatrixT<double> A{&dcA};
    int random_seed = 3872;

    auto init_tt =  ttg::make_tt<void>([&](std::tuple<ttg::Out<Key2, void>>& out) {
      for(int i = 0; i < A.rows(); i++) {
        for(int j = 0; inmatrix(i, j, uplo) && j < A.cols(); j++) {
          if(A.is_local(i, j)) {
            if(ttg::tracing()) ttg::print("init(", Key2{i, j}, ")");
            ttg::sendk<0>(Key2{i, j}, out);
          }
        }
      }
    }, ttg::edges(), ttg::edges(startup), "Startup Trigger", {}, {"startup"});
    init_tt->set_keymap([&]() {return world.rank();});

    auto plgsy_ttg = make_plgsy_ttg(A, N, random_seed, startup, totrtri, cow_hint);
    decltype(trtri_LOWER::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, result, cow_hint)) trtri_ttg;
    if(uplo == lapack::Uplo::Lower)
      trtri_ttg = trtri_LOWER::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, result, cow_hint);
    else
      trtri_ttg = trtri_UPPER::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, result, cow_hint);
    auto result_ttg = make_result_ttg(A, result, cow_hint);

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
                << elapsed / 1E3 << " : Flops " << (trtri_LOWER::FLOPS_DTRTRI(N)) << " " << (trtri_LOWER::FLOPS_DTRTRI(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
    }
  } else {
    ttg::Edge<Key2, void> startup("startup");
    ttg::Edge<Key2, MatrixTile<double>> result("To result");

    MatrixT<double> A{&dcA};
    int random_seed = 3872;

    auto init_tt =  ttg::make_tt<void>([&](std::tuple<ttg::Out<Key2, void>>& out) {
      for(int i = 0; i < A.rows(); i++) {
        for(int j = 0; inmatrix(i, j, uplo) && j < A.cols(); j++) {
          if(A.is_local(i, j)) {
            if(ttg::tracing()) ttg::print("init(", Key2{i, j}, ")");
            ttg::sendk<0>(Key2{i, j}, out);
          }
        }
      }
    }, ttg::edges(), ttg::edges(startup), "Startup Trigger", {}, {"startup"});
    init_tt->set_keymap([&]() {return world.rank();});

    auto plgsy_ttg = make_plgsy_ttg(A, N, random_seed, startup, result, cow_hint);
    auto result_ttg = make_result_ttg(A, result, cow_hint);

    auto connected = make_graph_executable(init_tt.get());
    assert(connected);
    TTGUNUSED(connected);

    init_tt->invoke();

    ttg::execute(world);
    ttg::fence(world);

    double *A0 = A.getLAPACKMatrix();

    ttg::Edge<Key2, MatrixTile<double>> totrtri("To TRTRI");
    ttg::Edge<Key2, MatrixTile<double>> result2("To result");
    auto load_plgsy = make_load_tt(A, totrtri, cow_hint);
    decltype(trtri_LOWER::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, result, cow_hint)) trtri_ttg;
    if(uplo == lapack::Uplo::Lower)
      trtri_ttg = trtri_LOWER::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, result2, cow_hint);
    else
      trtri_ttg = trtri_UPPER::make_trtri_ttg(A, lapack::Diag::NonUnit, totrtri, result2, cow_hint);
    auto result2_ttg = make_result_ttg(A, result2, cow_hint);

    connected = make_graph_executable(load_plgsy.get());
    assert(connected);
    TTGUNUSED(connected);

    load_plgsy->invoke();

    ttg::fence(world);

    double *Ainv = A.getLAPACKMatrix();

    if( check_dtrtri(lapack::Diag::NonUnit, uplo, A0, Ainv, N) == -1 ) {
      if(N < 32) {
        print_LAPACK_matrix(A0, N, "Original");
        auto info = lapack::trtri(uplo, lapack::Diag::NonUnit, N, A0, N);
        print_LAPACK_matrix(A0, N, "lapack::trtri(Original)");
        print_LAPACK_matrix(Ainv, N, "ttg::trtri(Original)");
      }
      ret = EXIT_FAILURE;
    }

    delete [] Ainv;
    delete [] A0;
  }

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

int check_dtrtri( lapack::Diag diag, lapack::Uplo uplo, double *A, double *Ainv, int N )
{
    int ret;
    double Anorm, Ainvnorm, Rnorm;
    double Rcond, result;
    double eps = std::numeric_limits< double >::epsilon();

    assert(lapack::Uplo::Upper == uplo || lapack::Uplo::Lower == uplo);

    double *Id = new double[N*N];
    lapack::laset(lapack::MatrixType::General, N, N, 0.0, 1.0, Id, N);

    double *A0 = new double[N*N];
    if(diag == lapack::Diag::NonUnit) {
      lapack::laset(lapack::MatrixType::General, N, N, 0.0, 1.0, A0, N);
      if(uplo == lapack::Uplo::Lower) {
        lapack::lacpy(lapack::MatrixType::Lower, N, N, Ainv, N, A0, N);
      } else {
        lapack::lacpy(lapack::MatrixType::Upper, N, N, Ainv, N, A0, N);
      }
    } else {
      lapack::MatrixType mnuplo = (uplo == lapack::Uplo::Lower) ? lapack::MatrixType::Upper : lapack::MatrixType::Lower ;
      lapack::MatrixType muplo = (uplo == lapack::Uplo::Lower) ? lapack::MatrixType::Lower : lapack::MatrixType::Upper ;
      lapack::lacpy(muplo, N, N, Ainv, N, A0, N);
      lapack::laset(mnuplo, N, N, 0., 1., A0, N );
    }

    blas::trmm(blas::Layout::ColMajor, blas::Side::Left, uplo, blas::Op::NoTrans, diag, N, N, 1.0, A, N, A0, N);

    /* compute Id <- Id - A0 */
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < N; j++) {
        Id[i+j*N] -= A0[i+j*N];
      }
    }

    Anorm    = lapack::lantr(lapack::Norm::One, uplo, diag, N, N, A, N);
    Ainvnorm = lapack::lantr(lapack::Norm::One, uplo, diag, N, N, Ainv, N);
    Rnorm    = lapack::lantr(lapack::Norm::One, uplo, lapack::Diag::NonUnit, N, N, Id, N);

    Rcond  = ( 1. / Anorm ) / Ainvnorm;
    result = (Rnorm * Rcond) / (eps * N);

    std::cout << "============" << std::endl;
    std::cout << "Checking TRTRI " << std::endl;
    std::cout <<  "-- ||A||_one = " << Anorm << " ||A^(-1)||_one = " << Ainvnorm << " ||I - A * A^(-1)||_one = " 
              << Rnorm << ", cond = " << Rcond << ", result = " << result << std::endl;

    if ( std::isinf(Ainvnorm) || std::isnan(result) || std::isinf(result) || (result > 10.0) ) {
        std::cout << "-- Inversion is suspicious !" << std::endl;
        ret = -1;
    }
    else {
        std::cout << "-- Inversion is CORRECT !" << std::endl;
        ret = 0;
    }

    delete [] Id;
    delete [] A0;

    return ret;
}
