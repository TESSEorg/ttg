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
    world.profile_on();
    world.dag_on(prof_filename);
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

  auto potrf_ttg = potrf::make_potrf_ttg(A, topotrf, result);
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
              << elapsed / 1E3 << " : Flops " << (potrf::FLOPS_DPOTRF(N)) << " " << (potrf::FLOPS_DPOTRF(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
  }

  world.dag_off();

#ifdef USE_DPLASMA
  if( check ) {
    /* Check the factorization */
    int loud = 10;
    int ret = 0;
    sym_two_dim_block_cyclic_t dcA0;
    sym_two_dim_block_cyclic_init(&dcA0, matrix_type::matrix_RealDouble,
                                  world.size(), world.rank(), NB, NB, N, M,
                                  0, 0, N, M, P, matrix_Lower);
    dcA0.mat = parsec_data_allocate((size_t)dcA0.super.nb_local_tiles *
                                  (size_t)dcA0.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcA0.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA0, "Matrix A0");
    dplasma_dplgsy( world.impl().context(), (double)(N), matrix_Lower,
                  (parsec_tiled_matrix_dc_t *)&dcA0, random_seed);

    ret |= check_dpotrf( world.impl().context(), (world.rank() == 0) ? loud : 0, matrix_Lower,
                          (parsec_tiled_matrix_dc_t *)&dcA,
                          (parsec_tiled_matrix_dc_t *)&dcA0);

    /* Check the solution */
    two_dim_block_cyclic_t dcB;
    two_dim_block_cyclic_init(&dcB, matrix_type::matrix_RealDouble, matrix_storage::matrix_Tile,
                              world.size(), world.rank(), NB, NB, N, M,
                              0, 0, N, M, 1, 1, P);
    dcB.mat = parsec_data_allocate((size_t)dcB.super.nb_local_tiles *
                                  (size_t)dcB.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcB.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcB, "Matrix B");
    dplasma_dplrnt( world.impl().context(), 0, (parsec_tiled_matrix_dc_t *)&dcB, random_seed+1);

    two_dim_block_cyclic_t dcX;
    two_dim_block_cyclic_init(&dcX, matrix_type::matrix_RealDouble, matrix_storage::matrix_Tile,
                              world.size(), world.rank(), NB, NB, N, M,
                              0, 0, N, M, 1, 1, P);
    dcX.mat = parsec_data_allocate((size_t)dcX.super.nb_local_tiles *
                                  (size_t)dcX.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcX.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcX, "Matrix X");
    dplasma_dlacpy( world.impl().context(), dplasmaUpperLower,
                    (parsec_tiled_matrix_dc_t *)&dcB, (parsec_tiled_matrix_dc_t *)&dcX );

    dplasma_dpotrs(world.impl().context(), matrix_Lower,
                    (parsec_tiled_matrix_dc_t *)&dcA,
                    (parsec_tiled_matrix_dc_t *)&dcX );

    ret |= check_daxmb( world.impl().context(), (world.rank() == 0) ? loud : 0, matrix_Lower,
                        (parsec_tiled_matrix_dc_t *)&dcA0,
                        (parsec_tiled_matrix_dc_t *)&dcB,
                        (parsec_tiled_matrix_dc_t *)&dcX);

    /* Cleanup */
    parsec_data_free(dcA0.mat); dcA0.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0 );
    parsec_data_free(dcB.mat); dcB.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB );
    parsec_data_free(dcX.mat); dcX.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX );
  }
#endif // USE_DPLASMA

  //delete A;
  /* cleanup allocated matrix before shutting down PaRSEC */
  parsec_data_free(dcA.mat); dcA.mat = NULL;
  parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

  world.profile_off();

  ttg::finalize();
  return 0;
}

#if defined(USE_DPLASMA)
/**
 *******************************************************************************
 *
 * @ingroup dplasma_double_check
 *
 * check_dpotrf - Check the correctness of the Cholesky factorization computed
 * Cholesky functions with the following criteria:
 *
 *    \f[ ||L'L-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  or
 *
 *    \f[ ||UU'-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  where A is the original matrix, and L, or U, the result of the Cholesky
 *  factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A and A0 are referenced;
 *          = dplasmaLower: Lower triangle of A and A0 are referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == dplasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == dplasmaLower, the
 *          lower part is referenced.
 *
 * @param[in] A0
 *          Descriptor of the original distributed matrix A before
 *          factorization. If uplo == dplasmaUpper, the only the upper part is
 *          referenced, otherwise if uplo == dplasmaLower, the lower part is
 *          referenced.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_dpotrf( parsec_context_t *parsec, int loud,
                  dplasma_enum_t uplo,
                  parsec_tiled_matrix_dc_t *A,
                  parsec_tiled_matrix_dc_t *A0 )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A0;
    two_dim_block_cyclic_t LLt;
    int info_factorization;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double result = 0.0;
    int M = A->m;
    int N = A->n;
    double eps = std::numeric_limits<double>::epsilon();
    dplasma_enum_t side;

    two_dim_block_cyclic_init(&LLt, matrix_RealDouble, matrix_Tile,
                              ttg::default_execution_context().size(), twodA->grid.rank,
                              A->mb, A->nb,
                              M, N,
                              0, 0,
                              M, N,
                              twodA->grid.krows, twodA->grid.kcols,
                              twodA->grid.rows /*twodA->grid.ip, twodA->grid.jq*/);

    LLt.mat = parsec_data_allocate((size_t)LLt.super.nb_local_tiles *
                                  (size_t)LLt.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(LLt.super.mtype));

    dplasma_dlaset( parsec, dplasmaUpperLower, 0., 0.,(parsec_tiled_matrix_dc_t *)&LLt );
    dplasma_dlacpy( parsec, uplo, A, (parsec_tiled_matrix_dc_t *)&LLt );

    /* Compute LL' or U'U  */
    side = (uplo == dplasmaUpper ) ? dplasmaLeft : dplasmaRight;
    dplasma_dtrmm( parsec, side, uplo, dplasmaTrans, dplasmaNonUnit, 1.0,
                   A, (parsec_tiled_matrix_dc_t*)&LLt);

    /* compute LL' - A or U'U - A */
    dplasma_dtradd( parsec, uplo, dplasmaNoTrans,
                    -1.0, A0, 1., (parsec_tiled_matrix_dc_t*)&LLt);

    Anorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo, A0);
    Rnorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo,
                           (parsec_tiled_matrix_dc_t*)&LLt);

    result = Rnorm / ( Anorm * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Cholesky factorization \n");

        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||L'L-A||_oo = %e\n", Anorm, Rnorm );

        printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", result);
    }

    if ( std::isnan(Rnorm)  || std::isinf(Rnorm)  ||
         std::isnan(result) || std::isinf(result) ||
         (result > 60.0) )
    {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else
    {
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    parsec_data_free(LLt.mat); LLt.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&LLt);

    return info_factorization;
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_double_check
 *
 * check_daxmb - Returns the result of the following test
 *
 *    \f[ (|| A x - b ||_oo / ((||A||_oo * ||x||_oo + ||b||_oo) * N * eps) ) < 60. \f]
 *
 *  where A is the original matrix, b the original right hand side, and x the
 *  solution computed through any factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A is referenced;
 *          = dplasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == dplasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == dplasmaLower, the
 *          lower part is referenced.
 *
 * @param[in,out] b
 *          Descriptor of the original distributed right hand side b.
 *          On exit, b is overwritten by (b - A * x).
 *
 * @param[in] x
 *          Descriptor of the solution to the problem, x.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_daxmb( parsec_context_t *parsec, int loud,
                 dplasma_enum_t uplo,
                 parsec_tiled_matrix_dc_t *A,
                 parsec_tiled_matrix_dc_t *b,
                 parsec_tiled_matrix_dc_t *x )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int N = b->m;
    double eps = std::numeric_limits<double>::epsilon();

    Anorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo, A);
    Bnorm = dplasma_dlange(parsec, dplasmaInfNorm, b);
    Xnorm = dplasma_dlange(parsec, dplasmaInfNorm, x);

    /* Compute b - A*x */
    dplasma_dsymm( parsec, dplasmaLeft, uplo, -1.0, A, x, 1.0, b);

    Rnorm = dplasma_dlange(parsec, dplasmaInfNorm, b);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (std::isnan(Xnorm) || std::isinf(Xnorm) || std::isnan(result) || std::isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
#endif /* USE_DPLASMA */

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
