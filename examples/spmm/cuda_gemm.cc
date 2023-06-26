
#include <cublas.h>


void my_cublas_init_because_cublas_is_stupid() {
    cublasInit();
}

void my_cublas_shutdown_because_cublas_is_stupid() {
    cublasShutdown();
}

void my_cublas_dgemm_because_cublas_is_stupid(
    char transa,
    char transb,
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc)
{
  cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
