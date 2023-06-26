

/* We need wrap cublas v1 functions because some header (blaspp)
 * includes cublas_v2 and cublas is braindead and overwrites the v1
 * signatures. What a clusterfuck. */

void my_cublas_init_because_cublas_is_stupid();

void my_cublas_shutdown_because_cublas_is_stupid();

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
    int ldc);