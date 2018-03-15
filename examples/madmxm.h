#ifndef MADMXM_H_INCL
#define MADMXM_H_INCL

#include <iostream>
#include <cassert>
#include <mkl.h>

namespace mad {

    namespace detail {
    
        // Need to add complex and mixed versions (the latter might require using the Fortran BLAS API)

        static inline void gemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                                 const MKL_INT m, const MKL_INT n, const MKL_INT k,
                                 const float alpha, const float *a, const MKL_INT lda, const float *b, const MKL_INT ldb,
                                 const float beta, float *c, const MKL_INT ldc) {
            cblas_sgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        
        static inline void gemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                                 const MKL_INT m, const MKL_INT n, const MKL_INT k,
                                 const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb,
                                 const double beta, double *c, const MKL_INT ldc) {
            cblas_dgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    }
    
    
    /// Matrix = Matrix transpose * matrix ... MKL interface version
    
    /// Does \c C=AT*B
    /// \code
    ///    c(i,j) = sum(k) a(k,i)*b(k,j)  <------ does not accumulate into C
    /// \endcode
    ///
    /// \c ldb is the last dimension of \c b in C-storage (the leading dimension
    /// in fortran storage).  It is here to accomodate multiplying by a matrix
    /// stored with \c ldb>dimj which happens in madness when transforming with
    /// low rank matrices.  A matrix in dense storage has \c ldb=dimj which is
    /// the default for backward compatibility.
    template <typename aT, typename bT, typename cT>
    void mTxmq(size_t dimi, size_t dimj, size_t dimk,
               cT* c, const aT* a, const bT* b, size_t ldb=std::numeric_limits<size_t>::max()) {
        if (ldb == std::numeric_limits<size_t>::max()) ldb=dimj;
        assert(ldb>=dimj);

        if (dimi==0 || dimj==0) return; // nothing to do and *GEMM will complain
        if (dimk==0) {
            for (size_t i=0; i<dimi*dimj; i++) c[i] = 0.0;
        }
        
        const cT one = 1.0;  // alpha in *gemm
        const cT zero = 0.0; // beta  in *gemm

        detail::gemm(CblasRowMajor, CblasTrans, CblasNoTrans, dimi, dimj, dimk, one, a, dimi, b, ldb, zero, c, dimj);
    }
}

#endif
