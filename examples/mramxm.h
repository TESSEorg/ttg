#ifndef MADMXM_H_INCL
#define MADMXM_H_INCL

#include <cassert>
#include <complex>
#include <iostream>
#include "blas.hh"

namespace mra {

    namespace detail {

#ifdef MKL_INT
        using cblas_int = MKL_INT;
#else
        using cblas_int = int;
#endif

        // Need to add complex and mixed versions (the latter might require using the Fortran BLAS API)

        static inline void gemm (const blas::Layout Layout, const blas::Op transa, const blas::Op transb,
                                 const cblas_int m, const cblas_int n, const cblas_int k,
                                 const float alpha, const float *a, const cblas_int lda, const float *b, const cblas_int ldb,
                                 const float beta, float *c, const cblas_int ldc) {
            blas::gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        
        static inline void gemm (const blas::Layout Layout, const blas::Op transa, const blas::Op transb,
                                 const cblas_int m, const cblas_int n, const cblas_int k,
                                 const double alpha, const double *a, const cblas_int lda, const double *b, const cblas_int ldb,
                                 const double beta, double *c, const cblas_int ldc) {
            blas::gemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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

        detail::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans, dimi, dimj, dimk, one, a, dimi, b, ldb, zero, c, dimj);
    }

    /*static inline void vscale(size_t N, float s, float* x) {
        cblas_sscal(detail::cblas_int(N), s, x, detail::cblas_int(1));
    }

    static inline void vscale(size_t N, double s, double* x) {
        cblas_dscal(detail::cblas_int(N), s, x, detail::cblas_int(1));
    }

    static inline void vscale(size_t N, float s, std::complex<float>* x) {
        cblas_csscal(detail::cblas_int(N), s, x, detail::cblas_int(1));
    }

    static inline void vscale(size_t N, double s, std::complex<double>* x) {
        cblas_zdscal(detail::cblas_int(N), s, x, detail::cblas_int(1));
    }

    static inline void vscale(size_t N, const std::complex<float>& s, std::complex<float>* x) {
        cblas_cscal(detail::cblas_int(N), &s, x, detail::cblas_int(1));
    }

    static inline void vscale(size_t N, const std::complex<double>& s, double* x) {
        cblas_zscal(detail::cblas_int(N), &s, x, detail::cblas_int(1));
    }

    static inline void vexp(size_t N, const float* a, float* y) {
        vsExp(detail::cblas_int(N), a, y);
    }

    static inline void vexp(size_t N, const double* a, double* y) {
        vdExp(detail::cblas_int(N), a, y);
    }

    static inline void vexp(size_t N, const std::complex<float>& a, std::complex<float>* y) {
        vcExp(detail::cblas_int(N), (const MKL_Complex8*)(&a), (MKL_Complex8*)(y));
    }

    static inline void vexp(size_t N, const std::complex<double>& a, std::complex<double>* y) {
        vzExp(detail::cblas_int(N), (const MKL_Complex16*)(&a), (MKL_Complex16*)(y));
    }*/
    
}

#endif
