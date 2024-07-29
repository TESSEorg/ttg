#ifndef MADGL_H_INCL
#define MADGL_H_INCL

#include <cstddef>

#ifdef __CUDA_ARCH__
#define DEVICE __device__
#else
#define DEVICE
#endif // __CUDA_ARCH__

namespace mra {

    namespace detail {
        /// Get the points and weights for the Gauss-Legendre quadrature on [0,1].  x and w should be arrays length at least N.
        DEVICE void GLget(size_t N, const double** x, const double** w);

        /// Get the points and weights for the Gauss-Legendre quadrature on [0,1].  x and w should be arrays length at least N.
        DEVICE void GLget(size_t N, const float** x, const float** w);

        /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
        void legendre_scaling_functions(double x, size_t k, double *p);

        /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
        void legendre_scaling_functions(float x, size_t k, float *p);

        /// Get the points and weights for the Gauss-Legendre quadrature on [0,1]
        template <typename T>
        DEVICE void GLget(const T** x, const T** w, std::size_t N)  {
            if (!(N>0 && N<=64)) {
                throw std::runtime_error("Gauss-Legendre quadrature only available for up to N=64");
            }
            GLget(N, x, w);
        }
    }
}

#endif
