#ifndef MADGL_H_INCL
#define MADGL_H_INCL

#include <array>

#include "mramxm.h"
#include "mrakey.h"
#include "mradomain.h"
#include "mrasimpletensor.h"

namespace mra {

    namespace detail {
        /// Get the points and weights for the Gauss-Legendre quadrature on [0,1].  x and w should be arrays length at least N.
        void GLget(size_t N, double* x, double* w);
        
        /// Get the points and weights for the Gauss-Legendre quadrature on [0,1].  x and w should be arrays length at least N.
        void GLget(size_t N, float* x, float* w);
        
        /// Get the points and weights for the Gauss-Legendre quadrature on [0,1]
        template <typename T, size_t N>
        void GLget(std::array<T,N>& x, std::array<T,N>& w)  {
            static_assert(N>0 && N<=64, "Gauss-Legendre quadrature only available for up to N=64");
            GLget(N, &x[0], &w[0]);
        }
        
        /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
        void legendre_scaling_functions(double x, size_t k, double *p);
        
        /// Evaluate the first k Legendre scaling functions. p should be an array of k elements.
        void legendre_scaling_functions(float x, size_t k, float *p);
    }
    
    /// Call this single threaded at start of program to initialize static data ... returns status of self-check (true = OK)
    bool GLinitialize();
}

#endif
