#ifndef HAVE_GAUSSIAN_H
#define HAVE_GAUSSIAN_H

#include "../../mratypes.h"
#include "../../mradomain.h"

/**
 * Defines a Gaussian functor that we use across this example.
 * We want to template the whole stack on the functor but we
 * have to verify that nvcc can compile all TTG code (incl. coroutines).
 */

namespace mra {
    // Test gaussian functor
    template <typename T, Dimension NDIM>
    class Gaussian {
        const T expnt;
        const Coordinate<T,NDIM> origin;
        const T fac;
        const T maxr;
        Level initlev;
    public:
        /* default construction required for ttg::Buffer */
        Gaussian() = default;

        Gaussian(T expnt, const Coordinate<T,NDIM>& origin)
        : expnt(expnt)
        , origin(origin)
        , fac(std::pow(T(2.0*expnt/M_PI),T(0.25*NDIM)))
        , maxr(std::sqrt(std::log(fac/1e-12)/expnt))
        {
            // Pick initial level such that average gap between quadrature points
            // will find a significant value
            const int N = 6; // looking for where exp(-a*x^2) < 10**-N
            const int K = 6; // typically the lowest order of the polyn
            const T log10 = std::log(10.0);
            const T log2 = std::log(2.0);
            const T L = Domain<NDIM>::get_max_width();
            const T a = expnt*L*L;
            double n = std::log(a/(4*K*K*(N*log10+std::log(fac))))/(2*log2);
            //std::cout << expnt << " " << a << " " << n << std::endl;
            initlev = Level(n<2 ? 2.0 : std::ceil(n));
        }

        /* default copy ctor and operator */
        Gaussian(const Gaussian&) = default;
        Gaussian& operator=(const Gaussian&) = default;

        // T operator()(const Coordinate<T,NDIM>& r) const {
        //     T rsq = 0.0;
        //     for (auto x : r) rsq += x*x;
        //     return fac*std::exp(-expnt*rsq);
        // }

        /**
         * Evaluate function at N points x and store result in \c values
         */
        void operator()(const TensorView<T,2>& x, T* values, std::size_t N) const {
            assert(x.dim(0) == NDIM);
            assert(x.dim(1) == N);
            distancesq(origin, x, values, N);
#ifdef __CUDA_ARCH__
            int tid = threadDim.x * ((threadDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
            for (size_t i = tid; i < N; i += blockDim.x*blockDim.y*blockDim.z) {
                values[i] = fac * std::exp(-expnt*values[i]);
            }
#else  // __CUDA_ARCH__
            for (T& value : values) {
                value = fac * std::exp(-expnt*value);
            }
#endif // __CUDA_ARCH__
        }

        Level initial_level() const {
            return this->initlev;
        }

        bool is_negligible(const std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>& box, T thresh) const {
            auto& lo = box.first;
            auto& hi = box.second;
            T rsq = 0.0;
            T maxw = 0.0; // max width of box
            for (Dimension d : range(NDIM)) {
                maxw = std::max(maxw,hi(d)-lo(d));
                T x = T(0.5)*(hi(d)+lo(d)) - origin(d);
                rsq += x*x;
            }
            T diagndim = T(0.5)*std::sqrt(T(NDIM));
            T boxradplusr = maxw*diagndim + maxr;
            // ttg::print(box, boxradplusr, bool(boxradplusr*boxradplusr < rsq));
            return (boxradplusr*boxradplusr < rsq);
        }
    };
} // namespace mra
#endif // HAVE_GAUSSIAN_H