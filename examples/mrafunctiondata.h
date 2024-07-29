#ifndef MADFUNCTIONDATA_H_INCL
#define MADFUNCTIONDATA_H_INCL

#include "mratypes.h"
#include "mradomain.h"
#include "mrasimpletensor.h"

namespace mra {

    /// Convenient co-location of frequently used data
    template <typename T, size_t K, size_t NDIM>
    class FunctionData {
        inline static bool initialized = false;
        inline static SimpleTensor<T,K,K> phi; // phi(mu,i) = phi(x[mu],i) --- value of scaling functions at quadrature points on level 0
        inline static SimpleTensor<T,K,K> phibar; // phibar(mu,i) = w[mu]*phi(x[mu],i)
        inline static SimpleTensor<T,2*K,2*K> HG; // Two scale filter applied from left to scaling function coeffs
        inline static SimpleTensor<T,2*K,2*K> HGT; // Two scale filter applied from right to scaling function coeffs
        inline static std::array<T,K> x, w; // Quadrature points and weights on level 0
        inline static std::array<std::array<Slice,NDIM>, Key<NDIM>::num_children> child_slices; // Maps index of child into sub-cube of parent coeffs
        inline static std::array<Slice,NDIM> child0_slice; // Slice for first child which is also scaling function coeffs in full tensor of difference coefficients
        inline static SimpleTensor<T,K,K> rm, r0, rp; // blocks of the ABGV central derivative operator

        static void make_abgv_diff_operator() {
            double iphase = 1.0;
            for (auto i: range(K)) {
                double jphase = 1.0;
                for (auto j : range(K)) {
                    double gammaij = std::sqrt(double((2*i+1)*(2*j+1)));
                    double Kij;
                    if (((i-j)>0) && (((i-j)%2)==1))
                        Kij = 2.0;
                    else
                        Kij = 0.0;

                    r0(i,j) = T(0.5*(1.0 - iphase*jphase - 2.0*Kij)*gammaij);
                    rm(i,j) = T(0.5*jphase*gammaij);
                    rp(i,j) = T(-0.5*iphase*gammaij);
                }
            }
        }
        
        
        /// Make slices that index child into sub-cube of parent
        static std::array<std::array<Slice,NDIM>, Key<NDIM>::num_children>  make_child_slices() {
            std::array<std::array<Slice,NDIM>, Key<NDIM>::num_children> result;
            for (size_t child : range(Key<NDIM>::num_children)) {
                std::array<Slice,NDIM>& slices = result[child];
                for (size_t d : range(NDIM)) {
                    size_t b = get_bit(child,d);
                    slices[d] = Slice(K*b, K*(b+1));
                }
            }
            return result;
        }
            
        /// Set phi(mu,i) to be phi(x[mu],i)
        static void make_phi(FixedTensor<T,K,2>& phi) {
            std::array<T,K> x, w, p;
            detail::GLget(x,w);
            for (size_t mu : range(K)) {
                detail::legendre_scaling_functions(x[mu], K, &p[0]);
                for (size_t i : range(K)) {
                    phi(mu,i) = p[i];
                }
            }
        }
        
        /// Set phibar(mu,i) to be w[mu]*phi(x[mu],i)
        static void make_phibar(FixedTensor<T,K,2>& phibar) {
            std::array<T,K> x, w, p;
            detail::GLget(x,w);
            for (size_t mu : range(K)) {
                detail::legendre_scaling_functions(x[mu], K, &p[0]);
                for (size_t i : range(K)) {
                    phibar(mu,i) = w[mu]*p[i];
                }
            }
            // FixedTensor<T,K,2> phi, r;
            // make_phi<T,K>(phi);
            // mTxmq(K, K, K, r.ptr(), phi.ptr(), phibar.ptr());
            // std::cout << r << std::endl; // should be identify matrix
        }
        
    public:

        static void initialize() {
            make_phi(phi);
            make_phibar(phibar);
            twoscale_get(K, HG.ptr());
            for (size_t i : range(2*K)) {
                for (size_t j : range(2*K)) {
                    HGT(j,i) = HG(i,j);
                }
            }
            detail::GLget(x,w);
            child_slices = make_child_slices();
            child0_slice = child_slices[0];
            make_abgv_diff_operator();
            initialized = true;
        }

        static const auto& get_phi() {assert(initialized); return phi;}
        static const auto& get_phibar() {assert(initialized); return phibar;}
        static const auto& get_hg() {assert(initialized); return HG;}
        static const auto& get_hgT() {assert(initialized); return HGT;}
        static const auto& get_x() {assert(initialized); return x;}
        static const auto& get_w() {assert(initialized); return w;}
        static const auto& get_child_slices() {assert(initialized); return child_slices;}
        static const auto& get_child0_slice() {assert(initialized); return child0_slice;}
        static const auto& get_rm() {assert(initialized); return rm;}
        static const auto& get_r0() {assert(initialized); return r0;}
        static const auto& get_rp() {assert(initialized); return rp;}

        /// Set X(d,mu) to be the mu'th quadrature point in dimension d for the box described by key
        static void make_quadrature_pts(const Key<NDIM>& key, SimpleTensor<T,NDIM,K>& X) {
            assert(initialized);
            const Level n = key.level();
            const std::array<Translation,NDIM>& l = key.translation();
            const T h = std::pow(T(0.5),T(n));
            for (Dimension d : range(NDIM)) {
                T lo, hi; std::tie(lo,hi) = Domain<NDIM>::get(d);
                T width = h*Domain<NDIM>::get_width(d);
                for (size_t i : range(K)) {
                    X(d,i) = lo + width*(l[d] + x[i]);
                }
            }
        }
    };

}


#endif
