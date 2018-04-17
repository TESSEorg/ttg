#ifndef MAD_FUNCTION_FUNCTOR_H_INCL
#define MAD_FUNCTION_FUNCTOR_H_INCL

#include "mratypes.h"
#include "mrasimpletensor.h"

namespace mra {
    

    /// Function functor is going away ... don't use!
    /// \code
    ///template <typename T, Dimension NDIM>
    /// class FunctionFunctor {
    ///public:
    /// T operator()(const Coordinate<T,NDIM>& r) const;
    /// template <size_t K> void operator()(const SimpleTensor<T,NDIM,K>& x, FixedTensor<T,K,NDIM>& values) const;
    /// Level initial_level() const;
    /// bool is_negligible(const std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>& box, T thresh) const;
    /// special point interface to be added
    ///  }
    /// \endcode
    
    /// Adapts a simple callable to the API needed for evaluation --- implement your own for full vectorization
    template <typename T, Dimension NDIM>
    class FunctionFunctor {
        std::function<T(const Coordinate<T,NDIM>&)> f;
        
    public:
        static const Level default_initial_level = 3; //< needs to become user configurable
        
        template <typename functionT>
        FunctionFunctor(functionT f) : f(f) {}
        
        /// Evaluate at a single point
        T operator()(const Coordinate<T,NDIM>& r) const {return f(r);} 

    };

    /// Evaluate at points formed by tensor product of npt points in each dimension
    template <typename functorT, typename T, size_t K> void eval_cube(const functorT& f, const SimpleTensor<T,1,K>& x, SimpleTensor<T,K>& values) {
        for (size_t i=0; i<K; i++) values(i) = f(Coordinate<T,1>{x(0,i)});
    }
        
    /// Evaluate at points formed by tensor product of npt points in each dimension
    template <typename functorT, typename T, size_t K> void eval_cube(const functorT& f, const SimpleTensor<T,2,K>& x, SimpleTensor<T,K,K>& values) {
        for (size_t i=0; i<K; i++) {
            for (size_t j=0; j<K; j++) {
                values(i,j) = f(Coordinate<T,2>{x(0,i),x(1,j)});
            }
        }
    }
        
    /// Evaluate at points formed by tensor product of K points in each dimension
    template <typename functorT, typename T, size_t K> void eval_cube(const functorT& f, const SimpleTensor<T,3,K>& x, SimpleTensor<T,K,K,K>& values) {
        for (size_t i=0; i<K; i++) {
            for (size_t j=0; j<K; j++) {
                for (size_t k=0; k<K; k++) {
                    values(i,j,k) = f(Coordinate<T,3>{x(0,i),x(1,j),x(2,k)});
                }
            }
        }
    }

    /// Evaluate at points formed by tensor product of K points in each dimension using vectorized form
    template <typename functorT, typename T, size_t K> void eval_cube_vec(const functorT& f, const SimpleTensor<T,1,K>& x, std::array<T,K>& values) {
        for (size_t i=0; i<K; i++) {
            values[i] = f(x(0,i));
        }
    }

    /// Evaluate at points formed by tensor product of K points in each dimension using vectorized form
    template <typename functorT, typename T, size_t K2NDIM> void eval_cube_vec(const functorT& f, const SimpleTensor<T,2,K2NDIM>& x, std::array<T,K2NDIM>& values) {
        for (size_t i=0; i<K2NDIM; i++) {
            values[i] = f(x(0,i),x(1,i));
        }
    }

    /// Evaluate at points formed by tensor product of K points in each dimension using vectorized form
    template <typename functorT, typename T, size_t K2NDIM> void eval_cube_vec(const functorT& f, const SimpleTensor<T,3,K2NDIM>& x, std::array<T,K2NDIM>& values) {
        for (size_t i=0; i<K2NDIM; i++) {
            values[i] = f(x(0,i),x(1,i),x(2,i));
        }
    }
    
    /// Make outer product of quadrature points for vectorized algorithms
    template <typename T, size_t K> void make_xvec(const SimpleTensor<T,1,K>& x, SimpleTensor<T,1,K>& xvec) {
        xvec = x;
    }

    /// Make outer product of quadrature points for vectorized algorithms
    template <typename T, size_t K> void make_xvec(const SimpleTensor<T,2,K>& x, SimpleTensor<T,2,K*K>& xvec) {
        size_t ij = 0;
        for (size_t i=0; i<K; i++) {
            for (size_t j=0; j<K; j++,ij++) {
                    xvec(0,ij) = x(0,i);
                    xvec(1,ij) = x(1,j);
            }
        }
    }

    /// Make outer product of quadrature points for vectorized algorithms
    template <typename T, size_t K> void make_xvec(const SimpleTensor<T,3,K>& x, SimpleTensor<T,3,K*K*K>& xvec) {
        size_t ijk = 0;
        for (size_t i=0; i<K; i++) {
            for (size_t j=0; j<K; j++) {
                for (size_t k=0; k<K; k++,ijk++) {
                    xvec(0,ijk) = x(0,i);
                    xvec(1,ijk) = x(1,j);
                    xvec(2,ijk) = x(2,k);
                }
            }
        }
    }

    namespace detail {
        template <class functorT> using initial_level_t =
            decltype(std::declval<const functorT>().initial_level());
        template <class functorT> using supports_initial_level =
            ::mra::is_detected<initial_level_t,functorT>;
        
        template <class functorT, class pairT> using is_negligible_t =
            decltype(std::declval<const functorT>().is_negligible(std::declval<pairT>(),std::declval<double>()));
        template <class functorT, class pairT> using supports_is_negligible =
            ::mra::is_detected<is_negligible_t,functorT,pairT>;
    }
    
    template <typename functionT> Level initial_level(const functionT& f) {
        if constexpr (detail::supports_initial_level<functionT>()) return f.initial_level();
        else return 2; // <<<<<<<<<<<<<<< needs updating to make user configurable
    }
    
    template <typename functionT, typename T, Dimension NDIM>
    bool is_negligible(const functionT& f, const std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>& box, T thresh) {
        using pairT = std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>;
        if constexpr (detail::supports_is_negligible<functionT,pairT>()) return f.is_negligible(box, thresh);
        else return false;
    }
}
#endif
