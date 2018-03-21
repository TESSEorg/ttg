#ifndef MAD_FUNCTION_FUNCTOR_H_INCL
#define MAD_FUNCTION_FUNCTOR_H_INCL

#include "madtypes.h"
#include "madsimpletensor.h"

namespace mra {
    
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
        
        static const Level default_initial_level = 2;
    public:
        template <typename functionT>
        FunctionFunctor(functionT f) : f(f) {}
        
        /// Evaluate at a single point
        T operator()(const Coordinate<T,NDIM>& r) const {return f(r);} 
        
        /// Evaluate at points formed by tensor product of npt points in each dimension
        template <size_t K>
        void operator()(const SimpleTensor<T,1,K>& x, SimpleTensor<T,K>& values) const {
            for (size_t i=0; i<K; i++) values(i) = f(Coordinate<T,1>{x(0,i)});
        }
        
        /// Evaluate at points formed by tensor product of npt points in each dimension
        template <size_t K>
        void operator()(const SimpleTensor<T,2,K>& x, SimpleTensor<T,K,K>& values) const {
            for (size_t i=0; i<K; i++) {
                for (size_t j=0; j<K; j++) {
                    values(i,j) = f(Coordinate<T,2>{x(0,i),x(1,j)});
                }
            }
        }
        
        /// Evaluate at points formed by tensor product of K points in each dimension
        template <size_t K>
        void operator()(const SimpleTensor<T,3,K>& x, SimpleTensor<T,K,K,K>& values) const {
            for (size_t i=0; i<K; i++) {
                for (size_t j=0; j<K; j++) {
                    for (size_t k=0; k<K; k++) {
                        values(i,j,k) = f(Coordinate<T,3>{x(0,i),x(1,j),x(2,k)});
                    }
                }
            }
        }
        
        Level initial_level() const {return default_initial_level;}
        
        /// Return true if the function is negligible in the volume described by the bounding box (opposite corners of the box)
        bool is_negligible(const std::pair<Coordinate<T,NDIM>,Coordinate<T,NDIM>>& box, T thresh) const {
            return false;
        }
    };
}

#endif
