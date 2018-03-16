#ifndef MADFUNCTIONNODE_H_INCL
#define MADFUNCTIONNODE_H_INCL

#include <cmath>
#include <iostream>

#include "madtypes.h"
#include "maddomain.h"
#include "madsimpletensor.h"
#include "madfunctiondata.h"

namespace mad {


    /// Applies two-scale filter (sum coeffs of children to sum+difference coeffs of parent)
    template <typename T, size_t K, Dimension NDIM>
    void filter(const FixedTensor<T,2*K,NDIM>& in, FixedTensor<T,2*K,NDIM>& out) {
        auto& hgT = FunctionData<T,K,NDIM>::get_hgT();
        transform<T,T,T,2*K,NDIM>(in,hgT,out);

    }
    
    /// Applies inverse of two-scale filter (sum+difference coeffs of parent to sum coeffs of children)
    template <typename T, size_t K, Dimension NDIM>
    void unfilter(const FixedTensor<T,2*K,NDIM>& in, FixedTensor<T,2*K,NDIM>& out) {
        auto& hg = FunctionData<T,K,NDIM>::get_hg();
        transform<T,T,T,2*K,NDIM>(in,hg,out);

    }
    
    /// In given box return the truncation tolerance for given threshold
    template <typename T, Dimension NDIM>
    T truncate_tol(const Key<NDIM>& key, const T thresh) {
        return thresh; // nothing clever for now
    }
    
    /// Evaluate the function within one cube with screening
    template <typename functorT, typename T, size_t K, Dimension NDIM>
    void fcube(const functorT& f, const Key<NDIM>& key, const T thresh, FixedTensor<T,K,NDIM>& values) {
        if (f.is_negligible(Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
            values = 0.0;
        }
        else {
            SimpleTensor<T,NDIM,K> x; // When have object structure will move outside
            FunctionData<T,K,NDIM>::make_quadrature_pts(key,x);
            f(x,values);
        }
    }
    
    /// Project the scaling coefficients using screening and test norm of difference coeffs.  Return true if difference coeffs negligible.
    template <typename functorT, typename T, size_t K, Dimension NDIM>
    bool fcoeffs(const functorT& f, const Key<NDIM>& key, const T thresh, FixedTensor<T,K,NDIM>& s) {
        if (f.is_negligible(Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
            s = 0.0;
            return true;
        }
        else {
            auto& child_slices = FunctionData<T,K,NDIM>::get_child_slices();
            auto& phibar = FunctionData<T,K,NDIM>::get_phibar();
            
            FixedTensor<T,2*K,NDIM> values;
            {
                FixedTensor<T,K,NDIM> child_values;
                FixedTensor<T,K,NDIM> r;
                KeyChildren<NDIM> children(key);
                for (auto it=children.begin(); it!=children.end(); ++it) {
                    const Key<NDIM>& child = *it;
                    fcube<functorT,T,K,NDIM>(f, child, thresh, child_values);
                    transform<T,T,T,K,NDIM>(child_values,phibar,r);
                    values(child_slices[it.index()]) = r;
                }
            }
            T fac = std::sqrt(Domain<NDIM>:: template get_volume<T>()*std::pow(T(0.5),T(NDIM*(1+key.level()))));
            values *= fac;
            FixedTensor<T,2*K,NDIM> r;
            filter<T,K,NDIM>(values,r);
            s = r(child_slices[0]); // extract sum coeffs
            r(child_slices[0]) = 0.0; // zero sum coeffs so can easily compute norm of difference coeffs
            return (r.normf() < truncate_tol(key,thresh)); // test norm of difference coeffs
        }
    }

    template <typename T, size_t K, Dimension NDIM>
    class FunctionReconstructedNode {
    public: // temporarily make everything public while we figure out what we are doing
        static constexpr bool is_function_node = true;
        Key<NDIM> key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        mutable T sum; //< If recurring up tree (e.g., in compress) can use this to also compute a scalar reduction
        bool is_leaf; //< True if node is leaf on tree (i.e., no children).
        FixedTensor<T,K,NDIM> coeffs; //< if !is_leaf these are junk (and need not be communicated)
        FunctionReconstructedNode() = default; // Default initializer does nothing so that class is POD
        FunctionReconstructedNode(const Key<NDIM>& key) : key(key), sum(0.0), is_leaf(false) {}
        T normf() const {return (is_leaf ? coeffs.normf() : 0.0);}
        bool has_children() const {return !is_leaf;}
    };
    
    template <typename T, size_t K, Dimension NDIM>
    class FunctionCompressedNode {
    public: // temporarily make everything public while we figure out what we are doing
        static constexpr bool is_function_node = true;
        Key<NDIM> key; //< Key associated with this node to facilitate computation from otherwise unknown parent/child
        std::array<bool,Key<NDIM>::num_children> is_leaf; //< True if that child is leaf on tree
        FixedTensor<T,2*K,NDIM> coeffs; //< Always significant
        FunctionCompressedNode() = default; // Default initializer does nothing so that class is POD
        FunctionCompressedNode(const Key<NDIM>& key) : key(key) {}
        T normf() const {return coeffs.normf();}
        bool has_children(size_t childindex) const {assert(childindex<Key<NDIM>::num_children); return !is_leaf[childindex];}
    };
    
    template <typename T, size_t K, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionReconstructedNode<T,K,NDIM>& node) {
        s << "FunctionReconstructedNode(" << node.key << "," << node.is_leaf << "," << node.normf() << ")";
        return s;
    }

    template <typename T, size_t K, Dimension NDIM, typename ostream>
    ostream& operator<<(ostream& s, const FunctionCompressedNode<T,K,NDIM>& node) {
        s << "FunctionCompressedNode(" << node.key << "," << node.is_leaf << "," << node.normf() << ")";
        return s;
    }

}
    
#endif
