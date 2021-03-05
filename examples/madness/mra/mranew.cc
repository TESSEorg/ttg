#include <map>
#include <cmath>
#include <array>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>
#include <type_traits>

#include "mragl.h"
#include "mrakey.h"
#include "mramxm.h"
#include "mramisc.h"
#include "mratypes.h"
#include "mradomain.h"
#include "mratwoscale.h"
#include "mrasimpletensor.h"
#include "mrafunctiondata.h"
#include "mrafunctionnode.h"
#include "mrafunctionfunctor.h"

using namespace mra;

// Sums Frobenius norm up the tree for testing
template <typename functorT, typename T, size_t K, Dimension NDIM>
T project_function_node(functorT& f, const Key<NDIM>& key, const T thresh) {

    FunctionNode<T,K,NDIM> node(key); // Our eventual result which right now is being discarded

    auto& coeffs = node.coeffs; // Need to clean up OO design
    auto& is_leaf = node.is_leaf;

    T normsq = 0.0;
    
    if (f.is_negligible(Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
        coeffs = 0.0;
        for (auto& x : is_leaf) x = true;
    }
    else {
        auto& child_slices = FunctionData<T,K,NDIM>::get_child_slices();
        FixedTensor<T,2*K,NDIM> values;
        FixedTensor<T,K,NDIM> child_coeffs;
        FixedTensor<T,K,NDIM> r;
        KeyChildren<NDIM> children(key);
        for (auto it=children.begin(); it!=children.end(); ++it) {
            const Key<NDIM>& child = *it;
            bool small = fcoeffs<functorT,T,K,NDIM>(f, child, thresh, child_coeffs);
            //std::cout << "snorm " << child_coeffs.normf() << std::endl;
            is_leaf[it.index()] = (small || key.level() == MAX_LEVEL);
            //std::cout << "x " << child << " " << small << " " << it.index() << " " << is_leaf[it.index()] << " " << child_coeffs.normf() << std::endl;
            if (is_leaf[it.index()]) {
                coeffs(child_slices[it.index()]) = child_coeffs;
                normsq += child_coeffs.sumabssq();
            }
            else {
                coeffs(child_slices[it.index()]) = 0.0;
                normsq += project_function_node<functorT,T,K,NDIM>(f, child, thresh);
            }
        }
    }
    //for (size_t n=0; n<key.level(); n++) std::cout << "  ";
    //std::cout << key << " " << normsq << std::endl;
    return normsq;
}

// For checking we haven't broken something while developing
template <typename T>
struct is_serializable {
    static const bool value = std::is_fundamental<T>::value || std::is_member_function_pointer<T>::value || std::is_function<T>::value  || std::is_function<typename std::remove_pointer<T>::type>::value || std::is_pod<T>::value;
};

// Test gaussian function
template <typename T, Dimension NDIM>
T g(const Coordinate<T,NDIM>& r) {
    static const T expnt = 3.0;
    static const T fac = std::pow(T(2.0*expnt/M_PI),T(0.25*NDIM)); // makes square norm over all space unity
    T rsq = 0.0;
    for (auto x : r) rsq += x*x;
    return fac*std::exp(-expnt*rsq);
}

// Test the numerics
template <typename T, size_t K, Dimension NDIM>
void test_gaussian(T thresh) {
    Domain<NDIM>::set_cube(-5.0,5.0);
    FunctionData<T,K,NDIM>::initialize();
    FunctionFunctor<T, NDIM> ff(g<T,NDIM>);
    Key<NDIM> root(0,{});
    T normsq = project_function_node<decltype(ff), T, K, NDIM>(ff,root,thresh);
    std::cout << "normsq error " << normsq-1.0 << std::endl;
}

int main() {

    // int fred[2];
    // static_assert(std::is_pod<int [2]>::value, "int[2] is not POD??"); // yes
    // static_assert(std::is_pod<std::array<float,2>>>::value, "std::array is not POD??");// no
    // static_assert(std::is_trivial<std::array<float,2>>>::value, "std::array is not trivial??"); // no
    // static_assert(is_serializable<std::array<float,2>>>::value, "std::array is not serializable??"); // no
    static_assert(is_serializable<Key<2>>::value, "You just did something that stopped Key from being serializable"); // yes
    static_assert(is_serializable<SimpleTensor<float,2,2>>::value,"You just did something that stopped SimpleTensor from being serializable"); // yes
    static_assert(is_serializable<FunctionNode<float,2,2>>::value,"You just did something that stopped FunctionNode from being serializable"); // yes
    
    GLinitialize();

    test_gaussian<double,5,3>(1e-6);
    test_gaussian<float,8,3>(1e-1);
    test_gaussian<float,8,3>(1e-2);
    test_gaussian<float,8,3>(1e-3);
    test_gaussian<float,8,3>(1e-4);
    test_gaussian<float,8,3>(1e-5);
    test_gaussian<float,8,3>(1e-6);
    
    return 0;
}

