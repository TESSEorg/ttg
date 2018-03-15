// TTG AND MADNESS RUNTIME STUFF

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/worldmutex.h>
#include "madness/ttg.h"
// using namespace madness; // don't want this to avoid collisions with new mad stuff  
using namespace madness::ttg;
using namespace ::ttg;

// APPLICATION STUFF BELOW
#include <cmath>
#include <array>
#include <mutex>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>
#include <type_traits>

#include "madgl.h"
#include "madkey.h"
#include "madmxm.h"
#include "madmisc.h"
#include "madtypes.h"
#include "maddomain.h"
#include "madtwoscale.h"
#include "madsimpletensor.h"
#include "madfunctiondata.h"
#include "madfunctionnode.h"
#include "madfunctionfunctor.h"

using namespace mad;

template <size_t NDIM>
struct KeyProcMap {
    const size_t size;
    KeyProcMap() : size(get_default_world().size()) {}
    std::size_t operator()(const Key<NDIM>& key) const {return key.hash() % size;}
};


/// An empty class used for pure control flows
struct Control {};
std::ostream& operator<<(std::ostream& s, const Control& ctl) {s << "Ctl"; return s;}

template <typename T, size_t K, Dimension NDIM> using rnodeEdge = Edge<Key<NDIM>, FunctionReconstructedNode<T,K,NDIM>>;
template <typename T, size_t K, Dimension NDIM> using cnodeEdge = Edge<Key<NDIM>, FunctionCompressedNode<T,K,NDIM>>;
template <Dimension NDIM> using doubleEdge = Edge<Key<NDIM>, double>;
template <Dimension NDIM> using ctlEdge = Edge<Key<NDIM>, Control>;

template <typename T, size_t K, Dimension NDIM> using rnodeOut = Out<Key<NDIM>, FunctionReconstructedNode<T,K,NDIM>>;
template <typename T, size_t K, Dimension NDIM> using cnodeOut = Out<Key<NDIM>, FunctionCompressedNode<T,K,NDIM>>;
template <Dimension NDIM> using doubleOut = Out<Key<NDIM>, double>;
template <Dimension NDIM> using ctlOut = Out<Key<NDIM>, Control>;

std::mutex printer_guard;
template <typename keyT, typename valueT>
auto make_printer(const Edge<keyT, valueT>& in, const char* str = "") {
    auto func = [str](const keyT& key, const valueT& value, std::tuple<>& out) {
        std::lock_guard<std::mutex> obolus(printer_guard);
        std::cout << str << " (" << key << "," << value << ")" << std::endl;
    };
    return wrap(func, edges(in), edges(), "printer", {"input"});
}

template <Dimension NDIM>
auto make_start(const ctlEdge<NDIM>& ctl) {
    auto func = [](const Key<NDIM>& key, std::tuple<ctlOut<NDIM>>& out) { send<0>(key, Control(), out); };
    return wrap<Key<NDIM>>(func, edges(), edges(ctl), "start", {}, {"control"});
}


// Factory function to assist in wrapping a callable with signature
//
// void op(const input_keyT&, std::tuple<input_valuesT...>&&, output_terminals_tuple_type&)
template<typename keyT, typename output_terminals_tuple_type, typename...input_valuesT>
auto wrapx(std::function<void (const keyT&,  std::tuple<input_valuesT...>&&, output_terminals_tuple_type&)>&& func,
           const std::string &name="wrapper",
           const std::vector<std::string> &innames = std::vector<std::string>(std::tuple_size<std::tuple<input_valuesT...>>::value, "input"),
           const std::vector<std::string> &outnames= std::vector<std::string>(std::tuple_size<output_terminals_tuple_type>::value, "output"))
{
    using funcT = std::function<void (const keyT&,  std::tuple<input_valuesT...>&&, output_terminals_tuple_type&)>;
    using wrapT = WrapOp<funcT, keyT, output_terminals_tuple_type, input_valuesT...>;
    return std::make_unique<wrapT>(std::forward<funcT>(func), name, innames, outnames);
}

/// Constructs an operator that adaptively projects the provided function into the basis

/// Returns an std::unique_ptr to the object
template <typename functorT, typename T, size_t K, Dimension NDIM>
auto make_project(functorT& f,
                  const T thresh,
                  ctlEdge<NDIM>& ctl,
                  rnodeEdge<T,K,NDIM>& result,
                  const std::string& name = "project") {
    
    auto F = [f, thresh](const Key<NDIM>& key, Control&& junk, std::tuple<ctlOut<NDIM>, rnodeOut<T,K,NDIM>>& out) {
        FunctionReconstructedNode<T,K,NDIM> node(key); // Our eventual result
        auto& coeffs = node.coeffs; // Need to clean up OO design
        
        if (key.level() < f.initial_level()) {
            for (auto child : children(key)) send<0>(child, Control(), out);
            node.is_leaf = false;
        }
        else if (f.is_negligible(Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
            node.is_leaf = true;
        }
        else {
            node.is_leaf = fcoeffs<functorT,T,K,NDIM>(f, key, thresh, coeffs);
            if (!node.is_leaf) {
                for (auto child : children(key)) send<0>(child,Control(),out); // should be broadcast ?
            }
        }
        send<1>(key, node, out); // always produce a result
    };
    ctlEdge<NDIM> refine("refine");
    return wrap(F, edges(fuse(refine, ctl)), edges(refine, result), name, {"control"}, {"refine", "result"});
}

namespace detail {
    template <typename T, size_t K, Dimension NDIM>  struct tree_types{};

    // Can get clever and do this recursively once we know what we want
    template <typename T, size_t K>  struct tree_types<T,K,1>{
        using Rout = rnodeOut<T,K,1>;
        using Rin = FunctionReconstructedNode<T,K,1>;
        using compress_out_type = std::tuple<Rout,Rout,cnodeOut<T,K,1>>;
        using compress_in_type  = std::tuple<Rin, Rin>;
        template <typename compfuncT>
        using compwrap_type = WrapOp<compfuncT, Key<1>, compress_out_type, Rin, Rin>;
    };

    template <typename T, size_t K>  struct tree_types<T,K,2>{
        using Rout = rnodeOut<T,K,2>;
        using Rin = FunctionReconstructedNode<T,K,2>;
        using compress_out_type = std::tuple<Rout,Rout,Rout,Rout,cnodeOut<T,K,2>>;
        using compress_in_type  = std::tuple<Rin, Rin, Rin, Rin>;
        template <typename compfuncT>
        using compwrap_type = WrapOp<compfuncT, Key<2>, compress_out_type, Rin, Rin, Rin, Rin>;
    };
    
    template <typename T, size_t K>  struct tree_types<T,K,3>{
        using Rout = rnodeOut<T,K,3>;
        using Rin = FunctionReconstructedNode<T,K,3>;
        using compress_out_type = std::tuple<Rout,Rout,Rout,Rout,Rout,Rout,Rout,Rout,cnodeOut<T,K,3>>;
        using compress_in_type  = std::tuple<Rin, Rin, Rin, Rin, Rin, Rin, Rin, Rin>;
        template <typename compfuncT>
        using compwrap_type = WrapOp<compfuncT, Key<3>, compress_out_type, Rin, Rin, Rin, Rin, Rin, Rin, Rin, Rin>;
    };
};    

// Stream leaf nodes up the tree as a prelude to compressing
template <typename T, size_t K, Dimension NDIM>
void send_leaves_up(const Key<NDIM>& key,
                    const std::tuple<FunctionReconstructedNode<T,K,NDIM>>& inputs,
                    typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
    const FunctionReconstructedNode<T,K,NDIM>& node = std::get<0>(inputs);
    node.sum = 0.0;   // 
    if (!node.has_children()) { // We are only interested in the leaves
        if (key.level() == 0) {  // Tree is just one node
            throw "not yet";
            // rnodeOut& result = std::get<Key<NDIM>::num_children>(out); // last one
            // FunctionCompressedNode<T,K,NDIM> c(key);
            // zero coeffs
            // insert coeffs in right space;
            // set have no children for all children;
            // result.send(key, c);
        } else {
            auto outs = ::mad::subtuple_to_array_of_ptrs<0,Key<NDIM>::num_children>(out);
            outs[key.childindex()]->send(key.parent(),node);
        }
    }
}


// With data streaming up the tree run compression
template <typename T, size_t K, Dimension NDIM>
void do_compress(const Key<NDIM>& key,
                 const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
                 typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
    auto& child_slices = FunctionData<T,K,NDIM>::get_child_slices();
    FunctionCompressedNode<T,K,NDIM> result(key); // The eventual result
    auto& d = result.coeffs;

    T sumsq = 0.0;
    {   // Collect child coeffs and leaf info
        FixedTensor<T,2*K,NDIM> s;
        auto ins = ::mad::tuple_to_array_of_ptrs_const(in); /// Ugh ... cannot get const to match
        for (size_t i : range(Key<NDIM>::num_children)) {
            s(child_slices[i]) = ins[i]->coeffs;
            result.is_leaf[i] = ins[i]->is_leaf;
            sumsq += ins[i]->sum; // Accumulate sumsq from child difference coeffs
        }
        filter<T,K,NDIM>(s,d);  // Apply twoscale transformation
    }

    // Recur up
    if (key.level() > 0) {
        FunctionReconstructedNode<T,K,NDIM> p(key);
        p.coeffs = d(child_slices[0]);
        d(child_slices[0]) = 0.0;
        p.sum = d.sumabssq() + sumsq; // Accumulate sumsq of difference coeffs from this node and children
        auto outs = ::mad::subtuple_to_array_of_ptrs<0,Key<NDIM>::num_children>(out);
        outs[key.childindex()]->send(key.parent(), p);
    }
    else {
        std::cout << "At root of compressed tree: total normsq is " << sumsq + d.sumabssq() << std::endl;
    }

    // Send result to output tree
    send<Key<NDIM>::num_children>(key,result,out);
}


/// Return a string with the binary encoding of the lowest \c width bits of the given integer \c i
std::string int2bitstring(size_t i, size_t width) {
    std::string s="";
    for (auto d : range(width)) {
        s = ((i&0x1) ? "1" : "0") + s;
        i>>=1;
        d=d; 
    }
    return s;
}

/// Make a composite operator that implements compression for a single function
template <typename T, size_t K, Dimension NDIM>
auto make_compress(rnodeEdge<T,K,NDIM>& in, cnodeEdge<T,K,NDIM>& out, const std::string& name = "compress") {

    constexpr size_t num_children = Key<NDIM>::num_children;
 
    using sendfuncT = decltype(&send_leaves_up<T,K,NDIM>);
    using sendwrapT = WrapOp<sendfuncT, Key<NDIM>, typename ::detail::tree_types<T,K,NDIM>::compress_out_type, FunctionReconstructedNode<T,K,NDIM> >;
    using compfuncT = decltype(&do_compress<T,K,NDIM>);
    using compwrapT = typename ::detail::tree_types<T,K,NDIM>::template compwrap_type<compfuncT>;
    
    // Make names for terminals that connect child boxes
    std::vector<std::string> outnames;
    for (auto i : range(num_children)) {
        outnames.push_back(std::string("child:")+int2bitstring(i,NDIM));
    }
    std::vector<std::string> innames=outnames;
    outnames.push_back("output");

    auto s = std::unique_ptr<sendwrapT>(new sendwrapT(&send_leaves_up<T,K,NDIM>, "send_leaves_up", {"input"}, outnames));
    auto c = std::unique_ptr<compwrapT>(new compwrapT(&do_compress<T,K,NDIM>, "do_compress", innames, outnames));

    in.set_out(s-> template in<0>()); // Connect input to s
    out.set_in(s-> template out<num_children>()); // Connect s result to output 
    out.set_in(c-> template out<num_children>()); // Connect c result to output

    // Connect send_leaves_up to do_compress and recurrence for do_compress
    for (auto i : range(num_children)) {
        connect(i,i,s.get(),c.get()); // this via base class terminals
        connect(i,i,c.get(),c.get());
    }
    
    auto ins = std::make_tuple(s-> template in<0>());
    auto outs = std::make_tuple(c-> template out<num_children>());
    std::vector<std::unique_ptr<OpBase>> ops(2);
    ops[0] = std::move(s); 
    ops[1] = std::move(c);
        
    return make_composite_op(std::move(ops), ins, outs, name);
}

// For checking we haven't broken something while developing
template <typename T>
struct is_serializable {
    static const bool value = std::is_fundamental<T>::value || std::is_member_function_pointer<T>::value || std::is_function<T>::value  || std::is_function<typename std::remove_pointer<T>::type>::value || std::is_pod<T>::value;
};
static_assert(is_serializable<Key<2>>::value, "You just did something that stopped Key from being serializable"); // yes
static_assert(is_serializable<SimpleTensor<float,2,2>>::value,"You just did something that stopped SimpleTensor from being serializable"); // yes
static_assert(is_serializable<FunctionReconstructedNode<float,2,2>>::value,"You just did something that stopped FunctionReconstructedNode from being serializable"); // yes

// Test gaussian function
template <typename T, Dimension NDIM>
T g(const Coordinate<T,NDIM>& r) {
    static const T expnt = 3.0;
    static const T fac = std::pow(T(2.0*expnt/M_PI),T(0.25*NDIM)); // makes square norm over all space unity
    T rsq = 0.0;
    for (auto x : r) rsq += x*x;
    return fac*std::exp(-expnt*rsq);
}

// // Test the numerics
// template <typename T, size_t K, Dimension NDIM>
// void test_gaussian(T thresh) {
//     Domain<NDIM>::set_cube(-5.0,5.0);
//     FunctionData<T,K,NDIM>::initialize();
//     FunctionFunctor<T, NDIM> ff(g<T,NDIM>);
//     Key<NDIM> root(0,{});
//     T normsq = project_function_node<decltype(ff), T, K, NDIM>(ff,root,thresh);
//     std::cout << "normsq error " << normsq-1.0 << std::endl;
// }

int main(int argc, char** argv) {
    SimpleTensor<float,1> sdjflkasjfdlk;
    
    ttg_initialize(argc, argv, 2);
    std::cout << "Hello from madttg\n";
  
    // test_gaussian<double,5,3>(1e-6);
    // test_gaussian<float,8,3>(1e-1);
    // test_gaussian<float,8,3>(1e-2);
    // test_gaussian<float,8,3>(1e-3);
    // test_gaussian<float,8,3>(1e-4);
    // test_gaussian<float,8,3>(1e-5);
    // test_gaussian<float,8,3>(1e-6);

    using T = float;
    constexpr size_t K = 8;
    constexpr size_t NDIM = 3;
    GLinitialize();
    FunctionData<T,K,NDIM>::initialize();
    Domain<NDIM>::set_cube(-6.0,6.0);
    
    FunctionFunctor<T, NDIM> ff(g<T,NDIM>);

    ctlEdge<NDIM> ctl("start");
    rnodeEdge<T,K,NDIM> a("a");
    cnodeEdge<T,K,NDIM> b("b");
    auto start = make_start(ctl);
    auto p1 = make_project(ff, T(1e-6), ctl, a, "project A");

    double aa = 1;
    auto compress = make_compress<T,K,NDIM>(a, b);
    aa += 1;
    
    auto printer = make_printer(a);
    auto printer2 = make_printer(b);
    auto connected = make_graph_executable(start.get());
    assert(connected);
    if (ttg_default_execution_context().rank() == 0) {
      std::cout << "Is everything connected? " << Verify()(start.get()) << std::endl;
      std::cout << "==== begin dot ====\n";
      std::cout << Dot()(start.get()) << std::endl;
      std::cout << "====  end dot  ====\n";

      // This kicks off the entire computation
      start->invoke(Key<NDIM>(0, {0}));
    }

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());

    // //using funcT = std::function<void (const Key<1>&, std::tuple<int,double>&&, std::tuple<Out<Key<1>,double>>&)>;
    // // auto xxx = wrapx(funcT(fred)); // yes

    // using funcT = decltype(&fred);
    // using wrapT = WrapOp<funcT, Key<1>, std::tuple<Out<Key<1>,double>>, int, double>;
    // auto xxx = std::unique_ptr<wrapT>(new wrapT(&fred, "fred", {"a","b"}, {"c"}));

    ttg_finalize();
    
    
    return 0;
}

