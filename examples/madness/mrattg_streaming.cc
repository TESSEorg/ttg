// TTG AND MADNESS RUNTIME STUFF

//#define WORLD_INSTANTIATE_STATIC_TEMPLATES
//#include "ttg/madness/ttg.h"
// using namespace madness; // don't want this to avoid collisions with new mad stuff

#include "ttg.h"
using namespace ttg;

// APPLICATION STUFF BELOW
#include <cmath>
#include <array>
#include <mutex>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <chrono>

#include "../mragl.h"
#include "../mrakey.h"
#include "../mrahash.h"
#include "../mramxm.h"
#include "../mramisc.h"
#include "../mratypes.h"
#include "../mradomain.h"
#include "../mratwoscale.h"
#include "../mrasimpletensor.h"
#include "../mrafunctiondata.h"
#include "../mrafunctionnode.h"
#include "../mrafunctionfunctor.h"

//#include "/usr/include/mkl/mkl.h" // assume for now but need to wrap
#include "mkl.h"

using namespace mra;

/// Random process map
template <Dimension NDIM>
struct KeyProcMap {
    const size_t size;
    KeyProcMap() : size(get_default_world().size()) {}
    std::size_t operator()(const Key<NDIM>& key) const {return key.hash() % size;}
};


// /// A pmap that locates children on odd levels with their even level parents .. needs a litte fixing
// template <typename keyT>
// class LevelPmap : public WorldDCPmapInterface<keyT> {
// private:
//     const int nproc;
// public:
//     LevelPmap() : nproc(0) {};

//     LevelPmap(World& world) : nproc(world.nproc()) {}

//     /// Find the owner of a given key
//     ProcessID owner(const keyT& key) const {
//         Level n = key.level();
//         if (n == 0) return 0;
//         hashT hash;
//         if (n <= 3 || (n&0x1)) hash = key.hash();
//         else hash = key.parent().hash();
//         return hash%nproc;
//     }
// };


/// A pmap that spatially decomposes the domain and by default slightly overdcomposes to attempt to load balance
template <Dimension NDIM>
class PartitionPmap {
private:
    const int nproc;
    Level target_level;
public:
    PartitionPmap()
        : nproc(1)
        , target_level(3)
    {};

    // Default is to try to optimize the target_level, but you can specify any value > 0
    PartitionPmap(size_t nproc, const Level target_level=0)
        : nproc(nproc)
    {
        if (target_level > 0) {
            this->target_level = target_level;
        }
        else {
            this->target_level = 1;
            int p = nproc-1;
            while (p) {
                p >>= NDIM;
                this->target_level++;
            }
        }            
    }

    /// Find the owner of a given key
    ProcessID operator()(const Key<NDIM>& key) const {
        HashValue hash;
        if (key.level() <= target_level) {
            hash = key.hash();
        }
        else {
            hash = key.parent(target_level - key.level()).hash();
        }
        return hash%nproc;
    }
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
auto make_printer(const Edge<keyT, valueT>& in, const char* str = "", const bool doprint=true) {
    auto func = [str,doprint](const keyT& key, const valueT& value, std::tuple<>& out) {
        if (doprint) {
            std::lock_guard<std::mutex> obolus(printer_guard);
            std::cout << str << " (" << key << "," << value << ")" << std::endl;
        }
    };
    return wrap(func, edges(in), edges(), "printer", {"input"});
}

template <Dimension NDIM>
auto make_start(const ctlEdge<NDIM>& ctl) {
    auto func = [](const Key<NDIM>& key, std::tuple<ctlOut<NDIM>>& out) { send<0>(key, Control(), out); };
    return wrap<Key<NDIM>>(func, edges(), edges(ctl), "start", {}, {"control"});
}


/// Constructs an operator that adaptively projects the provided function into the basis

/// Returns an std::unique_ptr to the object
template <typename functorT, typename T, size_t K, Dimension NDIM>
auto make_project(functorT& f,
                  const T thresh, /// should be scalar value not complex
                  ctlEdge<NDIM>& ctl,
                  rnodeEdge<T,K,NDIM>& result,
                  const std::string& name = "project") {

    auto F = [f, thresh](const Key<NDIM>& key, Control&& junk, std::tuple<ctlOut<NDIM>, rnodeOut<T,K,NDIM>>& out) {
        FunctionReconstructedNode<T,K,NDIM> node(key); // Our eventual result
        auto& coeffs = node.coeffs; // Need to clean up OO design

        if (key.level() < initial_level(f)) {
            for (auto child : children(key)) send<0>(child, Control(), out);
            coeffs = T(1e7); // set to obviously bad value to detect incorrect use
            node.is_leaf = false;
        }
        else if (is_negligible<functorT,T,NDIM>(f, Domain<NDIM>:: template bounding_box<T>(key),truncate_tol(key,thresh))) {
            coeffs = T(0.0);
            node.is_leaf = true;
        }
        else {
            node.is_leaf = fcoeffs<functorT,T,K>(f, key, thresh, coeffs); // cannot deduce K
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
                    FunctionReconstructedNode<T,K,NDIM>& inputs,//Why do we need a tuple here?
                    std::tuple<rnodeOut<T,K,NDIM>, cnodeOut<T,K,NDIM>>& out) {
  //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
  //Removed const from here!!
  FunctionReconstructedNode<T,K,NDIM>& node = inputs; //std::get<0>(inputs);
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
          //auto outs = ::mra::subtuple_to_array_of_ptrs<0,Key<NDIM>::num_children>(out);
          //outs[key.childindex()]->send(key.parent(),node);
          send<0>(key.parent(), node, out);
        }
    }
}

template <typename T, size_t K, Dimension NDIM>
void reduce_leaves(const Key<NDIM>& key, FunctionReconstructedNode<T,K,NDIM>& node, std::tuple<rnodeOut<T,K,NDIM>>& out) {
  //std::cout << "Reduce_leaves " << node.key.childindex() << " " << node.neighbor_sum[node.key.childindex()] << std::endl;
  std::get<0>(out).send(key, node);
}

// With data streaming up the tree run compression
template <typename T, size_t K, Dimension NDIM>
void do_compress(const Key<NDIM>& key,
                 FunctionReconstructedNode<T,K,NDIM> &in,
                 std::tuple<rnodeOut<T,K,NDIM>, cnodeOut<T,K,NDIM>> &out) {
  //const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
  //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
    auto& child_slices = FunctionData<T,K,NDIM>::get_child_slices();
    FunctionCompressedNode<T,K,NDIM> result(key); // The eventual result
    auto& d = result.coeffs;

    T sumsq = 0.0;
    {   // Collect child coeffs and leaf info
        FixedTensor<T,2*K,NDIM> s;
        //auto ins = ::mra::tuple_to_array_of_ptrs_const(in); /// Ugh ... cannot get const to match
        for (size_t i : range(Key<NDIM>::num_children)) {
            s(child_slices[i]) = in.neighbor_coeffs[i];
            result.is_leaf[i] = in.is_neighbor_leaf[i];
            sumsq += in.neighbor_sum[i]; // Accumulate sumsq from child difference coeffs
            //if (in.neighbor_sum[i] > 10000)
            //std::cout << i << " " << in.neighbor_sum[i] << " " << sumsq << std::endl;

        }
        filter<T,K,NDIM>(s,d);  // Apply twoscale transformation
    }

    // Recur up
    if (key.level() > 0) {
        FunctionReconstructedNode<T,K,NDIM> p(key);
        p.coeffs = d(child_slices[0]);
        d(child_slices[0]) = 0.0;
        p.sum = d.sumabssq() + sumsq; // Accumulate sumsq of difference coeffs from this node and children
        //auto outs = ::mra::subtuple_to_array_of_ptrs<0,Key<NDIM>::num_children>(out);
        //outs[key.childindex()]->send(key.parent(), p);
        send<0>(key.parent(), p, out);
    }
    else {
        std::cout << "At root of compressed tree: total normsq is " << sumsq + d.sumabssq() << std::endl;
    }

    // Send result to output tree
    //send<Key<NDIM>::num_children>(key,result,out);
    send<1>(key, result, out);
}


/// Return a string with the binary encoding of the lowest \c width bits of the given integer \c i
std::string int2bitstring(size_t i, size_t width) {
    std::string s="";
    for (auto d : range(width)) {
        s = (((i>>d)&0x1) ? "1" : "0") + s;
        //i>>=1;
    }
    return s;
}

/// Make a composite operator that implements compression for a single function
template <typename T, size_t K, Dimension NDIM>
auto make_compress(rnodeEdge<T,K,NDIM>& in, cnodeEdge<T,K,NDIM>& out, const std::string& name = "compress") {

  //constexpr size_t num_children = Key<NDIM>::num_children;

    /*using sendfuncT = decltype(&send_leaves_up<T,K,NDIM>);
    //using sendwrapT = WrapOp<sendfuncT, Key<NDIM>, typename ::detail::tree_types<T,K,NDIM>::compress_out_type, FunctionReconstructedNode<T,K,NDIM> >;
    using sendwrapT = WrapOp<sendfuncT, Key<NDIM>, std::tuple<rnodeOut<T,K,NDIM>>, FunctionReconstructedNode<T,K,NDIM> >;
    using compfuncT = decltype(&do_compress<T,K,NDIM>);
    //using compwrapT = typename ::detail::tree_types<T,K,NDIM>::template compwrap_type<compfuncT>;
    using compwrapT = WrapOp<compfuncT, Key<NDIM>, std::tuple<rnodeOut<T,K,NDIM>, cnodeOut<T,K,NDIM>>, FunctionReconstructedNode<T,K,NDIM> >;
    using reducefuncT = decltype(&reduce_leaves<T,K,NDIM>);
    using reducewrapT = WrapOp<reducefuncT, Key<NDIM>, std::tuple<rnodeOut<T,K,NDIM>>, FunctionReconstructedNode<T,K,NDIM> >;
    */
    // Make names for terminals that connect child boxes
    /*std::vector<std::string> outnames;
    for (auto i : range(num_children)) {
        outnames.push_back(std::string("child:")+int2bitstring(i,NDIM));
        }*/
    //std::vector<std::string> innames=outnames;
    //outnames.push_back("output");

  rnodeEdge<T,K,NDIM> children1("children1"), children2("children2");

    //Do we need to send output from send_leaves_up?
    /*auto s = std::unique_ptr<sendwrapT>(new sendwrapT(&send_leaves_up<T,K,NDIM>, "send_leaves_up", {"input"}, {"children1"}));
    auto r = std::unique_ptr<reducewrapT>(new reducewrapT(&reduce_leaves<T,K,NDIM>, "reduce_leaves", {"children1"}, {"children2"}));
    auto c = std::unique_ptr<compwrapT>(new compwrapT(&do_compress<T,K,NDIM>, "do_compress", {"children2"}, {"output"}));

    in.set_out(s-> template in<0>()); // Connect input to s
    //out.set_in(s-> template out<1>()); // Connect s result to output
    out.set_in(c-> template out<1>()); // Connect c result to output
    children1.set_out(r-> template in<0>());
    children2.set_in(r-> template out<0>());
    */
    // Connect send_leaves_up to do_compress and recurrence for do_compress
    /*for (auto i : range(num_children)) {
        connect(i,i,s.get(),c.get()); // this via base class terminals
        connect(i,i,c.get(),c.get());
        }*/

    //auto ins = std::make_tuple(s-> template in<0>());
    //auto outs = std::make_tuple(c-> template out<1>());
    //std::vector<std::unique_ptr<ttg::OpBase>> ops(3);
    //ops[0] = std::move(s);
    //ops[1] = std::move(r);
    //ops[2] = std::move(c);

    //return make_composite_op(std::move(ops), ins, outs, name);

  return std::make_tuple(wrap(&send_leaves_up<T,K,NDIM>, edges(in), edges(children1, out), "send_leaves_up", {"input"}, {"children1", "output"}),
                           wrap(&reduce_leaves<T,K,NDIM>, edges(children1), edges(children2), "reduce_leaves", {"children1"}, {"children2"}),
                           wrap(&do_compress<T,K,NDIM>, edges(children2), edges(children1,out), "do_compress", {"children2"}, {"recur","output"})
                           );
}

template <typename T, size_t K, Dimension NDIM>
void do_reconstruct(const Key<NDIM>& key,
                    const std::tuple<FunctionCompressedNode<T,K,NDIM>&,FixedTensor<T,K,NDIM>&>& t,
                    std::tuple<Out<Key<NDIM>,FixedTensor<T,K,NDIM>>,rnodeOut<T,K,NDIM>>& out) {
  const auto& child_slices = FunctionData<T,K,NDIM>::get_child_slices();
    auto& node = std::get<0>(t);
    const auto& from_parent = std::get<1>(t);
    if (key.level() != 0) node.coeffs(child_slices[0]) = from_parent;

    FixedTensor<T,2*K,NDIM> s;
    unfilter<T,K,NDIM>(node.coeffs, s);

    FunctionReconstructedNode<T,K,NDIM> r(key);
    r.coeffs = T(0.0);
    r.is_leaf = false;
    ::send<1>(key, r, out); // Send empty interior node to result tree

    KeyChildren<NDIM> children(key);
    for (auto it=children.begin(); it!=children.end(); ++it) {
        const Key<NDIM> child= *it;
        r.key = child;
        r.coeffs = s(child_slices[it.index()]);
        r.is_leaf = node.is_leaf[it.index()];
        if (r.is_leaf) {
            ::send<1>(child, r, out);
        }
        else {
            ::send<0>(child, r.coeffs, out);
        }
    }
}

template <typename T, size_t K, Dimension NDIM>
auto make_reconstruct(const cnodeEdge<T,K,NDIM>& in, rnodeEdge<T,K,NDIM>& out, const std::string& name = "reconstruct") {
    Edge<Key<NDIM>,FixedTensor<T,K,NDIM>> S("S");  // passes scaling functions down

    auto s = wrapt(&do_reconstruct<T,K,NDIM>, edges(in, S), edges(S, out), name, {"input", "s"}, {"s", "output"});

    if (get_default_world().rank() == 0) {
      s->template in<1>()->send(Key<NDIM>{0,{0}}, FixedTensor<T,K,NDIM>()); // Prime the flow of scaling functions
    }

    return s;
}

template <typename keyT, typename valueT>
auto make_sink(const Edge<keyT,valueT>& e) {
    return std::make_unique<OpSink<keyT,valueT>>(e);
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

// Test gaussian functor
template <typename T, Dimension NDIM>
class Gaussian {
    const T expnt;
    const Coordinate<T,NDIM> origin;
    const T fac;
    const T maxr;
    Level initlev;
public:
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

    // T operator()(const Coordinate<T,NDIM>& r) const {
    //     T rsq = 0.0;
    //     for (auto x : r) rsq += x*x;
    //     return fac*std::exp(-expnt*rsq);
    // }

    template <size_t N>
    void operator()(const SimpleTensor<T,NDIM,N>& x, std::array<T,N>& values) const {
        distancesq(origin, x, values);
        vscale(N, -expnt, &values[0]);
        vexp(N, &values[0], &values[0]);
        vscale(N, fac, &values[0]);
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
        static const T diagndim = T(0.5)*std::sqrt(T(NDIM));
        T boxradplusr = maxw*diagndim + maxr;
        // ttg::print(box, boxradplusr, bool(boxradplusr*boxradplusr < rsq));
        return (boxradplusr*boxradplusr < rsq);
    }
};


template <typename T, size_t K, Dimension NDIM>
void test0() {
    FunctionData<T,K,NDIM>::initialize();
    Domain<NDIM>::set_cube(-6.0,6.0);

    //auto ff = &g<T,NDIM>;
    auto ff = Gaussian<T,NDIM>(T(3.0), {T(0.0),T(0.0),T(0.0)});

    ctlEdge<NDIM> ctl("start");
    rnodeEdge<T,K,NDIM> a("a"), c("c");
    cnodeEdge<T,K,NDIM> b("b");

    auto start = make_start(ctl);
    auto p1 = make_project(ff, T(1e-6), ctl, a, "project A");
    auto compress = make_compress<T,K,NDIM>(a, b);
    auto recon = make_reconstruct<T,K,NDIM>(b,c);
    //recon->set_trace_instance(true);

    auto printer =   make_printer(a,"projected    ", false);
    auto printer2 =  make_printer(b,"compressed   ", false);
    auto printer3 =  make_printer(c,"reconstructed", false);
    auto connected = make_graph_executable(start.get());
    assert(connected);
    if (ttg_default_execution_context().rank() == 0) {
        std::cout << "Is everything connected? " << connected << std::endl;
        std::cout << "==== begin dot ====\n";
        std::cout << Dot()(start.get()) << std::endl;
        std::cout << "====  end dot  ====\n";

        // This kicks off the entire computation
        start->invoke(Key<NDIM>(0, {0}));
    }

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());
}


template <typename T, size_t K, Dimension NDIM>
void test1() {
    FunctionData<T,K,NDIM>::initialize();
    Domain<NDIM>::set_cube(-6.0,6.0);

    //auto ff = &g<T,NDIM>;
    auto ff = Gaussian<T,NDIM>(T(30000.0), {T(0.0),T(0.0),T(0.0)});
    ctlEdge<NDIM> ctl("start");
    auto start = make_start(ctl);
    std::vector<std::unique_ptr<ttg::OpBase>> ops;
    for (auto i : range(3)) {
        TTGUNUSED(i);
        rnodeEdge<T,K,NDIM> a("a"), c("c");
        cnodeEdge<T,K,NDIM> b("b");

        auto p1 = make_project(ff, T(1e-6), ctl, a, "project A");
        auto compress = make_compress<T,K,NDIM>(a, b);

        auto &reduce_leaves_op = std::get<1>(compress);
        reduce_leaves_op->template set_input_reducer<0>([](FunctionReconstructedNode<T,K,NDIM> &&node,
                                                           FunctionReconstructedNode<T,K,NDIM> &&another)
                                                        {
                                                          //Update self values into the array.
                                                          node.neighbor_coeffs[node.key.childindex()] = node.coeffs;
                                                          node.is_neighbor_leaf[node.key.childindex()] = node.is_leaf;
                                                          node.neighbor_sum[node.key.childindex()] = node.sum;
                                                          node.neighbor_coeffs[another.key.childindex()] = another.coeffs;
                                                          node.is_neighbor_leaf[another.key.childindex()] = another.is_leaf;
                                                          node.neighbor_sum[another.key.childindex()] = another.sum;
                                                          //std::cout << "Neighbor_sum[" << another.key.childindex() << "] : " << node.neighbor_sum <<std::endl;
                                                          return node;
                                                        });
        reduce_leaves_op->template set_static_argstream_size<0>(1 << NDIM);

        auto recon = make_reconstruct<T,K,NDIM>(b,c);

        // auto printer =   make_printer(a,"projected    ", false);
        // auto printer2 =  make_printer(b,"compressed   ", false);
        // auto printer3 =  make_printer(c,"reconstructed", false);
        auto printer =   make_sink(a);
        auto printer2 =  make_sink(b);
        auto printer3 =  make_sink(c);

        ops.push_back(std::move(p1));
        ops.push_back(std::move(std::get<0>(compress)));
        ops.push_back(std::move(std::get<1>(compress)));
        ops.push_back(std::move(std::get<2>(compress)));
        ops.push_back(std::move(recon));
        ops.push_back(std::move(printer));
        ops.push_back(std::move(printer2));
        ops.push_back(std::move(printer3));
    }
    
    std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
    auto connected = make_graph_executable(start.get());
    assert(connected);    
    if (ttg_default_execution_context().rank() == 0) {
        //std::cout << "Is everything connected? " << connected << std::endl;
        //std::cout << "==== begin dot ====\n";
        //std::cout << Dot()(start.get()) << std::endl;
        //std::cout << "====  end dot  ====\n";
        
        beg = std::chrono::high_resolution_clock::now();
        // This kicks off the entire computation
        start->invoke(Key<NDIM>(0, {0}));
    }

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());

    if (ttg_default_execution_context().rank() == 0) {
      end = std::chrono::high_resolution_clock::now();
      std::cout << "TTG Execution Time (milliseconds) : "
                << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 
                << std::endl; 
    }
}

template <typename T, size_t K, Dimension NDIM>
void test2(size_t nfunc, T thresh = 1e-6) {
    FunctionData<T,K,NDIM>::initialize();
    PartitionPmap<NDIM> pmap =  PartitionPmap<NDIM>(ttg_default_execution_context().size());
    Domain<NDIM>::set_cube(-6.0,6.0);

    srand48(5551212); // for reproducible results
    for (auto i : range(10000)) drand48(); // warmup generator

    ctlEdge<NDIM> ctl("start");
    auto start = make_start(ctl);
    std::vector<std::unique_ptr<ttg::OpBase>> ops;
    for (auto i : range(nfunc)) {
        T expnt = 30000.0;
        Coordinate<T,NDIM> r;
        for (size_t d=0; d<NDIM; d++) {
            r[d] = T(-6.0) + T(12.0)*drand48();
        }
        auto ff = Gaussian<T,NDIM>(expnt, r);
        
        TTGUNUSED(i);
        rnodeEdge<T,K,NDIM> a("a"), c("c");
        cnodeEdge<T,K,NDIM> b("b");

        auto p1 = make_project(ff, T(thresh), ctl, a, "project A"); //p1->set_keymap(pmap);
        auto compress = make_compress<T,K,NDIM>(a, b); //std::get<0>(compress)->set_keymap(pmap);std::get<1>(compress)->set_keymap(pmap);

        auto &reduce_leaves_op = std::get<1>(compress);
        reduce_leaves_op->template set_input_reducer<0>([](FunctionReconstructedNode<T,K,NDIM> &&node,
                                                           FunctionReconstructedNode<T,K,NDIM> &&another)
                                                        {
                                                          //Update self values into the array.
                                                          node.neighbor_coeffs[node.key.childindex()] = node.coeffs;
                                                          node.is_neighbor_leaf[node.key.childindex()] = node.is_leaf;
                                                          node.neighbor_sum[node.key.childindex()] = node.sum;
                                                          node.neighbor_coeffs[another.key.childindex()] = another.coeffs;
                                                          node.is_neighbor_leaf[another.key.childindex()] = another.is_leaf;
                                                          node.neighbor_sum[another.key.childindex()] = another.sum;
                                                          //std::cout << "Neighbor_sum[" << another.key.childindex() << "] : " << node.neighbor_sum <<std::endl;
                                                          return node;
                                                        });
        reduce_leaves_op->template set_static_argstream_size<0>(1 << NDIM);

        auto recon = make_reconstruct<T,K,NDIM>(b,c);// recon->set_keymap(pmap);

        //auto printer =   make_printer(a,"projected    ", true);
        // auto printer2 =  make_printer(b,"compressed   ", false);
        // auto printer3 =  make_printer(c,"reconstructed", false);
        auto printer =   make_sink(a);
        auto printer2 =  make_sink(b);
        auto printer3 =  make_sink(c);

        ops.push_back(std::move(p1));
        ops.push_back(std::move(std::get<0>(compress)));
        ops.push_back(std::move(std::get<1>(compress)));
        ops.push_back(std::move(std::get<2>(compress)));
        ops.push_back(std::move(recon));
        ops.push_back(std::move(printer));
        ops.push_back(std::move(printer2));
        ops.push_back(std::move(printer3));
    }
    
    std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
    auto connected = make_graph_executable(start.get());
    assert(connected);    
    if (ttg_default_execution_context().rank() == 0) {
        //std::cout << "Is everything connected? " << connected << std::endl;
        //std::cout << "==== begin dot ====\n";
        //std::cout << Dot()(start.get()) << std::endl;
        //std::cout << "====  end dot  ====\n";
	beg = std::chrono::high_resolution_clock::now();
        // This kicks off the entire computation
        start->invoke(Key<NDIM>(0, {0}));
    }

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());
    
    if (ttg_default_execution_context().rank() == 0) {
        end = std::chrono::high_resolution_clock::now();
        std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;

    }
}

int main(int argc, char** argv) {
    ttg_initialize(argc, argv, 2);
    //std::cout << "Hello from madttg\n";

    //vmlSetMode(VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT); // default
    //vmlSetMode(VML_EP | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT); // err is 10x default
    vmlSetMode(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_DEFAULT); // err is same as default little faster
    //vmlSetMode(VML_EP | VML_FTZDAZ_ON  | VML_ERRMODE_DEFAULT); // err is 10x default

    GLinitialize();

    {
        //test0<float,6,3>();
        //test1<float,6,3>();
        //test2<float,6,3>(20);
        test2<double,10,3>(1, 1e-8);
        //test1<double,6,3>();
    }

    ttg_fence(get_default_world());

    ttg_finalize();


    return 0;
}

