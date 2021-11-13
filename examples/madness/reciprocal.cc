// TTG AND MADNESS RUNTIME STUFF

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include "ttg/madness/ttg.h"
using namespace ttg;

// APPLICATION STUFF BELOW
#include <cmath>
#include <utility>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <functional>

// Iteration to compute y=1/x .... y <-- y*(2-x*y) ... for a vector of numbers with 1 task per vector element

/***

     ** design choice: if an TT is used in phases so that it is never
     ** possible for the same key to arrive at the same time it is
     ** permissible to reuse keys in op playing the same algorithmic role on
     ** successive iterations
     **
     ** e.g., a reduce operation in an iterative solve --- since it is
     ** a syncrhonization point it is not possible for multiple identical keys to
     ** arrive at the same time
     **
     ** e.g., instead of key being pair(iteration,vector-index) it
     ** suffices to just use key=vector-index.
     **
     ** this can fail at runtime in two ways --- multiple assignment
     ** (runtime detection) or left over tasks (runtime detection) or
     ** wrong numbers
     **
     ** This is essential to avoid the key potentially becoming the
     ** union of all keys within an algorithm.  Example here is
     ** iteration index being joined to summation index in inner loop.

     ** issue: x logically depends only on i, but needs to be forwarded to all iterations k --- must we do this explicitly?

     ** easiest is to send x along with y ... change made

     ** process ... make this logical flow ... iterate until consistent and complete

     variable names here are essentially edges

     start() ---> x(i) for i n [0,N)

     guess(x(i)) ---> xy[0](i)

     iteration(xy[k](i)) --> xynew[k][i]

     residual(xy[k][i], xynew[k][i]) --> errsq[i]

     binarysum(errsq[i],k) --> errsqsum[k]

     convergence_test[k](errsqsum[k],xynew[i]) --> sends xynew[k][i] to either result[i] or xy[k+1][i]

 ***/

template <typename X, typename Y>
std::ostream& operator<<(std::ostream& s, const std::pair<X,Y>& p) {
    s << "(" << p.first << "," << p.second << ")";
    return s;
}

static inline bool is_power_of_two(size_t N) {
    if (N==0) return true;
    if (N==1) return false;
    else return ((N-1)&N) == 0;
}

/// Makes an operation that reduces a vector of length N to a scalar that is sent to specified key
template <typename sumresultkeyT, typename valueT>
auto make_binary_reduce(Edge<size_t,valueT>& in, Edge<sumresultkeyT,valueT>& result, size_t N, const sumresultkeyT& result_key) {
    using iNT = std::pair<size_t,size_t>;
    using outT = std::tuple<Out<iNT,valueT>,Out<iNT,valueT>,Out<sumresultkeyT,valueT>>;
    // Handles logic of propagating value up reduction tree and enventually to output
    auto logic = [result_key](const iNT& iN, valueT sum, outT& out) {
        const size_t i=iN.first, N=iN.second, Nisodd=(N&0x1);
        iNT up {i>>1,(N+Nisodd)>>1};
        if (N == 1)     ::send<2>(result_key, sum, out); // result
        else if (i&0x1) ::send<1>(up, sum, out);         // right=odd
        else            ::send<0>(up, sum, out);         // left=even
        if (N>1 && i==(N-1) && Nisodd) ::send<1>(up, valueT(0.0), out); // Missing element
    };

    // Takes vec[i] (i in [0,N]) as input and initiates reduction
    auto startfn = [N,logic](const size_t& i, const valueT& value, outT& out) {logic(iNT(i,N), value, out);};

    auto reducefn = [logic](const iNT& iN, const valueT& left, const valueT& right, outT& out) {logic(iN, left+right, out);};

    Edge<iNT,valueT> left, right;
    auto start = make_tt<size_t>(startfn, edges(in), edges(left,right,result), "startsum", {"in"}, {"left","right","result"});
    auto reduce= make_tt<iNT>(reducefn, edges(left,right), edges(left,right,result), "reducesum", {"left","right"}, {"left","right","result"});
    auto ins = std::make_tuple(start-> template in<0>());
    auto outs = std::make_tuple(reduce-> template out<2>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
    ops[0] = std::move(start);
    ops[1] = std::move(reduce);
    return make_composite_op(std::move(ops), ins, outs, "reduce");
}

using xT = double; // type of x
using yT = double; // type of y
using xyT = std::pair<xT,yT>; // holds x[i],y[i]
using iT = size_t; // i is the index into the vector
using kT = size_t; // k is the iteration index
using kiT = std::pair<kT,iT>; // ki holds iteration k and vector index i
using sumreskeyT = int; // type of key for sum destination

namespace std {
    template <> struct hash<kiT> {
        size_t operator()(const kiT& s) const noexcept { return (s.first<<32) + s.second; }
    };
}


/// Start computation and produce x
auto make_start(Edge<iT,xT>& x, size_t N) {
    auto f = [N](const iT& unused, std::tuple<Out<iT,xT>>& out) {
        for (iT i=0; i<N; i++) ::send<0>(i, drand48(), out);
    };
    return make_tt<iT>(f, edges(), edges(x), "start", {}, {"x"});
}

// Produce initial guess for y and forward xy
auto make_guess(Edge<iT,xT>& x, Edge<kiT,xyT>& xy) {
    auto f = [](const iT& i, const xT& x, std::tuple<Out<kiT,xyT>>& out) {
        yT y = 6.0+(8.0*x-12.0)*x; // quadratic part of taylor expansion about 0.5 (assuming x in (0,1))
        ::send<0>(kiT(0,i), xyT(x,y), out);
    };
    return make_tt<iT>(f, edges(x), edges(xy), "start", {"x"}, {"xy"});
}

/// Update y and forward xy
auto make_iteration(Edge<kiT,xyT>&& xy, Edge<kiT,xyT>& xynew) {
    auto f = [](const kiT& ki, const xyT& xy, std::tuple<Out<kiT,xyT>>& out) {
        xT x = xy.first;
        yT y = xy.second;
        yT ynew = y*(2.0-x*y);
        ::send<0>(ki, xyT(x,ynew), out);
    };
    return make_tt<kiT>(f, edges(xy), edges(xynew), "iteration", {"xy"}, {"xynew"});
}

/// Compute (yold[i]-ynew[i])**2
auto make_residual(Edge<kiT,xyT>&& xy, Edge<kiT,xyT>& xynew, Edge<sumreskeyT,kT>& k, Edge<iT,yT>& errsq) {
    auto f = [](const kiT& ki, const xyT& xy, const xyT& xynew, std::tuple<Out<sumreskeyT,kT>,Out<iT,yT>>& out) {
        yT err = (xy.second - xynew.second);
        const kT k = ki.first;
        const iT i = ki.second;
        if (i == 0) ::send<0>(sumreskeyT(0), k, out);
        ::send<1>(i, err*err, out);
        //::send<1>(i, (err*err)>1e-10?1:0, out); // Counts #converged
    };
    return make_tt<kiT>(f, edges(xy,xynew), edges(k,errsq), "residual", {"xy","xynew"}, {"k","errsq"});
}

auto make_sum_result(Edge<sumreskeyT,kT>& k, Edge<sumreskeyT,yT>& sumresult, Edge<kT,yT>& errsqsum) {
    auto f = [](const sumreskeyT& unused, const kT& k, const yT& sumresult, std::tuple<Out<kT,yT>>& out) {
        ::send<0>(k, sumresult, out);
    };
    return make_tt<sumreskeyT>(f, edges(k,sumresult), edges(errsqsum), "sumres", {"k","sum"}, {"sum"});
}

/// Use residual for this iteration (k) to produce boolean converged flag
auto make_converged(Edge<kT,yT>& errsqsum, Edge<kT,bool>& converged) {
    auto f = [](const kT& k, const yT& errsqsum, std::tuple<Out<kT,bool>>& out) {
    bool converged = (errsqsum < 1e-10);
    ttg::print(" residual ", std::sqrt(errsqsum), " converged ", converged ? "yes" : "no");
    // ttg::print(" residual ", (errsqsum), " converged ", converged?"yes":"no");
    ::send<0>(k, converged, out);
  };
    return make_tt<kT>(f, edges(errsqsum), edges(converged), "convtest", {"errsqsum"}, {"converged"});
}

/// Broadcast convergence flag for this iteration [k] to all vector
/// elements ... broken out from make_convergence_test to make this
/// more explicit and visible ... we only need N because we don't yet
/// have the sparse/wildcard broadcast, so we are very inefficiently
/// broadcasting to all i in [0,N) ... once we have the broadcast operation
/// this step is eliminated.
auto make_broadcast(Edge<kT,bool>& converged, Edge<kiT,bool>& bcast, size_t N) {
    auto f = [N](const kT& k, const bool& converged, std::tuple<Out<kiT,bool>>& out) {
        for (iT i=0; i<N; i++) ::send<0>(kiT(k,i),converged,out);
    };
    return make_tt<kT>(f, edges(converged), edges(bcast), "bcast", {"converged"}, {"bcast"});
}

/// If converged produce result, otherwise continue to iterate
auto make_loop(Edge<kiT,bool>& bcast, Edge<kiT,xyT>& xynew, Edge<kiT,xyT>& xy, Edge<iT,xyT>& result) {
    auto f = [](const kiT& ki, const bool& converged, const xyT& xynew, std::tuple<Out<kiT,xyT>,Out<iT,xyT>>& out) {
        if (converged) {
            send<1>(ki.second, xynew, out); // result
        }
        else {
            send<0>(ki, xynew, out); // iterate
        }
    };
    return make_tt<kiT>(f, edges(bcast, xynew), edges(xy, result), "loop", {"bcast", "xynew"}, {"xy", "result"});
}

/// Makes an operation that prints a stream
template <typename keyT, typename valueT>
auto make_printer(const Edge<keyT, valueT>& in, const char* str = "") {
    auto func = [str](const keyT& key, const valueT& value, std::tuple<>& out) { ttg::print(str, ":", key, ":", value); };
    return make_tt(func, edges(in), edges(), "printer", {"input"});
}

/// Discards output
template <typename keyT, typename valueT>
auto make_dev_null(const Edge<keyT, valueT>& in) {
    auto func = [](const keyT& key, const valueT& value, std::tuple<>& out) {};
    return make_tt(func, edges(in), edges(), "devnull", {"input"});
}

int main(int argc, char** argv)
{
    initialize(argc, argv, 2);
    std::cout << "Hello from reciprocal\n";

    // Give random number generator process-dependent state.  Also
    // drand48 seems to need warming up.
    srand48(ttg::default_execution_context().rank());
    for (size_t i=0; i<100; i++) drand48();

    try {
        Edge<iT,xT> x;
        Edge<iT,xyT> result;
        Edge<kT,yT> errsq, errsqsum;
        Edge<kT,bool> converged;
        Edge<kiT,xyT> xy, xynew, xyguess;
        Edge<kiT,bool> bcast;
        Edge<sumreskeyT,yT> sumresult;
        Edge<sumreskeyT,kT> k;

        const size_t N = 900;

        auto start = make_start(x, N);
        auto guess = make_guess(x, xyguess);
        auto iteration = make_iteration(fuse(xy,xyguess), xynew);
        auto residual = make_residual(fuse(xy,xyguess), xynew, k, errsq);
        auto binsum = make_binary_reduce(errsq, sumresult, N, sumreskeyT(0));
        auto sumres = make_sum_result(k, sumresult, errsqsum);
        auto conv = make_converged(errsqsum, converged);
        auto bcastop = make_broadcast(converged, bcast, N);
        auto loop = make_loop(bcast, xynew, xy, result);
        //auto printer = make_printer(result, "result: ");
        auto devnull = make_dev_null(result);

        auto connected = make_graph_executable(start.get());
        assert(connected);
        if (ttg::default_execution_context().rank() == 0) {
            std::cout << "Is everything connected? " << connected << std::endl;
            std::cout << "==== begin dot ====\n";
            std::cout << Dot()(start.get()) << std::endl;
            std::cout << "====  end dot  ====\n";

            // This kicks off the entire computation
            start->invoke(0);
        }

        execute();
        fence();
    }
    catch (std::string e) {
        std::cout << "std::string Exception: " << e << std::endl;
    }
    catch (const char* e) {
        std::cout << "char* Exception: " << e << std::endl;
    }

    fence();
    ttg_finalize();
    return 0;
}

