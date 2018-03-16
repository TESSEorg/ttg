// TTG AND MADNESS RUNTIME STUFF

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/worldmutex.h>
#include "madness/ttg.h"
// using namespace madness; // don't want this to avoid collisions with new mad stuff  
using namespace madness::ttg;
using namespace ::ttg;

// APPLICATION STUFF BELOW
#include <cmath>
#include <utility>
#include <iostream>

// Iteration to compute y=1/x .... y <-- y*(2-x*y)

template <typename X, typename Y>
std::ostream& operator<<(std::ostream& s, const std::pair<X,Y>& p) {
    s << "(" << p.first << "," << p.second << ")";
    return s;
}

template <typename keyT, typename valueT>
auto make_printer(const Edge<keyT, valueT>& in, const char* str = "") {
  auto func = [str](const keyT& key, const valueT& value, std::tuple<>& out) {
    std::cout << str << " (" << key << "," << value << ")" << std::endl;
  };
  return wrap(func, edges(in), edges(), "printer", {"input"});
}

void iteration(const int& iter, const std::pair<double,double>& xy, std::tuple<Out<int,std::pair<double,double>>, Out<int,std::pair<double,double>>>& out) {
    double x, y; std::tie(x,y) = xy;
    double ynew = y*(2.0-x*y);
    std::cout << "iteration " << iter << " y " << y << " ynew " << ynew << std::endl;
    if (std::abs(y-ynew) < 1e-10*std::abs(y)) {
        ::send<1>(iter,std::make_pair(x,ynew),out);
    }
    else {
        ::send<0>(iter+1,std::make_pair(x,ynew),out);
    }
}    

auto make_start(Edge<int,std::pair<double,double>>& out, double x, double yguess) {
    using outT = std::tuple<Out<int,std::pair<double,double>>>;
    auto f = [x,yguess](const int& key, const std::tuple<>& junk, outT& out) {
        ::send<0>(0,std::make_pair(x,yguess),out);
    };
    // using funcT = decltype(f);
    // using wrapopT = WrapOp<funcT, int, outT>;
    // return std::unique_ptr<wrapopT>(new wrapopT(f, edges(), edges(out), "start", {}, {"guess"}));
    return wrapt<int>(f, edges(), edges(out), "start", {}, {"guess"});
}

int main(int argc, char** argv) {
    ttg_initialize(argc, argv, 2);

    {
        Edge<int,std::pair<double,double>> a("iteration"), b("result");
        auto start = make_start(a,2.0,0.9);
        auto iter = wrap(iteration, edges(a), edges(a,b), "iteration", {"xy"},{"xy","result"});
        auto prt = make_printer(b);

        auto connected = make_graph_executable(start.get());
        assert(connected);
        if (ttg_default_execution_context().rank() == 0) {
            std::cout << "Is everything connected? " << Verify()(start.get()) << std::endl;
            std::cout << "==== begin dot ====\n";
            std::cout << Dot()(start.get()) << std::endl;
            std::cout << "====  end dot  ====\n";
            
            // This kicks off the entire computation
            start->invoke(0);
        }
        
        ttg_execute(ttg_default_execution_context());
        ttg_fence(ttg_default_execution_context());
    }
    

    get_default_world().gop.fence();
    ttg_finalize();
    return 0;
}

