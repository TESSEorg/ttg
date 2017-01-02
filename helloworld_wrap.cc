#include <iostream>
#include <string>
#include <tuple>
#include "flow.h"

void h(int index, const std::tuple<std::string>& args, Flows<Flow<double,std::string>>& out) {
    out.send<0>(double(index), std::get<0>(args)+"hello");
}

void w(double index, const std::tuple<std::string>& args, Flows<>& out) {
    std::cout << std::get<0>(args) << " world!" << std::endl;
}

int main() {
    BaseOp::set_trace(true);
    Flow<int,std::string> control;
    Flow<double,std::string> pipe;
    Flows<> nothing;

    auto hello = make_wrapper(&h, make_inflows(control), make_flows(pipe), "hello");
    auto world = make_wrapper(&w, make_inflows(pipe), nothing, "world");

    control.send(0,"");

    return 0;
}



    
