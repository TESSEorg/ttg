#include <iostream>
#include <string>
#include <tuple>
#include "flow.h"

class Hello : public OpTuple<Flow<int,std::string>, Flow<double,std::string>, Hello> {
    using baseT = OpTuple<Flow<int,std::string>, Flow<double,std::string>, Hello>;

 public:

    Hello(Flow<int,std::string> ctl, Flow<double,std::string> pipe)
        : baseT(ctl, pipe, "hello") {}

    void op(int index, const baseT::input_values_tuple_type& args, baseT::output_type& out) {
        out.send<0>(double(index), std::get<0>(args)+"hello");
    }
};

class Nothing{}; // Just to test that outputs can be anything.  Flows<> is more natural choice for empty output.

class World : public OpTuple<Flow<double,std::string>, Nothing, World> {
    using baseT = OpTuple<Flow<double,std::string>, Nothing, World>;

 public:

    World(Flow<double,std::string> pipe)
        : baseT(pipe, Nothing(), "world") {}

    void op(double index, const baseT::input_values_tuple_type& args, baseT::output_type& out) {
        std::cout << std::get<0>(args) << " world!" << std::endl;
    }
};

int main() {
    BaseOp::set_trace(true);
    Flow<int,std::string> control;
    Flow<double,std::string> pipe;

    Hello hello(control, pipe);
    World world(pipe);

    control.send(0,"");

    return 0;
}



    
