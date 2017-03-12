#include <iostream>
#include "edge.h"

class A : public  Op<int, std::tuple<double>, std::tuple<>, A> {
    using baseT = Op<int, std::tuple<double>, std::tuple<>, A>;
 public:
    A(const std::string& name) : baseT(name) {}
    void op(const int& key, const std::tuple<double>& t) {
        std::cout << " A got " << key << " " << std::get<0>(t) << std::endl;
    }
};

class B : public  Op<int, std::tuple<double>, std::tuple<OutEdge<int,double>>, B> {
    using baseT = Op<int, std::tuple<double>, std::tuple<OutEdge<int,double>>, B>;
 public:
    B(const std::string& name) : baseT(name) {}
    void op(const int& key, const std::tuple<double>& t) {
        std::cout << " B got " << key << " " << std::get<0>(t) << std::endl;
        send<0>(key,std::get<0>(t));
    }
};

int main() {

    BaseOp::set_trace(true);

    A a("fred");
    B b("mary");

    b.out<0>().connect(a.in<0>());
    b.in<0>().send(1,3.14);

    return 0;
}



    
