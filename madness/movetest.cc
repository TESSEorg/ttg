#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <iostream>
#include <tuple>
#include "madness/ttg.h"

using namespace madness;
using namespace madness::ttg;
using namespace ::ttg;

class Value {
    double value;

public:
    Value () : value(0.0) {std::cout << "default con\n";}
    Value (double v) : value(v) {std::cout << "double con " << value << std::endl;}
    Value (const Value& v) : value(v.value)  {std::cout << "copy con " << value << std::endl;}
    Value(Value&& v) : value(v.value) {v.value=-1.0; std::cout << "move con " << value << std::endl;}
    Value& operator=(const Value& v) {
        value = v.value;
        std::cout << "copy assignment " << value << std::endl;
        return *this;
    }
    Value& operator=(Value&& v) {
        value = v.value; v.value=-1.0;
        std::cout << "move assignment " << value << std::endl;
        return *this;
    }

    double get() const {return value;}

    template <typename Archive>
    void serialize(Archive& ar) {ar & value; std::cout << "(de)serialize\n";}
        
    ~Value () {}
};

class A : public  Op<int, std::tuple<Out<int,Value>>, A, Value> {
    using baseT = Op<int, std::tuple<Out<int,Value>>, A, Value>;
 public:
    A(const std::string& name) : baseT(name, {"input"}, {"result"}) {}
    
    void op(const int& key, std::tuple<Value>&& t, baseT::output_terminals_type& out) {
        std::cout << "in A::op\n";
        Value value = std::get<0>(std::forward<std::tuple<Value>>(t));
        std::cout << "A::op got value " << value.get() << std::endl;
        ::send<0>(key, std::move(value), out);
    }

    ~A() {std::cout << " A destructor\n";}
};

class Printer : public Op<int, std::tuple<>, Printer, Value> {
    using baseT =      Op<int, std::tuple<>, Printer, Value>;

public:
    Printer() : baseT("printer", {"input"}, {}) {}

    void op(const int& key, std::tuple<Value>&& t, baseT::output_terminals_type& out) {
        std::cout << "in Printer::op\n";
        Value value = std::get<0>(std::forward<std::tuple<Value>>(t));
        std::cout << "Printer::op got value " << value.get() << std::endl;
    }
};

int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    set_default_world(world);

    std::cout << "a\n";
    Value a;
    std::cout << "b\n";
    Value b(2.0);
    std::cout << "c\n";
    Value c(a);
    std::cout << "d\n";
    Value d(Value(3.0));
    std::cout << "e\n";
    Value e(std::move(b));

    std::cout << "e=b\n";
    e = b;
    std::cout << "e=std::move(b)\n";
    e = std::move(b);
    std::cout << "e=rvalue\n";
    e = Value(99.0);

    A x("A");
    Printer p;
    connect<0,0>(&x,&p);
    x.in<0>()->send(0,Value(33));

    world.gop.fence();
    finalize();
    return 0;
}
