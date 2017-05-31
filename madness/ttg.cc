
#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <iostream>
#include <tuple>
#include "madness/ttg.h"

using namespace madness;

using keyT = double;

class A : public  TTGOp<keyT, std::tuple<TTGOut<keyT,int>,TTGOut<keyT,int>>, A, int> {
    using baseT = TTGOp<keyT, std::tuple<TTGOut<keyT,int>,TTGOut<keyT,int>>, A, int>;
 public:
    A(const std::string& name) : baseT(name, {"input"}, {"iterate","result"}) {}

    A(const typename baseT::input_edges_type& inedges, const typename baseT::output_edges_type& outedges, const std::string& name)
        : baseT(inedges, outedges, name, {"input"}, {"result", "iterate"}) {}
    
    void op(const keyT& key, const std::tuple<int>& t, baseT::output_terminals_type& out) {
        int value = std::get<0>(t);
        //std::cout << "A got value " << value << std::endl;
        if (value >= 100) {
            ::send<0>(key, value, out);
        }
        else {
            ::send<1>(key+1, value+1, out);
        }
    }
};

class Producer : public TTGOp<keyT, std::tuple<TTGOut<keyT,int>>, Producer> {
    using baseT =       TTGOp<keyT, std::tuple<TTGOut<keyT,int>>, Producer>;
 public:
    Producer(const std::string& name) : baseT(name, {}, {"output"}) {}
    
    Producer(const typename baseT::output_edges_type& outedges, const std::string& name)
        : baseT(edges(), outedges, name, {}, {"output"}) {}
    
    void op(const keyT& key, const std::tuple<>& t, baseT::output_terminals_type& out) {
        std::cout << "produced " << 0 << std::endl;
        ::send<0>((int)(key),0,out);
    }
};

class Consumer : public TTGOp<keyT, std::tuple<>, Consumer, int> {
    using baseT =       TTGOp<keyT, std::tuple<>, Consumer, int>;
public:
    Consumer(const std::string& name) : baseT(name, {"input"}, {}) {}
    void op(const keyT& key, const std::tuple<int>& t, baseT::output_terminals_type& out) {
        std::cout << "consumed " << std::get<0>(t) << std::endl;
    }

    Consumer(const typename baseT::input_edges_type& inedges, const std::string& name)
        : baseT(inedges, edges(), name, {"input"}, {}) {}
};


class Everything : public TTGOp<keyT, std::tuple<>, Everything> {
    using baseT =         TTGOp<keyT, std::tuple<>, Everything>;
    
    Producer producer;
    A a;
    Consumer consumer;
    
    World& world;
public:
    Everything()
        : baseT("everything",{},{})
        , producer("producer")
        , a("A")
        , consumer("consumer")
        , world(madness::World::get_default())
    {
        producer.out<0>().connect(a.in<0>());
        a.out<0>().connect(consumer.in<0>());
        a.out<1>().connect(a.in<0>());

        TTGVerify()(&producer);
        world.gop.fence();
    }
    
    void print() {TTGPrint()(&producer);}

    std::string dot() {return TTGDot()(&producer);}
    
    void start() {if (world.rank() == 0) producer.invoke(0);}
    
    void wait() {world.gop.fence();}
};


class Everything2 : public TTGOp<keyT, std::tuple<>, Everything2> {
    using baseT =          TTGOp<keyT, std::tuple<>, Everything2>;
    
    Edge<keyT,int> P2A, A2A, A2C; // !!!! Edges must be constructed before classes that use them
    Producer producer;
    A a;
    Consumer consumer;

    World& world;
public:
    Everything2()
        : baseT("everything", {}, {})
        , P2A("P2A"), A2A("A2A"), A2C("A2C")
        , producer(P2A, "producer")
        , a(fuse(P2A,A2A), edges(A2C,A2A), "A")
        , consumer(A2C, "consumer")
        , world(madness::World::get_default())
    {
        world.gop.fence();
    }
    
    void print() {TTGPrint()(&producer);}

    std::string dot() {return TTGDot()(&producer);}
    
    void start() {if (world.rank() == 0) producer.invoke(0);}
    
    void wait() {world.gop.fence();}
};

class Everything3 {

    static void p(const keyT& key, const std::tuple<>& t, std::tuple<TTGOut<keyT,int>>& out) {
        std::cout << "produced " << 0 << std::endl;
        send<0>(key,int(key),out);
    }

    static void a(const keyT& key, const std::tuple<int>& t, std::tuple<TTGOut<keyT,int>,TTGOut<keyT,int>>&  out) {
        int value = std::get<0>(t);
        if (value >= 100) {
            send<0>(key, value, out);
        }
        else {
            send<1>(key+1, value+1, out);
        }
    }

    static void c(const keyT& key, const std::tuple<int>& t, std::tuple<>& out) {
        std::cout << "consumed " << std::get<0>(t) << std::endl;
    }

    Edge<keyT,int> P2A, A2A, A2C; // !!!! Edges must be constructed before classes that use them

    decltype(wrapt<keyT>(&p, edges(), edges(P2A))) wp;
    decltype(wrapt(&a, edges(fuse(P2A,A2A)), edges(A2C,A2A))) wa;
    decltype(wrapt(&c, edges(A2C), edges())) wc;

public:
    Everything3()
        : P2A("P2A"), A2A("A2A"), A2C("A2C")
        , wp(wrapt<keyT>(&p, edges(), edges(P2A), "producer",{},{"start"}))
        , wa(wrapt(&a, edges(fuse(P2A,A2A)), edges(A2C,A2A), "A",{"input"},{"result","iterate"}))
        , wc(wrapt(&c, edges(A2C), edges(), "consumer",{"result"},{}))
    {
        madness::World::get_default().gop.fence();
    }
    
    void print() {TTGPrint()(wp.get());}

    std::string dot() {return TTGDot()(wp.get());}
    
    void start() {if (madness::World::get_default().rank() == 0) wp->invoke(0);}
    
    void wait() {madness::World::get_default().gop.fence();}
};
    
class Everything4 {

    static void p(const keyT& key, std::tuple<TTGOut<keyT,int>>& out) {
        std::cout << "produced " << 0 << std::endl;
        send<0>(key,int(key),out);
    }

    static void a(const keyT& key, int value, std::tuple<TTGOut<keyT,int>,TTGOut<keyT,int>>&  out) {
        if (value >= 100) {
            send<0>(key, value, out);
        }
        else {
            send<1>(key+1, value+1, out);
        }
    }

    static void c(const keyT& key, int value, std::tuple<>& out) {
        std::cout << "consumed " << value << std::endl;
    }

    Edge<keyT,int> P2A, A2A, A2C; // !!!! Edges must be constructed before classes that use them

    decltype(wrap<keyT>(&p, edges(), edges(P2A))) wp;
    decltype(wrap(&a, edges(fuse(P2A,A2A)), edges(A2C,A2A))) wa;
    decltype(wrap(&c, edges(A2C), edges())) wc;

public:
    Everything4()
        : P2A("P2A"), A2A("A2A"), A2C("A2C")
        , wp(wrap<keyT>(&p, edges(), edges(P2A), "producer",{},{"start"}))
        , wa(wrap(&a, edges(fuse(P2A,A2A)), edges(A2C,A2A), "A",{"input"},{"result","iterate"}))
        , wc(wrap(&c, edges(A2C), edges(), "consumer",{"result"},{}))
    {
        madness::World::get_default().gop.fence();
    }
    
    void print() {TTGPrint()(wp.get());}

    std::string dot() {return TTGDot()(wp.get());}
    
    void start() {if (madness::World::get_default().rank() == 0) wp->invoke(0);}
    
    void wait() {madness::World::get_default().gop.fence();}
};
    
int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    
    for (int arg=1; arg<argc; ++arg) {
        if (strcmp(argv[arg],"-dx")==0)
            xterm_debug(argv[0], 0);
    }

    TTGOpBase::set_trace_all(false);

    // First compose with manual classes and connections
    Everything x;
    x.print();
    std::cout << x.dot() << std::endl;

    x.start();
    x.wait();

    // Next compose with manual classes and edges
    Everything2 y;
    y.print();
    std::cout << y.dot() << std::endl;
    
    y.start();
    y.wait();

    // Next compose with wrappers using tuple API and edges
    Everything3 z;
    std::cout << z.dot() << std::endl;
    z.start();
    z.wait();
    
    // Next compose with wrappers using unpacked tuple API and edges
    Everything4 q;
    std::cout << q.dot() << std::endl;
    q.start();
    q.wait();
    
    finalize();
    return 0;
}



    
