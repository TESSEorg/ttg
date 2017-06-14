
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <iostream>
#include <tuple>
#include <memory>

#include "madness/ttg.h"

using namespace madness;
using namespace madness::ttg;
using namespace ::ttg;

using keyT = double;

class A : public  Op<keyT, std::tuple<Out<keyT,int>,Out<keyT,int>>, A, int> {
    using baseT = Op<keyT, std::tuple<Out<keyT,int>,Out<keyT,int>>, A, int>;
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

    ~A() {std::cout << "A destructor\n";}
};

class Producer : public Op<keyT, std::tuple<Out<keyT,int>>, Producer> {
    using baseT =       Op<keyT, std::tuple<Out<keyT,int>>, Producer>;
 public:
    Producer(const std::string& name) : baseT(name, {}, {"output"}) {}
    
    Producer(const typename baseT::output_edges_type& outedges, const std::string& name)
        : baseT(edges(), outedges, name, {}, {"output"}) {}
    
    void op(const keyT& key, const std::tuple<>& t, baseT::output_terminals_type& out) {
        std::cout << "produced " << 0 << std::endl;
        ::send<0>((int)(key),0,out);
    }

    ~Producer() {std::cout << "Producer destructor\n";}
};

class Consumer : public Op<keyT, std::tuple<>, Consumer, int> {
    using baseT =       Op<keyT, std::tuple<>, Consumer, int>;
public:
    Consumer(const std::string& name) : baseT(name, {"input"}, {}) {}
    void op(const keyT& key, const std::tuple<int>& t, baseT::output_terminals_type& out) {
        std::cout << "consumed " << std::get<0>(t) << std::endl;
    }

    Consumer(const typename baseT::input_edges_type& inedges, const std::string& name)
        : baseT(inedges, edges(), name, {"input"}, {}) {}

    ~Consumer() {std::cout << "Consumer destructor\n";}
};


class Everything : public Op<keyT, std::tuple<>, Everything> {
    using baseT =         Op<keyT, std::tuple<>, Everything>;
    
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

        Verify()(&producer);
        world.gop.fence();
    }
    
    void print() {Print()(&producer);}

    std::string dot() {return Dot()(&producer);}
    
    void start() {if (world.rank() == 0) producer.invoke(0);}
    
    void wait() {world.gop.fence();}
};


class EverythingBase {
    std::unique_ptr<OpBase> producer;
    std::unique_ptr<OpBase> a;
    std::unique_ptr<OpBase> consumer;
    
    World& world;
public:
    EverythingBase()
        : producer(new Producer("producer"))
        , a(new A("A"))
        , consumer(new Consumer("consumer"))
        , world(madness::World::get_default())
    {
        producer->out(0)->connect(a->in(0));
        a->out(0)->connect(consumer->in(0));
        a->out(1)->connect(a->in(0));

        Verify()(producer.get());
        world.gop.fence();
    }
    
    void print() {Print()(producer.get());}

    std::string dot() {return Dot()(producer.get());}
    
    void start() {if (world.rank() == 0) dynamic_cast<Producer*>(producer.get())->invoke(0);} // Ugh!
    
    void wait() {world.gop.fence();}
};


class Everything2 : public Op<keyT, std::tuple<>, Everything2> {
    using baseT =          Op<keyT, std::tuple<>, Everything2>;
    
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
        , world(::madness::World::get_default())
    {
        world.gop.fence();
    }
    
    void print() {Print()(&producer);}

    std::string dot() {return Dot()(&producer);}
    
    void start() {if (world.rank() == 0) producer.invoke(0);}
    
    void wait() {world.gop.fence();}
};

class Everything3 {

    static void p(const keyT& key, const std::tuple<>& t, std::tuple<Out<keyT,int>>& out) {
        std::cout << "produced " << 0 << std::endl;
        send<0>(key,int(key),out);
    }

    static void a(const keyT& key, const std::tuple<int>& t, std::tuple<Out<keyT,int>,Out<keyT,int>>&  out) {
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
        ::madness::World::get_default().gop.fence();
    }
    
    void print() {Print()(wp.get());}

    std::string dot() {return Dot()(wp.get());}
    
    void start() {if (::madness::World::get_default().rank() == 0) wp->invoke(0);}
    
    void wait() {::madness::World::get_default().gop.fence();}
};
    
class Everything4 {

    static void p(const keyT& key, std::tuple<Out<keyT,int>>& out) {
        std::cout << "produced " << 0 << std::endl;
        send<0>(key,int(key),out);
    }

    static void a(const keyT& key, int value, std::tuple<Out<keyT,int>,Out<keyT,int>>&  out) {
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
        ::madness::World::get_default().gop.fence();
    }
    
    void print() {Print()(wp.get());}

    std::string dot() {return Dot()(wp.get());}
    
    void start() {if (::madness::World::get_default().rank() == 0) wp->invoke(0);}
    
    void wait() { ::madness::World::get_default().gop.fence();}
};

template <typename input_terminalsT, typename output_terminalsT>
class CompositeOp {

public:

    static constexpr int numins = std::tuple_size< input_terminalsT>::value;  // number of input arguments
    static constexpr int numouts= std::tuple_size<output_terminalsT>::value;  // number of outputs or results

    using input_terminals_type = input_terminalsT;
    using input_edges_type = typename ::ttg::terminals_to_edges<input_terminalsT>::type;

    using output_terminals_type = output_terminalsT;
    using output_edges_type = typename ::ttg::terminals_to_edges<output_terminalsT>::type;
 
    template <typename...opTs, >;
    CompositeOp(std::tuple<std::unique_ptr<opTs>...>&& ops, std::array<



};
    


    

int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    
    for (int arg=1; arg<argc; ++arg) {
        if (strcmp(argv[arg],"-dx")==0)
            xterm_debug(argv[0], 0);
    }

    OpBase::set_trace_all(false);

    // First compose with manual classes and connections
    {
      Everything x;
      x.print();
      std::cout << x.dot() << std::endl;
      
      x.start();
      x.wait();
    }

    // Next compose with base class pointers and verify destruction
    {
      EverythingBase x;
      x.print();
      std::cout << x.dot() << std::endl;
      
      x.start();
      x.wait();
    }

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



    
