#include <cstdint>

using keyT = uint64_t;

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

class A : public Op<keyT, std::tuple<Out<keyT, int>, Out<keyT, int>>,
                       A, int> {
  using baseT =
      Op<keyT, std::tuple<Out<keyT, int>, Out<keyT, int>>, A, int>;

 public:
  A(const std::string& name) : baseT(name, {"input"}, {"result", "iterate"}) {}

  A(const typename baseT::input_edges_type& inedges,
    const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"input"}, {"result", "iterate"}) {}

  void op(const keyT& key, baseT::input_values_tuple_type&& t,
          baseT::output_terminals_type& out) {
    int value = baseT::get<0>(t);
    std::cout << "A got value " << value << std::endl;
    if (value >= 100) {
      ::send<0>(key, value, out);
    } else {
      ::send<1>(key + 1, value + 1, out);
    }
  }
};

class Producer : public Op<keyT, std::tuple<Out<keyT, int>>, Producer> {
  using baseT = Op<keyT, std::tuple<Out<keyT, int>>, Producer>;

 public:
  Producer(const std::string& name) : baseT(name, {}, {"output"}) {}

  Producer(const typename baseT::output_edges_type& outedges,
           const std::string& name)
      : baseT(edges(), outedges, name, {}, {"output"}) {}

  void op(const keyT& key, baseT::input_values_tuple_type&& t,
          baseT::output_terminals_type& out) {
    std::cout << "produced " << 0 << std::endl;
    ::send<0>((int)(key), 0, out);
  }
};

class Consumer : public Op<keyT, std::tuple<>, Consumer, int> {
  using baseT = Op<keyT, std::tuple<>, Consumer, int>;

 public:
  Consumer(const std::string& name) : baseT(name, {"input"}, {}) {}
  void op(const keyT& key, baseT::input_values_tuple_type&& t,
          baseT::output_terminals_type& out) {
    std::cout << "consumed " << baseT::get<0>(t) << std::endl;
  }

  Consumer(const typename baseT::input_edges_type& inedges,
           const std::string& name)
      : baseT(inedges, edges(), name, {"input"}, {}) {}
};

class Everything : public Op<keyT, std::tuple<>, Everything> {
  using baseT = Op<keyT, std::tuple<>, Everything>;

  Producer producer;
  A a;
  Consumer consumer;

 public:
  Everything()
      : baseT("everything", {}, {}),
        producer("producer"),
        a("A"),
        consumer("consumer") {
      connect<0,0>(&producer, &a);
      connect<0,0>(&a, &consumer);
      connect<1,0>(&a, &a);

      Verify()(&producer);
      // ctx->fence();
  }

  void print() { Print()(&producer); }

  std::string dot() { return Dot()(&producer); }

  void start() {
    // if (my rank = 0)
    producer.invoke(0);
  }
};

#if 0
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
        , world(madness::World::get_default())
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
        madness::World::get_default().gop.fence();
    }
    
    void print() {Print()(wp.get());}

    std::string dot() {return Dot()(wp.get());}
    
    void start() {if (madness::World::get_default().rank() == 0) wp->invoke(0);}
    
    void wait() {madness::World::get_default().gop.fence();}
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
        madness::World::get_default().gop.fence();
    }
    
    void print() {Print()(wp.get());}

    std::string dot() {return Dot()(wp.get());}
    
    void start() {if (madness::World::get_default().rank() == 0) wp->invoke(0);}
    
    void wait() {madness::World::get_default().gop.fence();}
};
#endif

int main(int argc, char** argv) {
  ttg_initialize(argc, argv, 2);
  {
    OpBase::set_trace_all(true);

    // First compose with manual classes and connections
    Everything x;
    x.print();
    std::cout << x.dot() << std::endl;

    x.start();

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());

#if 0
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
#endif
  }
  ttg_finalize();

  return 0;
}
