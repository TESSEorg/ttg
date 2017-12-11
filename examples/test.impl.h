#include <cstdint>
#include <memory>

//#include <mpqc/util/misc/bug.h>

using keyT = uint64_t;

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

class A : public Op<keyT, std::tuple<Out<keyT, int>, Out < keyT, int>>, A, const int
> {
using baseT = Op<keyT, std::tuple<Out<keyT, int>, Out < keyT, int>>, A, const int>;

public:
A(const std::string &name) : baseT(name, {"input"}, {"iterate", "result"}) {}

A(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
  const std::string &name)
    : baseT(inedges, outedges, name, {"input"}, {"result", "iterate"}) {}

void op(const keyT &key, baseT::input_values_tuple_type &&t, baseT::output_terminals_type &out) {
  // int& value = baseT::get<0>(t);  // !! ERROR, trying to get int& from const int
  auto& value = baseT::get<0>(t);
  ::ttg::print("A got value ", value);
  if (value >= 100) {
    ::send<0>(key, value, out);
  } else {
    ::send<1>(key + 1, value + 1, out);
  }
}

~A() { std::cout << " A destructor\n"; }
};

class Producer : public Op<keyT, std::tuple<Out < keyT, int>>, Producer
> {
using baseT = Op<keyT, std::tuple<Out < keyT, int>>, Producer>;

public:
Producer(const std::string &name) : baseT(name, {}, {"output"}) {}

Producer(const typename baseT::output_edges_type &outedges, const std::string &name)
    : baseT(edges(), outedges, name, {}, {"output"}) {}

void op(const keyT &key, baseT::input_values_tuple_type &&t, baseT::output_terminals_type &out) {
  ::ttg::print("produced ", 0);
  ::send<0>((int) (key), 0, out);
}

~Producer() { std::cout << " Producer destructor\n"; }
};

class Consumer : public Op<keyT, std::tuple<>, Consumer, const int> {
  using baseT = Op<keyT, std::tuple<>, Consumer, const int>;

 public:
  Consumer(const std::string &name) : baseT(name, {"input"}, {}) {}
  void op(const keyT &key, baseT::input_values_tuple_type &&t, baseT::output_terminals_type &out) {
    ::ttg::print("consumed ", baseT::get<0>(t));
  }

  Consumer(const typename baseT::input_edges_type &inedges, const std::string &name)
      : baseT(inedges, edges(), name, {"input"}, {}) {}

  ~Consumer() { std::cout << " Consumer destructor\n"; }
};

class Everything {
  Producer producer;
  A a;
  Consumer consumer;

 public:
  Everything() : producer("producer"), a("A"), consumer("consumer") {
    connect<0, 0>(&producer, &a);  // producer.out<0>()->connect(a.in<0>());
    connect<0, 0>(&a, &consumer);  // a.out<0>()->connect(consumer.in<0>());
    connect<1, 0>(&a, &a);         // a.out<1>()->connect(a.in<0>());

    Verify()(&producer);
  }

  void print() { Print()(&producer); }

  std::string dot() { return Dot()(&producer); }

  void start() {
    producer.invoke(0);
  }
};

class EverythingBase {
  std::unique_ptr<OpBase> producer;
  std::unique_ptr<OpBase> a;
  std::unique_ptr<OpBase> consumer;

 public:
  EverythingBase()
      : producer(new Producer("producer")), a(new A("A")), consumer(new Consumer("consumer")) {
    connect<0, 0>(producer, a);  // producer->out(0)->connect(a->in(0));
    connect<0, 0>(a, consumer);  // a->out(0)->connect(consumer->in(0));
    connect<1, 0>(a, a);         // a->out(1)->connect(a->in(0));

    Verify()(producer.get());
  }

  void print() { Print()(producer.get()); }

  std::string dot() { return Dot()(producer.get()); }

  void start() {
    if (ttg_default_execution_context().rank() == 0) dynamic_cast<Producer *>(producer.get())->invoke(0);
  }  // Ugh!
};

class Everything2 {
  Edge<keyT, int> P2A, A2A, A2C;  // !!!! Edges must be constructed before classes that use them
  Producer producer;
  A a;
  Consumer consumer;

 public:
  Everything2()
      : P2A("P2A"),
        A2A("A2A"),
        A2C("A2C"),
        producer(edges(P2A), "producer"),
        a(edges(fuse(P2A, A2A)), edges(A2C, A2A), "A"),
        consumer(edges(A2C), "consumer") {
  }

  void print() { Print()(&producer); }

  std::string dot() { return Dot()(&producer); }

  void start() {
    if (ttg_default_execution_context().rank() == 0) producer.invoke(0);
  }  // Ugh!
};

class Everything3 {
  static void p(const keyT &key, std::tuple<> &&t, std::tuple<Out < keyT, int>>
  & out) {
    ::ttg::print("produced ", 0);
    send<0>(key, int(0), out);
  }

  static void a(const keyT &key, std::tuple<const int> &&t, std::tuple<Out < keyT, int>, Out<keyT, int>>
  & out) {
    const auto value = std::get<0>(t);
    if (value >= 100) {
      send<0>(key, value, out);
    } else {
      send<1>(key + 1, value + 1, out);
    }
  }

  static void c(const keyT &key, std::tuple<const int> &&t, std::tuple<> &out) {
    ::ttg::print("consumed ", std::get<0>(t));
  }

  Edge<keyT, int> P2A, A2A, A2C;  // !!!! Edges must be constructed before classes that use them

  decltype(wrapt<keyT>(&p, edges(), edges(P2A))) wp;
  decltype(wrapt(&a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(wrapt(&c, edges(A2C), edges())) wc;

 public:
  Everything3()
      : P2A("P2A"),
        A2A("A2A"),
        A2C("A2C"),
        wp(wrapt<keyT>(&p, edges(), edges(P2A), "producer", {}, {"start"})),
        wa(wrapt(&a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"})),
        wc(wrapt(&c, edges(A2C), edges(), "consumer", {"result"}, {})) {
  }

  void print() { Print()(wp.get()); }

  std::string dot() { return Dot()(wp.get()); }

  void start() {
    if (ttg_default_execution_context().rank() == 0) wp->invoke(0);
  }
};

class Everything4 {
  static void p(const keyT &key, std::tuple<Out<keyT, int>>
  &out) {
    ::ttg::print("produced ", 0);
    send<0>(key, int(0), out);
  }

  static void a(const keyT &key, const int &value, std::tuple<Out<keyT, int>, Out<keyT, int>>
  &out) {
    if (value >= 100) {
      send<0>(key, value, out);
    } else {
      send<1>(key + 1, value + 1, out);
    }
  }

  static void c(const keyT &key, const int &value, std::tuple<> &out) {
    ::ttg::print("consumed ", value);
  }

  Edge<keyT, int> P2A, A2A, A2C;  // !!!! Edges must be constructed before classes that use them

  decltype(wrap<keyT>(&p, edges(), edges(P2A))) wp;
  decltype(wrap(&a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(wrap(&c, edges(A2C), edges())) wc;

 public:
  Everything4()
      : P2A("P2A"),
        A2A("A2A"),
        A2C("A2C"),
        wp(wrap<keyT>(&p, edges(), edges(P2A), "producer", {}, {"start"})),
        wa(wrap(&a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"})),
        wc(wrap(&c, edges(A2C), edges(), "consumer", {"result"}, {})) {
  }

  void print() { Print()(wp.get()); }

  std::string dot() { return Dot()(wp.get()); }

  void start() {
    if (ttg_default_execution_context().rank() == 0) wp->invoke(0);
  }
};

class EverythingComposite {
  std::unique_ptr<OpBase> P;
  std::unique_ptr<OpBase> AC;

 public:
  EverythingComposite() {
    auto p = std::make_unique<Producer>("P");
    auto a = std::make_unique<A>("A");
    auto c = std::make_unique<Consumer>("C");

    ::ttg::print("P out<0>", (void *) (TerminalBase *) (p->out<0>()));
    ::ttg::print("A  in<0>", (void *) (TerminalBase *) (a->in<0>()));
    ::ttg::print("A out<0>", (void *) (TerminalBase *) (a->out<0>()));
    ::ttg::print("C  in<0>", (void *) (TerminalBase *) (c->in<0>()));

    connect<1, 0>(a, a);  // a->out<1>()->connect(a->in<0>());
    connect<0, 0>(a, c);  // a->out<0>()->connect(c->in<0>());
    const auto q = std::make_tuple(a->in<0>());
    ::ttg::print("q  in<0>", (void *) (TerminalBase *) (std::get<0>(q)));

    // std::array<std::unique_ptr<OpBase>,2> ops{std::move(a),std::move(c)};
    std::vector<std::unique_ptr<OpBase>> ops(2);
    ops[0] = std::move(a);
    ops[1] = std::move(c);
    ::ttg::print("opsin(0)", (void *) (ops[0]->in(0)));
    ::ttg::print("ops size", ops.size());

    auto ac = make_composite_op(std::move(ops), q, std::make_tuple(), "Fred");

    ::ttg::print("AC in<0>", (void *) (TerminalBase *) (ac->in<0>()));
    connect<0, 0>(p, ac);  // p->out<0>()->connect(ac->in<0>());

    Verify()(p.get());

    P = std::move(p);
    AC = std::move(ac);
  }

  void print() { Print()(P.get()); }

  std::string dot() { return Dot()(P.get()); }

  void start() {
    Producer * p = dynamic_cast<Producer *>(P.get());
    p->invoke(0);
  }  // Ugh!
};

void hi() { ::ttg::print("hi"); }

int try_main(int argc, char **argv) {
  ttg_initialize(argc, argv, 2);

//  using mpqc::Debugger;
//  auto debugger = std::make_shared<Debugger>();
//  Debugger::set_default_debugger(debugger);
//  debugger->set_exec(argv[0]);
//  debugger->set_prefix(ttg_default_execution_context().rank());
//  debugger->set_cmd("xterm -title \"$(PREFIX)$(EXEC)\" -e lldb -p $(PID) &");

  {

    OpBase::set_trace_all(false);

    ttg_execute(ttg_default_execution_context());

    // First compose with manual classes and connections
    Everything x;
    x.print();
    std::cout << x.dot() << std::endl;

    x.start();  // myusleep(100);

    // Next compose with base class pointers and verify destruction
    EverythingBase x1;
    x1.print();
    std::cout << x1.dot() << std::endl;

    x1.start();  // myusleep(100);

    // Now try with the composite operator wrapping A and C
    EverythingComposite x2;

    std::cout << "\nComposite\n";
    std::cout << x2.dot() << std::endl << std::endl;
    x2.start();  // myusleep(100);
    std::cout << "\nComposite done\n";

    // Next compose with manual classes and edges
    Everything2 y;
    y.print();
    std::cout << y.dot() << std::endl;
    y.start();  // myusleep(100);

    // Next compose with wrappers using tuple API and edges
    Everything3 z;
    std::cout << z.dot() << std::endl;
    z.start();  // myusleep(100);

    // Next compose with wrappers using unpacked tuple API and edges
    Everything4 q;
    std::cout << q.dot() << std::endl;
    q.start();  // myusleep(100);

    // can we compose directly (with free functions/lambdas) and type-safely like this?
#if 0
    {
      ttg.let([]() {
           std::cout << "produced 0" << std::endl;
           return {"A", 0, 0};
          })
         .let([](auto &&key, auto &&value) {
           if (key <= 100) {
             return {"A", key + 1, value + 1};
           }
           else {
             return {"B", 0, value};
           }
          })
         .attach_inputs("A")
         .let([](auto &&key, auto &&value) {
           std::cout << "consumed" << value << std::endl;
          })
         .attach_inputs("B");
      ttg.submit(execution_context);
    }
#endif

    ttg_fence(ttg_default_execution_context());
    std::cout << "\nFence done\n";
  }
  ttg_finalize();
  return 0;
}

int main(int argc, char **argv) {
  try {
    try_main(argc, argv);
  }
  catch (std::exception &x) {
    std::cerr << "Caught a std::exception: " << x.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "Caught an unknown exception: " << std::endl;
    return 1;
  }
}