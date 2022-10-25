#include <cstdint>
#include <memory>

#include "ttg.h"

/* TODO: Get rid of using statement */
using namespace ttg;

//#include "../ttg/util/bug.h"

using keyT = uint64_t;

#include "ttg.h"

class A : public TT<keyT, std::tuple<Out<void, int>, Out<keyT, int>>, A, ttg::typelist<const int>> {
  using baseT = typename A::ttT;

 public:
  A(const std::string &name) : baseT(name, {"inputA"}, {"resultA", "iterateA"}) {}

  A(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
    const std::string &name)
      : baseT(inedges, outedges, name, {"inputA"}, {"resultA", "iterateA"}) {}

  static constexpr const bool have_cuda_op = true;

  void op(const keyT &key, const baseT::input_refs_tuple_type &t, baseT::output_terminals_type &out) {
    // int& value = baseT::get<0>(t);  // !! ERROR, trying to get int& from const int
    auto &value = baseT::get<0>(t);
    ttg::print("A got value ", value);
    if (value >= 100) {
      ::sendv<0>(value, out);
    } else {
      ::send<1>(key + 1, value + 1, out);
    }
  }

  void op_cuda(const keyT &key, const baseT::input_refs_tuple_type &t, baseT::output_terminals_type &out) {
    // int& value = baseT::get<0>(t);  // !! ERROR, trying to get int& from const int
    auto &value = baseT::get<0>(t);
    ttg::print("A got value ", value);
    if (value >= 100) {
      ::sendv<0>(value, out);
    } else {
      ::send<1>(key + 1, value + 1, out);
    }
  }

  ~A() { std::cout << " A destructor\n"; }
};

class Producer : public TT<void, std::tuple<Out<keyT, int>>, Producer> {
  using baseT = typename Producer::ttT;

 public:
  Producer(const std::string &name) : baseT(name, {}, {"output"}) {}

  Producer(const typename baseT::output_edges_type &outedges, const std::string &name)
      : baseT(edges(), outedges, name, {}, {"output"}) {}

  void op(baseT::output_terminals_type &out) {
    ttg::print("produced ", 0);
    ::send<0>(0, 0, out);
  }

  ~Producer() { std::cout << " Producer destructor\n"; }
};

class Consumer : public TT<void, std::tuple<>, Consumer, ttg::typelist<const int>> {
  using baseT = typename Consumer::ttT;

 public:
  Consumer(const std::string &name) : baseT(name, {"input"}, {}) {}
  void op(const baseT::input_refs_tuple_type &t, baseT::output_terminals_type &out) {
    ttg::print("consumed ", baseT::get<0>(t));
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

    verify(&producer);
  }

  void print() { print_ttg(&producer); }

  std::string dot() { return Dot{}(&producer); }

  void start() {
    producer.make_executable();
    a.make_executable();
    consumer.make_executable();
    if (default_execution_context().rank() == 0) producer.invoke();
  }
};

class EverythingBase {
  std::unique_ptr<TTBase> producer;
  std::unique_ptr<TTBase> a;
  std::unique_ptr<TTBase> consumer;

 public:
  EverythingBase() : producer(new Producer("producer")), a(new A("A")), consumer(new Consumer("consumer")) {
    connect<0, 0>(producer, a);  // producer->out(0)->connect(a->in(0));
    connect<0, 0>(a, consumer);  // a->out(0)->connect(consumer->in(0));
    connect<1, 0>(a, a);         // a->out(1)->connect(a->in(0));

    verify(producer.get());
  }

  void print() { print_ttg(producer.get()); }

  std::string dot() { return Dot{}(producer.get()); }

  void start() {
    producer->make_executable();
    a->make_executable();
    consumer->make_executable();
    // TODO need abstract base world? or TTBase need to provide TTBase::rank(), etc.
    if (default_execution_context().rank() == 0) dynamic_cast<Producer *>(producer.get())->invoke();
  }  // Ugh!
};

class Everything2 {
  // !!!! Edges must be constructed before classes that use them
  Edge<keyT, int> P2A, A2A;
  Edge<void, int> A2C;
  Producer producer;
  A a;
  Consumer consumer;

 public:
  Everything2()
      : P2A("P2A")
      , A2A("A2A")
      , A2C("A2C")
      , producer(edges(P2A), "producer")
      , a(edges(fuse(P2A, A2A)), edges(A2C, A2A), "A")
      , consumer(edges(A2C), "consumer") {}

  void print() { print_ttg(&producer); }

  std::string dot() { return Dot{}(&producer); }

  void start() {
    producer.make_executable();
    a.make_executable();
    consumer.make_executable();
    if (producer.get_world().rank() == 0) producer.invoke();
  }  // Ugh!
};

class Everything3 {
  static void p(const std::tuple<> &) {
    ttg::print("produced ", 0);
    // N.B.: send(0, 0, int(0)) will produce a runtime error since it will try to cast 0th TerminalBase* to
    // Out<int, int>, but p was attached to Out<keyT, int>
    send(0, keyT{0}, int(0));
  }

  static void a(const keyT &key, const std::tuple<const int &> &t) {
    const auto value = std::get<0>(t);
    if (value >= 100) {
      sendv(0, value);
    } else {
      send(1, key + 1, value + 1);
    }
  }

  static void c(const std::tuple<const int &> &t) { ttg::print("consumed ", std::get<0>(t)); }

  // !!!! Edges must be constructed before classes that use them
  Edge<keyT, int> P2A, A2A;
  Edge<void, int> A2C;

  decltype(make_tt_tpl<void>(p, edges(), edges(P2A))) wp;
  decltype(make_tt_tpl(a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(make_tt_tpl(c, edges(A2C), edges())) wc;

 public:
  Everything3()
      : P2A("P2A")
      , A2A("A2A")
      , A2C("A2C")
      , wp(make_tt_tpl<void>(p, edges(), edges(P2A), "producer", {}, {"start"}))
      , wa(make_tt_tpl(a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"}))
      , wc(make_tt_tpl(c, edges(A2C), edges(), "consumer", {"result"}, {})) {}

  void print() { print_ttg(wp.get()); }

  std::string dot() { return Dot{}(wp.get()); }

  void start() {
    wp->make_executable();
    wa->make_executable();
    wc->make_executable();
    if (wp->get_world().rank() == 0) wp->invoke();
  }
};

class Everything4 {
  static void p() {
    ttg::print("produced ", 0);
    // send<0>(0, 0); // error, deduces int for key, but must be keyT
    send<0>(keyT{0}, 0);
    // also ok:
    if (false) send(0, keyT{0}, 0);
  }

  static void a(const keyT &key, const int &value) {
    if (value >= 100) {
      sendv<0>(value);
      // also ok:
      if (false) sendv(0, value);
    } else {
      send(1, key + 1, value + 1);
      // also ok:
      if (false) send<1>(key + 1, value + 1);
    }
  }

  static void c(const int &value) { ttg::print("consumed ", value); }

  // !!!! Edges must be constructed before classes that use them
  Edge<keyT, int> P2A, A2A;
  Edge<void, int> A2C;

  decltype(make_tt<void>(p, edges(), edges(P2A))) wp;
  decltype(make_tt(a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(make_tt(c, edges(A2C), edges())) wc;

 public:
  Everything4()
      : P2A("P2A")
      , A2A("A2A")
      , A2C("A2C")
      , wp(make_tt<void>(p, edges(), edges(P2A), "producer", {}, {"start"}))
      , wa(make_tt(a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"}))
      , wc(make_tt(c, edges(A2C), edges(), "consumer", {"result"}, {})) {}

  void print() { print_ttg(wp.get()); }

  std::string dot() { return Dot{}(wp.get()); }

  void start() {
    wp->make_executable();
    wa->make_executable();
    wc->make_executable();
    if (wp->get_world().rank() == 0) wp->invoke();
  }
};

class Everything5 {
  static void p(std::tuple<Out<keyT, int>> &out) {
    ttg::print("produced ", 0);
    send<0>(0, 0, out);
  }

  static void a(const keyT &key, const int &value, std::tuple<Out<void, int>, Out<keyT, int>> &out) {
    if (value < 100) {
      send<1>(key + 1, value + 1, out);
      sendv<0>(value, out);
    }
  }

  static void c(const int &value, std::tuple<> &out) { ttg::print("consumed ", value); }

  // !!!! Edges must be constructed before classes that use them
  Edge<keyT, int> P2A, A2A;
  Edge<void, int> A2C;

  decltype(make_tt<void>(p, edges(), edges(P2A))) wp;
  decltype(make_tt(a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(make_tt(c, edges(A2C), edges())) wc;

 public:
  Everything5()
      : P2A("P2A")
      , A2A("A2A")
      , A2C("A2C")
      , wp(make_tt<void>(p, edges(), edges(P2A), "producer", {}, {"start"}))
      , wa(make_tt(a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"}))
      , wc(make_tt(c, edges(A2C), edges(), "consumer", {"result"}, {})) {
    wc->set_input_reducer<0>([](int &a, const int &b) { a += b; });
    if (wc->get_world().rank() == 0) wc->set_argstream_size<0>(100);
  }

  void print() { print_ttg(wp.get()); }

  std::string dot() { return Dot{}(wp.get()); }

  void start() {
    wp->make_executable();
    wa->make_executable();
    wc->make_executable();
    if (wp->get_world().rank() == 0) wp->invoke();
  }
};

class EverythingComposite {
  std::unique_ptr<TTBase> P;
  std::unique_ptr<TTBase> AC;

 public:
  EverythingComposite() {
    auto p = std::make_unique<Producer>("P");
    auto a = std::make_unique<A>("A");
    auto c = std::make_unique<Consumer>("C");

    ttg::print("P out<0>", (void *)(ttg::TerminalBase *)(p->out<0>()));
    ttg::print("A  in<0>", (void *)(ttg::TerminalBase *)(a->in<0>()));
    ttg::print("A out<0>", (void *)(ttg::TerminalBase *)(a->out<0>()));
    ttg::print("C  in<0>", (void *)(ttg::TerminalBase *)(c->in<0>()));

    connect<1, 0>(a, a);  // a->out<1>()->connect(a->in<0>());
    connect<0, 0>(a, c);  // a->out<0>()->connect(c->in<0>());
    const auto q = std::make_tuple(a->in<0>());
    ttg::print("q  in<0>", (void *)(ttg::TerminalBase *)(std::get<0>(q)));

    // std::array<std::unique_ptr<TTBase>,2> ops{std::move(a),std::move(c)};
    std::vector<std::unique_ptr<TTBase>> ops(2);
    ops[0] = std::move(a);
    ops[1] = std::move(c);
    ttg::print("opsin(0)", (void *)(ops[0]->in(0)));
    ttg::print("ops size", ops.size());

    auto ac = make_ttg(std::move(ops), q, std::make_tuple(), "Fred");

    ttg::print("AC in<0>", (void *)(ttg::TerminalBase *)(ac->in<0>()));
    connect<0, 0>(p, ac);  // p->out<0>()->connect(ac->in<0>());

    verify(p.get());

    P = std::move(p);
    AC = std::move(ac);
  }

  void print() { print_ttg(P.get()); }

  std::string dot() { return Dot{}(P.get()); }

  void start() {
    Producer *p = dynamic_cast<Producer *>(P.get());
    P->make_executable();
    AC->make_executable();
    if (default_execution_context().rank() == 0) p->invoke();
  }  // Ugh!
};

class ReductionTest {
  static void generator(const int &key, std::tuple<Out<int, int>> &out) {
    const auto value = std::rand();
    ttg::print("ReductionTest: produced ", value, " on rank ", ttg::default_execution_context().rank());
    send<0>(key, value, out);
  }
  static void consumer(const int &key, const int &value, std::tuple<> &out) {
    ttg::print("ReductionTest: consumed ", value);
  }

  Edge<int, int> G2R, R2C;  // !!!! Edges must be constructed before classes that use them

  decltype(make_tt<int>(generator, edges(), edges(G2R))) wg;
  BinaryTreeReduce<int, decltype(std::plus<int>{}), int> reduction;
  decltype(make_tt(consumer, edges(R2C), edges())) wc;

 public:
  ReductionTest()
      : G2R("G2R")
      , R2C("R2C")
      , wg(make_tt<int>(generator, edges(), edges(G2R), "producer", {}, {"start"}))
      , reduction(G2R, R2C, 0, 0, std::plus<int>{})
      , wc(make_tt(consumer, edges(R2C), edges(), "consumer", {"result"}, {})) {}

  void print() { print_ttg(wg.get()); }

  std::string dot() { return Dot{}(wg.get()); }

  void start() {
    wg->make_executable();
    reduction.make_executable();
    wc->make_executable();
    wg->invoke(ttg::default_execution_context().rank());
  }
};

class BroadcastTest {
  static void generator(const int &key, std::tuple<Out<int, int>> &out) {
    const auto value = std::rand();
    ttg::print("BroadcastTest: produced ", value, " on rank ", ttg::default_execution_context().rank());
    send<0>(key, value, out);
  }
  static void consumer(const int &key, const int &value, std::tuple<> &out) {
    ttg::print("BroadcastTest: consumed ", value, " on rank ", ttg::default_execution_context().rank());
  }

  Edge<int, int> G2B, B2C;

  int root;

  decltype(make_tt<int>(generator, edges(), edges(G2B))) wg;
  BinaryTreeBroadcast<int, int> broadcast;
  decltype(make_tt(consumer, edges(B2C), edges())) wc;

 public:
  BroadcastTest(int root = 0)
      : G2B("G2B")
      , B2C("B2C")
      , root(root)
      , wg(make_tt<int>(generator, edges(), edges(G2B), "producer", {}, {"start"}))
      , broadcast(G2B, B2C, {ttg::default_execution_context().rank()}, root)
      , wc(make_tt(consumer, edges(B2C), edges(), "consumer", {"result"}, {})) {}

  void print() { print_ttg(wg.get()); }

  std::string dot() { return Dot{}(wg.get()); }

  void start() {
    wg->make_executable();
    broadcast.make_executable();
    wc->make_executable();
    if (wg->get_world().rank() == root) wg->invoke(root);
  }
};

int try_main(int argc, char **argv) {
  ttg::initialize(argc, argv, 2);

  //  using ttg::Debugger;
  //  auto debugger = std::make_shared<Debugger>();
  //  Debugger::set_default_debugger(debugger);
  //  debugger->set_exec(argv[0]);
  //  debugger->set_prefix(ttg::default_execution_context().rank());
  //  debugger->set_cmd("lldb_xterm");
  //  debugger->set_cmd("gdb_xterm");

  {
    ttg::TTBase::set_trace_all(false);

    ttg::execute();

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

    // Next compose with make_ttpers using unpacked tuple API and edges
    Everything4 q;
    std::cout << q.dot() << std::endl;
    q.start();  // myusleep(100);

#if 0  // can we make something like this happen? ... edges are named, but terminals are indexed, so components are
       // "reusable"
    {
      TTG ttg;
      auto [A, B] = ttg.edges("A", "B");  // TODO: abstract Edge
      ttg.emplace([&]() {
        std::cout << "produced 0" << std::endl;
        sendk(A, 0);
      })
      .emplace([&](int key) {
           if (key <= 100) {
             sendk(A, key + 1);
           }
           else {
             sendk(B, key);
           }
          }, A)
       .emplace([](int key) {
           std::cout << "consumed" << key << std::endl;
          }, B);
      ttg.submit(execution_context);

      // how to compose with existing functions
      ttg.emplace(fn, /* inedges = */ ttg.edges("A", "B"), /* outedges = */ ttg.edges("C"));  // arguments of fn are used to deduce task id + argument types
      // this is equivalen to
      ttg.emplace([&](const T1& a, const T2& b) {
        auto c = fn(a, b);
        sendv(C, c);
      }, A, B);
    }
#endif

    if constexpr (runtime_traits<ttg_runtime>::supports_streaming_terminal) {
      // Everything5 = Everything4 with consumer summing all values from A using a stream reducer
      Everything5 q5;
      std::cout << q5.dot() << std::endl;
      q5.start();

      // must fence here to flush out all tasks associated with Everything5
      // TODO must fence in TT destructors to avoid compositional nightmares like this
      ttg::fence();
    }

    ReductionTest t;
    t.start();

    BroadcastTest b;
    b.start();

    ttg::fence();
    std::cout << "\nFence done\n";
  }
  ttg::finalize();
  return 0;
}

int main(int argc, char **argv) {
  try {
    try_main(argc, argv);
  } catch (std::exception &x) {
    std::cerr << "Caught a std::exception: " << x.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Caught an unknown exception: " << std::endl;
    return 1;
  }
}
