#include <cstdint>
#include <memory>

#include "ttg.h"

/* TODO: Get rid of using statement */
using namespace ttg;

//#include "../ttg/util/bug.h"

using keyT = uint64_t;

#include "ttg.h"

class A : public Op<keyT, std::tuple<Out<void, int>, Out<keyT, int>>, A, const int> {
  using baseT = Op<keyT, std::tuple<Out<void, int>, Out<keyT, int>>, A, const int>;

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

class Producer : public Op<void, std::tuple<Out<keyT, int>>, Producer> {
  using baseT = Op<void, std::tuple<Out<keyT, int>>, Producer>;

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

class Consumer : public Op<void, std::tuple<>, Consumer, const int> {
  using baseT = Op<void, std::tuple<>, Consumer, const int>;

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
    if (ttg_default_execution_context().rank() == 0) producer.invoke();
  }
};

class EverythingBase {
  std::unique_ptr<OpBase> producer;
  std::unique_ptr<OpBase> a;
  std::unique_ptr<OpBase> consumer;

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
    // TODO need abstract base world? or OpBase need to provide OpBase::rank(), etc.
    if (ttg_default_execution_context().rank() == 0) dynamic_cast<Producer *>(producer.get())->invoke();
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
  static void p(const std::tuple<> &, std::tuple<Out<keyT, int>> &out) {
    ttg::print("produced ", 0);
    send<0>(0, int(0), out);
  }

  static void a(const keyT &key, const std::tuple<const int &> &t, std::tuple<Out<void, int>, Out<keyT, int>> &out) {
    const auto value = std::get<0>(t);
    if (value >= 100) {
      sendv<0>(value, out);
    } else {
      send<1>(key + 1, value + 1, out);
    }
  }

  static void c(const std::tuple<const int &> &t, std::tuple<> &out) { ttg::print("consumed ", std::get<0>(t)); }

  // !!!! Edges must be constructed before classes that use them
  Edge<keyT, int> P2A, A2A;
  Edge<void, int> A2C;

  decltype(wrapt<void>(p, edges(), edges(P2A))) wp;
  decltype(wrapt(a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(wrapt(c, edges(A2C), edges())) wc;

 public:
  Everything3()
      : P2A("P2A")
      , A2A("A2A")
      , A2C("A2C")
      , wp(wrapt<void>(p, edges(), edges(P2A), "producer", {}, {"start"}))
      , wa(wrapt(a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"}))
      , wc(wrapt(c, edges(A2C), edges(), "consumer", {"result"}, {})) {}

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
  static void p(std::tuple<Out<keyT, int>> &out) {
    ttg::print("produced ", 0);
    send<0>(0, 0, out);
  }

  static void a(const keyT &key, const int &value, std::tuple<Out<void, int>, Out<keyT, int>> &out) {
    if (value >= 100) {
      sendv<0>(value, out);
    } else {
      send<1>(key + 1, value + 1, out);
    }
  }

  static void c(const int &value, std::tuple<> &out) { ttg::print("consumed ", value); }

  // !!!! Edges must be constructed before classes that use them
  Edge<keyT, int> P2A, A2A;
  Edge<void, int> A2C;

  decltype(wrap<void>(p, edges(), edges(P2A))) wp;
  decltype(wrap(a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(wrap(c, edges(A2C), edges())) wc;

 public:
  Everything4()
      : P2A("P2A")
      , A2A("A2A")
      , A2C("A2C")
      , wp(wrap<void>(p, edges(), edges(P2A), "producer", {}, {"start"}))
      , wa(wrap(a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"}))
      , wc(wrap(c, edges(A2C), edges(), "consumer", {"result"}, {})) {}

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

  decltype(wrap<void>(p, edges(), edges(P2A))) wp;
  decltype(wrap(a, edges(fuse(P2A, A2A)), edges(A2C, A2A))) wa;
  decltype(wrap(c, edges(A2C), edges())) wc;

 public:
  Everything5()
      : P2A("P2A")
      , A2A("A2A")
      , A2C("A2C")
      , wp(wrap<void>(p, edges(), edges(P2A), "producer", {}, {"start"}))
      , wa(wrap(a, edges(fuse(P2A, A2A)), edges(A2C, A2A), "A", {"input"}, {"result", "iterate"}))
      , wc(wrap(c, edges(A2C), edges(), "consumer", {"result"}, {})) {
    wc->set_input_reducer<0>([](int &&a, int &&b) { return a + b; });
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
  std::unique_ptr<OpBase> P;
  std::unique_ptr<OpBase> AC;

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

    // std::array<std::unique_ptr<OpBase>,2> ops{std::move(a),std::move(c)};
    std::vector<std::unique_ptr<OpBase>> ops(2);
    ops[0] = std::move(a);
    ops[1] = std::move(c);
    ttg::print("opsin(0)", (void *)(ops[0]->in(0)));
    ttg::print("ops size", ops.size());

    auto ac = make_composite_op(std::move(ops), q, std::make_tuple(), "Fred");

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
    if (ttg_default_execution_context().rank() == 0) p->invoke();
  }  // Ugh!
};

class ReductionTest {
  static void generator(const int &key, std::tuple<Out<int, int>> &out) {
    const auto value = std::rand();
    ttg::print("ReductionTest: produced ", value, " on rank ", ttg_default_execution_context().rank());
    send<0>(key, value, out);
  }
  static void consumer(const int &key, const int &value, std::tuple<> &out) {
    ttg::print("ReductionTest: consumed ", value);
  }

  Edge<int, int> G2R, R2C;  // !!!! Edges must be constructed before classes that use them

  decltype(wrap<int>(generator, edges(), edges(G2R))) wg;
  BinaryTreeReduce<int, decltype(std::plus<int>{}), int> reduction;
  decltype(wrap(consumer, edges(R2C), edges())) wc;

 public:
  ReductionTest()
      : G2R("G2R")
      , R2C("R2C")
      , wg(wrap<int>(generator, edges(), edges(G2R), "producer", {}, {"start"}))
      , reduction(G2R, R2C, 0, 0, std::plus<int>{})
      , wc(wrap(consumer, edges(R2C), edges(), "consumer", {"result"}, {})) {}

  void print() { print_ttg(wg.get()); }

  std::string dot() { return Dot{}(wg.get()); }

  void start() {
    wg->make_executable();
    reduction.make_executable();
    wc->make_executable();
    wg->invoke(ttg_default_execution_context().rank());
  }
};

class BroadcastTest {
  static void generator(const int &key, std::tuple<Out<int, int>> &out) {
    const auto value = std::rand();
    ttg::print("BroadcastTest: produced ", value, " on rank ", ttg_default_execution_context().rank());
    send<0>(key, value, out);
  }
  static void consumer(const int &key, const int &value, std::tuple<> &out) {
    ttg::print("BroadcastTest: consumed ", value, " on rank ", ttg_default_execution_context().rank());
  }

  Edge<int, int> G2B, B2C;

  int root;

  decltype(wrap<int>(generator, edges(), edges(G2B))) wg;
  BinaryTreeBroadcast<int, int> broadcast;
  decltype(wrap(consumer, edges(B2C), edges())) wc;

 public:
  BroadcastTest(int root = 0)
      : G2B("G2B")
      , B2C("B2C")
      , root(root)
      , wg(wrap<int>(generator, edges(), edges(G2B), "producer", {}, {"start"}))
      , broadcast(G2B, B2C, {ttg_default_execution_context().rank()}, root)
      , wc(wrap(consumer, edges(B2C), edges(), "consumer", {"result"}, {})) {}

  void print() { print_ttg(wg.get()); }

  std::string dot() { return Dot{}(wg.get()); }

  void start() {
    wg->make_executable();
    broadcast.make_executable();
    wc->make_executable();
    if (wg->get_world().rank() == root) wg->invoke(root);
  }
};

// Computes Fibonacci numbers up to some value
class Fibonacci {
  // compute all numbers up to this value
  static constexpr const int max() { return 1000; }

  // computes next value: F_{n+2} = F_{n+1} + F_{n}, seeded by F_1 = 2, F_0 = 1
  static void next(const int &F_np1 /* aka key */, const int &F_n, std::tuple<Out<int, int>, Out<int, int>> &outs) {
    // if this is first call reduce F_np1 and F_n also
    if (F_np1 == 2 && F_n == 1) send<1>(0, F_np1 + F_n, outs);

    const auto F_np2 = F_np1 + F_n;
    if (F_np2 < max()) {
      // on 1 process the right order of sends can avoid the race iff reductions are inline (on-current-thread) and not
      // async (nthread>1):
      // - send<1> will call wc->set_arg which will eagerly reduce the argument
      // - send<0> then will call wa->set_arg which will create task for key F_np2 ... that can potentially call
      // finalize<1> in the other clause
      // - reversing the order of sends will create a race between wc->set_arg->send<1> executing on this thread and
      // wa->set_arg->finalize<1> executing in thread pool
      // - there is no way to detect the "undesired" outcome of the race without keeping expired OpArgs from the cache
      // there is no way currently to avoid race if there is more than 1 process ... need to add the number of messages
      // that the reducing terminal will receive. The order of operations will still matter.
      send<1>(0, F_np2, outs);
      send<0>(F_np2, F_np1, outs);
    } else
      finalize<1>(0, outs);
  }

  static void consume(const int &key, const int &value, std::tuple<> &out) {
    ttg::print("sum of Fibonacci numbers up to ", max(), " = ", value);
  }

  Edge<int, int> N2N, N2C;  // !!!! Edges must be constructed before classes that use them

  decltype(wrap(next, edges(N2N), edges(N2N, N2C))) wa;
  decltype(wrap(consume, edges(N2C), edges())) wc;

 public:
  Fibonacci()
      : N2N("N2N")
      , N2C("N2C")
      , wa(wrap(next, edges(N2N), edges(N2N, N2C), "next", {"input"}, {"iterate", "sum"}))
      , wc(wrap(consume, edges(N2C), edges(), "consumer", {"result"}, {})) {
    wc->set_input_reducer<0>([](int &&a, int &&b) { return a + b; });
  }

  void print() { print_ttg(wa.get()); }

  std::string dot() { return Dot{}(wa.get()); }

  void start() {
    wa->make_executable();
    wc->make_executable();
    if (wa->get_world().rank() == 0) wa->invoke(2, 1);
  }
};

int try_main(int argc, char **argv) {
  ttg_initialize(argc, argv, 2);

  //  using mpqc::Debugger;
  //  auto debugger = std::make_shared<Debugger>();
  //  Debugger::set_default_debugger(debugger);
  //  debugger->set_exec(argv[0]);
  //  debugger->set_prefix(ttg_default_execution_context().rank());
  //  debugger->set_cmd("lldb_xterm");
  //  debugger->set_cmd("gdb_xterm");

  {
    ttg::OpBase::set_trace_all(false);

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

#if 0  // can we make something like this happen? ... edges are named, but terminals are indexed, so components are
       // "reusable"
    {
      ttg.let([]() {
           std::cout << "produced 0" << std::endl;
           // sends {Void{}, 0} to the first terminal
           return 0;
           // this is equivalent to:
           // send<0>({Void{}, 0}); return;
          })
         .attach_outputs({"A"})
         // {void,value} is implicitly converted to {key,void}?
         .let([](auto &&key) {
           if (key <= 100) {
             send<0>(key + 1);
           }
           else {
             send<1>(key);
           }
          })
         .attach_inputs({"A"})
         .attach_outputs({"A", "B"})
         .let([](auto &&key) {
           std::cout << "consumed" << key << std::endl;
          })
         .attach_inputs({"B"});
      ttg.submit(execution_context);
    }
#endif

    if constexpr (runtime_traits<ttg_runtime>::supports_streaming_terminal) {
      // Everything5 = Everything4 with consumer summing all values from A using a stream reducer
      Everything5 q5;
      std::cout << q5.dot() << std::endl;
      q5.start();

      Fibonacci fi;
      std::cout << fi.dot() << std::endl << std::endl;
      if (ttg_default_execution_context().size() == 1)
        fi.start();  // see Fibonacci::next() for why there is a race here when nproc>1 (works most of the time)

      // must fence here to flush out all tasks associated with Everything5 and Fibonacci
      // TODO must fence in Op destructors to avoid compositional nightmares like this
      ttg_fence(ttg_default_execution_context());

      // compose Fibonacci from free functions
      {
        const auto max = 1000;
        // Sum Fibonacci numbers up to max
        auto fib = [&](const int &F_n_plus_1, const int &F_n, std::tuple<Out<int, int>, Out<Void, int>> &outs) {
          // if this is first call reduce F_n also
          if (F_n_plus_1 == 2 && F_n == 1) sendv<1>(F_n, outs);

          const auto F_n_plus_2 = F_n_plus_1 + F_n;
          if (F_n_plus_1 < max) {
            sendv<1>(F_n_plus_1, outs);
            send<0>(F_n_plus_2, F_n_plus_1, outs);
          } else
            finalize<1>(outs);
        };

        auto print = [max](const int &value, std::tuple<> &out) {
          ttg::print("sum of Fibonacci numbers up to ", max, " = ", value);
          {  // validate the result
            auto ref_value = 0;
            // recursive lambda pattern from http://pedromelendez.com/blog/2015/07/16/recursive-lambdas-in-c14/
            auto validator = [max, &ref_value](int f_np1, int f_n) {
              auto impl = [max, &ref_value](int f_np1, int f_n, const auto &impl_ref) -> void {
                assert(f_n < max);
                ref_value += f_n;
                if (f_np1 < max) {
                  const auto f_np2 = f_np1 + f_n;
                  impl_ref(f_np2, f_np1, impl_ref);
                }
              };
              impl(f_np1, f_n, impl);
            };
            validator(2, 1);
            assert(value == ref_value);
          }
        };

        Edge<int, int> F2F;
        Edge<Void, int> F2P;

        auto f = make_op(fib, edges(F2F), edges(F2F, F2P), "next");
        auto p = make_op(print, edges(F2P), edges(), "print");
        p->set_input_reducer<0>([](int &&a, int &&b) { return a + b; });
        make_graph_executable(f.get());
        if (ttg_default_execution_context().rank() == 0) f->invoke(2, 1);

        ttg_fence(ttg_default_execution_context());
      }
    }

    ReductionTest t;
    t.start();

    BroadcastTest b;
    b.start();

    ttg_fence(ttg_default_execution_context());
    std::cout << "\nFence done\n";
  }
  ttg_finalize();
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
