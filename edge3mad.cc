#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/MADworld.h>
#include <iostream>
#include <tuple>
#include "edgemad.h"

using namespace madness;

using keyT = int;

class A : public Op<keyT, std::tuple<int>, std::tuple<OutEdge<keyT, int>, OutEdge<keyT, int>>, A> {
  using baseT = Op<keyT, std::tuple<int>, std::tuple<OutEdge<keyT, int>, OutEdge<keyT, int>>, A>;

 public:
  A(World& world, const std::string& name) : baseT(world, name) {}
  void op(const keyT& key, const std::tuple<int>& t) {
    int value = std::get<0>(t);
    std::cout << "A got value " << value << std::endl;
    if (value < 100) {
      send<0>(key + 1, value + 1);
    } else {
      send<1>(key, value);
    }
  }
};

class Producer : public Op<keyT, std::tuple<>, std::tuple<OutEdge<keyT, int>>, Producer> {
  using baseT = Op<keyT, std::tuple<>, std::tuple<OutEdge<keyT, int>>, Producer>;

 public:
  Producer(World& world, const std::string& name) : baseT(world, name) {}
  void op(const keyT& key, const std::tuple<>& t) {
    std::cout << "produced " << 0 << std::endl;
    send<0>(0, 0);
  }
};

class Consumer : public Op<keyT, std::tuple<int>, std::tuple<>, Consumer> {
  using baseT = Op<keyT, std::tuple<int>, std::tuple<>, Consumer>;

 public:
  Consumer(World& world, const std::string& name) : baseT(world, name) {}
  void op(const keyT& key, const std::tuple<int>& t) { std::cout << "consumed " << std::get<0>(t) << std::endl; }
};

class Everything : public Op<keyT, std::tuple<>, std::tuple<>, Everything> {
  using baseT = Op<keyT, std::tuple<>, std::tuple<>, Everything>;

  Producer producer;
  A a;
  Consumer consumer;
  Merge<keyT, int> merge;

  World& world;

 public:
  Everything(World& world)
      : baseT(world, "everything")
      , producer(world, "producer")
      , a(world, "A")
      , consumer(world, "consumer")
      , merge(world, "merge")
      , world(world) {
    producer.out<0>().connect(merge.in<0>());
    merge.out<0>().connect(a.in<0>());
    a.out<0>().connect(merge.in<1>());
    a.out<1>().connect(consumer.in<0>());
    world.gop.fence();
  }

  void start() {
    if (world.rank() == 0) producer.op(0, std::tuple<>());
  }

  void wait() { world.gop.fence(); }
};

int main(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);

  for (int arg = 1; arg < argc; ++arg) {
    if (strcmp(argv[arg], "-dx") == 0) xterm_debug(argv[0], 0);
  }

  BaseOp::set_trace(true);
  Everything x(world);
  x.start();
  x.wait();

  finalize();
  return 0;
}
