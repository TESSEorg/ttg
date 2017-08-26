#include <iostream>
#include <tuple>
#include "edge_original.h"

using keyT = int;

class A : public Op<keyT, std::tuple<int>, std::tuple<OutEdge<keyT, int>, OutEdge<keyT, int>>, A> {
  using baseT = Op<keyT, std::tuple<int>, std::tuple<OutEdge<keyT, int>, OutEdge<keyT, int>>, A>;

 public:
  A(const std::string& name) : baseT(name) {}
  void op(const keyT& key, const std::tuple<int>& t) {
    int value = std::get<0>(t);
    std::cout << "A got value " << value << std::endl;
    if (value < 3) {
      send<0>(key + 1, value + 1);
    } else {
      send<1>(key, value);
    }
  }
};

class Producer : public Op<keyT, std::tuple<>, std::tuple<OutEdge<keyT, int>>, Producer> {
  using baseT = Op<keyT, std::tuple<>, std::tuple<OutEdge<keyT, int>>, Producer>;

 public:
  Producer(const std::string& name) : baseT(name) {}
  void op(const keyT& key, const std::tuple<>& t) {
    std::cout << "produced " << 0 << std::endl;
    send<0>(0, 0);
  }
};

class Consumer : public Op<keyT, std::tuple<int>, std::tuple<>, Consumer> {
  using baseT = Op<keyT, std::tuple<int>, std::tuple<>, Consumer>;

 public:
  Consumer(const std::string& name) : baseT(name) {}
  void op(const keyT& key, const std::tuple<int>& t) { std::cout << "consumed " << std::get<0>(t) << std::endl; }
};

class Everything : public Op<keyT, std::tuple<>, std::tuple<>, Everything> {
  Producer producer;
  A a;
  Consumer consumer;
  Merge<keyT, int> merge;

 public:
  Everything() : producer("producer"), a("A"), consumer("consumer"), merge("merge") {
    producer.out<0>().connect(merge.in<0>());
    merge.out<0>().connect(a.in<0>());
    a.out<0>().connect(merge.in<1>());
    a.out<1>().connect(consumer.in<0>());
  }

  void start() { producer.op(0, std::tuple<>()); }

  void wait() {}
};

int main() {
  // BaseOp::set_trace(true);
  Everything x;
  x.start();
  x.wait();
  return 0;
}
