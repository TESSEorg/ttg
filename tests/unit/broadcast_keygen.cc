// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include "ttg.h"

#include <atomic>

TEST_CASE("Broadcast-Keygen", "[broadcast][keygen]") {
  constexpr int N = 10;
  constexpr double value = 42.0;

  auto world = ttg::default_execution_context();

  std::atomic<int> nA = 0, nB = 0, nC = 0, nD = 0;
  std::atomic<bool> failed = false;

  ttg::Edge<int, double> start, toA, toB, toD;
  ttg::Edge<void, double> toC;

  auto producer = ttg::make_tt(
      [](const int &key, const double &v) {
        /* send the value to all keys produced by the generators set below */
        ttg::broadcast_keygen(key, v);
      },
      ttg::edges(start), ttg::edges(toA, toB, toC, toD), "PRODUCER");

  producer->set_broadcast_keygen([](const int &key) {
    return std::make_tuple(
        /* toA: keys 0, ..., N-1 */
        [=]() -> ttg::generator<int> {
          for (int i = 0; i < N; ++i) co_yield i;
        },
        /* toB: keys 0, 10, ..., 10*(N-1) */
        [=]() -> ttg::generator<int> {
          for (int i = 0; i < N; ++i) co_yield 10 * i;
        },
        /* toC: void key, send */
        [=]() -> ttg::generator<bool> { co_yield true; },
        /* toD: no keys */
        [=]() -> ttg::generator<int> { co_return; });
  });

  auto make_consumer = [&](ttg::Edge<int, double> &in, std::atomic<int> &counter, const char *name) {
    const int nprocs = world.size();
    auto tt = ttg::make_tt(
        [&counter, &failed, world, nprocs](const int &key, const double &v) {
          if (v != value) failed = true;
          /* every key must arrive on the process it is mapped to */
          if (key % nprocs != world.rank()) failed = true;
          ++counter;
        },
        ttg::edges(in), ttg::edges(), name);
    tt->set_keymap([nprocs](const int &key) { return key % nprocs; });
    return tt;
  };
  auto consumerA = make_consumer(toA, nA, "CONSUMER_A");
  auto consumerB = make_consumer(toB, nB, "CONSUMER_B");
  auto consumerD = make_consumer(toD, nD, "CONSUMER_D");

  auto consumerC = ttg::make_tt<void>(
      [&](const double &v) {
        if (v != value) failed = true;
        ++nC;
      },
      ttg::edges(toC), ttg::edges(), "CONSUMER_C");

  ttg::make_graph_executable(producer.get());
  ttg::execute(world);
  if (producer->get_keymap()(0) == world.rank()) {
    producer->invoke(0, value);
  }
  ttg::fence(world);

  CHECK(!failed);
  CHECK(nD == 0);
  if (world.size() == 1) {
    CHECK(nA == N);
    CHECK(nB == N);
    CHECK(nC == 1);
  }
}
