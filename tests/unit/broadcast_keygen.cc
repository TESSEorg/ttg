// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include "ttg.h"

#include <atomic>
#include <stdexcept>

namespace {
  /* sum a counter across all processes */
  double global_sum(ttg::World world, double value) {
    TTG_IMPL_NS::ttg_sum(world, value);
    return value;
  }
}  // namespace

TEST_CASE("Broadcast-Keygen", "[broadcast][keygen]") {
  constexpr int N = 10;  // keys per producer and output

  auto world = ttg::default_execution_context();
  const int P = world.size();
  /* several producers so that every process is the root of a broadcast */
  const int num_producers = 2 * P + 1;
  auto value_of = [](int producer) { return 100.0 + producer; };
  /* consumer keys encode the producer and the process they are mapped to */
  auto encode = [P](int producer, int i, int rank) { return (producer * N + i) * P + rank; };
  auto producer_of = [P](int key) { return key / (N * P); };

  std::atomic<int> nA = 0, nB = 0, nC = 0, nD = 0, nfailed = 0;

  ttg::Edge<int, double> start, toA, toB, toD;
  ttg::Edge<void, double> toC;

  auto producer = ttg::make_tt(
      [](const int &key, const double &v) {
        /* send the value to all keys produced by the generators set below */
        ttg::broadcast_keygen(key, v);
      },
      ttg::edges(start), ttg::edges(toA, toB, toC, toD), "PRODUCER");
  producer->set_keymap([P](const int &key) { return key % P; });

  producer->set_broadcast_keygen([=](const int &k) {
    return std::make_tuple(
        /* toA: even producers send to keys on all processes, including their own */
        [=]() -> ttg::generator<int> {
          if (k % 2 == 0) {
            for (int i = 0; i < N; ++i) co_yield encode(k, i, (k + i) % P);
          }
        },
        /* toB: keys on the next process only, so odd producers do not send to their own process */
        [=]() -> ttg::generator<int> {
          for (int i = 0; i < N; ++i) co_yield encode(k, i, (k + 1) % P);
        },
        /* toC: void key, sent once by producer 1 (on process 1 if there is more than one) */
        [=]() -> ttg::generator<bool> { co_yield k == 1; },
        /* toD: no keys */
        [=]() -> ttg::generator<int> { co_return; });
  });

  auto make_consumer = [&](ttg::Edge<int, double> &in, std::atomic<int> &counter, const char *name) {
    auto tt = ttg::make_tt(
        [&counter, &nfailed, world, P, value_of, producer_of](const int &key, const double &v) {
          if (v != value_of(producer_of(key))) ++nfailed;
          /* every key must arrive on the process it is mapped to */
          if (key % P != world.rank()) ++nfailed;
          ++counter;
        },
        ttg::edges(in), ttg::edges(), name);
    tt->set_keymap([P](const int &key) { return key % P; });
    return tt;
  };
  auto consumerA = make_consumer(toA, nA, "CONSUMER_A");
  auto consumerB = make_consumer(toB, nB, "CONSUMER_B");
  auto consumerD = make_consumer(toD, nD, "CONSUMER_D");

  auto consumerC = ttg::make_tt<void>(
      [&, value_of](const double &v) {
        if (v != value_of(1)) ++nfailed;
        ++nC;
      },
      ttg::edges(toC), ttg::edges(), "CONSUMER_C");

  ttg::make_graph_executable(producer.get());
  ttg::execute(world);
  for (int k = 0; k < num_producers; ++k) {
    if (k % P == world.rank()) producer->invoke(k, value_of(k));
  }
  ttg::fence(world);

  const int num_even_producers = (num_producers + 1) / 2;
  CHECK(global_sum(world, nfailed) == 0);
  CHECK(global_sum(world, nA) == num_even_producers * N);
  CHECK(global_sum(world, nB) == num_producers * N);
  CHECK(global_sum(world, nC) == 1);
  CHECK(global_sum(world, nD) == 0);
}

TEST_CASE("Broadcast-Keygen-Mismatched-Value-Type", "[broadcast][keygen]") {
  auto world = ttg::default_execution_context();
  const int P = world.size();

  std::atomic<int> nthrown = 0, nX = 0, nY = 0;

  ttg::Edge<int, double> start, toX;
  ttg::Edge<int, int> toY;

  auto producer = ttg::make_tt(
      [&](const int &key, const double &v) {
        try {
          ttg::broadcast_keygen(key, v);
        } catch (const std::runtime_error &) {
          ++nthrown;
        }
      },
      ttg::edges(start), ttg::edges(toX, toY), "PRODUCER");

  /* keys for toY cannot receive the double value, so nothing must be sent */
  producer->set_broadcast_keygen([P](const int &k) {
    return std::make_tuple([=]() -> ttg::generator<int> { co_yield (k + 1) % P; },
                           [=]() -> ttg::generator<int> { co_yield (k + 1) % P; });
  });

  auto consumerX = ttg::make_tt([&](const int &key, const double &v) { ++nX; }, ttg::edges(toX), ttg::edges(),
                                "CONSUMER_X");
  auto consumerY = ttg::make_tt([&](const int &key, const int &v) { ++nY; }, ttg::edges(toY), ttg::edges(),
                                "CONSUMER_Y");
  consumerX->set_keymap([P](const int &key) { return key % P; });
  consumerY->set_keymap([P](const int &key) { return key % P; });

  ttg::make_graph_executable(producer.get());
  ttg::execute(world);
  if (producer->get_keymap()(0) == world.rank()) producer->invoke(0, 1.0);
  ttg::fence(world);

  CHECK(global_sum(world, nthrown) == 1);
  CHECK(global_sum(world, nX) == 0);
  CHECK(global_sum(world, nY) == 0);
}
