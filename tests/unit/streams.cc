#include <catch2/catch_all.hpp>
#include <ctime>

#include "ttg.h"

#include "ttg/serialization/std/pair.h"
#include "ttg/util/hash/std/pair.h"



TEST_CASE("streams", "[streams][core]") {

  // in distributed memory we must count how many messages the reducer will receive
  SECTION("concurrent-stream-size") {
    ttg::Edge<int, int> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    constexpr std::size_t N = 12000;
    constexpr std::size_t SLICE = 600;
    constexpr const timespec ts = { .tv_sec = 0, .tv_nsec = 10000 };
    constexpr int VALUE = 1;
    std::atomic<std::size_t> reduce_ops = 0;

    auto op = ttg::make_tt(
        [&](const int &n, int&& i,
           std::tuple<ttg::Out<int, int>> &outs) {
          int key = n/SLICE;
          nanosleep(&ts, nullptr);
          if (n < N-1) {
            ttg::send<0>(key, std::forward<int>(i), outs);
            //ttg::print("sent to sink ", key);
          } else {
            // set the size of the last reducer
            if (N%SLICE > 0) {
              ttg::set_size<0>(key, N%SLICE, outs);
              std::cout << "set_size key " << key << " size " << N%SLICE << std::endl;
            }
            // forward the value
            ttg::send<0>(key, std::forward<int>(i), outs);
            //ttg::print("finalized last sink ", key);
          }
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int key, const int &value) {
          std::cout << "sink " << key << std::endl;
          if (!(value == SLICE || key == (N/SLICE))) {
            std::cout << "SINK ERROR: key " << key << " value " << value << " SLICE " << SLICE << " N " << N << std::endl;
          }
          CHECK((value == SLICE || key == (N/SLICE)));
          reduce_ops++;
        },
        ttg::edges(O2S), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    op->set_trace_instance(true);
    sink_op->set_input_reducer<0>([&](int &a, const int &b) {
      a += 1; // we count invocations
      CHECK(b == VALUE);
      reduce_ops++;
    }, SLICE);

    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i, VALUE);
      }
    }

    ttg::ttg_fence(ttg::default_execution_context());
    CHECK(reduce_ops == N/nranks);
  }

  SECTION("streams-readonly-input") {
    ttg::Edge<int, int> I2O;
    ttg::Edge<int, int> O2S;
    ttg::Edge<int, int> O2D;
    const auto nranks = ttg::default_execution_context().size();

    constexpr std::size_t N = 12000;
    constexpr std::size_t SLICE = 600;
    constexpr const timespec ts = { .tv_sec = 0, .tv_nsec = 10000 };
    constexpr int VALUE = 1;
    std::atomic<std::size_t> reduce_ops = 0;

    auto dummy = ttg::make_tt([&](const int &n, const int &i){
      CHECK(i == VALUE);
    }, ttg::edges(O2D));

    auto op = ttg::make_tt(
        [&](const int &n, const int& i,
           std::tuple<ttg::Out<int, int>, ttg::Out<int, int>> &outs) {
          int key = n/SLICE;
          nanosleep(&ts, nullptr);
          ttg::send<1>(n, i, outs); // send to a dummy to check ref-counting
          if (n < N-1) {
            ttg::send<0>(key, i, outs);
            //ttg::print("sent to sink ", key);
          } else {
            // set the size of the last reducer
            if (N%SLICE > 0) {
              ttg::set_size<0>(key, N%SLICE, outs);
              std::cout << "set_size key " << key << " size " << N%SLICE << std::endl;
            }
            // forward the value
            ttg::send<0>(key, i, outs);
            //ttg::print("finalized last sink ", key);
          }
        },
        ttg::edges(I2O), ttg::edges(O2S, O2D));

    auto sink_op = ttg::make_tt(
        [&](const int key, const int &value) {
          std::cout << "sink " << key << std::endl;
          if (!(value == SLICE || key == (N/SLICE))) {
            std::cout << "SINK ERROR: key " << key << " value " << value << " SLICE " << SLICE << " N " << N << std::endl;
          }
          CHECK((value == SLICE || key == (N/SLICE)));
          reduce_ops++;
        },
        ttg::edges(O2S), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    op->set_trace_instance(true);
    sink_op->set_input_reducer<0>([&](int &a, const int &b) {
      a += 1; // we count invocations
      CHECK(b == VALUE);
      reduce_ops++;
    }, SLICE);

    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i, VALUE);
      }
    }

    ttg::ttg_fence(ttg::default_execution_context());
    CHECK(reduce_ops == N/nranks);
  }

  SECTION("streams-temporary-input") {
    ttg::Edge<int, int> I2O;
    ttg::Edge<int, int> O2S;
    const auto nranks = ttg::default_execution_context().size();

    constexpr std::size_t N = 12000;
    constexpr std::size_t SLICE = 600;
    constexpr const timespec ts = { .tv_sec = 0, .tv_nsec = 10000 };
    constexpr int VALUE = 1;
    std::atomic<std::size_t> reduce_ops = 0;

    auto op = ttg::make_tt(
        [&](const int &n, const int& i,
           std::tuple<ttg::Out<int, int>> &outs) {
          int key = n/SLICE;
          nanosleep(&ts, nullptr);
          int tmp = i; // temporary data, not tracked
          if (n < N-1) {
            std::get<0>(outs).send(key, int{i});
            //ttg::send<0>(key, int{i}, outs);
            //ttg::print("sent to sink ", key);
          } else {
            // set the size of the last reducer
            if (N%SLICE > 0) {
              ttg::set_size<0>(key, N%SLICE, outs);
              std::cout << "set_size key " << key << " size " << N%SLICE << std::endl;
            }
            // forward the value
            ttg::send<0>(key, int{i}, outs);
            //ttg::print("finalized last sink ", key);
          }
        },
        ttg::edges(I2O), ttg::edges(O2S));

    auto sink_op = ttg::make_tt(
        [&](const int key, const int &value) {
          std::cout << "sink " << key << std::endl;
          if (!(value == SLICE || key == (N/SLICE))) {
            std::cout << "SINK ERROR: key " << key << " value " << value << " SLICE " << SLICE << " N " << N << std::endl;
          }
          CHECK((value == SLICE || key == (N/SLICE)));
          reduce_ops++;
        },
        ttg::edges(O2S), ttg::edges());

    op->set_keymap([=](const auto &key) { return nranks - 1; });
    op->set_trace_instance(true);
    sink_op->set_input_reducer<0>([&](int &a, const int &b) {
      a += 1; // we count invocations
      CHECK(b == VALUE);
      reduce_ops++;
    }, SLICE);

    make_graph_executable(op);
    ttg::execute(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) {
      for (std::size_t i = 0; i < N; ++i) {
        op->invoke(i, VALUE);
      }
    }

    ttg::ttg_fence(ttg::default_execution_context());
    CHECK(reduce_ops == N/nranks);
  }
}  // TEST_CASE("streams")
