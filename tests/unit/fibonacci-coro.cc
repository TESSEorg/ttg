#include <catch2/catch_all.hpp>

#include "ttg.h"

#include "ttg/serialization/std/pair.h"
#include "ttg/util/hash/std/pair.h"

#include "ttg/util/coroutine.h"

constexpr int64_t N = 1000;

TEST_CASE("Fibonacci-coroutines", "[fib][core]") {
  // compute the reference result
  int reference_result = 0;
  {
    // recursive lambda pattern from http://pedromelendez.com/blog/2015/07/16/recursive-lambdas-in-c14/
    auto compute_reference_result = [&reference_result](int f_np1, int f_n) {
      auto impl = [&reference_result](int f_np1, int f_n, const auto &impl_ref) -> void {
        assert(f_n < N);
        reference_result += f_n;
        if (f_np1 < N) {
          const auto f_np2 = f_np1 + f_n;
          impl_ref(f_np2, f_np1, impl_ref);
        }
      };
      impl(f_np1, f_n, impl);
    };
    compute_reference_result(1, 0);
  }

  SECTION("shared-memory") {
    if (ttg::default_execution_context().size() == 1) {
      ttg::Edge<int, int> F2F;
      ttg::Edge<void, int> F2P;

      // N.B. wrap a trivial (nonsuspending) coroutine using make_tt!
      auto fib_op = ttg::make_tt(
          // computes next value: F_{n+2} = F_{n+1} + F_{n}, seeded by F_1 = 1, F_0 = 0
          // N.B. can't autodeduce return type, must explicitly declare the return type
          [](const int &F_n_plus_1, const int &F_n) -> ttg::resumable_task {
            // on 1 process the right order of sends can avoid the race iff reductions are inline (on-current-thread)
            // and not async (nthread>1):
            // - send<1> will call wc->set_arg which will eagerly reduce the argument
            // - send<0> then will call wa->set_arg which will create task for key F_np2 ... that can potentially call
            // finalize<1> in the other clause
            // - reversing the order of sends will create a race between wc->set_arg->send<1> executing on this thread
            // and wa->set_arg->finalize<1> executing in thread pool
            // - there is no way to detect the "undesired" outcome of the race without keeping expired TTArgs from the
            // cache there is no way currently to avoid race if there is more than 1 process ... need to track the
            // number of messages that the reducing terminal will receive, that's what distributed example demonstrates.
            // The order of operations will still matter.
            if (F_n_plus_1 < N) {
              const auto F_n_plus_2 = F_n_plus_1 + F_n;
              // cool, if there are no events to wait for co_await is no-op
              co_await ttg::resumable_task_events{};
              ttg::sendv<1>(F_n_plus_1);
              ttg::send<0>(F_n_plus_2, F_n_plus_1);
            } else
              ttg::finalize<1>();

            // to test coro-based task lifecycle introduce fake events
            ttg::event null_event;
            co_await ttg::resumable_task_events{null_event};

            // N.B. return void just as normal TT op
            co_return;
          },
          ttg::edges(F2F), ttg::edges(F2F, F2P));
      auto print_op = ttg::make_tt(
          [reference_result](const int &value, std::tuple<> &out) {
            ttg::print("sum of Fibonacci numbers up to ", N, " = ", value);
            CHECK(value == reference_result);
          },
          ttg::edges(F2P), ttg::edges());
      print_op->set_input_reducer<0>([](int &a, const int &b) { a = a + b; });
      make_graph_executable(fib_op);
      if (ttg::default_execution_context().rank() == 0) fib_op->invoke(1, 0);
      ttg::ttg_fence(ttg::default_execution_context());
    }
  }

  // in distributed memory we must count how many messages the reducer will receive
  SECTION("distributed-memory") {
    ttg::Edge<int, std::pair<int, int>> F2F;
    ttg::Edge<void, int> F2P;
    const auto nranks = ttg::default_execution_context().size();

    auto fib_op = ttg::make_tt(
        // computes next value: F_{n+2} = F_{n+1} + F_{n}, seeded by F_1 = 1, F_0 = 0
        [](const int &n, const std::pair<int, int> &F_np1_n) {
          const auto &[F_n_plus_1, F_n] = F_np1_n;
          if (F_n_plus_1 < N) {
            const auto F_n_plus_2 = F_n_plus_1 + F_n;
            ttg::print("sent ", F_n_plus_1, " to fib reducer");
            ttg::sendv<1>(F_n_plus_1);
            ttg::send<0>(n + 1, std::make_pair(F_n_plus_2, F_n_plus_1));
          } else {
            // how many messages the reducer should expect to receive
            ttg::set_size<1>(n);
            ttg::print("fib reducer will expect ", n, " messages");
          }
        },
        ttg::edges(F2F), ttg::edges(F2F, F2P));
    auto print_op = ttg::make_tt(
        [reference_result](const int &value, std::tuple<> &out) {
          ttg::print("sum of Fibonacci numbers up to ", N, " = ", value);
          CHECK(value == reference_result);
        },
        ttg::edges(F2P), ttg::edges());
    // move all fib tasks to last rank, all reductions will happen on 0 => for some reason no reductions occur!
    fib_op->set_keymap([=](const auto &key) { return nranks - 1; });
    fib_op->set_trace_instance(true);
    print_op->set_input_reducer<0>([](int &a, const int &b) {
      ttg::print("fib reducer: current value = ", a, ", incremented by ", b, " set to ", a + b);
      a = a + b;
    });
    make_graph_executable(fib_op);
    ttg::ttg_fence(ttg::default_execution_context());
    if (ttg::default_execution_context().rank() == 0) fib_op->invoke(0, std::make_pair(1, 0));
    ttg::ttg_fence(ttg::default_execution_context());
  }
}  // TEST_CAST("Fibonacci")
