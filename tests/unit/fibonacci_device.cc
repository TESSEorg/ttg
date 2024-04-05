#include <ttg.h>

#if defined(TTG_HAVE_CUDA)
#define ES ttg::ExecutionSpace::CUDA
#include "cuda_runtime.h"
#include "fibonacci_cuda_kernel.h"
#else
#error " CUDA  is required to build this test!"
#endif

#include "ttg/serialization.h"

const int64_t F_n_max = 1000;
/// N.B. contains values of F_n and F_{n-1}
struct Fn : public ttg::TTValue<Fn> {
  int64_t F[2] = {1, 0};  // F[0] = F_n, F[1] = F_{n-1}
  ttg::Buffer<int64_t> b;

  Fn() : b(&F[0], 2) {}

  Fn(const Fn&) = delete;
  Fn(Fn&& other) = default;
  Fn& operator=(const Fn& other) = delete;
  Fn& operator=(Fn&& other) = default;

  template <typename Archive>
  void serialize(Archive& ar) {
    ttg::ttg_abort();
  }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ttg::ttg_abort();
  }
};
extern ttg::Edge<int64_t, Fn> f2f;
extern ttg::Edge<void, Fn> f2p;
auto create_fib_task() {
  return ttg::make_tt<ES>(
      [=](int64_t n, Fn&& f_n) -> ttg::device::Task {
        assert(n > 0);

        co_await ttg::device::select(f_n.b);

        next_value(f_n.b.current_device_ptr());

        // wait for the task to complete and the values to be brought back to the host
        co_await ttg::device::wait(f_n.b);

        if (f_n.F[0] < F_n_max) {
          co_await ttg::device::forward(ttg::device::send<0>(n + 1, std::move(f_n)));
        } else {
          co_await ttg::device::forward(ttg::device::sendv<1>(std::move(f_n)));
        }
      },
      ttg::edges(f2f), ttg::edges(f2f, f2p), "fib");
  auto print = ttg::make_tt(
      [](Fn f_n) {
        std::cout << "The largest Fibonacci number smaller than " << F_n_max << " is " << f_n.F[1] << std::endl;
      },
      ttg::edges(f2p), ttg::edges(), "print");
}

int main(int argc, char* argv[]) {
  ttg::initialize(argc, argv, -1);
  auto fib = create_fib_task();

  ttg::make_graph_executable(fib.get());
  if (ttg::default_execution_context().rank() == 0) fib->invoke(1, Fn{});

  ttg::execute(ttg::ttg_default_execution_context());
  ttg::fence(ttg::ttg_default_execution_context());

  ttg::finalize();
  return 0;
}
