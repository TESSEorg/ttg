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
  std::unique_ptr<int64_t[]> F;  // F[0] = F_n, F[1] = F_{n-1}
  ttg::Buffer<int64_t> b;

  Fn() : F(std::make_unique<int64_t[]>(2)), b(F.get(), 2) { F[0] = 1; F[1] = 0; }

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

auto make_ttg_fib_lt(const int64_t F_n_max = 1000) {
  ttg::Edge<int64_t, Fn> f2f;
  ttg::Edge<void, Fn> f2p;

  auto fib = ttg::make_tt<ES>(
      [=](int64_t n, Fn&& f_n) -> ttg::device::Task {
        assert(n > 0);
        ttg::trace("in fib: n=", n, " F_n=", f_n.F[0]);

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
      [=](Fn&& f_n) {
        std::cout << "The largest Fibonacci number smaller than " << F_n_max << " is " << f_n.F[1] << std::endl;
      },
      ttg::edges(f2p), ttg::edges(), "print");

  auto ins = std::make_tuple(fib->template in<0>());
  std::vector<std::unique_ptr<::ttg::TTBase>> ops;
  ops.emplace_back(std::move(fib));
  ops.emplace_back(std::move(print));
  return make_ttg(std::move(ops), ins, std::make_tuple(), "Fib_n < N");
}

int main(int argc, char* argv[]) {
  ttg::initialize(argc, argv, -1);
  ttg::trace_on();
  int64_t N = 1000;
  if (argc > 1) N = std::atol(argv[1]);
  auto fib = make_ttg_fib_lt(N);  // computes largest F_n < N

  ttg::make_graph_executable(fib.get());
  if (ttg::default_execution_context().rank() == 0)
    fib->template in<0>()->send(1, Fn{});;

  ttg::execute(ttg::ttg_default_execution_context());
  ttg::fence(ttg::ttg_default_execution_context());

  ttg::finalize();
  return 0;
}
