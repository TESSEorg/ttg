#include <ttg.h>
#include "ttg/serialization.h"

const int64_t F_n_max = 1000;
/// N.B. contains values of F_n and F_{n-1}
struct Fn : public ttg::TTValue<Fn> {
  std::unique_ptr<int64_t[]> F;  // F[0] = F_n, F[1] = F_{n-1}

  Fn() : F(std::make_unique<int64_t[]>(2)) { F[0] = 1; F[1] = 0; }

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
auto make_ttg_fib_lt(const int64_t) {
  ttg::Edge<int64_t, Fn> f2f;
  ttg::Edge<void, Fn> f2p;

  auto fib = ttg::make_tt(
      [=](int64_t n, Fn&& f_n) {
        int64_t next_f_n = f_n.F[0] + f_n.F[1];
        f_n.F[1] = f_n.F[0];
        f_n.F[0] = next_f_n;
        if (next_f_n < F_n_max) {
          ttg::send<0>(n + 1, std::move(f_n));
        } else {
          ttg::send<1>(n, std::move(f_n));
        }
      },
      ttg::edges(f2f), ttg::edges(f2f, f2p), "fib");

  auto print = ttg::make_tt(
      [=](Fn&& f_n) {
        std::cout << "The largest Fibonacci number smaller than " << F_n_max << " is " << f_n.F[1] << std::endl;
      },
      ttg::edges(f2p), ttg::edges(), "print");
  auto ins = std::make_tuple(fib->template in<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops;
  ops.emplace_back(std::move(fib));
  ops.emplace_back(std::move(print));
  return make_ttg(std::move(ops), ins, std::make_tuple(), "Fib_n < N");
}


int main(int argc, char* argv[]) {
  ttg::initialize(argc, argv, -1);
  ttg::trace_on();
  int64_t N = 1000;
  if (argc > 1) N = std::atol(argv[1]);

  auto fib = make_ttg_fib_lt(N);
  ttg::make_graph_executable(fib.get());
  if (ttg::default_execution_context().rank() == 0)
    fib->template in<0>()->send(1, Fn{});;

  ttg::execute();
  ttg::fence();
  ttg::finalize();
  return 0;
}
