#include <ttg.h>
#include "ttg/serialization.h"

/// N.B. contains values of F_n and F_{n-1}
struct Fn {
  int64_t F[2];  // F[0] = F_n, F[1] = F_{n-1}
  Fn() { F[0] = 1; F[1] = 0; }
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & F;
  }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & F;
  }
};
auto make_ttg_fib_lt(const int64_t F_n_max = 1000) {
  ttg::Edge<int64_t, Fn> f2f;
  ttg::Edge<void, Fn> f2p;

  auto fib = ttg::make_tt(
      [=](int64_t n, Fn&& f_n) {
        int64_t next_f_n = f_n.F[0] + f_n.F[1];
        f_n.F[1] = f_n.F[0];
        f_n.F[0] = next_f_n;
        if (next_f_n < F_n_max) {
          ttg::send<0>(n + 1, f_n);
        } else {
          ttg::sendv<1>(f_n);
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
  int64_t N = (argc > 1) ? std::atol(argv[1]) : 1000;

  // make TTG
  auto fib = make_ttg_fib_lt(N);
  // program complete, declare it executable
  ttg::make_graph_executable(fib.get());
  // start execution
  ttg::execute();
  // start the computation by sending the first message
  if (ttg::default_execution_context().rank() == 0)
    fib->template in<0>()->send(1, Fn{});;
  // wait for the computation to finish
  ttg::fence();

  ttg::finalize();
  return 0;
}