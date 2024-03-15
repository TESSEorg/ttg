////

#include <ttg.h>
// Define TTG_USE_CUDA only if CUDA support is desired and available
#ifdef TTG_USE_CUDA
#include "cuda_runtime.h"
#include "cuda_kernel.h"
#endif

#include "ttg/serialization.h"

// Default to CUDA if available, can be overridden by defining TTG_USE_XXX for other backends
#define ES ttg::default_execution_space()

struct A : public ttg::TTValue<A> {
  int64_t value;
  ttg::Buffer<int64_t> buffer;

  A() : value(0), buffer(&value, 1) {}
  A(int64_t val) : value(val), buffer(&value, 1) {}

  A(A&& other) = default;
  A& operator=(A&& other) = default;

  template <typename Archive>
  void serialize(Archive& ar) {
    ttg_abort();
  }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ttg_abort();
  }
};

int main(int argc, char* argv[]) {
  ttg::initialize(argc, argv, -1);
  const int64_t N = 20;

  ttg::Edge<int64_t, A> f2f;
  ttg::Edge<void, A> f2p;

  auto fib = ttg::make_tt<ES>(
      [=](int64_t n, A& F_nms) -> ttg::device::Task {
        if (n <= N) {
          co_await ttg::device::select(F_nms.buffer);

          int64_t result = calculate_fibonacci(n);

          A F_n(result);
          if (n < N) {
            co_await ttg::device::send<0>(n + 1, F_n);
          } else {
            co_await ttg::device::sendv<1>(F_n);
          }
        }
      },
      ttg::edges(f2f),
      ttg::edges(f2f, f2p),
      "fib");

  auto print = ttg::make_tt([](A F_N) {
    std::cout << "The " << N << "th Fibonacci number is " << F_N.value << std::endl;
  },
                            ttg::edges(f2p),
                            ttg::edges(),
                            "print");

  ttg::make_graph_executable(fib.get());
  if (ttg::default_execution_context().rank() == 0) fib->invoke(2, A(1));

  ttg::execute(ttg_default_execution_context());
  ttg::fence(ttg_default_execution_context());

  ttg::finalize();
  return 0;
}

//// Created by Hyndavi Venkatreddygari on 3/13/24.
////
//#include <ttg.h>
//#define TTG_USE_CUDA
//#include "cuda_runtime.h"
//#include "cuda_kernel.h"
//
//#include "ttg/serialization.h"
//
//#define ES ttg::ExecutionSpace::CUDA
//
//struct A : public ttg::TTValue<A> {
//  int64_t value;
//  ttg::Buffer<int64_t> buffer;
//
//  A() : value(0), buffer(&value, 1) {}
//  A(int64_t val) : value(val), buffer(&value, 1) {}
//
//  A(A&& other) = default;
//  A& operator=(A&& other) = default;
//
//  template <typename Archive>
//  void serialize(Archive& ar) {
//    ttg_abort();
//  }
//  template <typename Archive>
//  void serialize(Archive& ar, const unsigned int) {
//    ttg_abort();
//  }
//};
//
//int main(int argc, char* argv[]) {
//  ttg::initialize(argc, argv, -1);
//  const int64_t N = 20;
//
//  ttg::Edge<int64_t, A> f2f;
//  ttg::Edge<void, A> f2p;
//
//  auto fib = ttg::make_tt<ES>(
//      [=](int64_t n, A& F_nms) -> ttg::device::Task {
//        if (n <= N) {
//          co_await ttg::device::select(F_nms.buffer);
//
//          int64_t* d_result;
//          cudaMalloc(&d_result, sizeof(int64_t));
//
//          calculate_fibonacci(d_result, n);
//
//          co_await ttg::wait_kernel();
//
//          int64_t h_result;
//          cudaMemcpy(&h_result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);
//
//          A F_n(h_result);
//          if (n < N) {
//            co_await ttg::device::send<0>(n + 1, F_n);
//          } else {
//            co_await ttg::device::sendv<1>(F_n);
//          }
//        }
//      },
//      ttg::edges(f2f),
//      ttg::edges(f2f, f2p),
//      "fib");
//
//  auto print = ttg::make_tt([](A F_N) {
//    std::cout << "The " << N << "th Fibonacci number is " << F_N.value << std::endl;
//  },
//                            ttg::edges(f2p),
//                            ttg::edges(),
//                            "print");
//
//  ttg::make_graph_executable(fib.get());
//  if (ttg::default_execution_context().rank() == 0) fib->invoke(2, A(1));
//
//  ttg::execute(ttg_default_execution_context());
//  ttg::fence(ttg_default_execution_context());
//
//  ttg::finalize();
//  return 0;
//}
