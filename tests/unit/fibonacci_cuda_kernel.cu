#include "cuda_kernel.h"

#ifdef TTG_HAVE_CUDA

__global__ void cu_calculate_fibonacci(int64_t* results, std::size_t n) {
  int tx = threadIdx.x; // Thread index

  if (tx == 0) {
    int64_t a = 0, b = 1, c;
    if (n == 0) {
      results[tx] = a;
      return;
    }
    for (int i = 2; i <= n; i++) {
      c = a + b;
      a = b;
      b = c;
    }
    results[tx] = b;
  }
}

void calculate_fibonacci(int64_t* results, std::size_t n) {
  cu_calculate_fibonacci<<<1, 1>>>(results, n); // Adjust <<<1, 1>>> as needed for parallel computation
}

#endif // TTG_HAVE_CUDA
