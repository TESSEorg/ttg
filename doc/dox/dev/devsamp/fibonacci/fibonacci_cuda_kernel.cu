#include "fibonacci_cuda_kernel.h"

#ifdef TTG_HAVE_CUDA

__global__ void cu_next_value(int64_t* fn_and_fnm1) {
  int64_t fnp1 = fn_and_fnm1[0] + fn_and_fnm1[1];
  fn_and_fnm1[1] = fn_and_fnm1[0];
  fn_and_fnm1[0] = fnp1;
}

void next_value(int64_t* fn_and_fnm1) {
  cu_next_value<<<1, 1>>>(fn_and_fnm1);
}

#endif // TTG_HAVE_CUDA
