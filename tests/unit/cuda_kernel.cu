
#include "cuda_kernel.h"

#ifdef TTG_HAVE_CUDA

static __global__ void cu_increment_buffer(double* buffer, double* scratch) {
  // Thread index
  int tx = threadIdx.x;

  buffer[tx] += 1.0;
  if (tx == 0 && scratch != nullptr) {
    *scratch += 1.0;
  }
}

void increment_buffer_cuda(
  double* buffer, std::size_t buffer_size,
  double* scratch, std::size_t scratch_size)
{

  cu_increment_buffer<<<1, buffer_size>>>(buffer, scratch);

}

#endif // TTG_HAVE_CUDA