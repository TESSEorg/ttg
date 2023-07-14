#ifndef TTG_DEVICE_CUBLAS_HELPER_H
#define TTG_DEVICE_CUBLAS_HELPER_H

#include "ttg/config.h"

#ifdef TTG_HAVE_CUDART
#include <cublas_v2.h>

namespace ttg::detail {

/// \brief Returns the current CUDA stream used by cuBLAS
void cublas_set_kernel_stream(cudaStream_t stream);

} // namespace ttg::detail
#endif // TTG_HAVE_CUDART


#endif // TTG_DEVICE_CUBLAS_HELPER_H