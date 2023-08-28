#include "ttg/config.h"
#include "ttg/device/cublas_helper.h"

#include <memory>
#include <stdexcept>
#include <optional>

namespace ttg::detail {

#ifdef TTG_HAVE_CUDART
/// \brief Returns the cuBLAS handle to be used for launching cuBLAS kernels from the current thread
/// \return the cuBLAS handle for the current thread
const cublasHandle_t& cublas_get_handle() {
  static thread_local std::optional<cublasHandle_t> handle;
  if (!handle.has_value()) {
    auto status = cublasCreate_v2(&handle.emplace());
    if (CUBLAS_STATUS_SUCCESS != status) {
      throw std::runtime_error("cublasCreate_v2 failed");
    }
  }
  return *handle;
}

void cublas_set_kernel_stream(cudaStream_t stream) {
    cublasStatus_t status = cublasSetStream_v2(cublas_get_handle(), stream);
    if (CUBLAS_STATUS_SUCCESS != status) {
        throw std::runtime_error("cublasSetStream_v2 failed");
    }
}
#endif // TTG_HAVE_CUDART

#ifdef TTG_HAVE_HIPBLAS
/// \brief Returns the rocBLAS handle to be used for launching rocBLAS kernels from the current thread
/// \return the rocBLAS handle for the current thread
const hipblasHandle_t& hipblas_get_handle() {
  static thread_local std::optional<hipblasHandle_t> handle;
  if (!handle.has_value()) {
    hipblasStatus_t status = hipblasCreate(&handle.emplace());
    if (HIPBLAS_STATUS_SUCCESS != status) {
      throw std::runtime_error("hipblasCreate failed");
    }
  }
  return *handle;
}

void hipblas_set_kernel_stream(hipStream_t stream) {
    hipblasStatus_t status = hipblasSetStream(hipblas_get_handle(), stream);
    if (HIPBLAS_STATUS_SUCCESS != status) {
        throw std::runtime_error("hipblasSetStream failed");
    }
}

#endif // TTG_HAVE_HIPBLAS


} // namespace