#include "ttg/config.h"
#include "ttg/device/cublas_helper.h"

#include <memory>
#include <stdexcept>
#include <optional>

namespace ttg::detail {

#ifdef TTG_HAVE_CUDART
/// \brief Returns the cuBLAS handle to be used for launching cuBLAS kernels from the current thread
/// \return the cuBLAS handle for the current thread
inline const cublasHandle_t& cublas_get_handle() {
  static thread_local std::optional<cublasHandle_t> handle;
  if (!handle.has_value()) {
    auto status = cublasCreate_v2(&handle.emplace());
    if (CUBLAS_STATUS_SUCCESS != status) {
      throw std::runtime_error("cublasCreate_v2 failed");
    }
  }
  return *handle;
}
#endif // TTG_HAVE_CUDART

void cublas_set_kernel_stream(cudaStream_t stream) {
#ifdef TTG_HAVE_CUDART
    cublasStatus_t status = cublasSetStream_v2(cublas_get_handle(), stream);
    if (CUBLAS_STATUS_SUCCESS != status) {
        throw std::runtime_error("cublasSetStream_v2 failed");
    }
#else
    throw std::runtime_error("Support for cublas missing during installation!");
#endif // TTG_HAVE_CUDART
}

} // namespace