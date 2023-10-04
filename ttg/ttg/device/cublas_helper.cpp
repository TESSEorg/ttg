#include "ttg/config.h"
#include "ttg/device/cublas_helper.h"

#include <memory>
#include <stdexcept>
#include <optional>
#include <map>
#ifdef TTG_HAVE_CUDART
#include <cuda_runtime.h>
#endif // TTG_HAVE_CUDART

#ifdef TTG_HAVE_HIPBLAS
#include <hip/hip_runtime.h>
#endif // TTG_HAVE_HIPBLAS

namespace ttg::detail {

#ifdef TTG_HAVE_CUDART
/// \brief Returns the cuBLAS handle to be used for launching cuBLAS kernels from the current thread
/// \return the cuBLAS handle for the current thread
const cublasHandle_t& cublas_get_handle() {
  static thread_local std::map<int, cublasHandle_t> handles;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)){
      throw std::runtime_error("cudaGetDevice failed");
  }
  std::map<int, cublasHandle_t>::iterator it;
  if ((it = handles.find(device)) == handles.end()){
    cublasHandle_t handle;
    auto status = cublasCreate_v2(&handle);
    if (CUBLAS_STATUS_SUCCESS != status) {
      throw std::runtime_error("cublasCreate_v2 failed");
    }
    auto [iterator, success] = handles.insert({device, handle});
    it = iterator;
  }

  return it->second;
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
  static thread_local std::map<int, hipblasHandle_t> handles;
  int device;
  if (hipSuccess != hipGetDevice(&device)){
      throw std::runtime_error("hipGetDevice failed");
  }
  std::map<int, hipblasHandle_t>::iterator it;
  if ((it = handles.find(device)) == handles.end()){
    hipblasHandle_t handle;
    auto status = hipblasCreate(&handle);
    if (HIPBLAS_STATUS_SUCCESS != status) {
      throw std::runtime_error("hipblasCreate failed");
    }
    auto [iterator, success] = handles.insert({device, handle});
    it = iterator;
  }
  return it->second;
}

void hipblas_set_kernel_stream(hipStream_t stream) {
    hipblasStatus_t status = hipblasSetStream(hipblas_get_handle(), stream);
    if (HIPBLAS_STATUS_SUCCESS != status) {
        throw std::runtime_error("hipblasSetStream failed");
    }
}

#endif // TTG_HAVE_HIPBLAS


} // namespace