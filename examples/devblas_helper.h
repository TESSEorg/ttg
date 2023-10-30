#include "ttg/config.h"

#include <memory>
#include <stdexcept>
#include <optional>
#include <map>

#ifdef TTG_HAVE_CUDART

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

/// \brief Returns the cuBLAS handle to be used for launching cuBLAS kernels from the current thread
/// \return the cuBLAS handle for the current thread
template<typename T = int>
inline const cublasHandle_t& cublas_handle(T _ = 0) {
  using map_type = std::map<std::pair<int, cudaStream_t>, cublasHandle_t>;
  static thread_local map_type handles;

  auto d = ttg::device::current_device();
  int device = 0; // assume 0 if we don't have a device
  if (d.is_device()) {
    device = d;
  }

  cudaStream_t stream = ttg::device::current_stream();

  map_type::iterator it;
  if ((it = handles.find({device, stream})) == handles.end()){
    cublasHandle_t handle;
    auto status = cublasCreate_v2(&handle);
    if (CUBLAS_STATUS_SUCCESS != status) {
      std::cerr << "cublasCreate_v2 failed: " << status << std::endl;
      throw std::runtime_error("cublasCreate_v2 failed");
    }
    status = cublasSetStream_v2(handle, ttg::device::current_stream());
    if (CUBLAS_STATUS_SUCCESS != status) {
      std::cerr << "cublasSetStream_v2 failed: " << status << std::endl;
      throw std::runtime_error("cublasSetStream_v2 failed");
    }
    auto [iterator, success] = handles.insert({{device, stream}, handle});
    it = iterator;
  }
  return it->second;
}

template<typename T = int>
inline const cusolverDnHandle_t& cusolver_handle(T _ = 0) {

  using map_type = std::map<std::pair<int, cudaStream_t>, cusolverDnHandle_t>;
  static thread_local map_type handles;

  auto d = ttg::device::current_device();
  int device = 0; // assume 0 if we don't have a device
  if (d.is_device()) {
    device = d;
  }
  cudaStream_t stream = ttg::device::current_stream();

  map_type::iterator it;
  if ((it = handles.find({device, stream})) == handles.end()){
    cusolverDnHandle_t handle;
    std::cout << "Creating cusolver handle " << handle << " for device " << device << " stream " << stream << std::endl;

    auto status = cusolverDnCreate(&handle);
    if (CUSOLVER_STATUS_SUCCESS != status) {
      std::cerr << "cusolverDnCreate failed: " << status << std::endl;
      throw std::runtime_error("cusolverDnCreate failed");
    }
    status = cusolverDnSetStream(handle, stream);
    if (CUSOLVER_STATUS_SUCCESS != status) {
      std::cerr << "cusolverDnSetStream failed: " << status << std::endl;
      throw std::runtime_error("cusolverDnSetStream failed");
    }

    auto [iterator, success] = handles.insert({{device, stream}, handle});
    it = iterator;
  } else {
    std::cout << "Found cusolver handle " << it->second << " for device " << device << " stream " << stream << std::endl;
  }

  return it->second;
}
#endif // TTG_HAVE_CUDART

#ifdef TTG_HAVE_HIPBLAS

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

/// \brief Returns the rocBLAS handle to be used for launching rocBLAS kernels from the current thread
/// \return the rocBLAS handle for the current thread
template<typename T = int>
const hipblasHandle_t& hipblas_handle(T _ = 0) {
  static thread_local std::map<int, hipblasHandle_t> handles;
  int device = ttg::device::current_device();
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
  hipblasStatus_t status = hipblasSetStream(it->second, ttg::device::current_stream());
  if (HIPBLAS_STATUS_SUCCESS != status) {
    throw std::runtime_error("hipblasSetStream failed");
  }
  return it->second;
}

/// \brief Returns the hipsolver handle to be used for launching rocBLAS kernels from the current thread
/// \return the hipsolver handle for the current thread
template<typename T = int>
const hipsolverDnHandle_t& hipsolver_handle(T _ = 0) {
  static thread_local std::map<int, hipsolverDnHandle_t> handles;
  int device = ttg::device::current_device();
  std::map<int, hipsolverDnHandle_t>::iterator it;
  if ((it = handles.find(device)) == handles.end()){
    hipsolverDnHandle_t handle;
    auto status = hipsolverDnCreate(&handle);
    if (HIPSOLVER_STATUS_SUCCESS != status) {
      throw std::runtime_error("hipsolverCreate failed");
    }
    auto [iterator, success] = handles.insert({device, handle});
    it = iterator;
  }
  hipsolverStatus_t status = hipsolverDnSetStream(it->second, ttg::device::current_stream());
  if (HIPSOLVER_STATUS_SUCCESS != status) {
    throw std::runtime_error("hipsolverSetStream failed");
  }
  return it->second;
}
#endif // TTG_HAVE_HIPBLAS
