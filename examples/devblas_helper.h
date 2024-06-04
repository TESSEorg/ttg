#pragma once

#include "ttg/config.h"

#include "ttg/device/device.h"

#include <memory>
#include <stdexcept>
#include <optional>
#include <map>
#include <mutex>

#ifdef TTG_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

/// \brief Returns the cuBLAS handle to be used for launching cuBLAS kernels from the current thread
/// \return the cuBLAS handle for the current thread
template<typename T = int>
inline const cublasHandle_t& cublas_handle(T _ = 0) {
  using map_type = std::map<std::pair<int, cudaStream_t>, cublasHandle_t>;
  static map_type handles;
  static std::mutex handle_mtx;

  auto d = ttg::device::current_device();
  int device = 0; // assume 0 if we don't have a device
  if (d.is_device()) {
    device = d;
  }

  cudaStream_t stream = ttg::device::current_stream();

  std::lock_guard g(handle_mtx);
  map_type::iterator it;
  if ((it = handles.find({device, stream})) == handles.end()){
    cublasHandle_t handle;
    auto status = cublasCreate_v2(&handle);
    if (CUBLAS_STATUS_SUCCESS != status) {
      std::cerr << "cublasCreate_v2 failed: " << status << std::endl;
      throw std::runtime_error("cublasCreate_v2 failed");
    }
    status = cublasSetStream_v2(handle, stream);
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
  static map_type handles;
  static std::mutex handle_mtx;

  auto d = ttg::device::current_device();
  int device = 0; // assume 0 if we don't have a device
  if (d.is_device()) {
    device = d;
  }
  cudaStream_t stream = ttg::device::current_stream();

  std::lock_guard g(handle_mtx);
  map_type::iterator it;
  if ((it = handles.find({device, stream})) == handles.end()){
    cusolverDnHandle_t handle;
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
  }

  return it->second;
}
#endif // TTG_ENABLE_CUDA

#ifdef TTG_ENABLE_HIP

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

/// \brief Returns the rocBLAS handle to be used for launching rocBLAS kernels from the current thread
/// \return the rocBLAS handle for the current thread
template<typename T = int>
inline const hipblasHandle_t& hipblas_handle(T _ = 0) {
  using map_type = std::map<std::pair<int, hipStream_t>, hipblasHandle_t>;
  static map_type handles;
  static std::mutex handle_mtx;

  auto d = ttg::device::current_device();
  int device = 0; // assume 0 if we don't have a device
  if (d.is_device()) {
    device = d;
  }

  hipStream_t stream = ttg::device::current_stream();

  std::lock_guard g(handle_mtx);
  map_type::iterator it;
  if ((it = handles.find({device, stream})) == handles.end()){
    hipblasHandle_t handle;
    auto status = hipblasCreate(&handle);
    if (HIPBLAS_STATUS_SUCCESS != status) {
      throw std::runtime_error("hipblasCreate failed");
    }
    status = hipblasSetStream(handle, stream);
    if (HIPBLAS_STATUS_SUCCESS != status) {
      throw std::runtime_error("hipblasSetStream failed");
    }
    auto [iterator, success] = handles.insert({{device, stream}, handle});
    it = iterator;
  }

  return it->second;
}

/// \brief Returns the hipsolver handle to be used for launching rocBLAS kernels from the current thread
/// \return the hipsolver handle for the current thread
template<typename T = int>
inline const hipsolverDnHandle_t& hipsolver_handle(T _ = 0) {
  using map_type = std::map<std::pair<int, hipStream_t>, hipsolverDnHandle_t>;
  static map_type handles;
  static std::mutex handle_mtx;
  auto d = ttg::device::current_device();
  int device = 0; // assume 0 if we don't have a device
  if (d.is_device()) {
    device = d;
  }

  hipStream_t stream = ttg::device::current_stream();

  std::lock_guard g(handle_mtx);
  map_type::iterator it;
  if ((it = handles.find({device, stream})) == handles.end()){
    hipsolverDnHandle_t handle;
    auto status = hipsolverDnCreate(&handle);
    if (HIPSOLVER_STATUS_SUCCESS != status) {
      throw std::runtime_error("hipsolverCreate failed");
    }
    status = hipsolverDnSetStream(handle, stream);
    if (HIPSOLVER_STATUS_SUCCESS != status) {
      throw std::runtime_error("hipsolverSetStream failed");
    }
    auto [iterator, success] = handles.insert({{device, stream}, handle});
    it = iterator;
  }
  return it->second;
}
#endif // TTG_ENABLE_HIP
