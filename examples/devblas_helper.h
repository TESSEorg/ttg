#include "ttg/config.h"

#include <memory>
#include <stdexcept>
#include <optional>
#include <map>

#ifdef TTG_HAVE_CUDART

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace detail {
  template<typename Stream>
  struct device_stream_t {
    int device;
    Stream stream;
    device_stream_t(int device, Stream stream)
    : device(device)
    , stream(stream)
    { }

    bool operator<(const device_stream_t& ds) const {
      bool result = ((device < ds.device) && (reinterpret_cast<uintptr_t>(stream) < reinterpret_cast<uintptr_t>(ds.stream)));
      std::cout << *this << " < " << ds << ": " << result << std::endl;
      return result;
    }

    bool operator==(const device_stream_t& ds) const {
      bool result = ((device == ds.device) && (stream == ds.stream));
      std::cout << *this << " == " << ds << ": " << result << std::endl;
      return result;
    }
  };
} // namespace detail

namespace std {
template<typename Stream>
  std::ostream& operator<<(std::ostream& os, const ::detail::device_stream_t<Stream>& ds) {
    os << "[" << ds.device << ", " << ds.stream << "]";
    return os;
  }

} //namespace std

/// \brief Returns the cuBLAS handle to be used for launching cuBLAS kernels from the current thread
/// \return the cuBLAS handle for the current thread
inline const cublasHandle_t& cublas_handle() {
  using map_type = std::map<std::pair<int, cudaStream_t>, cublasHandle_t>;
  static thread_local map_type handles;

  int device = ttg::device::current_device();
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

inline const cusolverDnHandle_t& cusolver_handle() {

  //using map_type = std::map<detail::device_stream_t<cudaStream_t>, cusolverDnHandle_t>;
  using map_type = std::map<std::pair<int, cudaStream_t>, cusolverDnHandle_t>;
  static thread_local map_type handles;

  int device = ttg::device::current_device();
  cudaStream_t stream = ttg::device::current_stream();

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

    std::cout << "Creating cusolver handle " << handle << " for device " << device << " stream " << stream << std::endl;
    auto [iterator, success] = handles.insert({{device, stream}, handle});
    it = iterator;
  } else {
    std::cout << "Found cusolver handle " << it->second << " for device " << device << " stream " << stream << std::endl;
  }

  return it->second;
}
#endif // TTG_HAVE_CUDART

#ifdef TTG_HAVE_HIPBLAS

#include <hip_runtime.h>
#include <hipblas.h>
#include <hipsolverDn.h>

/// \brief Returns the rocBLAS handle to be used for launching rocBLAS kernels from the current thread
/// \return the rocBLAS handle for the current thread
const hipblasHandle_t& hipblas_handle() {
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
const hipsolverDnHandle_t& hipsolver_handle() {
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
