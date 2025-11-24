// SPDX-License-Identifier: BSD-3-Clause

#if defined(TTG_ENABLE_LEVEL_ZERO)
#include <oneapi/mkl.hpp>
#include <sys/time.h>
#endif

#include "../devblas_helper.h"


template <typename Blk>
inline void device_gemm(Blk &C, const Blk &A, const Blk &B) {
  using blk_t = Blk;
  using T = typename blk_t::value_type;
  static_assert(std::is_same_v<T,double> || std::is_same_v<T,float>);
  static const T alpha = 1.0;
  static const T beta  = 1.0;
  // make sure all memory is on the device
  // TODO: A and B are read-only so the owner device will be 0. How to fix?
  //assert(A.b.get_current_device() != 0);
  //assert(B.b.get_current_device() != 0);
  auto device = ttg::device::current_device();
  assert(device.is_device());
#if defined(TTG_ENABLE_CUDA)
  if constexpr (std::is_same_v<T,double>) {
      cublasDgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, C.extent(0), C.extent(1), A.extent(1),
                  &alpha, A.b.current_device_ptr(), A.extent(0), B.b.current_device_ptr(), B.extent(0), &beta,
                  C.b.current_device_ptr(), C.extent(0));
  }
  else if constexpr (std::is_same_v<T,float>) {
      cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, C.extent(0), C.extent(1), A.extent(1),
                  &alpha, A.b.current_device_ptr(), A.extent(0), B.b.current_device_ptr(), B.extent(0), &beta,
                  C.b.current_device_ptr(), C.extent(0));
  }
#elif defined(TTG_ENABLE_HIP)
  if constexpr (std::is_same_v<T,double>) {
    hipblasDgemm(hipblas_handle(),
                 HIPBLAS_OP_N, HIPBLAS_OP_N,
                 C.extent(0), C.extent(1), A.extent(1), &alpha,
                 A.b.current_device_ptr(), A.extent(0),
                 B.b.current_device_ptr(), B.extent(0), &beta,
                 C.b.current_device_ptr(), C.extent(0));
  } else if constexpr (std::is_same_v<T,float>) {
    hipblasSgemm(hipblas_handle(),
                 HIPBLAS_OP_N, HIPBLAS_OP_N,
                 C.extent(0), C.extent(1), A.extent(1), &alpha,
                 A.b.current_device_ptr(), A.extent(0),
                 B.b.current_device_ptr(), B.extent(0), &beta,
                 C.b.current_device_ptr(), C.extent(0));
  }
#elif defined(TTG_ENABLE_LEVEL_ZERO)

#if defined(DEBUG_SYNCHRONOUS)
  try {
#endif /* DEBUG_SYNCHRONOUS */
    cl::sycl::event gemm_event;
    gemm_event = oneapi::mkl::blas::gemm(ttg::device::current_stream(),
         oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
         C.extent(0), C.extent(1), A.extent(1),
         alpha, A.b.current_device_ptr(), A.extent(0),
                B.b.current_device_ptr(), B.extent(0),
         beta,  C.b.current_device_ptr(), C.extent(0));
#if defined(DEBUG_SYNCHRONOUS)
    gemm_event.wait();
  } catch (const oneapi::mkl::invalid_argument &e) {
    std::cerr << "OneAPI MKL BLAS GEMM throws invalid argument exception" << std::endl;
  } catch (const oneapi::mkl::unsupported_device &e) {
    std::cerr << "OneAPI MKL BLAS GEMM throws unsuported device exception" << std::endl;
  } catch (const oneapi::mkl::host_bad_alloc &e) {
    std::cerr << "OneAPI MKL BLAS GEMM throws host bad allocation exception" << std::endl;
  } catch (const oneapi::mkl::device_bad_alloc &e) {
    std::cerr << "OneAPI MKL BLAS GEMM throws device bad allocation exception" << std::endl;
  } catch (const oneapi::mkl::unimplemented &e) {
    std::cerr << "OneAPI MKL BLAS GEMM throws unimplemented exception" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "OneAPI MKL BLAS GEMM throws unexpected exception" << std::endl;
  } catch (...) {
    std::cerr << "OneAPI MKL BLAS GEMM throws unexpected exception that is also badly formatted..." << std::endl;
  }
#endif /* DEBUG_SYNCHRONOUS */
#endif
}
