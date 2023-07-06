#ifdef TTG_HAVE_CUDART
#include <cublas.h>
#endif // TTG_HAVE_CUDART

#include <exception>
#include <stdexcept>

#include "ttg/config.h"
#include "ttg/device/cublas_helper.h"

namespace ttg::detail {

/* shim wrapper to work around the fact that cublas
 * deliberately breaks its API depending on the order
 * in which header are included */
void cublas_set_kernel_stream(cudaStream_t stream) {
#ifdef TTG_HAVE_CUDART
    cublasStatus_t status = cublasSetKernelStream(stream);
    if (CUBLAS_STATUS_SUCCESS != status) {
        throw std::runtime_error("cublasSetKernelStream failed");
    }
#else
    throw std::runtime_error("Support for cublas missing during installation!");
#endif // TTG_HAVE_CUDART
}

} // namespace