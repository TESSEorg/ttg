
#include <ttg/config.h>

#if defined(TTG_HAVE_CUDA)
#include <cuda_runtime.h>

namespace ttg::device {
  namespace detail {
    inline thread_local int current_device_ts = 0;
    inline thread_local cudaStream_t current_stream_ts = 0; // default stream

    void reset_current() {
      current_device_ts = 0;
      current_stream_ts = 0;
    }

    void set_current(int device, cudaStream_t stream) {
      current_device_ts = device;
      current_stream_ts = stream;
    }
  } // namespace detail

  int current_device() {
    return detail::current_device_ts;
  }

  cudaStream_t current_stream() {
    return detail::current_stream_ts;
  }
} // namespace ttg

#elif defined(TTG_HAVE_HIP)

#include <hip/hip_runtime.h>

namespace ttg::device {
  namespace detail {
    inline thread_local int current_device_ts = 0;
    inline thread_local hipStream_t current_stream_ts = 0; // default stream

    void reset_current() {
      current_device_ts = 0;
      current_stream_ts = 0;
    }

    void set_current(int device, hipStream_t stream) {
      current_device_ts = device;
      current_stream_ts = stream;
    }
  } // namespace detail

  int current_device() {
    return detail::current_device_ts;
  }

  hipStream_t current_stream() {
    return detail::current_stream_ts;
  }
} // namespace ttg

#endif // defined(TTG_HAVE_HIP)
