#pragma once

#include <ttg/config.h>


namespace ttg::device {
  class Device {
    int m_id = 0;
    ttg::ExecutionSpace m_space = ttg::ExecutionSpace::Host;

  public:
    Device() = default;
    Device(int id, ttg::ExecutionSpace space)
    : m_id(id)
    , m_space(space)
    { }

    int id() const {
      if (is_host()) {
        throw std::runtime_error("No valid ID for Host execution space!");
      }
      if (is_invalid()) {
        throw std::runtime_error("Invalid execution space!");
      }
      return m_id;
    }

    operator int() const {
      return id();
    }

    ttg::ExecutionSpace space() const {
      return m_space;
    }

    bool is_device() const {
      return ((!is_invalid()) && (m_space != ttg::ExecutionSpace::Host));
    }

    bool is_host() const {
      return (m_space == ttg::ExecutionSpace::Host);
    }

    bool is_invalid() const {
      return (m_space == ttg::ExecutionSpace::Invalid);
    }
  };
} // namespace ttg::device

namespace std {
  std::ostream& operator<<(std::ostream& os, ttg::device::Device device) {
    os << ttg::detail::execution_space_name(device.space());
    if (device.is_device()) {
      os << "(" << device.id() << ")";
    }
    return os;
  }
} // namespace std

#if defined(TTG_HAVE_CUDA)
#include <cuda_runtime.h>

namespace ttg::device {
  namespace detail {
    inline thread_local ttg::device::Device current_device_ts = {};
    inline thread_local cudaStream_t current_stream_ts = 0; // default stream

    void reset_current() {
      current_device_ts = {};
      current_stream_ts = 0;
    }

    void set_current(int device, cudaStream_t stream) {
      current_device_ts = ttg::device::Device(device, ttg::ExecutionSpace::CUDA);
      current_stream_ts = stream;
    }
  } // namespace detail

  inline
  Device current_device() {
    return detail::current_device_ts;
  }

  inline
  cudaStream_t current_stream() {
    return detail::current_stream_ts;
  }
} // namespace ttg

#elif defined(TTG_HAVE_HIP)

#include <hip/hip_runtime.h>

namespace ttg::device {
  namespace detail {
    inline thread_local ttg::device::Device current_device_ts = {};
    inline thread_local hipStream_t current_stream_ts = 0; // default stream

    void reset_current() {
      current_device_ts = {};
      current_stream_ts = 0;
    }

    void set_current(int device, hipStream_t stream) {
      current_device_ts = ttg::device::Device(device, ttg::ExecutionSpace::HIP);
      current_stream_ts = stream;
    }
  } // namespace detail

  inline
  Device current_device() {
    return detail::current_device_ts;
  }

  inline
  hipStream_t current_stream() {
    return detail::current_stream_ts;
  }
} // namespace ttg

#endif // defined(TTG_HAVE_HIP)
