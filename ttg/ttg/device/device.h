#pragma once

#include "ttg/config.h"
#include "ttg/execution.h"
#include "ttg/impl_selector.h"
#include "ttg/fwd.h"



namespace ttg::device {

#if defined(TTG_HAVE_CUDA)
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::CUDA;
#elif defined(TTG_HAVE_HIP)
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::HIP;
#elif defined(TTG_HAVE_LEVEL_ZERO)
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::L0;
#else
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::Invalid;
#endif

  /// Represents a device in a specific execution space
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
      return !is_host();
    }

    bool is_host() const {
      return !is_invalid() && (m_space == ttg::ExecutionSpace::Host);
    }

    bool is_invalid() const {
      return (m_space == ttg::ExecutionSpace::Invalid);
    }
  };
} // namespace ttg::device

namespace std {
  inline
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

    inline void reset_current() {
      current_device_ts = {};
      current_stream_ts = 0;
    }

    inline void set_current(int device, cudaStream_t stream) {
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

    inline void reset_current() {
      current_device_ts = {};
      current_stream_ts = 0;
    }

    inline void set_current(int device, hipStream_t stream) {
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

#elif defined(TTG_HAVE_LEVEL_ZERO)

#include <CL/sycl.hpp>

namespace ttg::device {
  namespace detail {
    inline thread_local ttg::device::Device current_device_ts = {};
    inline thread_local sycl::queue* current_stream_ts = nullptr; // default stream


    inline void reset_current() {
      current_device_ts = {};
      current_stream_ts = nullptr;
    }

    inline void set_current(int device, sycl::queue& stream) {
      current_device_ts = ttg::device::Device(device, ttg::ExecutionSpace::HIP);
      current_stream_ts = &stream;
    }
  } // namespace detail

  inline
  Device current_device() {
    return detail::current_device_ts;
  }

  inline
  sycl::queue& current_stream() {
    return *detail::current_stream_ts;
  }
} // namespace ttg

#else

namespace ttg::device {
  inline Device current_device() {
    return {};
  }

  template<ttg::ExecutionSpace Space = ttg::ExecutionSpace::Invalid>
  inline const void* current_stream() {
    static_assert(Space != ttg::ExecutionSpace::Invalid,
                  "TTG was built without any known device support so we cannot provide a current stream!");
    return nullptr;
  }
} // namespace ttg
#endif // defined(TTG_HAVE_HIP)

namespace ttg::device {
  inline int num_devices() {
    return TTG_IMPL_NS::num_devices();
  }
}
