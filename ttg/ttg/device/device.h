#pragma once

#include "ttg/config.h"
#include "ttg/execution.h"
#include "ttg/impl_selector.h"
#include "ttg/fwd.h"
#include "ttg/util/meta.h"



namespace ttg::device {

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

namespace ttg::device {

  namespace detail {
    template<typename Stream>
    struct default_stream {
      static constexpr const Stream value = 0;
    };
    template<typename Stream>
    constexpr const Stream default_stream_v = default_stream<Stream>::value;
  } // namespace detail

} // namespace ttg

#if defined(TTG_HAVE_CUDA)
#include <cuda_runtime.h>
namespace ttg::device {
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::CUDA;
  using Stream = cudaStream_t;
} // namespace ttg::device
#elif defined(TTG_HAVE_HIP)
#include <hip/hip_runtime.h>
namespace ttg::device {
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::HIP;
  using Stream = hipStream_t;
} // namespace ttg::device
#elif defined(TTG_HAVE_LEVEL_ZERO)
#include <CL/sycl.hpp>
namespace ttg::device {
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::L0;
  using Stream = std::add_reference_t<sycl::queue>;
} // namespace ttg::device
#else
namespace ttg::device {
  struct Stream { };
  namespace detail {
    template<>
    struct default_stream<Stream> {
      static constexpr const Stream value = {};
    };
  } // namespace detail
  constexpr ttg::ExecutionSpace available_execution_space = ttg::ExecutionSpace::Host;
} // namespace ttg::device
#endif

namespace ttg::device {

#if !defined(TTG_HAVE_LEVEL_ZERO)
  namespace detail {
    inline thread_local ttg::device::Device current_device_ts = {};
    inline thread_local Stream current_stream_ts = detail::default_stream_v<Stream>; // default stream

    inline void reset_current() {
      current_device_ts = {};
      current_stream_ts = detail::default_stream_v<Stream>;
    }

    inline void set_current(int device, Stream stream) {
      current_device_ts = ttg::device::Device(device, available_execution_space);
      current_stream_ts = stream;
    }
  } // namespace detail

  inline
  Device current_device() {
    return detail::current_device_ts;
  }

  inline
  Stream current_stream() {
    return detail::current_stream_ts;
  }

  inline int num_devices() {
    return TTG_IMPL_NS::num_devices();
  }

#else // TTG_HAVE_LEVEL_ZERO
  /* SYCL needs special treatment because it uses pointers/references */
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
#endif // TTG_HAVE_LEVEL_ZERO

} // namespace ttg
