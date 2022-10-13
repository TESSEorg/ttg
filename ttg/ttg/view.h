#ifndef TTG_VIEW_H
#define TTG_VIEW_H

#include <array>
#include "ttg/util/iovec.h"

namespace ttg {

  namespace detail {

    template<typename T>
    struct typed_iov {
      T* ptr;
      std::size_t size;
    };

    template<typename... Ts>
    struct typed_iovs {
      std::tuple<typed_iov<Ts...>> iovs;
    };

  } // namespace detail

  template<typename HostT, typename... DevTs>
  struct view_t {

    view_t(HostT& obj, detail::typed_iovs<DevTs...> iovs)
    : m_obj(obj)
    , m_iovs(iovs)
    { }

    template<std::size_t i>
    auto get_device_ptr() {
      return std::get<i>(m_iovs).ptr;
    }

    template<std::size_t i>
    std::size_t get_device_size() {
      return std::get<i>(m_iovs).size;
    }

    HostT& get_host_object() {
      return m_obj;
    }

  private:
    HostT& m_obj;
    detail::typed_iovs<DevTs...> m_iovs;

  };

  template<typename HostT, typename... DevTs>
  auto make_view(HostT&& obj, detail::typed_iov<DevTs...> iovs) {
    /* TODO: allocate memory on the device and transfer the data to it */
    return view_t(obj, std::move(iovs));
  }

  template<typename HostT, typename... DevTs>
  auto new_view(HostT&& obj, detail::typed_iov<DevTs...> iovs) {
    /* TODO: allocate memory on the device, no copying needed */
    return view_t(obj, std::move(iovs));
  }


} // namespace ttg


#endif // TTG_VIEW_H
