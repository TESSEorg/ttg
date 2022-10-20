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
  struct View {

    View()
    : m_obj(nullptr)
    , m_spans(ttg::span<DevTs>(nullptr, std::size_t{0})...)
    {}

    View(HostT& obj, std::tuple<ttg::span<DevTs>...> spans)
    : m_obj(&obj)
    , m_spans(std::move(spans))
    { }

    template<std::size_t i>
    auto get_device_ptr() {
      return std::get<i>(m_spans).data();
    }

    template<std::size_t i>
    std::size_t get_device_size() {
      return std::get<i>(m_spans).size();
    }

    HostT& get_host_object() {
      return *m_obj;
    }

  private:
    HostT* m_obj;
    std::tuple<ttg::span<DevTs>...> m_spans;

  };

  template<typename HostT, typename... Spans>
  auto make_view(HostT& obj, std::tuple<Spans...> spans) {
    /* TODO: allocate memory on the device and transfer the data to it */
    return View(obj, std::move(spans));
  }

  template<typename HostT, typename... Spans>
  auto new_view(HostT& obj, std::tuple<Spans...> spans) {
    /* TODO: allocate memory on the device, no copying needed */
    return View(obj, std::move(spans));
  }


} // namespace ttg


#endif // TTG_VIEW_H
