#ifndef TTG_VIEW_H
#define TTG_VIEW_H

#include <array>
#include "ttg/util/iovec.h"

namespace ttg {

  enum class ViewScope {
    Allocate  = 0x0,
    SyncIn    = 0x1,
    SyncOut   = 0x2
  };

  template<typename T>
  struct ViewSpan {

    using element_type = T;
    using value_type   = std::remove_cv_t<T>;

    constexpr ViewSpan() = default;
    constexpr ViewSpan(T* ptr, std::size_t size, ViewScope scope = ViewScope::SyncIn)
    : m_ptr(ptr)
    , m_size(size)
    , m_scope(scope)
    { }

    constexpr std::size_t size() const {
      return m_size;
    }

    constexpr T* data() const {
      return m_ptr;
    }

    constexpr bool is_allocate() const {
      return (m_scope == ViewScope::Allocate);
    }

    constexpr bool is_sync_in() const {
      return !!(static_cast<int>(m_scope) & static_cast<int>(ViewScope::SyncIn));
    }

    constexpr bool is_sync_out() const {
      return !!(static_cast<int>(m_scope) & static_cast<int>(ViewScope::SyncOut));
    }

    constexpr ViewScope scope() const {
      return m_scope;
    }

  private:
    T *m_ptr = nullptr;
    std::size_t m_size = 0;
    ViewScope m_scope = ViewScope::Allocate;
  };

  template<typename HostT, typename... DevTypeTs>
  struct View {

    using span_tuple_type  = std::tuple<ttg::ViewSpan<DevTypeTs>...>;
    using host_type = HostT;

    using view_type = View<HostT, DevTypeTs...>;

    constexpr View() = default;

    View(HostT& obj, span_tuple_type spans)
    : m_obj(&obj)
    , m_spans(std::move(spans))
    { }

    View(view_type&&) = default;

    View(const view_type&) = default;

    view_type& operator=(view_type&&) = default;
    view_type& operator=(const view_type&) = default;

    template<std::size_t i>
    auto get_device_ptr() {
      return std::get<i>(m_spans).data();
    }

    template<std::size_t i>
    const auto get_device_ptr() const {
      return std::get<i>(m_spans).data();
    }

    template<std::size_t i>
    std::size_t get_device_size() const {
      return std::get<i>(m_spans).size();
    }

    template<std::size_t i>
    auto get_span() const {
      return std::get<i>(m_spans);
    }

    HostT& get_host_object() {
      return *m_obj;
    }

    const HostT& get_host_object() const {
      return *m_obj;
    }

    template<std::size_t i>
    ViewScope get_scope() const {
      return std::get<i>(m_spans).scope();
    }

    constexpr static std::size_t size() {
      return std::tuple_size_v<decltype(m_spans)>;
    }

  private:
    HostT* m_obj = nullptr;
    span_tuple_type m_spans{};
  };

  template<typename HostT, typename... ViewSpanTs>
  auto make_view(HostT& obj, ViewSpan<ViewSpanTs>... spans) {
    return View(obj, std::make_tuple(std::move(spans)...));
  }

  /// std::span mirrored between host and device memory
  struct HDSpan {
    HDSpan() = default;
    HDSpan(std::byte* ptr, std::size_t nbytes) {
      ptrs_[0] = ptr;
      nbytes_ = nbytes;
      last_touched_space_ = 0;
    }

    std::size_t nbytes() const { return nbytes_; }

    const std::byte* host_data() const { return ptrs_[0]; }

    std::byte* host_data() {
      last_touched_space_ = 0;
      return ptrs_[0];
    }

    const std::byte* device_data() const { return ptrs_[1]; }

    std::byte* device_data() {
      last_touched_space_ = 1;
      return ptrs_[1];
    }

    void mark_synched() { last_touched_space_ = 2; }

   private:
    std::array<std::byte*, 2> ptrs_ = {nullptr, nullptr};
    std::size_t nbytes_ = 0;
    std::size_t last_touched_space_ = 2;
  };

  /// set of std::span's mirrored between host and device memory
  template <std::size_t N = 10>
  struct HDSpans {
    HDSpans() = default;

   private:
    std::array<HDSpan, N> spans_;
  };

}  // namespace ttg


#endif // TTG_VIEW_H
