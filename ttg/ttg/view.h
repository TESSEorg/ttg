#ifndef TTG_VIEW_H
#define TTG_VIEW_H

#include <array>
#include <type_traits>
#include <span>

#include "ttg/ptr.h"

namespace ttg {

  enum class ViewScope {
    Allocate     = 0x0,  //< memory allocated as scratch, but not moved in or out
    Available    = 0x1,  //< data will be reused on device if available, transferred to device otherwise
    SyncIn       = 0x2,  //< data will be allocated on and transferred to device
                         //< if latest version resides on the device (no previous sync-out) the data will
                         //< not be transferred again
    SyncOut      = 0x4,  //< value will be transferred from device to host after kernel completes
    SyncInOut    = 0x8,  //< data will be moved in and synchronized back out after the kernel completes
    AvailableOut = 0x16, //< similar to Available and data is transferred back to device after kernel completes
  };

  /**
   * A view span that can be type-punned.
   * We use it instead of a std::span to be able
   * to remove the type and convert to void pointers instead.
   */
  template<typename T, typename = void>
  struct ViewPart;

  template<>
  struct ViewPart<void, void> {

    using element_type  = void;
    using value_type    = void;
    using view_part_type = ViewPart<value_type>;

    constexpr ViewPart() = default;
    constexpr ViewPart(void* ptr, std::size_t size, ViewScope scope = ViewScope::SyncIn)
    : m_data_ptr(ptr)
    , m_size(size)
    , m_scope(scope)
    { }

    constexpr std::size_t size() const {
      return m_size;
    }

    constexpr void* data() const {
      return m_data_ptr;
    }

    void set_data(void *ptr) {
      m_data_ptr = ptr;
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
    void *m_data_ptr = nullptr;
    std::size_t m_size = 0;
    ViewScope m_scope = ViewScope::Allocate;
  };


  template<typename T>
  struct ViewPart<T, void> : public ViewPart<void, void> {

    using element_type  = T;
    using value_type    = std::remove_cv_t<T>;
    using view_part_type = ViewPart<value_type>;

    constexpr ViewPart() = default;
    constexpr ViewPart(T* ptr, std::size_t size, ViewScope scope = ViewScope::SyncIn)
    : ViewPart<void, void>(ptr, size, scope)
    { }

    constexpr T* data() const {
      return static_cast<T*>(m_data_ptr);
    }

  };

  namespace detail {
    template<typename T>
    struct view_trait
    {
      static constexpr bool is_view = false;
      static constexpr bool is_persistent = false;
    };
  } // namespace detail

  template<typename HostT, typename DevT = HostT, typename... DevTypeTs>
  struct View {

    using span_tuple_type  = std::tuple<ttg::ViewPart<DevT>, ttg::ViewPart<DevTypeTs>...>;
    using host_type = HostT;

    using view_type = View<HostT, DevT, DevTypeTs...>;

    constexpr static std::size_t num_spans = std::tuple_size_v<span_tuple_type>;

  private:
    template<std::size_t... Is>
    View(HostT& obj, span_tuple_type& spans, std::index_sequence<Is...>)
    : m_obj(&obj)
    , m_spans({std::get<Is>(spans)...})
    { }

    View(HostT& obj, span_tuple_type spans)
    : View(obj, spans, std::make_index_sequence<sizeof...(DevTypeTs)+1>{})
    {
      /* TODO: let the runtime handle the view */
      //ttg::detail::register_view(*this);
    }

    /* hidden so that users cannot create views outside of a task */
    template<typename T, typename... ArgsT>
    friend auto make_view(T& obj, ViewPart<ArgsT>... spans);

  public:

    constexpr View() = delete;

    /* move ctor deleted to prevent moving out of a task */
    View(view_type&&) = delete;

    /* copy ctor deleted to prevent copying out of a task */
    View(const view_type&) = delete;

    ~View() {
      /* TODO: let the runtime remove the view */
      //ttg::detail::drop_view(*this);
    }

    /* move operator deleted to prevent moving out of a task */
    view_type& operator=(view_type&&) = delete;
    /* copy operator deleted to prevent moving out of a task */
    view_type& operator=(const view_type&) = delete;

    template<std::size_t i>
    auto get_device_ptr() {
      return static_cast<std::tuple_element_t<i, span_tuple_type>::value_type*>(std::get<i>(m_spans).data());
    }

    template<std::size_t i>
    const auto get_device_ptr() const {
      return static_cast<std::tuple_element_t<i, span_tuple_type>::value_type>(std::get<i>(m_spans).data());
    }

    template<std::size_t i>
    std::size_t get_device_size() const {
      return std::get<i>(m_spans).size();
    }

    template<std::size_t i>
    auto& get_ViewPart() const {
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
      return num_spans;
    }

    /* return a std::span of type-punned ViewParts */
    std::span<ViewPart<void>> view_spans() {
      return {m_spans.begin(), m_spans.end()};
    }

  private:
    HostT* m_obj = nullptr;
    /* type-punned storage, cast to actual types in get_device_ptr */
    std::array<ViewPart<void>, num_spans> m_spans;
    //span_tuple_type m_spans{};
  };

  template<typename HostT, typename... ViewPartTs>
  auto make_view(HostT& obj, ViewPart<ViewPartTs>... spans) {
    return View(obj, std::make_tuple(std::move(spans)...));
  }

  /* overload for trivially-copyable host objects */
  template<typename HostT, typename = std::enable_if_t<std::is_trivially_copyable_v<HostT>>>
  auto make_view(HostT& obj, ViewScope scope = ViewScope::SyncIn) {
    return make_view(obj, ViewPart<HostT>(&obj, sizeof(HostT), scope));
  }

  namespace detail {
    template<typename HostT, typename... DevTs>
    struct view_trait<View<HostT, DevTs...>>
    {
      static constexpr bool is_view = true;
      static constexpr bool is_persistent = false;
    };
  } // namespace detail

  /* yielded when waiting on a kernel to complete */
  struct device_op_wait_kernel
  { };

  enum ttg_device_coro_state {
    TTG_DEVICE_CORO_STATE_NONE,
    TTG_DEVICE_CORO_INIT,
    TTG_DEVICE_CORO_WAIT_TRANSFER,
    TTG_DEVICE_CORO_WAIT_KERNEL,
    TTG_DEVICE_CORO_COMPLETE
  };

  /* type-punned version of the View, providing access to the object
   * pointer and a std::span over the views of that object */
  struct device_obj_view {
    using span_type = std::span<ViewPart<void>>;
    using iterator = typename span_type::iterator;

    device_obj_view(void *obj, span_type&& span)
    : m_obj(obj)
    , m_span(std::forward<span_type>(span))
    { }

    void *host_obj() {
      return m_obj;
    }

    auto begin() {
      return m_span.begin();
    }

    auto end() {
      return m_span.end();
    }

  private:
    void *m_obj;
    span_type m_span;
  };


  /**
   * A view that is persistent and can be copied in and out of the TTG
   */
  template<typename HostT, typename DevT = HostT, typename... DevTypeTs>
  struct PersistentView {

    using span_tuple_type  = std::tuple<ttg::ViewPart<DevT>, ttg::ViewPart<DevTypeTs>...>;
    using host_type = HostT;

    using view_type = PersistentView<HostT, DevT, DevTypeTs...>;

    using ptr_type = ttg::ptr<host_type>;

    constexpr static std::size_t num_spans = std::tuple_size_v<span_tuple_type>;

  private:
    template<std::size_t... Is>
    PersistentView(ptr_type&& ptr, span_tuple_type& spans, std::index_sequence<Is...>)
    : m_ptr(std::forward<ptr_type>(ptr))
    , m_spans({std::get<Is>(spans)...})
    { }

  public:

    constexpr PersistentView() = default;

    PersistentView(ptr_type ptr, span_tuple_type spans)
    : PersistentView(std::move(ptr), spans, std::make_index_sequence<sizeof...(DevTypeTs)+1>{})
    {
      /* TODO: let the runtime handle the view */
      //ttg::detail::register_view(*this);
    }

    PersistentView(view_type&&) = default;

    PersistentView(const view_type&) = default;

    ~PersistentView() {
      /* TODO: let the runtime remove the view */
      //ttg::detail::drop_view(*this);
    }

    view_type& operator=(view_type&&) = default;
    view_type& operator=(const view_type&) = default;

    template<std::size_t i>
    auto get_device_ptr() {
      return static_cast<std::tuple_element_t<i, span_tuple_type>::value_type*>(std::get<i>(m_spans).data());
    }

    template<std::size_t i>
    const auto get_device_ptr() const {
      return static_cast<std::tuple_element_t<i, span_tuple_type>::value_type>(std::get<i>(m_spans).data());
    }

    template<std::size_t i>
    std::size_t get_device_size() const {
      return std::get<i>(m_spans).size();
    }

    template<std::size_t i>
    auto& get_ViewPart() const {
      return std::get<i>(m_spans);
    }

    HostT& get_host_object() {
      return *m_ptr;
    }

    const HostT& get_host_object() const {
      return *m_ptr;
    }

    template<std::size_t i>
    ViewScope get_scope() const {
      return std::get<i>(m_spans).scope();
    }

    constexpr static std::size_t size() {
      return num_spans;
    }

    ptr_type get_ptr() const {
      return m_ptr;
    }

    /* return a std::span of type-punned ViewParts */
    std::span<ViewPart<void>> view_spans() {
      return {m_spans.begin(), m_spans.end()};
    }

  private:
    ptr_type m_ptr;
    /* type-punned storage, cast to actual types in get_device_ptr */
    std::array<ViewPart<void>, num_spans> m_spans;
    //span_tuple_type m_spans{};
  };

  template<typename HostT, typename... ViewPartTs>
  auto make_persistent_view(HostT& obj, ViewPart<ViewPartTs>... spans) {
    return PersistentView(obj, std::make_tuple(std::move(spans)...));
  }

  /* overload for trivially-copyable host objects */
  template<typename HostT, typename = std::enable_if_t<std::is_trivially_copyable_v<HostT>>>
  auto make_persistent_view(HostT& obj, ViewScope scope = ViewScope::SyncIn) {
    return PersistentView<HostT, HostT>(obj, std::make_tuple(ViewPart<HostT>(&obj, sizeof(HostT), scope)));
  }


  namespace detail {
    template<typename HostT, typename... DevTs>
    struct view_trait<PersistentView<HostT, DevTs...>>
    {
      static constexpr bool is_view = true;
      static constexpr bool is_persistent = true;
    };
  } // namespace detail


  namespace detail {
    /* TODO: is this still needed? */
    template<typename... Ts>
    struct await_t {
      std::tuple<Ts&...> ties;
    };
  }

  template<typename... Args>
  inline auto make_await(Args&&... args) {
    return detail::await_t{std::tie(std::forward<Args>(args)...)};
  }

  namespace detail {
    template<typename... Ts>
    struct to_device_t {
      std::tuple<Ts&...> ties;
    };
  } // namespace detail

  template<typename... Args>
  inline auto to_device(Args&&... args) {
    return detail::to_device_t{std::tie(std::forward<Args>(args)...)};
  }

  namespace detail {
    template<typename... Ts>
    struct to_host_t {
      std::tuple<Ts&...> ties;
    };
  } // namespace detail

  template<typename... Args>
  inline auto to_host(Args&&... args) {
    return detail::to_host_t{std::tie(std::forward<Args>(args)...)};
  }

  namespace detail {
    template<typename... T>
    struct wait_kernel_t;
    template<>
    struct wait_kernel_t<>
    { };
    template<typename T, typename... Ts>
    struct wait_kernel_t<T, Ts...> {
      std::tuple<T&, Ts&...> ties;
    };
  } // namespace detail

  /* Wait for the kernel to complete */
  inline auto wait_kernel() {
    return detail::wait_kernel_t<>{};
  }

  /* Wait for kernel to complete and provided ttg::buffer
   * to be transferred back to host */
  template<typename... Buffers>
  inline auto wait_kernel_out(Buffers&&... args) {
    static_assert((ttg::detail::is_buffer_v<std::decay_t<Buffers>>&&...),
                  "Only ttg::buffer can be explicitly waited on!");
    return detail::wait_kernel_t<std::decay_t<Buffers>...>{std::tie(std::forward<Buffers>(args)...)};
  }

  struct device_task_promise_type;

  using device_task_handle_type = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<device_task_promise_type>;

  /// task that can be resumed after some events occur
  struct device_task : public device_task_handle_type {
    using base_type = device_task_handle_type;

    /// these are members mandated by the promise_type concept
    ///@{

    using promise_type = device_task_promise_type;

    ///@}

    device_task(base_type base) : base_type(std::move(base)) {}

    base_type& handle() { return *this; }

    /// @return true if ready to resume
    inline bool ready() {
      return true;
    }

    /// @return true if task completed and can be destroyed
    inline bool completed();
  };

  /* The promise type that stores the views provided by the
   * application task coroutine on the first co_yield. It subsequently
   * tracks the state of the task when it moves from waiting for transfers
   * to waiting for the submitted kernel to complete. */
  struct device_task_promise_type {

    /* do not suspend the coroutine on first invocation, we want to run
     * the coroutine immediately and suspend when we get the device transfers.
     */
    TTG_CXX_COROUTINE_NAMESPACE::suspend_never initial_suspend() {
      m_state = TTG_DEVICE_CORO_INIT;
      return {};
    }

    /* suspend the coroutine at the end of the execution
     * so we can access the promise.
     * TODO: necessary? maybe we can save one suspend here
     */
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always final_suspend() noexcept {
      m_state = TTG_DEVICE_CORO_COMPLETE;
      return {};
    }

    /* waiting for transfers to complete should always suspend
     * TODO: as an optimization, we could check here if all data
     *       is already available and avoid suspending...
     */
    template<typename... Views>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always yield_value(std::tuple<Views&...> &views) {
      /* gather all the views (host object + view spans, type-punned) into a vector */
      constexpr static std::size_t view_count = std::tuple_size_v<std::tuple<Views...>>;
      std::cout << "yield_value: views" << std::endl;
      m_spans.clear(); // in case we ever come back here
      m_spans.reserve(view_count);
      if constexpr(view_count > 0) {
        auto unpack_lambda = [&](Views&... view){
                                ((m_spans.push_back(device_obj_view(&view.get_host_object(),
                                                                    view.view_spans()))),
                                  ...);
                              };
        std::apply(unpack_lambda, views);
      }
      m_state = TTG_DEVICE_CORO_WAIT_TRANSFER;
      return {};
    }

    /* convenience-function to yield a single view */
    template<typename HostT, typename... DeviceViewTs>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always yield_value(View<HostT, DeviceViewTs...> &view) {
      auto tmp_tuple = std::tie(view);
      return yield_value(tmp_tuple);
    }

    /* convenience-function to yield a single view */
    template<typename HostT, typename... DeviceViewTs>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always yield_value(PersistentView<HostT, DeviceViewTs...> &view) {
      auto tmp_tuple = std::tie(view);
      return yield_value(tmp_tuple);
    }

    /* waiting for the kernel to complete should always suspend */
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always yield_value(device_op_wait_kernel) {
      std::cout << "yield_value: device_op_wait_kernel" << std::endl;
      m_state = TTG_DEVICE_CORO_WAIT_KERNEL;
      return {};
    }

    /* Allow co_await on a tuple */
    template<typename... Views>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(std::tuple<Views&...> &views) {
      return yield_value(views);
    }

    /* convenience-function to await a single view */
    template<typename HostT, typename... DeviceViewTs>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(View<HostT, DeviceViewTs...> &view) {
      auto tmp_tuple = std::tie(view);
      return yield_value(tmp_tuple);
    }

    /* convenience-function to await a single view */
    template<typename HostT, typename... DeviceViewTs>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(PersistentView<HostT, DeviceViewTs...> &view) {
      auto tmp_tuple = std::tie(view);
      return yield_value(tmp_tuple);
    }

    /* co_await for the kernel to complete should always suspend */
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(device_op_wait_kernel) {
      std::cout << "yield_value: device_op_wait_kernel" << std::endl;
      m_state = TTG_DEVICE_CORO_WAIT_KERNEL;
      return {};
    }

    template<typename... Ts>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(detail::await_t<Ts...>&& a) {
      bool need_transfer = !(TTG_IMPL_NS::register_device_memory(a.ties));
      /* TODO: are we allowed to not suspend here and launch the kernel directly? */
      m_state = TTG_DEVICE_CORO_WAIT_TRANSFER;
      return {};
    }

    template<typename... Ts>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(detail::to_device_t<Ts...>&& a) {
      bool need_transfer = !(TTG_IMPL_NS::register_device_memory(a.ties));
      /* TODO: are we allowed to not suspend here and launch the kernel directly? */
      m_state = TTG_DEVICE_CORO_WAIT_TRANSFER;
      return {};
    }

    template<typename... Ts>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(detail::wait_kernel_t<ttg::buffer<Ts>...>&& a) {
      std::cout << "yield_value: wait_kernel_t" << std::endl;
      if constexpr (sizeof...(Ts) > 0) {
        TTG_IMPL_NS::mark_device_out(a.ties);
      }
      m_state = TTG_DEVICE_CORO_WAIT_KERNEL;
      return {};
    }

#if 0
    template<typename... Ts>
    auto await_transform(ttg::detail::get_ptr_tpl_t<Ts...>&& a) {
      return a;
    }

    template<typename T>
    auto await_transform(ttg::detail::get_ptr_t<T>&& a) {
      return a;
    }
#endif // 0

    void return_void() {
      m_state = TTG_DEVICE_CORO_COMPLETE;
    }

    bool complete() const {
      return m_state == TTG_DEVICE_CORO_COMPLETE;
    }

    device_task get_return_object() { return device_task{device_task_handle_type::from_promise(*this)}; }

    void unhandled_exception() {

    }

    using iterator = std::vector<device_obj_view>::iterator;

    auto begin() {
      return m_spans.begin();
    }

    auto end() {
      return m_spans.end();
    }

    auto state() {
      return m_state;
    }

  private:

    std::vector<device_obj_view> m_spans;
    ttg_device_coro_state m_state = TTG_DEVICE_CORO_STATE_NONE;

  };

  bool device_task::completed() { return base_type::promise().state() == TTG_DEVICE_CORO_COMPLETE; }

  struct device_wait_kernel
  { };

}  // namespace ttg


#endif // TTG_VIEW_H
