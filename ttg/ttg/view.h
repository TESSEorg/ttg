#ifndef TTG_VIEW_H
#define TTG_VIEW_H

#include <array>
#include <type_traits>
#include <span>

namespace ttg {

  enum class ViewScope {
    Allocate  = 0x0,
    SyncIn    = 0x1,
    SyncOut   = 0x2
  };

  /**
   * A view span that can be type-punned.
   * We use it instead of a std::span to be able
   * to remove the type and convert to void pointers instead.
   */
  template<typename T, typename = void>
  struct ViewSpan;

  template<>
  struct ViewSpan<void, void> {

    using element_type  = void;
    using value_type    = void;
    using viewspan_type = ViewSpan<value_type>;

    constexpr ViewSpan() = default;
    constexpr ViewSpan(void* ptr, std::size_t size, ViewScope scope = ViewScope::SyncIn)
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
  struct ViewSpan<T, void> : public ViewSpan<void, void> {

    using element_type  = T;
    using value_type    = std::remove_cv_t<T>;
    using viewspan_type = ViewSpan<value_type>;

    constexpr ViewSpan() = default;
    constexpr ViewSpan(T* ptr, std::size_t size, ViewScope scope = ViewScope::SyncIn)
    : ViewSpan<void, void>(ptr, size, scope)
    { }

    constexpr T* data() const {
      return static_cast<T*>(m_data_ptr);
    }

  };

  template<typename HostT, typename... DevTypeTs>
  struct View {

    using span_tuple_type  = std::tuple<ttg::ViewSpan<DevTypeTs>...>;
    using host_type = HostT;

    using view_type = View<HostT, DevTypeTs...>;

    constexpr static std::size_t num_spans = std::tuple_size_v<span_tuple_type>;

  private:
    template<std::size_t... Is>
    View(HostT& obj, span_tuple_type& spans, std::index_sequence<Is...>)
    : m_obj(&obj)
    , m_spans({std::get<Is>(m_spans)...})
    { }

  public:

    constexpr View() = default;

    View(HostT& obj, span_tuple_type spans)
    : View(obj, spans, std::index_sequence_for<DevTypeTs...>{})
    { }

    View(view_type&&) = default;

    View(const view_type&) = default;

    view_type& operator=(view_type&&) = default;
    view_type& operator=(const view_type&) = default;

    template<std::size_t i>
    auto get_device_ptr() {
      return static_cast<std::tuple_element<i, span_tuple_type>>(std::get<i>(m_spans).data());
    }

    template<std::size_t i>
    const auto get_device_ptr() const {
      return static_cast<std::tuple_element<i, span_tuple_type>>(std::get<i>(m_spans).data());
    }

    template<std::size_t i>
    std::size_t get_device_size() const {
      return std::get<i>(m_spans).size();
    }

    template<std::size_t i>
    auto& get_viewspan() const {
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

    /* return a std::span of type-punned ViewSpans */
    std::span<ViewSpan<void>> view_spans() {
      return {m_spans.begin(), m_spans.end()};
    }

  private:
    HostT* m_obj = nullptr;
    /* type-punned storage, cast to actual types in get_device_ptr */
    std::array<ViewSpan<void>, num_spans> m_spans;
    //span_tuple_type m_spans{};
  };

  template<typename HostT, typename... ViewSpanTs>
  auto make_view(HostT& obj, ViewSpan<ViewSpanTs>... spans) {
    return View(obj, std::make_tuple(std::move(spans)...));
  }

  /* overload for trivially-copyable host objects */
  template<typename HostT, typename = std::enable_if_t<std::is_trivially_copyable_v<HostT>>>
  auto make_view(HostT& obj) {
    return View(obj, std::make_tuple(ViewSpan{&obj, sizeof(HostT)}));
  }

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
    using span_type = std::span<ViewSpan<void>>;
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

  struct device_task_promise_type;

  /// task that can be resumed after some events occur
  struct device_task : public TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<device_task_promise_type> {
    using base_type = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<device_task_promise_type>;

    /// these are members mandated by the promise_type concept
    ///@{

    using promise_type = device_task_promise_type;

    ///@}

    device_task(base_type base) : base_type(std::move(base)) {}

    base_type handle() { return *this; }

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
    TTG_CXX_COROUTINE_NAMESPACE::suspend_never final_suspend() noexcept {
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
      m_spans.clear(); // in case we ever come back here
      m_spans.reserve(view_count);
      if constexpr(view_count > 0) {
        auto unpack_lambda = [&](Views&... view){
                                ((m_spans.push_back(device_obj_view(view.host_obj(), view.view_spans()))),...);
                              };
        std::apply(unpack_lambda, views);
      }
      m_state = TTG_DEVICE_CORO_WAIT_TRANSFER;
      return {};
    }

    /* convenience-function to yield a single view */
    template<typename HostT, typename... DeviceViewTs>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always yield_value(View<HostT, DeviceViewTs...> &view) {
      return yield_value(std::tie(view));
    }

    /* waiting for the kernel to complete should always suspend */
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always yield_value(device_op_wait_kernel) {
      m_state = TTG_DEVICE_CORO_WAIT_KERNEL;
      return {};
    }

    void return_void() {
      m_state = TTG_DEVICE_CORO_COMPLETE;
    }

    bool complete() const {
      return m_state == TTG_DEVICE_CORO_COMPLETE;
    }

    using handle_type = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<device_task_promise_type>;
    device_task get_return_object() { return device_task{handle_type::from_promise(*this)}; }

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
