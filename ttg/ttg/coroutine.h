//
// Created by Eduard Valeyev on 10/31/22.
//

#ifndef TTG_COROUTINE_H
#define TTG_COROUTINE_H

#include "ttg/config.h"

#ifdef TTG_HAVE_COROUTINE
#include TTG_CXX_COROUTINE_HEADER

#include <algorithm>
#include <array>


namespace ttg {

  // import std coroutine API into ttg namespace

  using suspend_always = TTG_CXX_COROUTINE_NAMESPACE::suspend_always;
  using suspend_never = TTG_CXX_COROUTINE_NAMESPACE::suspend_never;
  template <typename Promise>
  using coroutine_handle = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<Promise>;

  /// @defgroup resumable_task resumable_task coroutine

  /// resumable_task is the original prototype TTG coroutine that awaits on generic events.
  /// There is no proper support for it by TTG runtimes, but it can be useful for understanding how
  /// coroutines work with TTG and potentially in the future as a model for universal resumable tasks

  /// @{

  // fwd-declares

  struct resumable_task_state;

  template <std::size_t N>
  struct resumable_task_events;

  /// represents a generic one-time event
  struct event {
    void finish() { finished_ = true; }

    /// @return true if the event has occurred
    bool finished() const { return finished_; }

   private:
    std::atomic<bool> finished_ = false;
  };

  /// task that can be resumed after some events occur
  struct resumable_task : public ttg::coroutine_handle<resumable_task_state> {
    using base_type = ttg::coroutine_handle<resumable_task_state>;

    /// @name members mandated by the promise_type concept
    /// @{

    using promise_type = struct resumable_task_state;

    /// @}

    resumable_task(base_type base) : base_type(std::move(base)) {}

    base_type handle() { return *this; }

    /// @return true if ready to resume
    inline bool ready() const;

    /// @return true if task completed and can be destroyed
    inline bool completed() const;

    /// @return ttg::span of events that this task depends on
    inline ttg::span<event*> events();
  };

  /// encapsulates the state of the coroutine object visible to the outside world
  /// @note this is the `promise_type` for resumable_task coroutine
  struct resumable_task_state {
    resumable_task_state() noexcept = default;
    // these only live on coroutine frames so make noncopyable and nonmovable
    resumable_task_state(const resumable_task_state&) = delete;
    resumable_task_state& operator=(const resumable_task_state&) = delete;
    resumable_task_state(resumable_task_state&&) = delete;
    resumable_task_state& operator=(resumable_task_state&&) = delete;

    constexpr static inline std::size_t MaxNumEvents = 20;
    using handle_type = coroutine_handle<resumable_task_state>;

    /// @name members mandated by the promise_type concept
    /// @{

    resumable_task get_return_object() { return resumable_task{handle_type::from_promise(*this)}; }

    /// @note start task eagerly
    suspend_never initial_suspend() noexcept { return {}; }

    /// @note suspend task before destroying it so the runtime can know that the task is completed
    suspend_always final_suspend() noexcept {
      completed_ = true;
      return {};
    }
    void return_void() {}
    void unhandled_exception() {}

    /// @}

    /// @name optional members of the promise_type concept
    /// @{

    // these can be used to use optional storage provided by the runtime (e.g. part of the runtime's task data struct)
    // N.B. the existing buffer must be passed to operator new via TLS
    //    void* operator new(std::size_t size)
    //    {
    //      return ::operator new(size);
    //    }

    // N.B. whether the external buffer was used by operator new must be passed via TLS
    //    void operator delete(void* ptr, std::size_t size)
    //    {
    //      ::operator delete(ptr, size);
    //    }

    /// @}

    /// @return true if ready to resume
    constexpr bool ready() const {
      for (std::size_t e = 0; e != nevents_; ++e)
        if (!events_storage_[e]->finished()) return false;
      return true;
    }

    /// @return true if the task is completed
    constexpr bool completed() const { return completed_; }

    ttg::span<event*> events() { return ttg::span(events_storage_.data(), nevents_); }

   private:
    std::array<event*, MaxNumEvents> events_storage_;
    std::size_t nevents_;
    bool completed_ = false;

    template <std::size_t N>
    friend struct resumable_task_events;

    void reset_events() {
      std::fill(events_storage_.begin(), events_storage_.begin() + nevents_, nullptr);
      nevents_ = 0;
    }

    template <std::size_t N>
    void set_events(const std::array<event*, N> events) {
      static_assert(N <= MaxNumEvents);
      std::copy(events.begin(), events.end(), events_storage_.begin());
      nevents_ = N;
    }
  };

  bool resumable_task::ready() const { return base_type::promise().ready(); }
  bool resumable_task::completed() const { return base_type::promise().completed(); }
  ttg::span<event*> resumable_task::events() { return base_type::promise().events(); }

  /// statically-sized sequence of events on whose completion progress of a given task depends on
  /// @note this is the `Awaiter` for resumable_task coroutine
  ///       (the concept is not defined in the standard, see
  ///       https://lewissbaker.github.io/2017/11/17/understanding-operator-co-await instead )
  template <std::size_t N>
  struct resumable_task_events {
   private:
    template <std::size_t... I>
    constexpr bool await_ready(std::index_sequence<I...>) const {
      return (std::get<I>(events_)->finished() && ...);
    }

   public:
    template <typename... Events>
    constexpr resumable_task_events(Events&&... events) : events_{(&events)...} {}

    /// @name members mandated by the Awaiter concept
    /// @{

    constexpr bool await_ready() const { return await_ready(std::make_index_sequence<N>{}); }

    void await_suspend(coroutine_handle<resumable_task_state> pending_task) {
      pending_task_ = pending_task;
      pending_task_.promise().set_events(events_);
    }

    void await_resume() {
      if (pending_task_) {
        pending_task_.promise().reset_events();
        pending_task_ = {};
      }
    }

    /// @}

   private:
    std::array<event*, N> events_;
    coroutine_handle<resumable_task_state> pending_task_;
  };  // resumable_task_events

  // deduce the number of events properly
  template <typename... Events>
  resumable_task_events(Events&&...) -> resumable_task_events<sizeof...(Events)>;

  static_assert(resumable_task_events<0>{}.await_ready() == true);

  /// @}

  /////////////////////////////////////////////////////////////////////////////
  // describe all types of coroutine tasks known to TTG
  /////////////////////////////////////////////////////////////////////////////

  // fwd declare all coro promise types that have not been declared yet
  namespace device::detail {
    template<ttg::ExecutionSpace Space>
    struct device_task_promise_type;
  }  // namespace device::detail

  /// describes all types of coroutine tasks known to TTG
  /// @internal only exists to simplify metaprogramming in the backend code
  enum class TaskCoroutineID {
    /// not a coroutine, i.e. a standard task function, -> void
    Invalid,
    /// -> ttg::resumable_task
    ResumableTask,
    /// -> ttg::device::Task
    DeviceTask
  };

}  // namespace ttg

#endif // TTG_HAVE_COROUTINE

#endif  // TTG_COROUTINE_H
