//
// Created by Eduard Valeyev on 10/31/22.
//

#ifndef TTG_UTIL_COROUTINE_H
#define TTG_UTIL_COROUTINE_H

#include "ttg/config.h"
#include TTG_CXX_COROUTINE_HEADER

#include <algorithm>
#include <array>

namespace ttg {

  struct resumable_task_state;

  template <std::size_t N>
  struct resumable_task_events;

  /// represents a one-time event
  struct event {
    void finish() { finished_ = true; }

    /// @return true if the event has occurred
    bool finished() const { return finished_; }

   private:
    std::atomic<bool> finished_ = false;
  };

  /// task that can be resumed after some events occur
  struct resumable_task : TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<resumable_task_state> {
    using base_type = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<resumable_task_state>;

    using promise_type = struct resumable_task_state;

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
    constexpr static inline std::size_t MaxNumEvents = 20;
    using handle_type = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<resumable_task_state>;

    resumable_task get_return_object() { return resumable_task{handle_type::from_promise(*this)}; }

    /// @note start task eagerly
    TTG_CXX_COROUTINE_NAMESPACE::suspend_never initial_suspend() noexcept { return {}; }

    /// @note suspend task before destroying it so the runtime can know that the task is completed
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always final_suspend() noexcept {
      completed_ = true;
      return {};
    }
    void return_void() {}
    void unhandled_exception() {}

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
  /// @note this is the `Awaitable` for resumable_task coroutine
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

    constexpr bool await_ready() const { return await_ready(std::make_index_sequence<N>{}); }

    void await_suspend(TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<resumable_task_state> pending_task) {
      pending_task_ = pending_task;
      pending_task_.promise().set_events(events_);
    }

    void await_resume() {
      if (pending_task_) {
        pending_task_.promise().reset_events();
        pending_task_ = {};
      }
    }

   private:
    std::array<event*, N> events_;
    TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<resumable_task_state> pending_task_;
  };  // resumable_task_events

  // deduce the number of events properly
  template <typename... Events>
  resumable_task_events(Events&&...) -> resumable_task_events<sizeof...(Events)>;

  static_assert(resumable_task_events<0>{}.await_ready() == true);
}  // namespace ttg

#endif  // TTG_UTIL_COROUTINE_H
