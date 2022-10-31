//
// Created by Eduard Valeyev on 10/31/22.
//

#ifndef TTG_UTIL_COROUTINE_H
#define TTG_UTIL_COROUTINE_H

#include "ttg/config.h"
#include TTG_CXX_COROUTINE_HEADER

#include "ttg/view.h"

// coroutines in TTG can return void
template <typename... Args>
struct TTG_CXX_COROUTINE_NAMESPACE::coroutine_traits<void, Args...> {
  struct promise_type {
    auto get_return_object() noexcept { return this; }

    TTG_CXX_COROUTINE_NAMESPACE::suspend_never initial_suspend() const noexcept { return {}; }
    TTG_CXX_COROUTINE_NAMESPACE::suspend_never final_suspend() const noexcept { return {}; }

    void return_void() noexcept {}
    void unhandled_exception() noexcept {}
  };
};

#include "ttg/util/void.h"

// coroutines in TTG can return Void
template <typename... Args>
struct TTG_CXX_COROUTINE_NAMESPACE::coroutine_traits<ttg::Void, Args...> {
  struct promise_type {
    auto get_return_object() noexcept { return this; }

    TTG_CXX_COROUTINE_NAMESPACE::suspend_never initial_suspend() const noexcept { return {}; }
    TTG_CXX_COROUTINE_NAMESPACE::suspend_never final_suspend() const noexcept { return {}; }

    void return_void() noexcept {}
    void unhandled_exception() noexcept {}
  };
};

namespace ttg {

  struct resumable_task_state;
  struct resumable_task : std::coroutine_handle<resumable_task_state> {
    using promise_type = struct resumable_task_state;
  };

  struct resumable_task_state {
    resumable_task get_return_object() { return {resumable_task::from_promise(*this)}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    resumable_task_state& return_value(std::initializer_list<HDSpan> views) {
      // push views onto work queue
      return *this;
    }
    void unhandled_exception() {}

   private:
    ViewSpan<std::byte> views[20];
  };

}  // namespace ttg

#endif  // TTG_UTIL_COROUTINE_H
