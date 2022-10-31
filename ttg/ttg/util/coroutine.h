//
// Created by Eduard Valeyev on 10/31/22.
//

#ifndef TTG_UTIL_COROUTINE_H
#define TTG_UTIL_COROUTINE_H

#include "ttg/config.h"
#include TTG_CXX_COROUTINE_HEADER

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

#endif  // TTG_UTIL_COROUTINE_H
