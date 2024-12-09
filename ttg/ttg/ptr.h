#ifndef TTG_PTR_H
#define TTG_PTR_H

#include "ttg/fwd.h"

#include "ttg/util/meta.h"

namespace ttg {

template<typename T>
using Ptr = TTG_IMPL_NS::Ptr<T>;

template<typename T, typename... Args>
inline Ptr<T> make_ptr(Args&&... args) {
  return TTG_IMPL_NS::make_ptr(std::forward<Args>(args)...);
}

template<typename T>
inline Ptr<std::decay_t<T>> get_ptr(T&& obj) {
  return TTG_IMPL_NS::get_ptr(std::forward<T>(obj));
}

namespace meta {

  /* specialize some traits */

  template<typename T>
  struct is_ptr<ttg::Ptr<T>> : std::true_type
  { };

} // namespace ptr

#if 0
namespace detail {

    /* awaiter for ttg::get_ptr with multiple arguments
     * operator co_wait will return the tuple of ttg::Ptr
     */
    template<typename... Ts>
    struct get_ptr_tpl_t {
    private:
      std::tuple<ttg::Ptr<Ts>...> m_ptr_tuple;
      bool m_is_ready = false;
    public:
      get_ptr_tpl_t(bool is_ready, std::tuple<ttg::ptr<Ts>...>&& ptrs)
      : m_ptr_tuple(std::forward<std::tuple<ttg::Ptr<Ts>...>>(ptrs))
      , m_is_ready(is_ready)
      { }

      bool await_ready() const noexcept {
        return m_is_ready;
      }

      constexpr void await_suspend( std::coroutine_handle<> ) const noexcept {
        /* TODO: anything to be done here? */
      }

      auto await_resume() const noexcept {
        return std::move(m_ptr_tuple);
      }
    };


    /* awaiter for ttg::get_ptr for a single argument */
    template<typename T>
    struct get_ptr_t {
    private:
      ttg::Ptr<T> m_ptr;
      bool m_is_ready = false;
    public:
      get_ptr_t(bool is_ready, ttg::Ptr<T>&& ptr)
      : m_ptr(std::forward<ttg::Ptr<T>>(ptr))
      , m_is_ready(is_ready)
      { }

      bool await_ready() const noexcept {
        return m_is_ready;
      }

      constexpr void await_suspend( std::coroutine_handle<> ) const noexcept {
        /* TODO: anything to be done here? */
      }

      auto await_resume() const noexcept {
        return std::move(m_ptr);
      }
    };
  } // namespace detail

  /**
   * Get an awaiter that results in a ttg::Ptr to a task argument.
   * Must only be called inside a task on a value that was passed
   * to the task and has not yet been moved on.
   * Should be used in conjunction with co_await, e.g.,
   * ttg::Ptr<double> ptr = co_await ttg::get_ptr(val);
   *
   * Multiple value can be passed, which results in a tuple of ptr:
   * ttg::Ptr<double> ptr1, ptr2;
   * std::tie(ptr1, ptr2) = co_await ttg::get_ptr(val1, val2);
   */
  template<typename Arg, typename... Args>
  auto get_ptr(Arg&& arg, Args&&... args) {
    bool is_ready;
    using tpl_type    = std::tuple<ttg::Ptr<std::decay_t<Arg>, std::decay<Args>...>>;
    using result_type = std::pair<bool, tpl_type>;
    result_type p = TTG_IMPL_NS::get_ptr(std::forward<Arg>(arg), std::forward<Args>(args)...);
    if constexpr (sizeof...(Args) > 0) {
      return detail::get_ptr_tpl_t<std::decay_t<Arg>, std::decay_t<Args>...>(p.first, std::move(p.second));
    } else if constexpr (sizeof...(Args) == 0) {
      return detail::get_ptr_t<std::decay_t<Arg>>(p.first, std::move(std::get<0>(p.second)));
    }
  }
#endif // 0
} // namespace ttg

#endif // TTG_PTR_H