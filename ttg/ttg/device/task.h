#ifndef TTG_DEVICE_TASK_H
#define TTG_DEVICE_TASK_H

#include <array>
#include <type_traits>
#include <span>

#include "ttg/fwd.h"
#include "ttg/impl_selector.h"
#include "ttg/ptr.h"

namespace ttg {


  /* yielded when waiting on a kernel to complete */
  struct device_op_wait_kernel
  { };

  enum ttg_device_coro_state {
    TTG_DEVICE_CORO_STATE_NONE,
    TTG_DEVICE_CORO_INIT,
    TTG_DEVICE_CORO_WAIT_TRANSFER,
    TTG_DEVICE_CORO_WAIT_KERNEL,
    TTG_DEVICE_CORO_SENDOUT,
    TTG_DEVICE_CORO_COMPLETE
  };


  namespace detail {
    template<typename... Ts>
    struct to_device_t {
      std::tuple<std::add_lvalue_reference_t<Ts>...> ties;
    };
  } // namespace detail

  template<typename... Args>
  [[nodiscard]]
  inline auto to_device(Args&&... args) {
    return detail::to_device_t<std::remove_reference_t<Args>...>{std::tie(std::forward<Args>(args)...)};
  }

  namespace detail {
    template<typename... Ts>
    struct wait_kernel_t {
      std::tuple<Ts&...> ties;

      /* always suspend */
      constexpr bool await_ready() const noexcept { return false; }

      /* always suspend */
      constexpr void await_suspend( TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<> ) const noexcept {}

      void await_resume() noexcept {
        if constexpr (sizeof...(Ts) > 0) {
          /* hook to allow the backend to handle the data after pushout */
          TTG_IMPL_NS::post_device_out(ties);
        }
      }
    };
  } // namespace detail

  /* Wait for kernel to complete and provided ttg::buffer and ttg::devicescratch
   * to be transferred back to host */
  template<typename... Buffers>
  [[nodiscard]]
  inline auto wait_kernel(Buffers&&... args) {
    static_assert(((ttg::detail::is_buffer_v<std::decay_t<Buffers>>
                    ||ttg::detail::is_devicescratch_v<std::decay_t<Buffers>>)&&...),
                  "Only ttg::buffer and ttg::devicescratch can be waited on!");
    return detail::wait_kernel_t<std::remove_reference_t<Buffers>...>{std::tie(std::forward<Buffers>(args)...)};
  }


  /* TODO: move all device code into ttg::device */
  namespace device {

    /******************************
     * Send/Broadcast handling
     * We pass the value returned by the backend's copy handler into a coroutine
     * and execute the first part (prepare), before suspending it.
     * The second part (send/broadcast) is executed after the task completed.
     ******************************/

    namespace detail {
      struct send_coro_promise_type;

      using send_coro_handle_type = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<send_coro_promise_type>;

      /// task that can be resumed after some events occur
      struct send_coro_state : public send_coro_handle_type {
        using base_type = send_coro_handle_type;

        /// these are members mandated by the promise_type concept
        ///@{

        using promise_type = send_coro_promise_type;

        ///@}

        send_coro_state(base_type base) : base_type(std::move(base)) {}

        base_type& handle() { return *this; }

        /// @return true if ready to resume
        inline bool ready() {
          return true;
        }

        /// @return true if task completed and can be destroyed
        inline bool completed();
      };

      struct send_coro_promise_type {

        /* do not suspend the coroutine on first invocation, we want to run
        * the coroutine immediately and suspend only once.
        */
        TTG_CXX_COROUTINE_NAMESPACE::suspend_never initial_suspend() {
          return {};
        }

        /* we don't suspend the coroutine at the end.
        * it can be destroyed once the send/broadcast is done
        */
        TTG_CXX_COROUTINE_NAMESPACE::suspend_never final_suspend() noexcept {
          return {};
        }

        send_coro_state get_return_object() { return send_coro_state{send_coro_handle_type::from_promise(*this)}; }

        /* the send coros only have an empty co_await */
        TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(ttg::Void) {
          return {};
        }

        void unhandled_exception() {

        }

      };

      template<typename Key, typename Value, ttg::Runtime Runtime = ttg::ttg_runtime>
      inline send_coro_state send_coro (const Key& key, Value&& value, ttg::Out<Key, std::decay_t<Value>> &t,
                                        ttg::detail::value_copy_handler<Runtime>& ch) {
        ttg::detail::value_copy_handler<Runtime> copy_handler = std::move(ch); // destroyed at the end of the coro
        Key k = key;
        t.prepare_send(k, std::forward<Value>(value));
        co_await ttg::Void{}; // we'll come back once the task is done
        t.send(k, std::forward<Value>(value));
      };

      template<typename Value, ttg::Runtime Runtime = ttg::ttg_runtime>
      inline send_coro_state sendv_coro (Value&& value, ttg::Out<void, std::decay_t<Value>> &t,
                                         ttg::detail::value_copy_handler<Runtime>& ch) {
        ttg::detail::value_copy_handler<Runtime> copy_handler = std::move(ch); // destroyed at the end of the coro
        t.prepare_send(std::forward<Value>(value));
        co_await ttg::Void{}; // we'll come back once the task is done
        t.sendv(std::forward<Value>(value));
      };
    } // namespace detail


    /* functionality to deal with sending/broadcasting */
    namespace detail {
      struct send_t {
        send_coro_state coro;
      };
    } // namespace detail

    template <size_t i, typename keyT, typename valueT, typename... out_keysT, typename... out_valuesT,
              ttg::Runtime Runtime = ttg::ttg_runtime>
    inline detail::send_t send(const keyT &key, valueT &&value,
                                        std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      ttg::detail::value_copy_handler<Runtime> copy_handler;
      return detail::send_t{detail::send_coro(key, copy_handler(std::forward<valueT>(value)), std::get<i>(t), copy_handler)};
    }

    template <size_t i, typename valueT, typename... out_keysT, typename... out_valuesT,
              ttg::Runtime Runtime = ttg::ttg_runtime>
    inline detail::send_t sendv(
        valueT &&value, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      ttg::detail::value_copy_handler<Runtime> copy_handler;
      return detail::send_t{detail::sendv_coro(copy_handler(std::forward<valueT>(value)), std::get<i>(t), copy_handler)};
    }


    // clang-format off
    /// \brief Sends a task id and a value to the template tasks attached to the output terminal of this template task
    /// \param[in] i Identifies which output terminal of this template task to select for sending
    /// \param[in] key: the id of the task(s) receiving the value
    /// \param[in] value: the value to send to the receiving task(s)
    // clang-format on
    template <typename keyT, typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
    inline detail::send_t send(size_t i, const keyT &key, valueT &&value) {
      ttg::detail::value_copy_handler<Runtime> copy_handler;
      auto *terminal_ptr = ttg::detail::get_out_terminal<keyT, valueT>(i, "ttg::device::send(i, key, value)");
      return detail::send_t{detail::send_coro(key, copy_handler(std::forward<valueT>(value)), *terminal_ptr, copy_handler)};
    }

    // clang-format off
    /// \brief Sends a task id and a value to the template tasks attached to the output terminal of this template task
    /// \note this is provided to support `send<i>` with and without explicitly-passed terminal tuple
    /// \tparam <i> Identifies which output terminal of this template task to select for sending
    /// \param[in] key: the id of the task(s) receiving the value
    /// \param[in] value: the value to send to the receiving task(s)
    // clang-format on
    template <size_t i, typename keyT, typename valueT>
    inline auto send(const keyT &key, valueT &&value) {
      return ttg::device::send(i, key, std::forward<valueT>(value));
    }



    namespace detail {

      template<typename T, typename Enabler = void>
      struct broadcast_keylist_trait {
        using type = T;
      };

      /* overload for iterable types that extracts the type of the first element */
      template<typename T>
      struct broadcast_keylist_trait<T, std::enable_if_t<ttg::meta::is_iterable_v<T>>> {
        using key_type = decltype(*std::begin(std::get<0>(std::declval<T>())));
      };

      template <size_t KeyId, size_t I, size_t... Is, typename... RangesT, typename valueT,
                typename... out_keysT, typename... out_valuesT>
      inline void prepare_broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
                                    std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
        std::get<I>(t)->prepare_send(std::get<KeyId>(keylists), std::forward<valueT>(value));
        if constexpr (sizeof...(Is) > 0) {
          detail::prepare_broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value), t);
        }
      }

      template <size_t KeyId, size_t I, size_t... Is, typename... RangesT, typename valueT,
                typename... out_keysT, typename... out_valuesT>
      inline void prepare_broadcast(const std::tuple<RangesT...> &keylists, valueT &&value) {
        using key_t = typename broadcast_keylist_trait<
                        std::tuple_element_t<KeyId, std::tuple<std::remove_reference_t<RangesT>...>>
                      >::key_type;
        auto *terminal_ptr = ttg::detail::get_out_terminal<key_t, valueT>(I, "ttg::device::broadcast(keylists, value)");
        terminal_ptr->prepare_send(std::get<KeyId>(keylists), value);
        if constexpr (sizeof...(Is) > 0) {
          detail::prepare_broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value));
        }
      }

      template <size_t KeyId, size_t I, size_t... Is, typename... RangesT, typename valueT,
                typename... out_keysT, typename... out_valuesT>
      inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
                                    std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
        std::get<I>(t)->broadcast(std::get<KeyId>(keylists), std::forward<valueT>(value));
        if constexpr (sizeof...(Is) > 0) {
          detail::broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value), t);
        }
      }

      template <size_t KeyId, size_t I, size_t... Is, typename... RangesT, typename valueT,
                typename... out_keysT, typename... out_valuesT>
      inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value) {
        using key_t = typename broadcast_keylist_trait<
                        std::tuple_element_t<KeyId, std::tuple<std::remove_reference_t<RangesT>...>>
                      >::key_type;
        auto *terminal_ptr = ttg::detail::get_out_terminal<key_t, valueT>(I, "ttg::device::broadcast(keylists, value)");
        terminal_ptr->broadcast(std::get<KeyId>(keylists), value);
        if constexpr (sizeof...(Is) > 0) {
          detail::broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value));
        }
      }

      /* overload with explicit terminals */
      template <size_t I, size_t... Is, typename RangesT, typename valueT,
                typename... out_keysT, typename... out_valuesT,
                ttg::Runtime Runtime = ttg::ttg_runtime>
      inline send_coro_state
      broadcast_coro(RangesT &&keylists, valueT &&value,
                     std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t,
                     ttg::detail::value_copy_handler<Runtime>&& ch) {
        ttg::detail::value_copy_handler<Runtime> copy_handler = std::move(ch); // destroyed at the end of the coro
        RangesT kl = std::forward<RangesT>(keylists); // capture the keylist(s)
        if constexpr (ttg::meta::is_tuple_v<RangesT>) {
          // treat as tuple
          prepare_broadcast<0, I, Is...>(kl, std::forward<std::decay_t<decltype(value)>>(value), t);
          co_await ttg::Void{}; // we'll come back once the task is done
          broadcast<0, I, Is...>(kl, std::forward<std::decay_t<decltype(value)>>(value), t);
        } else if constexpr (!ttg::meta::is_tuple_v<RangesT>) {
          // create a tie to the captured keylist
          prepare_broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value), t);
          co_await ttg::Void{}; // we'll come back once the task is done
          broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value), t);
        }
      }

      /* overload with implicit terminals */
      template <size_t I, size_t... Is, typename RangesT, typename valueT,
                ttg::Runtime Runtime = ttg::ttg_runtime>
      inline send_coro_state
      broadcast_coro(RangesT &&keylists, valueT &&value,
                     ttg::detail::value_copy_handler<Runtime>&& ch) {
        ttg::detail::value_copy_handler<Runtime> copy_handler = std::move(ch); // destroyed at the end of the coro
        RangesT kl = std::forward<RangesT>(keylists); // capture the keylist(s)
        if constexpr (ttg::meta::is_tuple_v<RangesT>) {
          // treat as tuple
          static_assert(sizeof...(Is)+1 == std::tuple_size_v<RangesT>,
                        "Size of keylist tuple must match the number of output terminals");
          prepare_broadcast<0, I, Is...>(kl, std::forward<std::decay_t<decltype(value)>>(value));
          co_await ttg::Void{}; // we'll come back once the task is done
          broadcast<0, I, Is...>(kl, std::forward<std::decay_t<decltype(value)>>(value));
        } else if constexpr (!ttg::meta::is_tuple_v<RangesT>) {
          // create a tie to the captured keylist
          prepare_broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value));
          co_await ttg::Void{}; // we'll come back once the task is done
          broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value));
        }
      }
    }  // namespace detail

    /* overload with explicit terminals and keylist passed by const reference */
    template <size_t I, size_t... Is, typename rangeT, typename valueT, typename... out_keysT, typename... out_valuesT,
              ttg::Runtime Runtime = ttg::ttg_runtime>
    [[nodiscard]]
    inline detail::send_t broadcast(rangeT &&keylist,
                                    valueT &&value,
                                    const std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      ttg::detail::value_copy_handler<Runtime> copy_handler;
      return detail::send_t{
              broadcast_coro<0, I, Is...>(std::forward<rangeT>(keylist),
                                          copy_handler(std::forward<valueT>(value)),
                                          t, std::move(copy_handler))};
    }

    /* overload with implicit terminals and keylist passed by const reference */
    template <size_t i, typename rangeT, typename valueT,
              ttg::Runtime Runtime = ttg::ttg_runtime>
    inline detail::send_t broadcast(rangeT &&keylist, valueT &&value) {
      ttg::detail::value_copy_handler<Runtime> copy_handler;
      return detail::send_t{broadcast_coro<0, i>(std::tie(keylist), copy_handler(std::forward<valueT>(value)),
                                                 std::move(copy_handler))};
    }

    template<typename... Args, ttg::Runtime Runtime = ttg::ttg_runtime>
    [[nodiscard]]
    std::vector<device::detail::send_t> forward(Args&&... args) {
      // TODO: check the cost of this!
      return std::vector{std::forward<Args>(args)...};
    }

  } // namespace device

  /*******************************************
   * Device task promise and coroutine handle
   *******************************************/


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

#if 0
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
#endif // 0

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

#if 0
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
#endif // 0

    /* co_await for the kernel to complete should always suspend */
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(device_op_wait_kernel) {
      std::cout << "yield_value: device_op_wait_kernel" << std::endl;
      m_state = TTG_DEVICE_CORO_WAIT_KERNEL;
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
    auto await_transform(detail::wait_kernel_t<Ts...>&& a) {
      //std::cout << "yield_value: wait_kernel_t" << std::endl;
      if constexpr (sizeof...(Ts) > 0) {
        TTG_IMPL_NS::mark_device_out(a.ties);
      }
      m_state = TTG_DEVICE_CORO_WAIT_KERNEL;
      return a;
    }

    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(std::vector<device::detail::send_t>&& v) {
      m_sends = std::forward<std::vector<device::detail::send_t>>(v);
      m_state = TTG_DEVICE_CORO_SENDOUT;
      return {};
    }

#if 0
    void return_value(std::vector<device::detail::send_t>&& sends) {
      m_sends = std::forward<std::vector<device::detail::send_t>>(sends);
      m_state = TTG_DEVICE_CORO_SENDOUT;
    }
#endif // 0

    void return_void() {
      m_state = TTG_DEVICE_CORO_COMPLETE;
    }

    bool complete() const {
      return m_state == TTG_DEVICE_CORO_COMPLETE;
    }

    device_task get_return_object() { return device_task{device_task_handle_type::from_promise(*this)}; }

    void unhandled_exception() { }

    //using iterator = std::vector<device_obj_view>::iterator;

    /* execute all pending send and broadcast operations */
    void do_sends() {
      for (auto& send : m_sends) {
        send.coro();
      }
      m_sends.clear();
    }

    auto state() {
      return m_state;
    }

  private:
    std::vector<device::detail::send_t> m_sends;
    ttg_device_coro_state m_state = TTG_DEVICE_CORO_STATE_NONE;

  };

  bool device_task::completed() { return base_type::promise().state() == TTG_DEVICE_CORO_COMPLETE; }

  struct device_wait_kernel
  { };


  /**************************
   * Device reduction coros *
   **************************/

  struct device_reducer_promise_type;

  using device_reducer_handle_type = TTG_CXX_COROUTINE_NAMESPACE::coroutine_handle<device_reducer_promise_type>;

  /// task that can be resumed after some events occur
  struct device_reducer : public device_reducer_handle_type {
    using base_type = device_reducer_handle_type;

    /// these are members mandated by the promise_type concept
    ///@{

    using promise_type = device_reducer_promise_type;

    ///@}

    device_reducer(base_type base) : base_type(std::move(base)) {}

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
  struct device_reducer_promise_type {

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

    template<typename... Ts>
    TTG_CXX_COROUTINE_NAMESPACE::suspend_always await_transform(detail::to_device_t<Ts...>&& a) {
      bool need_transfer = !(TTG_IMPL_NS::register_device_memory(a.ties));
      /* TODO: are we allowed to not suspend here and launch the kernel directly? */
      m_state = TTG_DEVICE_CORO_WAIT_TRANSFER;
      return {};
    }

    void return_void() {
      m_state = TTG_DEVICE_CORO_COMPLETE;
    }

    bool complete() const {
      return m_state == TTG_DEVICE_CORO_COMPLETE;
    }

    device_reducer get_return_object() { return device_reducer{device_reducer_handle_type::from_promise(*this)}; }

    void unhandled_exception() { }

    auto state() {
      return m_state;
    }


  private:
    ttg_device_coro_state m_state = TTG_DEVICE_CORO_STATE_NONE;

  };

  bool device_reducer::completed() { return base_type::promise().state() == TTG_DEVICE_CORO_COMPLETE; }

}  // namespace ttg

#endif // TTG_DEVICE_TASK_H