#ifndef TTG_DEVICE_TASK_H
#define TTG_DEVICE_TASK_H

#include <array>
#include <type_traits>
#include <span>


#include "ttg/fwd.h"
#include "ttg/impl_selector.h"
#include "ttg/ptr.h"
#include "ttg/devicescope.h"

#ifdef TTG_HAVE_COROUTINE

namespace ttg::device {

  namespace detail {

    struct device_input_data_t {
      using impl_data_t = decltype(TTG_IMPL_NS::buffer_data(std::declval<ttg::Buffer<int>>()));

      device_input_data_t(impl_data_t data, ttg::scope scope, bool isconst, bool isscratch)
      : impl_data(data), scope(scope), is_const(isconst), is_scratch(isscratch)
      { }
      impl_data_t impl_data;
      ttg::scope scope;
      bool is_const;
      bool is_scratch;
    };

    template <typename... Ts>
    struct to_device_t {
      std::tuple<std::add_lvalue_reference_t<Ts>...> ties;
    };

    /* extract buffer information from to_device_t */
    template<typename... Ts, std::size_t... Is>
    auto extract_buffer_data(detail::to_device_t<Ts...>& a, std::index_sequence<Is...>) {
      using arg_types = std::tuple<Ts...>;
      return std::array<device_input_data_t, sizeof...(Ts)>{
                device_input_data_t{TTG_IMPL_NS::buffer_data(std::get<Is>(a.ties)),
                                    std::get<Is>(a.ties).scope(),
                                    ttg::meta::is_const_v<std::tuple_element_t<Is, arg_types>>,
                                    ttg::meta::is_devicescratch_v<std::tuple_element_t<Is, arg_types>>}...};
    }
  }  // namespace detail

  struct Input {
  private:
    std::vector<detail::device_input_data_t> m_data;

  public:
    Input() { }
    template<typename... Args>
    Input(Args&&... args)
    : m_data{{TTG_IMPL_NS::buffer_data(args), args.scope(),
              std::is_const_v<std::remove_reference_t<Args>>,
              ttg::meta::is_devicescratch_v<std::decay_t<Args>>}...}
    { }

    template<typename T>
    void add(T&& v) {
      using type = std::remove_reference_t<T>;
      m_data.emplace_back(TTG_IMPL_NS::buffer_data(v), v.scope(), std::is_const_v<type>,
                          ttg::meta::is_devicescratch_v<type>);
    }

    ttg::span<detail::device_input_data_t> span() {
      return ttg::span(m_data);
    }
  };

  namespace detail {
    // overload for Input
    template <>
    struct to_device_t<Input> {
      Input& input;
    };
  } // namespace detail

  /**
   * Select a device to execute on based on the provided buffer and scratchspace objects.
   * Returns an object that should be awaited on using \c co_await.
   * Upon resume, the device is selected (i.e., \sa ttg::device::current_device and
   * \sa ttg::device::current_stream are available) and the buffers are available on the
   * selected device.
   */
  template <typename... Args>
  [[nodiscard]]
  inline auto select(Args &&...args) {
    return detail::to_device_t<std::remove_reference_t<Args>...>{std::tie(std::forward<Args>(args)...)};
  }

  [[nodiscard]]
  inline auto select(Input& input) {
    return detail::to_device_t<Input>{input};
  }

  namespace detail {

    enum ttg_device_coro_state {
      TTG_DEVICE_CORO_STATE_NONE,
      TTG_DEVICE_CORO_INIT,
      TTG_DEVICE_CORO_WAIT_TRANSFER,
      TTG_DEVICE_CORO_WAIT_KERNEL,
      TTG_DEVICE_CORO_SENDOUT,
      TTG_DEVICE_CORO_COMPLETE
    };

    template <typename... Ts>
    struct wait_kernel_t {
      std::tuple<std::add_lvalue_reference_t<Ts>...> ties;

      /* always suspend */
      constexpr bool await_ready() const noexcept { return false; }

      /* always suspend */
      template <typename Promise>
      constexpr void await_suspend(ttg::coroutine_handle<Promise>) const noexcept {}

      void await_resume() noexcept {
        if constexpr (sizeof...(Ts) > 0) {
          /* hook to allow the backend to handle the data after pushout */
          TTG_IMPL_NS::post_device_out(ties);
        }
      }
    };
  }  // namespace detail

  /**
   * Wait for previously submitted kernels to complete and provided
   * ttg::Buffer and ttg::devicescratch to be transferred back to host.
   * Must only be called after awaiting \sa ttg::device::select has resumed.
   */
  template <typename... Buffers>
  [[nodiscard]]
  inline auto wait(Buffers &&...args) {
    static_assert(
        ((ttg::meta::is_buffer_v<std::decay_t<Buffers>> || ttg::meta::is_devicescratch_v<std::decay_t<Buffers>>) &&
         ...),
        "Only ttg::Buffer and ttg::devicescratch can be waited on!");
    return detail::wait_kernel_t<std::remove_reference_t<Buffers>...>{std::tie(std::forward<Buffers>(args)...)};
  }

  /******************************
   * Send/Broadcast handling
   * We pass the value returned by the backend's copy handler into a coroutine
   * and execute the first part (prepare), before suspending it.
   * The second part (send/broadcast) is executed after the task completed.
   ******************************/

  namespace detail {
    struct send_coro_promise_type;

    using send_coro_handle_type = ttg::coroutine_handle<send_coro_promise_type>;

    /// a coroutine for sending data from the device
    struct send_coro_state : public send_coro_handle_type {
      using base_type = send_coro_handle_type;

      /// these are members mandated by the promise_type concept
      ///@{

      using promise_type = send_coro_promise_type;

      ///@}

      send_coro_state(base_type base) : base_type(std::move(base)) {}

      base_type &handle() { return *this; }

      /// @return true if ready to resume
      inline bool ready() { return true; }

      /// @return true if task completed and can be destroyed
      inline bool completed();
    };

    /// the promise type for the send coroutine
    struct send_coro_promise_type {
      /* do not suspend the coroutine on first invocation, we want to run
      * the coroutine immediately and suspend only once.
       */
      ttg::suspend_never initial_suspend() { return {}; }

      /* we don't suspend the coroutine at the end.
      * it can be destroyed once the send/broadcast is done
       */
      ttg::suspend_never final_suspend() noexcept { return {}; }

      send_coro_state get_return_object() { return send_coro_state{send_coro_handle_type::from_promise(*this)}; }

      /* the send coros only have an empty co_await */
      ttg::suspend_always await_transform(ttg::Void) { return {}; }

      void unhandled_exception() {
        std::cerr << "Send coroutine caught an unhandled exception!" << std::endl;
        throw;  // fwd
      }

      void return_void() {}
    };

    template <typename Key, typename Value, ttg::Runtime Runtime = ttg::ttg_runtime>
    inline send_coro_state send_coro(const Key &key, Value &&value, ttg::Out<Key, std::decay_t<Value>> &t,
                                     ttg::detail::value_copy_handler<Runtime> &ch) {
      ttg::detail::value_copy_handler<Runtime> copy_handler = std::move(ch);  // destroyed at the end of the coro
      Key k = key;
      t.prepare_send(k, std::forward<Value>(value));
      co_await ttg::Void{};  // we'll come back once the task is done
      t.send(k, std::forward<Value>(value));
    };

    template <typename Value, ttg::Runtime Runtime = ttg::ttg_runtime>
    inline send_coro_state sendv_coro(Value &&value, ttg::Out<void, std::decay_t<Value>> &t,
                                      ttg::detail::value_copy_handler<Runtime> &ch) {
      ttg::detail::value_copy_handler<Runtime> copy_handler = std::move(ch);  // destroyed at the end of the coro
      t.prepare_send(std::forward<Value>(value));
      co_await ttg::Void{};  // we'll come back once the task is done
      t.sendv(std::forward<Value>(value));
    };

    template <typename Key, ttg::Runtime Runtime = ttg::ttg_runtime>
    inline send_coro_state sendk_coro(const Key &key, ttg::Out<Key, void> &t) {
      // no need to prepare the send but we have to suspend once
      Key k = key;
      co_await ttg::Void{};  // we'll come back once the task is done
      t.sendk(k);
    };

    template <ttg::Runtime Runtime = ttg::ttg_runtime>
    inline send_coro_state send_coro(ttg::Out<void, void> &t) {
      // no need to prepare the send but we have to suspend once
      co_await ttg::Void{};  // we'll come back once the task is done
      t.send();
    };

    struct send_t {
      send_coro_state coro;
    };
  }  // namespace detail

  template <size_t i, typename keyT, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t send(const keyT &key, valueT &&value, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    ttg::detail::value_copy_handler<Runtime> copy_handler;
    return detail::send_t{
        detail::send_coro(key, copy_handler(std::forward<valueT>(value)), std::get<i>(t), copy_handler)};
  }

  template <size_t i, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t sendv(valueT &&value, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    ttg::detail::value_copy_handler<Runtime> copy_handler;
    return detail::send_t{detail::sendv_coro(copy_handler(std::forward<valueT>(value)), std::get<i>(t), copy_handler)};
  }

  template <size_t i, typename Key, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t sendk(const Key &key, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    return detail::send_t{detail::sendk_coro(key, std::get<i>(t))};
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


  template <typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t sendv(std::size_t i, valueT &&value) {
    auto *terminal_ptr = ttg::detail::get_out_terminal<void, valueT>(i, "ttg::device::send(i, key, value)");
    ttg::detail::value_copy_handler<Runtime> copy_handler;
    return detail::send_t{detail::sendv_coro(copy_handler(std::forward<valueT>(value)), *terminal_ptr, copy_handler)};
  }

  template <typename Key, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t sendk(std::size_t i, const Key& key) {
    auto *terminal_ptr = ttg::detail::get_out_terminal<Key, void>(i, "ttg::device::send(i, key, value)");
    return detail::send_t{detail::sendk_coro(key, *terminal_ptr)};
  }

  template <ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t send(std::size_t i) {
    auto *terminal_ptr = ttg::detail::get_out_terminal<void, void>(i, "ttg::device::send(i, key, value)");
    return detail::send_t{detail::send_coro(*terminal_ptr)};
  }


  template <std::size_t i, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t sendv(valueT &&value) {
    return sendv(i, std::forward<valueT>(value));
  }

  template <size_t i, typename Key, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t sendk(const Key& key) {
    return sendk(i, key);
  }

  template <size_t i, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t sendk() {
    return send(i);
  }

  namespace detail {

    template<typename T, typename Enabler = void>
    struct broadcast_keylist_trait {
      using type = T;
    };

    /* overload for iterable types that extracts the type of the first element */
    template<typename T>
    struct broadcast_keylist_trait<T, std::enable_if_t<ttg::meta::is_iterable_v<T>>> {
      using key_type = decltype(*std::begin(std::declval<T>()));
    };

    template <size_t KeyId, size_t I, size_t... Is, typename... RangesT, typename valueT,
              typename... out_keysT, typename... out_valuesT>
    inline void prepare_broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
                                  std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      std::get<I>(t).prepare_send(std::get<KeyId>(keylists), std::forward<valueT>(value));
      if constexpr (sizeof...(Is) > 0) {
        prepare_broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value), t);
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
        prepare_broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value));
      }
    }

    template <size_t KeyId, size_t I, size_t... Is, typename... RangesT, typename valueT,
              typename... out_keysT, typename... out_valuesT>
    inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
                                  std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      std::get<I>(t).broadcast(std::get<KeyId>(keylists), std::forward<valueT>(value));
      if constexpr (sizeof...(Is) > 0) {
        detail::broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value), t);
      }
    }

    template <size_t KeyId, size_t I, size_t... Is, typename... RangesT, typename valueT>
    inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value) {
      using key_t = typename broadcast_keylist_trait<
                      std::tuple_element_t<KeyId, std::tuple<std::remove_reference_t<RangesT>...>>
                    >::key_type;
      auto *terminal_ptr = ttg::detail::get_out_terminal<key_t, valueT>(I, "ttg::device::broadcast(keylists, value)");
      terminal_ptr->broadcast(std::get<KeyId>(keylists), value);
      if constexpr (sizeof...(Is) > 0) {
        ttg::device::detail::broadcast<KeyId+1, Is...>(keylists, std::forward<valueT>(value));
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
        ttg::device::detail::broadcast<0, I, Is...>(kl, std::forward<std::decay_t<decltype(value)>>(value), t);
      } else if constexpr (!ttg::meta::is_tuple_v<RangesT>) {
        // create a tie to the captured keylist
        prepare_broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value), t);
        co_await ttg::Void{}; // we'll come back once the task is done
        ttg::device::detail::broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value), t);
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
        ttg::device::detail::broadcast<0, I, Is...>(kl, std::forward<std::decay_t<decltype(value)>>(value));
      } else if constexpr (!ttg::meta::is_tuple_v<RangesT>) {
        // create a tie to the captured keylist
        prepare_broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value));
        co_await ttg::Void{}; // we'll come back once the task is done
        ttg::device::detail::broadcast<0, I, Is...>(std::tie(kl), std::forward<std::decay_t<decltype(value)>>(value));
      }
    }

    /**
     * broadcastk
     */

    template <size_t KeyId, size_t I, size_t... Is, typename... RangesT,
              typename... out_keysT, typename... out_valuesT>
    inline void broadcastk(const std::tuple<RangesT...> &keylists,
                           std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      std::get<I>(t).broadcast(std::get<KeyId>(keylists));
      if constexpr (sizeof...(Is) > 0) {
        detail::broadcastk<KeyId+1, Is...>(keylists, t);
      }
    }

    template <size_t KeyId, size_t I, size_t... Is, typename... RangesT>
    inline void broadcastk(const std::tuple<RangesT...> &keylists) {
      using key_t = typename broadcast_keylist_trait<
                      std::tuple_element_t<KeyId, std::tuple<std::remove_reference_t<RangesT>...>>
                    >::key_type;
      auto *terminal_ptr = ttg::detail::get_out_terminal<key_t, void>(I, "ttg::device::broadcastk(keylists)");
      terminal_ptr->broadcast(std::get<KeyId>(keylists));
      if constexpr (sizeof...(Is) > 0) {
        ttg::device::detail::broadcastk<KeyId+1, Is...>(keylists);
      }
    }

    /* overload with explicit terminals */
    template <size_t I, size_t... Is, typename RangesT,
              typename... out_keysT, typename... out_valuesT,
              ttg::Runtime Runtime = ttg::ttg_runtime>
    inline send_coro_state
    broadcastk_coro(RangesT &&keylists,
                    std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      RangesT kl = std::forward<RangesT>(keylists); // capture the keylist(s)
      if constexpr (ttg::meta::is_tuple_v<RangesT>) {
        // treat as tuple
        co_await ttg::Void{}; // we'll come back once the task is done
        ttg::device::detail::broadcastk<0, I, Is...>(kl, t);
      } else if constexpr (!ttg::meta::is_tuple_v<RangesT>) {
        // create a tie to the captured keylist
        co_await ttg::Void{}; // we'll come back once the task is done
        ttg::device::detail::broadcastk<0, I, Is...>(std::tie(kl), t);
      }
    }

    /* overload with implicit terminals */
    template <size_t I, size_t... Is, typename RangesT,
              ttg::Runtime Runtime = ttg::ttg_runtime>
    inline send_coro_state
    broadcastk_coro(RangesT &&keylists) {
      RangesT kl = std::forward<RangesT>(keylists); // capture the keylist(s)
      if constexpr (ttg::meta::is_tuple_v<RangesT>) {
        // treat as tuple
        static_assert(sizeof...(Is)+1 == std::tuple_size_v<RangesT>,
                      "Size of keylist tuple must match the number of output terminals");
        co_await ttg::Void{}; // we'll come back once the task is done
        ttg::device::detail::broadcastk<0, I, Is...>(kl);
      } else if constexpr (!ttg::meta::is_tuple_v<RangesT>) {
        // create a tie to the captured keylist
        co_await ttg::Void{}; // we'll come back once the task is done
        ttg::device::detail::broadcastk<0, I, Is...>(std::tie(kl));
      }
    }
  }  // namespace detail

  /* overload with explicit terminals and keylist passed by const reference */
  template <size_t I, size_t... Is, typename rangeT, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  [[nodiscard]]
  inline detail::send_t broadcast(rangeT &&keylist,
                                  valueT &&value,
                                  std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    ttg::detail::value_copy_handler<Runtime> copy_handler;
    return detail::send_t{
            detail::broadcast_coro<I, Is...>(std::forward<rangeT>(keylist),
                                            copy_handler(std::forward<valueT>(value)),
                                            t, std::move(copy_handler))};
  }

  /* overload with implicit terminals and keylist passed by const reference */
  template <size_t i, typename rangeT, typename valueT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t broadcast(rangeT &&keylist, valueT &&value) {
    ttg::detail::value_copy_handler<Runtime> copy_handler;
    return detail::send_t{detail::broadcast_coro<i>(std::tie(keylist),
                                                    copy_handler(std::forward<valueT>(value)),
                                                    std::move(copy_handler))};
  }

  /* overload with explicit terminals and keylist passed by const reference */
  template <size_t I, size_t... Is, typename rangeT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  [[nodiscard]]
  inline detail::send_t broadcastk(rangeT &&keylist,
                                   std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    ttg::detail::value_copy_handler<Runtime> copy_handler;
    return detail::send_t{
            detail::broadcastk_coro<I, Is...>(std::forward<rangeT>(keylist), t)};
  }

  /* overload with implicit terminals and keylist passed by const reference */
  template <size_t i, typename rangeT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline detail::send_t broadcastk(rangeT &&keylist) {
    if constexpr (std::is_rvalue_reference_v<decltype(keylist)>) {
      return detail::send_t{detail::broadcastk_coro<i>(std::forward<rangeT>(keylist))};
    } else {
      return detail::send_t{detail::broadcastk_coro<i>(std::tie(keylist))};
    }
  }

  template<typename... Args, ttg::Runtime Runtime = ttg::ttg_runtime>
  [[nodiscard]]
  std::vector<device::detail::send_t> forward(Args&&... args) {
    // TODO: check the cost of this!
    return std::vector<device::detail::send_t>{std::forward<Args>(args)...};
  }

  /*******************************************
   * Device task promise and coroutine handle
   *******************************************/

  namespace detail {
    // fwd-decl
    struct device_task_promise_type;
    // base type for ttg::device::Task
    using device_task_handle_type = ttg::coroutine_handle<device_task_promise_type>;
  } // namespace detail

  /// A device::Task is a coroutine (a callable that can be suspended and resumed).

  /// Since task execution in TTG is not preempable, tasks should not block.
  /// The purpose of suspending a task is to yield control back to the runtime until some events occur;
  /// in the meantime its executor (e.g., a user-space thread) can perform other work.
  /// Once the task function reaches a point where further progress is pending completion of one or more asynchronous
  /// actions the function needs to be suspended via a coroutine await (`co_await`).
  /// Resumption will be handled by the runtime.
  struct Task : public detail::device_task_handle_type {
    using base_type = detail::device_task_handle_type;

    /// these are members mandated by the promise_type concept
    ///@{

    using promise_type = detail::device_task_promise_type;

    ///@}

    Task(base_type base) : base_type(std::move(base)) {}

    base_type& handle() { return *this; }

    /// @return true if ready to resume
    inline bool ready() {
      return true;
    }

    /// @return true if task completed and can be destroyed
    inline bool completed();
  };

  namespace detail {

    /* The promise type that stores the views provided by the
    * application task coroutine on the first co_yield. It subsequently
    * tracks the state of the task when it moves from waiting for transfers
    * to waiting for the submitted kernel to complete. */
    struct device_task_promise_type {

      /* do not suspend the coroutine on first invocation, we want to run
      * the coroutine immediately and suspend when we get the device transfers.
      */
      ttg::suspend_never initial_suspend() {
        m_state = ttg::device::detail::TTG_DEVICE_CORO_INIT;
        return {};
      }

      /* suspend the coroutine at the end of the execution
      * so we can access the promise.
      * TODO: necessary? maybe we can save one suspend here
      */
      ttg::suspend_always final_suspend() noexcept {
        m_state = ttg::device::detail::TTG_DEVICE_CORO_COMPLETE;
        return {};
      }

      /* Allow co_await on a tuple */
      template<typename... Views>
      ttg::suspend_always await_transform(std::tuple<Views&...> &views) {
        return yield_value(views);
      }

      template<typename... Ts>
      ttg::suspend_always await_transform(detail::to_device_t<Ts...>&& a) {
        auto arr = detail::extract_buffer_data(a, std::make_index_sequence<sizeof...(Ts)>{});
        bool need_transfer = !(TTG_IMPL_NS::register_device_memory(ttg::span(arr)));
        /* TODO: are we allowed to not suspend here and launch the kernel directly? */
        m_state = ttg::device::detail::TTG_DEVICE_CORO_WAIT_TRANSFER;
        return {};
      }

      ttg::suspend_always await_transform(detail::to_device_t<Input>&& a) {
        bool need_transfer = !(TTG_IMPL_NS::register_device_memory(a.input.span()));
        /* TODO: are we allowed to not suspend here and launch the kernel directly? */
        m_state = ttg::device::detail::TTG_DEVICE_CORO_WAIT_TRANSFER;
        return {};
      }

      template<typename... Ts>
      auto await_transform(detail::wait_kernel_t<Ts...>&& a) {
        //std::cout << "yield_value: wait_kernel_t" << std::endl;
        if constexpr (sizeof...(Ts) > 0) {
          TTG_IMPL_NS::mark_device_out(a.ties);
        }
        m_state = ttg::device::detail::TTG_DEVICE_CORO_WAIT_KERNEL;
        return a;
      }

      ttg::suspend_always await_transform(std::vector<device::detail::send_t>&& v) {
        m_sends = std::forward<std::vector<device::detail::send_t>>(v);
        m_state = ttg::device::detail::TTG_DEVICE_CORO_SENDOUT;
        return {};
      }

      ttg::suspend_always await_transform(device::detail::send_t&& v) {
        m_sends.clear();
        m_sends.push_back(std::forward<device::detail::send_t>(v));
        m_state = ttg::device::detail::TTG_DEVICE_CORO_SENDOUT;
        return {};
      }

      void return_void() {
        m_state = ttg::device::detail::TTG_DEVICE_CORO_COMPLETE;
      }

      bool complete() const {
        return m_state == ttg::device::detail::TTG_DEVICE_CORO_COMPLETE;
      }

      ttg::device::Task get_return_object() { return {detail::device_task_handle_type::from_promise(*this)}; }

      void unhandled_exception() {
        std::cerr << "Task coroutine caught an unhandled exception!" << std::endl;
        throw; // fwd
      }

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
      ttg_device_coro_state m_state = ttg::device::detail::TTG_DEVICE_CORO_STATE_NONE;

    };

  } // namespace detail

  bool Task::completed() { return base_type::promise().state() == ttg::device::detail::TTG_DEVICE_CORO_COMPLETE; }

  struct device_wait_kernel
  { };


  /* NOTE: below is preliminary for reductions on the device, which is not available yet */
#if 0
  /**************************
   * Device reduction coros *
   **************************/

  struct device_reducer_promise_type;

  using device_reducer_handle_type = ttg::coroutine_handle<device_reducer_promise_type>;

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
    ttg::suspend_never initial_suspend() {
      m_state = ttg::device::detail::TTG_DEVICE_CORO_INIT;
      return {};
    }

    /* suspend the coroutine at the end of the execution
    * so we can access the promise.
    * TODO: necessary? maybe we can save one suspend here
    */
    ttg::suspend_always final_suspend() noexcept {
      m_state = ttg::device::detail::TTG_DEVICE_CORO_COMPLETE;
      return {};
    }

    template<typename... Ts>
    ttg::suspend_always await_transform(detail::to_device_t<Ts...>&& a) {
      bool need_transfer = !(TTG_IMPL_NS::register_device_memory(a.ties));
      /* TODO: are we allowed to not suspend here and launch the kernel directly? */
      m_state = ttg::device::detail::TTG_DEVICE_CORO_WAIT_TRANSFER;
      return {};
    }

    void return_void() {
      m_state = ttg::device::detail::TTG_DEVICE_CORO_COMPLETE;
    }

    bool complete() const {
      return m_state == ttg::device::detail::TTG_DEVICE_CORO_COMPLETE;
    }

    device_reducer get_return_object() { return device_reducer{device_reducer_handle_type::from_promise(*this)}; }

    void unhandled_exception() { }

    auto state() {
      return m_state;
    }


  private:
    ttg::device::detail::ttg_device_coro_state m_state = ttg::device::detail::TTG_DEVICE_CORO_STATE_NONE;

  };

  bool device_reducer::completed() { return base_type::promise().state() == ttg::device::detail::TTG_DEVICE_CORO_COMPLETE; }
#endif // 0

}  // namespace ttg::device

#endif // TTG_HAVE_COROUTINE

#endif // TTG_DEVICE_TASK_H
