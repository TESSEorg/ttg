#ifndef TTG_FUNC_H
#define TTG_FUNC_H

#include "ttg/fwd.h"

#include "ttg/edge.h"
#include "ttg/impl_selector.h"
#include "ttg/terminal.h"
#include "ttg/traverse.h"
#include "ttg/tt.h"

namespace ttg {

  namespace detail {
    /** Wrapper allowing implementations to provide copies of data the user
     * passed to send and broadcast. The value returned by operator() will
     * be passed to all terminals. By default, the given reference is (perfectly) forwarded.
     * Implementations may provide specializations using the ttg::Runtime tag.
     */
    template <ttg::Runtime Runtime>
    struct value_copy_handler {
      template <typename Value>
      inline constexpr decltype(auto) operator()(Value &&value) const {
        return std::forward<Value>(value);
      }
    };

    template <typename keyT, typename valueT>
    inline auto get_out_terminal(size_t i, const char *func) {
#ifndef NDEBUG
      auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->at(i);
      auto *terminal_ptr = dynamic_cast<Out<std::decay_t<keyT>, std::decay_t<valueT>> *>(base_terminal_ptr);
      if (terminal_ptr == nullptr) {
        std::stringstream ss;
        ss << func
           << ": invalid type of ith output terminal, most likely due to mismatch between its type "
              "and the type of key/value; make sure that the arguments to "
           << func
           << "() match the types encoded in the output "
              "terminals, or pass the output terminal tuple to the task function explicitly";
        throw std::runtime_error(ss.str());
      }
#else
      auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->operator[](i);
      auto *terminal_ptr = static_cast<Out<std::decay_t<keyT>, std::decay_t<valueT>> *>(base_terminal_ptr);
#endif
      return terminal_ptr;
    }

    template <typename keyT>
    inline auto get_out_base_terminal(size_t i, const char *func) {
#ifndef NDEBUG
      auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->at(i);
      auto *terminal_ptr = dynamic_cast<OutTerminalBase<std::decay_t<keyT>> *>(base_terminal_ptr);
      if (terminal_ptr == nullptr) {
        std::stringstream ss;
        ss << func
           << ": invalid type of ith output terminal, most likely due to mismatch between its type "
              "and the type of key; make sure that the arguments to "
           << func
           << "() match the types encoded in the output "
              "terminals, or pass the output terminal tuple to the task function explicitly";
        throw std::runtime_error(ss.str());
      }
#else
      auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->operator[](i);
      auto *terminal_ptr = static_cast<OutTerminalBase<std::decay_t<keyT>> *>(base_terminal_ptr);
#endif
      return terminal_ptr;
    }

  }  // namespace detail

  /// \brief Make the TTG \c tts executable.
  /// Applies \sa make_executable method to every op in the graph
  /// \param tts The task graph to make executable.
  /// \return true if there are no dangling out terminals
  template <typename... TTBasePtrs>
  inline std::enable_if_t<(std::is_convertible_v<decltype(*(std::declval<TTBasePtrs>())), TTBase &> && ...), bool>
  make_graph_executable(TTBasePtrs &&...tts) {
    return ttg::make_traverse([](auto &&x) { std::forward<decltype(x)>(x)->make_executable(); })(
        std::forward<TTBasePtrs>(tts)...);
  }

  /// \brief Connect output terminal to successor input terminal
  /// \param out The output terminal.
  /// \param in The input terminal.
  template <typename keyT, typename valueT>
  inline void connect(ttg::Out<keyT, valueT> *out, ttg::In<keyT, valueT> *in) {
    out->connect(in);
  }

  /// Connect output terminal to successor input terminal
  inline void connect(ttg::TerminalBase *out, ttg::TerminalBase *in) { out->connect(in); }

  /// \brief Connect producer output terminal outindex to consumer input terminal inindex (via unique or otherwise
  /// wrapped pointers to TTs) \tparam outindex The index of the output terminal on the producer. \tparam inindex  The
  /// index of the input terminal on the consumer. \param p The producer TT \param c The consumer TT
  template <std::size_t outindex, std::size_t inindex, typename producer_tt_ptr, typename successor_tt_ptr>
  inline void connect(producer_tt_ptr &p, successor_tt_ptr &s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// \brief Connect producer output terminal outindex to consumer input terminal inindex (via bare pointers to TTs)
  /// \tparam outindex The index of the output terminal on the producer.
  /// \tparam inindex  The index of the input terminal on the consumer.
  /// \param p The producer TT
  /// \param c The consumer TT
  template <std::size_t outindex, std::size_t inindex, typename producer_tt_ptr, typename successor_tt_ptr>
  inline void connect(producer_tt_ptr *p, successor_tt_ptr *s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// \brief Connect producer output terminal outindex to consumer input terminal inindex (via TTBase pointers)
  /// \tparam outindex The index of the output terminal on the producer.
  /// \tparam inindex  The index of the input terminal on the consumer.
  /// \param producer The producer TT
  /// \param consumer The consumer TT
  inline void connect(size_t outindex, size_t inindex, TTBase *producer, TTBase *consumer) {
    connect(producer->out(outindex), consumer->in(inindex));
  }

  /// \brief Fuse edges into one
  /// This allows receiving one data from either of the combined edges.
  /// \note All the types of the edges have to have the same prototype.
  /// \note The valuesT template argument is used only for variadic arguments.
  /// \param args The edges to combine one edge.
  /// \return One edge with the same type, combining the input edges.
  template <typename keyT, typename... valuesT>
  inline auto fuse(const Edge<keyT, valuesT> &...args) {
    using valueT = std::tuple_element_t<0, std::tuple<valuesT...>>;  // grab first type
    return Edge<keyT, valueT>(args...);                              // This will force all valuesT to be the same
  }

  /// \brief Make a tuple of Edges to pass to \sa ttg::make_tt.
  /// \param args: variable argument list of Edges
  /// \return A tuple of Edges.
  /// \note All Edges must have the same prototype.
  template <typename... inedgesT>
  inline auto edges(inedgesT &&...args) {
    return std::make_tuple(std::forward<inedgesT>(args)...);
  }

  // clang-format off
  /// \brief Sends a task id and a value to the given output terminal
  /// \param[in] key: the id of the task(s) receiving the value
  /// \param[in] value: the value to send to the receiving task(s)
  /// \param[in] out: the output terminal
  // clang-format on
  template <typename keyT, typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void send(const keyT &key, valueT &&value, ttg::Out<keyT, valueT> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.send(key, copy_handler(std::forward<valueT>(value)));
  }

  // clang-format off
  /// \brief Sends a task id (without an accompanying value) to the given output terminal
  /// \param[in] key: the id of the task(s) receiving the value
  /// \param[in] out: the output terminal
  // clang-format on
  template <typename keyT>
  inline void sendk(const keyT &key, ttg::Out<keyT, void> &t) {
    t.sendk(key);
  }

  // clang-format off
  /// \brief Sends a value (without an accompanying task id) to the given output terminal
  /// \param[in] value: the value to send to the receiving task(s)
  /// \param[in] out: the output terminal
  // clang-format on
  template <typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void sendv(valueT &&value, ttg::Out<void, valueT> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.sendv(copy_handler(std::forward<valueT>(value)));
  }

  // clang-format off
  /// \brief Sends a control message (message without an accompanying task id or a value) to the given output terminal
  /// \param[in] out: the output terminal
  // clang-format on
  inline void send(ttg::Out<void, void> &t) { t.send(); }

  // clang-format off
  /// \brief Sends a task id and a value to the template tasks attached to the output terminal selected in the explicitly given terminal tuple \p t
  /// \tparam <i> Identifies which output terminal in \p t to select for sending
  /// \param[in] key: the id of the task(s) receiving the value
  /// \param[in] value: the value to send to the receiving task(s)
  /// \param[in] out: a tuple of output terminals (typically, this is the output terminal of the template task where this is invoked)
  // clang-format on
  template <size_t i, typename keyT, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline std::enable_if_t<meta::is_none_void_v<keyT, std::decay_t<valueT>>, void> send(
      const keyT &key, valueT &&value, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).send(key, copy_handler(std::forward<valueT>(value)));
  }

  // clang-format off
  /// \brief Sends a task id and a value to the template tasks attached to the output terminal of this template task
  /// \param[in] i Identifies which output terminal of this template task to select for sending
  /// \param[in] key: the id of the task(s) receiving the value
  /// \param[in] value: the value to send to the receiving task(s)
  // clang-format on
  template <typename keyT, typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline std::enable_if_t<meta::is_none_void_v<keyT, std::decay_t<valueT>>, void> send(size_t i, const keyT &key,
                                                                                       valueT &&value) {
    detail::value_copy_handler<Runtime> copy_handler;
    auto *terminal_ptr = detail::get_out_terminal<keyT, valueT>(i, "ttg::send(i, key, value)");
    terminal_ptr->send(key, copy_handler(std::forward<valueT>(value)));
  }

  // clang-format off
  /// \brief Sends a task id and a value to the template tasks attached to the output terminal of this template task
  /// \note this is provided to support `send<i>` with and without explicitly-passed terminal tuple
  /// \tparam <i> Identifies which output terminal of this template task to select for sending
  /// \param[in] key: the id of the task(s) receiving the value
  /// \param[in] value: the value to send to the receiving task(s)
  // clang-format on
  template <size_t i, typename keyT, typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline std::enable_if_t<meta::is_none_void_v<keyT, std::decay_t<valueT>>, void> send(const keyT &key,
                                                                                       valueT &&value) {
    send(i, key, std::forward<valueT>(value));
  }

  // clang-format off
  /// \brief Sends a task id (without an accompanying value) to the template tasks attached to the output terminal selected in the explicitly given terminal tuple \p t
  /// \tparam <i> Identifies which output terminal in \p t to select for sending
  /// \param[in] key: the id of the task(s) receiving the value
  /// \param[in] out: a tuple of output terminals (typically, this is the output terminal of the template task where this is invoked)
  // clang-format on
  template <size_t i, typename keyT, typename... out_keysT, typename... out_valuesT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> sendk(const keyT &key,
                                                              std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).sendk(key);
  }

  // clang-format off
  /// \brief Sends a task id (without an accompanying value) to the template tasks attached to the output terminal of this template task
  /// \param[in] i Identifies which output terminal of this template task to select for sending
  /// \param[in] key: the id of the task(s) receiving the value
  // clang-format on
  template <typename keyT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> sendk(std::size_t i, const keyT &key) {
    auto *terminal_ptr = detail::get_out_terminal<keyT, void>(i, "ttg::sendk(i, key)");
    terminal_ptr->sendk(key);
  }

  // clang-format off
  /// \brief Sends a task id (without an accompanying value) to the template tasks attached to the output terminal of this template task
  /// \note this is provided to support `sendk<i>` with and without explicitly-passed terminal tuple
  /// \tparam <i> Identifies which output terminal of this template task to select for sending
  /// \param[in] key: the id of the task(s) receiving the value
  // clang-format on
  template <size_t i, typename keyT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> sendk(const keyT &key) {
    sendk(i, key);
  }

  // clang-format off
  /// \brief Sends a value (without an accompanying task id) to the template tasks attached to the output terminal selected in the explicitly given terminal tuple \p t
  /// \tparam <i> Identifies which output terminal in \p t to select for sending
  /// \param[in] value: the value to send to the receiving task(s)
  /// \param[in] out: a tuple of output terminals (typically, this is the output terminal of the template task where this is invoked)
  // clang-format on
  template <size_t i, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline std::enable_if_t<!meta::is_void_v<std::decay_t<valueT>>, void> sendv(
      valueT &&value, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).sendv(copy_handler(std::forward<valueT>(value)));
  }

  // clang-format off
  /// \brief Sends a value (without an accompanying task id) to the template tasks attached to the output terminal of this template task
  /// \param[in] i Identifies which output terminal of this template task to select for sending
  /// \param[in] value: the value to send to the receiving task(s)
  // clang-format on
  template <typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline std::enable_if_t<!meta::is_void_v<std::decay_t<valueT>>, void> sendv(std::size_t i, valueT &&value) {
    detail::value_copy_handler<Runtime> copy_handler;
    auto *terminal_ptr = detail::get_out_terminal<void, valueT>(i, "ttg::sendv(i, value)");
    terminal_ptr->sendv(copy_handler(std::forward<valueT>(value)));
  }

  // clang-format off
  /// \brief Sends a value (without an accompanying task id) to the template tasks attached to the output terminal of this template task
  /// \note this is provided to support `sendv<i>` with and without explicitly-passed terminal tuple
  /// \tparam <i> Identifies which output terminal of this template task to select for sending
  /// \param[in] value: the value to send to the receiving task(s)
  // clang-format on
  template <size_t i, typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline std::enable_if_t<!meta::is_void_v<std::decay_t<valueT>>, void> sendv(valueT &&value) {
    sendv(i, std::forward<valueT>(value));
  }

  // clang-format off
  /// \brief Sends a control message (message without an accompanying task id or a value) to the template tasks attached to the output terminal selected in the explicitly given terminal tuple \p t
  /// \tparam <i> Identifies which output terminal in \p t to select for sending
  /// \param[in] out: a tuple of output terminals (typically, this is the output terminal of the template task where this is invoked)
  // clang-format on
  template <size_t i, typename... out_keysT, typename... out_valuesT>
  inline void send(std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).send();
  }

  // clang-format off
  /// \brief Sends a control message (message without an accompanying task id or a value) to the template tasks attached to the output terminal of this template task
  /// \param[in] i Identifies which output terminal of this template task to select for sending
  // clang-format on
  inline void send(std::size_t i) {
    auto *terminal_ptr = detail::get_out_terminal<void, void>(i, "ttg::send(i)");
    terminal_ptr->send();
  }

  // clang-format off
  /// \brief Sends a control message (message without an accompanying task id or a value) to the template tasks attached to the output terminal of this template task
  /// \note this is provided to support `send<i>` with and without explicitly-passed terminal tuple
  /// \tparam <i> Identifies which output terminal of this template task to select for sending
  // clang-format on
  template <size_t i>
  inline void send() {
    send(i);
  }

  namespace detail {
    template <size_t KeyId, size_t i, size_t... I, typename... RangesT, typename valueT, typename... out_keysT,
              typename... out_valuesT>
    inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
                          std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      if constexpr (ttg::meta::is_iterable_v<std::tuple_element_t<KeyId, std::tuple<RangesT...>>>) {
        if (std::distance(std::begin(std::get<KeyId>(keylists)), std::end(std::get<KeyId>(keylists))) > 0) {
          std::get<i>(t).broadcast(std::get<KeyId>(keylists), value);
        }
      } else {
        std::get<i>(t).broadcast(std::get<KeyId>(keylists), value);
      }
      if constexpr (sizeof...(I) > 0) {
        detail::broadcast<KeyId + 1, I...>(keylists, value, t);
      }
    }

    template <size_t KeyId, size_t i, size_t... I, typename... RangesT, typename valueT>
    inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value) {
      if constexpr (ttg::meta::is_iterable_v<std::tuple_element_t<KeyId, std::tuple<RangesT...>>>) {
        if (std::distance(std::begin(std::get<KeyId>(keylists)), std::end(std::get<KeyId>(keylists))) > 0) {
          using key_t = decltype(*std::begin(std::get<KeyId>(keylists)));
          auto *terminal_ptr = detail::get_out_terminal<key_t, valueT>(i, "ttg::broadcast(keylists, value)");
          terminal_ptr->broadcast(std::get<KeyId>(keylists), value);
        }
      } else {
        using key_t = decltype(std::get<KeyId>(keylists));
        auto *terminal_ptr = detail::get_out_terminal<key_t, valueT>(i, "ttg::broadcast(keylists, value)");
        terminal_ptr->broadcast(std::get<KeyId>(keylists), value);
      }
      if constexpr (sizeof...(I) > 0) {
        detail::broadcast<KeyId + 1, I...>(keylists, value);
      }
    }

    template <size_t KeyId, size_t i, size_t... I, typename... RangesT, typename... out_keysT, typename... out_valuesT>
    inline void broadcast(const std::tuple<RangesT...> &keylists, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
      if constexpr (ttg::meta::is_iterable_v<std::tuple_element_t<KeyId, std::tuple<RangesT...>>>) {
        if (std::distance(std::begin(std::get<KeyId>(keylists)), std::end(std::get<KeyId>(keylists))) > 0) {
          std::get<i>(t).broadcast(std::get<KeyId>(keylists));
        }
      } else {
        std::get<i>(t).broadcast(std::get<KeyId>(keylists));
      }
      if constexpr (sizeof...(I) > 0) {
        detail::broadcast<KeyId + 1, I...>(keylists, t);
      }
    }

    template <size_t KeyId, size_t i, size_t... I, typename... RangesT>
    inline void broadcast(const std::tuple<RangesT...> &keylists) {
      if constexpr (ttg::meta::is_iterable_v<std::tuple_element_t<KeyId, std::tuple<RangesT...>>>) {
        if (std::distance(std::begin(std::get<KeyId>(keylists)), std::end(std::get<KeyId>(keylists))) > 0) {
          using key_t = decltype(*std::begin(std::get<KeyId>(keylists)));
          auto *terminal_ptr = detail::get_out_terminal<key_t, void>(i, "ttg::broadcast(keylists)");
          terminal_ptr->broadcast(std::get<KeyId>(keylists));
        }
      } else {
        using key_t = decltype(std::get<KeyId>(keylists));
        auto *terminal_ptr = detail::get_out_terminal<key_t, void>(i, "ttg::broadcast(keylists)");
        terminal_ptr->broadcast(std::get<KeyId>(keylists));
      }
      if constexpr (sizeof...(I) > 0) {
        detail::broadcast<KeyId + 1, I...>(keylists);
      }
    }
  }  // namespace detail

  template <size_t i, typename rangeT, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcast(const rangeT &keylist, valueT &&value, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).broadcast(keylist, copy_handler(std::forward<valueT>(value)));
  }

  template <typename rangeT, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcast(std::size_t i, const rangeT &keylist, valueT &&value) {
    detail::value_copy_handler<Runtime> copy_handler;
    using key_t = decltype(*std::begin(keylist));
    auto *terminal_ptr = detail::get_out_terminal<key_t, valueT>(i, "ttg::broadcast(keylist, value)");
    terminal_ptr->broadcast(keylist, copy_handler(std::forward<valueT>(value)));
  }

  template <size_t i, typename rangeT, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcast(const rangeT &keylist, valueT &&value) {
    broadcast(i, keylist, std::forward<valueT>(value));
  }

  template <size_t i, size_t... I, typename... RangesT, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
                        std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    static_assert(sizeof...(I) + 1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::value_copy_handler<Runtime> copy_handler;
    detail::broadcast<0, i, I...>(keylists, copy_handler(std::forward<valueT>(value)), t);
  }

  template <size_t i, size_t... I, typename... RangesT, typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value) {
    static_assert(sizeof...(I) + 1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::value_copy_handler<Runtime> copy_handler;
    detail::broadcast<0, i, I...>(keylists, copy_handler(std::forward<valueT>(value)));
  }

  template <size_t i, typename rangeT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcastk(const rangeT &keylist, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).broadcast(keylist);
  }

  template <typename rangeT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcastk(std::size_t i, const rangeT &keylist) {
    using key_t = decltype(*std::begin(keylist));
    auto *terminal_ptr = detail::get_out_terminal<key_t, void>(i, "ttg::broadcastk(keylist)");
    terminal_ptr->broadcast(keylist);
  }

  template <size_t i, typename rangeT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcastk(const rangeT &keylist) {
    broadcastk(i, keylist);
  }

  template <size_t i, size_t... I, typename... RangesT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcastk(const std::tuple<RangesT...> &keylists, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    static_assert(sizeof...(I) + 1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::broadcast<0, i, I...>(keylists, t);
  }

  template <size_t i, size_t... I, typename... RangesT, ttg::Runtime Runtime = ttg::ttg_runtime>
  inline void broadcastk(const std::tuple<RangesT...> &keylists) {
    static_assert(sizeof...(I) + 1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::broadcast<0, i, I...>(keylists);
  }

  template <typename keyT, typename out_valueT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(const keyT &key, const std::size_t size,
                                                                 ttg::Out<keyT, out_valueT> &t) {
    t.set_size(key, size);
  }

  /// \brief Set the size of all streaming input terminals connected to the output terminal for a task identified by \c
  /// key. \tparam <i> The index of the terminal through which to set the size. \param key The key identifying the task
  /// (or tasks) for which to set the streaming terminal size. \param size THe size to set (i.e., the number of elements
  /// to accumulate). \param t The tasks's output terminals.
  template <size_t i, typename keyT, typename... out_keysT, typename... out_valuesT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(const keyT &key, const std::size_t size,
                                                                 std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).set_size(key, size);
  }

  template <typename keyT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(std::size_t i, const keyT &key,
                                                                 const std::size_t size) {
    auto *terminal_ptr = detail::get_out_base_terminal<keyT>(i, "ttg::set_size(i, key, size)");
    terminal_ptr->set_size(size);
  }

  template <size_t i, typename keyT, typename... out_keysT, typename... out_valuesT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(const keyT &key, const std::size_t size) {
    set_size(i, key, size);
  }

  /// \brief Set the size of all streaming input terminals connected to an output terminal with \c void key.
  /// \param size THe size to set (i.e., the number of elements to accumulate).
  /// \param t The output terminal through which to set the size.
  template <typename out_keyT, typename out_valueT>
  inline void set_size(const std::size_t size, ttg::Out<out_keyT, out_valueT> &t) {
    t.set_size(size);
  }

  /// \brief Set the size of all streaming input terminals connected to an output terminal for a task, with \c void key.
  /// \tparam <i> The index of the terminal for which to set the size.
  /// \param size THe size to set (i.e., the number of elements to accumulate).
  /// \param t The task's output terminals.
  template <size_t i, typename... out_keysT, typename... out_valuesT>
  inline void set_size(const std::size_t size, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).set_size(size);
  }

  inline void set_size(std::size_t i, const std::size_t size) {
    auto *terminal_ptr = detail::get_out_base_terminal<void>(i, "ttg::set_size(i, size)");
    terminal_ptr->set_size(size);
  }

  template <std::size_t i>
  inline void set_size(const std::size_t size) {
    set_size<i>(size);
  }

  /// \brief Finalize streaming input terminals connecting to the given output terminal for tasks
  ///        identified by \c key.
  /// \param key The key identifying the tasks for which to finalize the streaming terminal.
  /// \param t The output terminal through which to finalize connected streaming terminals.
  template <typename keyT, typename out_keyT, typename out_valueT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(const keyT &key, ttg::Out<out_keyT, out_valueT> &t) {
    t.finalize(key);
  }

  /// \brief Finalize streaming input terminals connected to the given output terminal; use this to finalize terminals
  /// with non-`void` key. \tparam <i> The index of the output terminal through which to finalize connected streaming
  /// terminals. \param key The key identifying the tasks for which to finalize the streaming terminal. \param t The
  /// task's output terminals.
  template <size_t i, typename keyT, typename... out_keysT, typename... out_valuesT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(const keyT &key,
                                                                 std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).finalize(key);
  }

  template <typename keyT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(std::size_t i, const keyT &key) {
    auto *terminal_ptr = detail::get_out_base_terminal<keyT>(i, "ttg::finalize(i, key)");
    terminal_ptr->finalize(key);
  }

  template <std::size_t i, typename keyT>
  inline std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(const keyT &key) {
    finalize(i, key);
  }

  /// \brief Finalize streaming input terminals connected to the given output terminal; use this to finalize terminals
  /// with \c void key. \param t The output terminal through which to finalize connected streaming terminals.
  template <typename out_keyT, typename out_valueT>
  inline void finalize(ttg::Out<out_keyT, out_valueT> &t) {
    t.finalize();
  }

  /// \brief Finalize streaming input terminals connected to the `i`th output terminal in the tuple; use this to
  /// finalize terminals with \c void key. \tparam <i> The index of the output terminal through which to finalize
  /// connected streaming terminals. \param t The task's output terminals.
  template <size_t i, typename... out_keysT, typename... out_valuesT>
  inline void finalize(std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).finalize();
  }

  inline void finalize(std::size_t i) {
    auto *terminal_ptr = detail::get_out_base_terminal<void>(i, "ttg::finalize(i)");
    terminal_ptr->finalize();
  }

  template <std::size_t i>
  inline void finalize() {
    finalize<i>();
  }

}  // namespace ttg

#endif  // TTG_FUNC_H
