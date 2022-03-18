#ifndef TTG_FUNC_H
#define TTG_FUNC_H

#include <memory>
#include <tuple>

#include "ttg/edge.h"
#include "ttg/impl_selector.h"
#include "ttg/terminal.h"
#include "ttg/traverse.h"
#include "ttg/tt.h"

namespace ttg {

  namespace detail {

    /* Wrapper allowing implementations to provide copies of data the user
     * passed to send and broadcast. The value returned by operator() will
     * be passed to all terminals. By default, the user-provided copy is
     * returned.
     * Implementations may provide specializations using the ttg::Runtime tag.
     * TODO: can we avoid the three overloads?
     */
    template <ttg::Runtime Runtime>
    struct value_copy_handler {
      template <typename Value>
      inline constexpr Value &&operator()(Value &&value) const {
        return std::forward<Value>(value);
      }

      template <typename Value>
      inline constexpr const Value &operator()(const Value &value) const {
        return value;
      }

      template <typename Value>
      inline constexpr Value &operator()(Value &value) const {
        return value;
      }
    };

  }  // namespace detail

  /// applies @c make_executable method to every op in the graph
  /// return true if there are no dangling out terminals
  template <typename... TTBasePtrs>
  std::enable_if_t<(std::is_convertible_v<decltype(*(std::declval<TTBasePtrs>())), TTBase &> && ...), bool>
  make_graph_executable(TTBasePtrs &&...ops) {
    return ttg::make_traverse([](auto &&x) { std::forward<decltype(x)>(x)->make_executable(); })(
        std::forward<TTBasePtrs>(ops)...);
  }

  template <typename keyT, typename valueT>
  class Edge;  // Forward decl.

  /// Connect output terminal to successor input terminal
  template <typename keyT, typename valueT>
  void connect(ttg::Out<keyT, valueT> *out, ttg::In<keyT, valueT> *in) {
    out->connect(in);
  }

  /// Connect output terminal to successor input terminal
  void connect(ttg::TerminalBase *out, ttg::TerminalBase *in) {
    out->connect(in);
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via unique or otherwise wrapped
  /// pointers to Ops)
  template <std::size_t outindex, std::size_t inindex, typename producer_op_ptr, typename successor_op_ptr>
  void connect(producer_op_ptr &p, successor_op_ptr &s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via bare pointers to Ops)
  template <std::size_t outindex, std::size_t inindex, typename producer_op_ptr, typename successor_op_ptr>
  void connect(producer_op_ptr *p, successor_op_ptr *s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via TTBase pointers)
  inline void connect(size_t outindex, size_t inindex, TTBase *producer, TTBase *consumer) {
    connect(producer->out(outindex), consumer->in(inindex));
  }

  // Fuse edges into one ... all the types have to be the same ... just using
  // valuesT for variadic args
  template <typename keyT, typename... valuesT>
  auto fuse(const Edge<keyT, valuesT> &...args) {
    using valueT = std::tuple_element_t<0, std::tuple<valuesT...>>;  // grab first type
    return Edge<keyT, valueT>(args...);                              // This will force all valuesT to be the same
  }

  // Make a tuple of Edges ... needs some type checking injected
  template <typename... inedgesT>
  auto edges(inedgesT &&...args) {
    return std::make_tuple(std::forward<inedgesT>(args)...);
  }

  template <typename keyT, typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void send(const keyT &key, valueT &&value, ttg::Out<keyT, valueT> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.send(key, copy_handler(std::forward<valueT>(value)));
  }

  template <typename keyT>
  void sendk(const keyT &key, ttg::Out<keyT, void> &t) {
    t.sendk(key);
  }

  // TODO if sendk is removed, rename to send
  template <typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void sendv(valueT &&value, ttg::Out<void, valueT> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.sendv(copy_handler(std::forward<valueT>(value)));
  }

  void send(ttg::Out<void, void> &t) {
    t.send();
  }

  template <size_t i, typename keyT, typename valueT,
            typename... out_keysT,
            typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<meta::is_none_void_v<keyT, std::decay_t<valueT>>, void>
  send(const keyT &key, valueT &&value,
       std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).send(key, copy_handler(std::forward<valueT>(value)));
  }

  template <typename keyT, typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<meta::is_none_void_v<keyT, std::decay_t<valueT>>, void> send(size_t i, const keyT &key,
                                                                                valueT &&value) {
    detail::value_copy_handler<Runtime> copy_handler;
#ifndef NDEBUG
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->at(i);
    auto *terminal_ptr = dynamic_cast<Out<keyT, std::decay_t<valueT>> *>(base_terminal_ptr);
    if (terminal_ptr == nullptr) {
      throw std::runtime_error(
          "ttg::send(i, key, value): invalid type of ith output terminal, most likely due to mismatch between its type "
          "and the type of key/value; make sure that the arguments to send() match the types encoded in the output "
          "terminals, or pass the output terminal tuple to the task function explicitly");
    }
#else
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->operator[](i);
    auto *terminal_ptr = static_cast<Out<keyT, std::decay_t<valueT>> *>(base_terminal_ptr);
#endif
    terminal_ptr->send(key, copy_handler(std::forward<valueT>(value)));
  }

  template <size_t i, typename keyT,
            typename... out_keysT,
            typename... out_valuesT>
  std::enable_if_t<!meta::is_void_v<keyT>, void>
  sendk(const keyT &key, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).sendk(key);
  }

  template <typename keyT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> sendk(std::size_t i, const keyT &key) {
#ifndef NDEBUG
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->at(i);
    auto *terminal_ptr = dynamic_cast<Out<keyT, void> *>(base_terminal_ptr);
    if (terminal_ptr == nullptr) {
      throw std::runtime_error(
          "ttg::sendk(i, key): invalid type of ith output terminal, most likely due to mismatch between its type and "
          "the type of key; make sure that the arguments to sendk() match the types encoded in the output terminals, "
          "or pass the output terminal tuple to the task function explicitly");
    }
#else
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->operator[](i);
    auto *terminal_ptr = static_cast<Out<keyT, void> *>(base_terminal_ptr);
#endif
    terminal_ptr->sendk(key);
  }

  template <size_t i, typename valueT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<!meta::is_void_v<valueT>, void>
  sendv(valueT &&value, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).sendv(copy_handler(std::forward<valueT>(value)));
  }

  template <typename valueT, ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<!meta::is_void_v<valueT>, void> sendv(std::size_t i, valueT &&value) {
    detail::value_copy_handler<Runtime> copy_handler;
#ifndef NDEBUG
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->at(i);
    auto *terminal_ptr = dynamic_cast<Out<void, std::decay_t<valueT>> *>(base_terminal_ptr);
    if (terminal_ptr == nullptr) {
      throw std::runtime_error(
          "ttg::sendv(i, value): invalid type of ith output terminal, most likely due to mismatch between its type and "
          "the type of value; make sure that the arguments to sendv() match the types encoded in the output terminals, "
          "or pass the output terminal tuple to the task function explicitly");
    }
#else
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->operator[](i);
    auto *terminal_ptr = static_cast<Out<void, std::decay_t<valueT>> *>(base_terminal_ptr);
#endif
    terminal_ptr->sendv(copy_handler(std::forward<valueT>(value)));
  }

  template <size_t i, typename... out_keysT, typename... out_valuesT>
  void send(std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).send();
  }

  inline void send(std::size_t i) {
#ifndef NDEBUG
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->at(i);
    auto *terminal_ptr = dynamic_cast<Out<void, void> *>(base_terminal_ptr);
    if (terminal_ptr == nullptr) {
      throw std::runtime_error(
          "ttg::send(i): invalid type of ith output terminal due to internal TTG runtime error; try passing the output "
          "terminal tuple to the task function explicitly");
    }
#else
    auto *base_terminal_ptr = TTBase::get_outputs_tls_ptr()->operator[](i);
    auto *terminal_ptr = static_cast<Out<void, void> *>(base_terminal_ptr);
#endif
    terminal_ptr->send();
  }

  namespace detail {
    template <size_t KeyId, size_t i, size_t... I, typename... RangesT, typename valueT,
              typename... out_keysT, typename... out_valuesT>
    void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
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

    template <size_t KeyId, size_t i, size_t... I, typename... RangesT,
              typename... out_keysT, typename... out_valuesT>
    void broadcast(const std::tuple<RangesT...> &keylists,
                   std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
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
  }  // namespace detail

  template <size_t i, typename rangeT, typename valueT,
            typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcast(const rangeT &keylist, valueT &&value,
                 std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).broadcast(keylist, copy_handler(std::forward<valueT>(value)));
  }

  template <size_t i, size_t... I, typename... RangesT, typename valueT,
            typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value,
                 std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    static_assert(sizeof...(I) + 1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::value_copy_handler<Runtime> copy_handler;
    detail::broadcast<0, i, I...>(keylists, copy_handler(std::forward<valueT>(value)), t);
  }

  template <size_t i, typename rangeT, typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcastk(const rangeT &keylist,
                  std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).broadcast(keylist);
  }

  template <size_t i, size_t... I, typename... RangesT,
            typename... out_keysT, typename... out_valuesT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcastk(const std::tuple<RangesT...> &keylists,
                  std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    static_assert(sizeof...(I) + 1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::broadcast<0, i, I...>(keylists, t);
  }

  template <typename keyT, typename out_valueT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(
      const keyT &key, const std::size_t size, ttg::Out<keyT, out_valueT> &t) {
    t.set_size(key, size);
  }

  template <size_t i, typename keyT,
            typename... out_keysT, typename... out_valuesT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(const keyT &key, const std::size_t size,
                                                          std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).set_size(key, size);
  }

  template <typename out_keyT, typename out_valueT>
  void set_size(const std::size_t size, ttg::Out<out_keyT, out_valueT> &t) {
    t.set_size(size);
  }

  template <size_t i, typename... out_keysT, typename... out_valuesT>
  void set_size(const std::size_t size, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).set_size(size);
  }

  template <typename keyT, typename out_keyT, typename out_valueT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(
      const keyT &key, ttg::Out<out_keyT, out_valueT> &t) {
    t.finalize(key);
  }

  template <size_t i, typename keyT, typename... out_keysT, typename... out_valuesT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(
      const keyT &key, std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).finalize(key);
  }

  template <typename out_keyT, typename out_valueT>
  void finalize(ttg::Out<out_keyT, out_valueT> &t) {
    t.finalize();
  }

  template <size_t i, typename... out_keysT, typename... out_valuesT>
  void finalize(std::tuple<ttg::Out<out_keysT, out_valuesT>...> &t) {
    std::get<i>(t).finalize();
  }

}  // namespace ttg

#endif  // TTG_FUNC_H
