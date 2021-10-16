#ifndef TTG_UTIL_FUNC_H
#define TTG_UTIL_FUNC_H

#include <tuple>
#include <memory>

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
    template<ttg::Runtime Runtime>
    struct value_copy_handler {
      template<typename Value>
      inline constexpr
      Value&& operator()(Value&& value) const {
        return std::forward<Value>(value);
      }

      template<typename Value>
      inline constexpr
      const Value& operator()(const Value& value) const {
        return value;
      }

      template<typename Value>
      inline constexpr
      Value& operator()(Value& value) const {
        return value;
      }
    };

  } // namespace detail

  /// applies @c make_executable method to every op in the graph
  /// return true if there are no dangling out terminals
  template <typename... OpBasePtrs>
  std::enable_if_t<(std::is_convertible_v<std::remove_const_t<std::remove_reference_t<OpBasePtrs>>, OpBase *> && ...),
                   bool>
  make_graph_executable(OpBasePtrs &&... ops) {
    return ttg::make_traverse([](auto &&x) { std::forward<decltype(x)>(x)->make_executable(); })(
        std::forward<OpBasePtrs>(ops)...);
  }

  template <typename keyT, typename valueT>
  class Edge;  // Forward decl.

  /// Connect output terminal to successor input terminal
  template <typename out_terminalT, typename in_terminalT>
  void connect(out_terminalT *out, in_terminalT *in) {
    out->connect(in);
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via unique or otherwise wrapped pointers to Ops)
  template <std::size_t outindex, std::size_t inindex, typename producer_op_ptr, typename successor_op_ptr>
  void connect(producer_op_ptr &p, successor_op_ptr &s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via bare pointers to Ops)
  template <std::size_t outindex, std::size_t inindex, typename producer_op_ptr, typename successor_op_ptr>
  void connect(producer_op_ptr *p, successor_op_ptr *s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via OpBase pointers)
  inline void connect(size_t outindex, size_t inindex, OpBase *producer, OpBase *consumer) {
    connect(producer->out(outindex), consumer->in(inindex));
  }

  // Fuse edges into one ... all the types have to be the same ... just using
  // valuesT for variadic args
  template <typename keyT, typename... valuesT>
  auto fuse(const Edge<keyT, valuesT> &... args) {
    using valueT = typename std::tuple_element<0, std::tuple<valuesT...>>::type;  // grab first type
    return Edge<keyT, valueT>(args...);  // This will force all valuesT to be the same
  }

  // Make a tuple of Edges ... needs some type checking injected
  template <typename... inedgesT>
  auto edges(const inedgesT &... args) {
    return std::make_tuple(args...);
  }

  template <typename keyT, typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void send(const keyT &key, const valueT &value, output_terminalT &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.send(key, copy_handler(value));
  }

  template <typename keyT, typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void send(const keyT &key, valueT &&value, output_terminalT &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.send(key, copy_handler(std::forward<valueT>(value)));
  }

  template <typename keyT, typename output_terminalT>
  void sendk(const keyT &key, output_terminalT &t) {
    t.sendk(key);
  }

  // TODO if sendk is removed, rename to send
  template <typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void sendv(valueT &&value, output_terminalT &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.sendv(copy_handler(std::forward<valueT>(value)));
  }

  template <typename keyT, typename valueT, typename output_terminalT>
  void send(output_terminalT &t) {
    t.send();
  }

  template <size_t i, typename keyT, typename valueT, typename... output_terminalsT, ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<meta::is_none_void_v<keyT,std::decay_t<valueT>>,void>
      send(const keyT &key, valueT &&value, std::tuple<output_terminalsT...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).send(key, copy_handler(std::forward<valueT>(value)));
  }

  // TODO decide whether we need this ... (how common will be pure control flow?)
  template <size_t i, typename keyT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<keyT>,void>
  sendk(const keyT &key, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).sendk(key);
  }

  // TODO if sendk is removed, rename to send
  template <size_t i, typename valueT, typename... output_terminalsT, ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<!meta::is_void_v<valueT>,void>
  sendv(valueT &&value, std::tuple<output_terminalsT...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).sendv(copy_handler(std::forward<valueT>(value)));
  }

  template <size_t i, typename... output_terminalsT>
  void send(std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).send();
  }

  namespace detail {
    template <size_t KeyId, size_t i, size_t... I, typename ...RangesT, typename valueT, typename... output_terminalsT>
    void broadcast(const std::tuple<RangesT...>& keylists, valueT&& value, std::tuple<output_terminalsT...> &t) {
      if (std::get<KeyId>(keylists).size() > 0) {
        std::get<i>(t).broadcast(std::get<KeyId>(keylists), value);
      }
      if constexpr(sizeof...(I) > 0) {
        detail::broadcast<KeyId+1, I...>(keylists, value, t);
      }
    }
  } // namespace detail

  template <size_t i, typename rangeT, typename valueT, typename... output_terminalsT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcast(const rangeT &keylist, valueT &&value, std::tuple<output_terminalsT...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).broadcast(keylist, copy_handler(std::forward<valueT>(value)));
  }

  template <size_t i, size_t... I, typename ...RangesT, typename valueT, typename... output_terminalsT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcast(const std::tuple<RangesT...>& keylists, valueT &&value, std::tuple<output_terminalsT...> &t) {
    static_assert(sizeof...(I)+1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::value_copy_handler<Runtime> copy_handler;
    detail::broadcast<0, i, I...>(keylists, copy_handler(std::forward<valueT>(value)), t);
  }

  template <typename keyT, typename output_terminalT>
  void set_size(const keyT &key, const std::size_t size, output_terminalT &t) {
    t.set_size(key, size);
  }

  template <size_t i, typename keyT, typename... output_terminalsT>
  void set_size(const keyT &key, const std::size_t size, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).set_size(key, size);
  }

  template <typename keyT, typename output_terminalT>
  std::enable_if_t<!meta::is_void_v<keyT>,void>
  finalize(const keyT &key, output_terminalT &t) {
    t.finalize(key);
  }

  template <size_t i, typename keyT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<keyT>,void>
  finalize(const keyT &key, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).finalize(key);
  }

  template <typename output_terminalT>
  void finalize(output_terminalT &t) {
    t.finalize();
  }

  template <size_t i, typename... output_terminalsT>
  void finalize(std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).finalize();
  }

}  // namespace ttg

#endif // TTG_UTIL_FUNC_H
