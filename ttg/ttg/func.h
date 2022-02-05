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

  /// \brief Make the TTG \c tts executable.
  /// Applies \sa make_executable method to every op in the graph
  /// \param tts The task graph to make executable.
  /// \return true if there are no dangling out terminals
  template <typename... TTBasePtrs>
  std::enable_if_t<(std::is_convertible_v<decltype(*(std::declval<TTBasePtrs>())), TTBase &> && ...), bool>
  make_graph_executable(TTBasePtrs &&... tts) {
    return ttg::make_traverse([](auto &&x) { std::forward<decltype(x)>(x)->make_executable(); })(
        std::forward<TTBasePtrs>(tts)...);
  }

  template <typename keyT, typename valueT>
  class Edge;  // Forward decl.

  /// \brief Connect output terminal to successor input terminal
  /// \param out The output terminal.
  /// \param in The input terminal.
  template <typename out_terminalT, typename in_terminalT>
  void connect(out_terminalT *out, in_terminalT *in) {
    out->connect(in);
  }

  /// \brief Connect producer output terminal outindex to consumer input terminal inindex (via unique or otherwise wrapped
  /// pointers to TTs)
  /// \tparam outindex The index of the output terminal on the producer.
  /// \tparam inindex  The index of the input terminal on the consumer.
  /// \param p The producer TT
  /// \param c The consumer TT
  template <std::size_t outindex, std::size_t inindex, typename producer_tt_ptr, typename successor_tt_ptr>
  void connect(producer_tt_ptr &p, successor_tt_ptr &c) {
    connect(p->template out<outindex>(), c->template in<inindex>());
  }

  /// \brief Connect producer output terminal outindex to consumer input terminal inindex (via bare pointers to TTs)
  /// \tparam outindex The index of the output terminal on the producer.
  /// \tparam inindex  The index of the input terminal on the consumer.
  /// \param p The producer TT
  /// \param c The consumer TT
  template <std::size_t outindex, std::size_t inindex, typename producer_op_ptr, typename successor_op_ptr>
  void connect(producer_op_ptr *p, successor_op_ptr *c) {
    connect(p->template out<outindex>(), c->template in<inindex>());
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
  auto fuse(const Edge<keyT, valuesT> &... args) {
    using valueT = typename std::tuple_element<0, std::tuple<valuesT...>>::type;  // grab first type
    return Edge<keyT, valueT>(args...);  // This will force all valuesT to be the same
  }

  /// \brief Make a tuple of Edges to pass to \sa ttg::make_tt.
  /// \param args: variable argumetn list of Edges
  /// \return A tuple of Edges.
  /// \note All Edges must have the same prototype.
  template <typename... inedgesT>
  auto edges(inedgesT &&... args) {
    return std::make_tuple(std::forward<inedgesT>(args)...);
  }

  /// \brief Output a task identifier and value on a given output terminal
  /// Overload for const data.
  /// \tparam <i> The output terminal index on which key and value are output
  /// \param[in] key: the key that identifies the task receiving the value
  /// \param[in] value: the value to propagate through this terminal
  /// \param[in] out: the tuple of output  terminals
  template <typename keyT, typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void send(const keyT &key, const valueT &value, output_terminalT &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.send(key, copy_handler(value));
  }

  /// \brief Output a task identifier and value on a given output terminal
  /// Overload for move data.
  /// \tparam <i> The output terminal index on which key and value are output
  /// \param[in] key: the key that identifies the task receiving the value
  /// \param[in] value: the value to propagate through this terminal
  /// \param[in] out: the tuple of output  terminals
  template <typename keyT, typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void send(const keyT &key, valueT &&value, output_terminalT &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.send(key, copy_handler(std::forward<valueT>(value)));
  }

  /// \brief Send void data to a terminal.
  /// This overload is used for TTs with \c void identifier and \c void values.
  /// \note Short-cut for \c t.send()
  /// \param[in] t: the tuple of output terminal
  template <typename output_terminalT>
  void send(output_terminalT &t) {
    t.send();
  }

  /// \brief Send void data to a TT on terminal \c i.
  /// This overload is used for TTs with \c void identifier and \c void values.
  /// \tparam i The index of the terminal to send to.
  /// \param[in] t: the tuple of output terminal
  template <size_t i, typename... output_terminalsT>
  void send(std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).send();
  }

  /// \brief Send data to a TT on output terminal \c i.
  /// \tparam i The index of the terminal to send to.
  /// \param key The key identifiying the receiving task or tasks.
  /// \param valye The value to send.
  /// \param[in] t: the tuple of output terminal
  template <size_t i, typename keyT, typename valueT, typename... output_terminalsT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<meta::is_none_void_v<keyT, std::decay_t<valueT>>, void> send(const keyT &key, valueT &&value,
                                                                                std::tuple<output_terminalsT...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).send(key, copy_handler(std::forward<valueT>(value)));
  }


  /// \brief Send on a control-flow only output terminal.
  /// @param[in] key: the task identifier of the destination task
  /// @param[in] t: the output terminal
  template <typename keyT, typename output_terminalT>
  void sendk(const keyT &key, output_terminalT &t) {
    t.sendk(key);
  }

  /// Trigger a control flow-only output terminal
  ///  @tparam <i>: index of the output terminal to trigger
  ///  @param[in]   key: the task identifier of the destination task
  ///  @param[in,out] t: the output terminals of the op
  // TODO decide whether we need this ... (how common will be pure control flow?)
  template <size_t i, typename keyT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> sendk(const keyT &key, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).sendk(key);
  }

  /// \brief Output a value on a given terminal that does not take a task identifier.
  /// @param[in] value: rvalue reference to the object that must be output on the output terminal
  /// @param[in] t: the output terminals of the op
  // TODO if sendk is removed, rename to send
  template <typename valueT, typename output_terminalT, ttg::Runtime Runtime = ttg::ttg_runtime>
  void sendv(valueT &&value, output_terminalT &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    t.sendv(copy_handler(std::forward<valueT>(value)));
  }

  /// \brief Output a value on an output terminal that does not take a task identifier.
  /// @tparam <i>: index of the output terminal to trigger
  /// @param[in] value: rvalue reference to the object to send on the terminal
  /// @param[in,out] t: the output terminals of the op
  // TODO if sendk is removed, rename to send
  template <size_t i, typename valueT, typename... output_terminalsT, ttg::Runtime Runtime = ttg::ttg_runtime>
  std::enable_if_t<!meta::is_void_v<valueT>, void> sendv(valueT &&value, std::tuple<output_terminalsT...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).sendv(copy_handler(std::forward<valueT>(value)));
  }

  namespace detail {
    template <size_t KeyId, size_t i, size_t... I, typename... RangesT, typename valueT, typename... output_terminalsT>
    void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value, std::tuple<output_terminalsT...> &t) {
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
  }  // namespace detail

  /// \brief Broadcast a value to multiple tasks on an output terminal.
  /// \tparam <i>: index of the output terminal to broadcast on.
  /// \param[in] keylist: Iterable of keys identifiying receiving tasks.
  /// \param[in] value: Value to send on the terminal
  /// \param[in] t: the output terminals of the task
  template <size_t i, typename rangeT, typename valueT, typename... output_terminalsT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcast(const rangeT &keylist, valueT &&value, std::tuple<output_terminalsT...> &t) {
    detail::value_copy_handler<Runtime> copy_handler;
    std::get<i>(t).broadcast(keylist, copy_handler(std::forward<valueT>(value)));
  }

  /// \brief Broadcast a value to multiple task identifer on an output terminal.
  /// \tparam <i>: First index of the output terminals to broadcast on.
  /// \tparam <I>: Remaining indexes of the output terminals to broadcast on.
  /// \param[in] keylist: Tuple of iterables of keys identifiying receiving tasks.
  /// \param[in] value: Value to send on the terminals
  /// \param[in] t: the output terminals of the task
  template <size_t i, size_t... I, typename... RangesT, typename valueT, typename... output_terminalsT,
            ttg::Runtime Runtime = ttg::ttg_runtime>
  void broadcast(const std::tuple<RangesT...> &keylists, valueT &&value, std::tuple<output_terminalsT...> &t) {
    static_assert(sizeof...(I) + 1 == sizeof...(RangesT),
                  "Number of selected output terminals must match the number of keylists!");
    detail::value_copy_handler<Runtime> copy_handler;
    detail::broadcast<0, i, I...>(keylists, copy_handler(std::forward<valueT>(value)), t);
  }

  /// \brief Set the size of all streaming input terminals connected to the output terminal for a task identified by \c key.
  /// \param key The key identifying the task (or tasks) for which to set the streaming terminal size.
  /// \param size THe size to set (i.e., the number of elements to accumulate).
  /// \param t The output terminal through which to set the size.
  template <typename keyT, typename output_terminalT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(const keyT &key, const std::size_t size,
                                                          output_terminalT &t) {
    t.set_size(key, size);
  }

  /// \brief Set the size of all streaming input terminals connected to the output terminal for a task identified by \c key.
  /// \tparam <i> The index of the terminal through which to set the size.
  /// \param key The key identifying the task (or tasks) for which to set the streaming terminal size.
  /// \param size THe size to set (i.e., the number of elements to accumulate).
  /// \param t The tasks's output terminals.
  template <size_t i, typename keyT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> set_size(const keyT &key, const std::size_t size,
                                                          std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).set_size(key, size);
  }

  /// \brief Set the size of all streaming input terminals connected to an output terminal with \c void key.
  /// \param size THe size to set (i.e., the number of elements to accumulate).
  /// \param t The output terminal through which to set the size.
  template <typename output_terminalT>
  void set_size(const std::size_t size, output_terminalT &t) {
    t.set_size(size);
  }

  /// \brief Set the size of all streaming input terminals connected to an output terminal for a task, with \c void key.
  /// \tparam <i> The index of the terminal for which to set the size.
  /// \param size THe size to set (i.e., the number of elements to accumulate).
  /// \param t The task's output terminals.
  template <size_t i, typename... output_terminalsT>
  void set_size(const std::size_t size, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).set_size(size);
  }

  /// \brief Finalize streaming input terminals connecting to the given output terminal for tasks
  ///        identified by \c key.
  /// \param key The key identifying the tasks for which to finalize the streaming terminal.
  /// \param t The output terminal through which to finalize connected streaming terminals.
  template <typename keyT, typename output_terminalT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(const keyT &key, output_terminalT &t) {
    t.finalize(key);
  }

  /// \brief Finalize streaming input terminals connecting to the given output terminal for tasks
  ///        identified by \c key.
  /// \tparam <i> The index of the output terminal through which to finalize connected streaming terminals.
  /// \param key The key identifying the tasks for which to finalize the streaming terminal.
  /// \param t The task's output terminals.
  template <size_t i, typename keyT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<keyT>, void> finalize(const keyT &key, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).finalize(key);
  }

  /// \brief Finalize streaming input terminals connecting to the given output terminal for tasks, with \c void key.
  /// \param t The output terminal through which to finalize connected streaming terminals.
  template <typename output_terminalT>
  void finalize(output_terminalT &t) {
    t.finalize();
  }

  /// \brief Finalize streaming input terminals connecting to the given output terminal for tasks, with \c void key.
  /// \tparam <i> The index of the output terminal through which to finalize connected streaming terminals.
  /// \param t The task's output terminals.
  template <size_t i, typename... output_terminalsT>
  void finalize(std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).finalize();
  }

}  // namespace ttg

#endif  // TTG_FUNC_H
