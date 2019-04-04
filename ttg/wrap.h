// to be #include'd within runtime::ttg namespace

#ifndef CXXAPI_WRAP_H
#define CXXAPI_WRAP_H

// Class to wrap a callable with signature
//
// case 1 (keyT != void): void op(auto&& key, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
//
template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
class WrapOp
    : public Op<keyT, output_terminalsT, WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>, input_valuesT...> {
  using baseT = Op<keyT, output_terminalsT, WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>, input_valuesT...>;

  using input_values_tuple_type = typename baseT::input_values_tuple_type;
  using input_edges_type = typename baseT::input_edges_type;
  using output_edges_type = typename baseT::output_edges_type;

  std::function<boost::callable_traits::function_type_t<funcT>> func;

  template <typename Key, typename Tuple, std::size_t... S>
  void call_func(Key &&key, Tuple &&args, output_terminalsT &out, std::index_sequence<S...>) {
    // this is the tuple of values
    using func_args_t = std::remove_reference_t<std::tuple_element_t<1, boost::callable_traits::args_t<funcT>>>;
    // NB cannot use std::forward_as_tuple since that makes a tuple of refs!
    func_args_t unwrapped_args(
        baseT::template get<S, std::tuple_element_t<S, func_args_t>>(std::forward<Tuple>(args))...);
    func(std::forward<Key>(key), std::move(unwrapped_args), out);
  }

  template <typename Tuple, std::size_t... S>
  void call_func(Tuple &&args, output_terminalsT &out, std::index_sequence<S...>) {
    // this is the tuple of values
    using func_args_t = std::remove_reference_t<std::tuple_element_t<0, boost::callable_traits::args_t<funcT>>>;
    // NB cannot use std::forward_as_tuple since that makes a tuple of refs!
    func_args_t unwrapped_args(
        baseT::template get<S, std::tuple_element_t<S, func_args_t>>(std::forward<Tuple>(args))...);
    func(std::move(unwrapped_args), out);
  }

  template <typename Key>
  void call_func(Key &&key, output_terminalsT &out) {
    using func_args_t = std::remove_reference_t<std::tuple_element_t<1, boost::callable_traits::args_t<funcT>>>;
    func(std::forward<Key>(key), func_args_t{}, out);
  }

  void call_func(output_terminalsT &out) {
    using func_args_t = std::remove_reference_t<std::tuple_element_t<0, boost::callable_traits::args_t<funcT>>>;
    func(func_args_t{}, out);
  }

 public:
  template <typename funcT_>
  WrapOp(funcT_ &&f, const input_edges_type &inedges, const output_edges_type &outedges,
         const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template<typename funcT_>
  WrapOp(funcT_ &&f,
         const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames)
      : baseT(name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template<typename Key, typename ArgsTuple>
  std::enable_if_t<std::is_same_v<ArgsTuple,input_values_tuple_type> &&
      !::ttg::meta::is_empty_tuple_v<ArgsTuple> &&
      !::ttg::meta::is_void_v<Key>,void>
  op(Key&& key, ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<Key>(key), std::forward<ArgsTuple>(args_tuple), out,
              std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{});
  }

  template<typename ArgsTuple, typename Key = keyT>
  std::enable_if_t<std::is_same_v<ArgsTuple,input_values_tuple_type> &&
      !::ttg::meta::is_empty_tuple_v<ArgsTuple> &&
      ::ttg::meta::is_void_v<Key>,void>
  op(ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<ArgsTuple>(args_tuple), out,
              std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{});
  }

  template<typename Key, typename ArgsTuple = input_values_tuple_type>
  std::enable_if_t<::ttg::meta::is_empty_tuple_v<ArgsTuple> &&
      !::ttg::meta::is_void_v<Key>,void>
  op(Key&& key, output_terminalsT &out) {
    call_func(std::forward<Key>(key), out);
  }

  template<typename Key = keyT, typename ArgsTuple = input_values_tuple_type>
  std::enable_if_t<::ttg::meta::is_empty_tuple_v<ArgsTuple> &&
      ::ttg::meta::is_void_v<Key>,void>
  op(output_terminalsT &out) {
    call_func(out);
  }
};

template <typename funcT, typename keyT, typename output_terminalsT, typename input_values_tupleT>
struct WrapOpUnwrapTuple;

template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
struct WrapOpUnwrapTuple<funcT, keyT, output_terminalsT, std::tuple<input_valuesT...>> {
  using type = WrapOp<funcT, keyT, output_terminalsT, std::remove_reference_t<input_valuesT>...>;
};

// Class to wrap a callable with signature
//
// case 1 (keyT != void): void op(auto&& key, input_valuesT&&..., std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(input_valuesT&&..., std::tuple<output_terminalsT...>&)
//
template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
class WrapOpArgs : public Op<keyT, output_terminalsT, WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>,
                             input_valuesT...> {
  using baseT =
      Op<keyT, output_terminalsT, WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>, input_valuesT...>;

  using input_values_tuple_type = typename baseT::input_values_tuple_type;
  using input_edges_type = typename baseT::input_edges_type;
  using output_edges_type = typename baseT::output_edges_type;
  using input_unwrapped_values_tuple_type = typename baseT::input_unwrapped_values_tuple_type;

  std::function<boost::callable_traits::function_type_t<funcT>> func;

  template <typename Key, typename Tuple, std::size_t... S>
  void call_func(Key &&key, Tuple &&args_tuple, output_terminalsT &out, std::index_sequence<S...>) {
    using func_args_t = boost::callable_traits::args_t<funcT>;
    func(std::forward<Key>(key),
         baseT::template get<S, std::tuple_element_t<S + 1, func_args_t>>(std::forward<Tuple>(args_tuple))..., out);
  }

  template <typename Tuple, std::size_t... S>
  void call_func(Tuple &&args_tuple, output_terminalsT &out, std::index_sequence<S...>) {
    using func_args_t = boost::callable_traits::args_t<funcT>;
    func(baseT::template get<S, std::tuple_element_t<S, func_args_t>>(std::forward<Tuple>(args_tuple))..., out);
  }

  template <typename Key>
  void call_func(Key &&key, output_terminalsT &out) {
    func(std::forward<Key>(key), out);
  }

  template <typename OutputTerminals>
  void call_func(OutputTerminals & out) {
    func(out);
  }

 public:

  template <typename funcT_>
  WrapOpArgs(funcT_ &&f, const input_edges_type &inedges,
             const typename baseT::output_edges_type &outedges, const std::string &name,
             const std::vector<std::string> &innames, const std::vector<std::string> &outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template<typename funcT_>
  WrapOpArgs(funcT_ &&f, const std::string &name,
             const std::vector<std::string> &innames, const std::vector<std::string> &outnames)
      : baseT(name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template<typename Key, typename ArgsTuple>
  std::enable_if_t<std::is_same_v<ArgsTuple,input_values_tuple_type> &&
      !::ttg::meta::is_empty_tuple_v<input_unwrapped_values_tuple_type> &&
      !::ttg::meta::is_void_v<Key>,void>
      op(Key &&key, ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<Key>(key), std::forward<ArgsTuple>(args_tuple), out,
              std::make_index_sequence<std::tuple_size<ArgsTuple>::value>{});
  };

  template<typename ArgsTuple, typename Key = keyT>
  std::enable_if_t<std::is_same_v<ArgsTuple,input_values_tuple_type> &&
      !::ttg::meta::is_empty_tuple_v<input_unwrapped_values_tuple_type> &&
      ::ttg::meta::is_void_v<Key>,void>
  op(ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<ArgsTuple>(args_tuple), out,
              std::make_index_sequence<std::tuple_size<ArgsTuple>::value>{});
  };

  template<typename Key, typename ArgsTuple = input_unwrapped_values_tuple_type>
  std::enable_if_t<::ttg::meta::is_empty_tuple_v<ArgsTuple> &&
      !::ttg::meta::is_void_v<Key>,void>
  op(Key &&key, output_terminalsT &out) {
    call_func(std::forward<Key>(key), out);
  };

  template<typename Key = keyT, typename ArgsTuple = input_unwrapped_values_tuple_type>
  std::enable_if_t<::ttg::meta::is_empty_tuple_v<ArgsTuple> &&
      ::ttg::meta::is_void_v<Key>,void>
  op(output_terminalsT &out) {
    call_func(out);
  };

};

template <typename funcT, typename keyT, typename output_terminalsT, typename input_values_tupleT>
struct WrapOpArgsUnwrapTuple;

template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
struct WrapOpArgsUnwrapTuple<funcT, keyT, output_terminalsT, std::tuple<input_valuesT...>> {
  using type = WrapOpArgs<funcT, keyT, output_terminalsT, std::remove_reference_t<input_valuesT>...>;
};

// Factory function to assist in wrapping a callable with signature
//
// case 1 (keyT != void): void op(const input_keyT&, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
template <typename keyT, typename funcT, typename... input_valuesT, typename... output_edgesT>
auto wrapt(funcT &&func, const std::tuple<::ttg::Edge<keyT, input_valuesT>...> &inedges,
           const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
           const std::vector<std::string> &innames = std::vector<std::string>(
               std::tuple_size<std::tuple<::ttg::Edge<keyT, input_valuesT>...>>::value, "input"),
           const std::vector<std::string> &outnames =
               std::vector<std::string>(std::tuple_size<std::tuple<output_edgesT...>>::value, "output")) {
  using output_terminals_type = typename ::ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;

  // Op needs actual types of arguments to func ... extract them and pass to WrapOpArgs
  // 1. func_args_t = {const input_keyT&, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&}
  using func_args_t = boost::callable_traits::args_t<funcT>;
  constexpr const auto num_args = std::tuple_size<func_args_t>::value;
  constexpr const auto void_key = ::ttg::meta::is_void_v<keyT>;
  static_assert(num_args == (void_key ? 2 : 3), "ttg::wrapt(func, ...): func must take 3 arguments (or 2, if keyT=void)");
  // 2. input_args_t = {input_valuesT&&...}
  using input_args_t = std::decay_t<typename std::tuple_element<void_key ? 0 : 1, func_args_t>::type>;
  using decayed_input_args_t = ::ttg::meta::decayed_tuple_t<input_args_t>;
  using wrapT = typename WrapOpUnwrapTuple<funcT, keyT, output_terminals_type, input_args_t>::type;
  // not sure if we need this level of type checking ...
  // TODO determine the generic signature of func
  if constexpr (!void_key) {
    static_assert(std::is_same_v<typename std::tuple_element<0, func_args_t>::type, const keyT &>,
                  "ttg::wrapt(func, inedges, outedges): first argument of func must be const keyT& (unless keyT = void)");
  }
  static_assert(std::is_same_v<decayed_input_args_t, std::tuple<input_valuesT...>>,
                "ttg::wrapt(func, inedges, outedges): inedges value types do not match argument types of func");
  static_assert(
      std::is_same_v<typename std::tuple_element<num_args - 1, func_args_t>::type, output_terminals_type &>,
      "ttg::wrapt(func, inedges, outedges): last argument of func must be std::tuple<output_terminals_type>&");

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

// Factory function to assist in wrapping a callable with signature
//
// case 1 (keyT != void): void op(const input_keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(input_valuesT&&..., std::tuple<output_terminalsT...>&)
template <typename keyT, typename funcT, typename... input_edge_valuesT, typename... output_edgesT>
auto wrap(funcT &&func, const std::tuple<::ttg::Edge<keyT, input_edge_valuesT>...> &inedges,
          const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
          const std::vector<std::string> &innames = std::vector<std::string>(
              std::tuple_size<std::tuple<::ttg::Edge<keyT, input_edge_valuesT>...>>::value, "input"),
          const std::vector<std::string> &outnames =
              std::vector<std::string>(std::tuple_size<std::tuple<output_edgesT...>>::value, "output")) {
  using output_terminals_type = typename ::ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;

  // Op needs actual types of arguments to func ... extract them and pass to WrapOpArgs
  // 1. func_args_t = {const input_keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&}
  using func_args_t = boost::callable_traits::args_t<funcT>;
  constexpr const auto num_args = std::tuple_size<func_args_t>::value;
  constexpr const auto void_key = ::ttg::meta::is_void_v<keyT>;
  static_assert(num_args == sizeof...(input_edge_valuesT) + (void_key ? 1 : 2),
                "ttg::wrap(func, inedges): func's # of args != # of inedges");
  // 2. input_args_t = {input_valuesT&&...}
  using input_args_t =
      typename ::ttg::meta::take_first_n<typename ::ttg::meta::drop_first_n<func_args_t, std::size_t(void_key ? 0 : 1)>::type,
                                         std::tuple_size<func_args_t>::value - (void_key ? 1 : 2)>::type;
  using decayed_input_args_t = ::ttg::meta::decayed_tuple_t<input_args_t>;
  using wrapT = typename WrapOpArgsUnwrapTuple<funcT, keyT, output_terminals_type, input_args_t>::type;
  // not sure if we need this level of type checking ...
  // TODO determine the generic signature of func
  if constexpr (!void_key) {
    static_assert(std::is_same_v<typename std::tuple_element<0, func_args_t>::type, const keyT &>,
                  "ttg::wrap(func, inedges, outedges): first argument of func must be const keyT& (unless keyT = void)");
  }
  static_assert(std::is_same_v<decayed_input_args_t, std::tuple<input_edge_valuesT...>>,
                "ttg::wrap(func, inedges, outedges): inedges value types do not match argument types of func");
  static_assert(
      std::is_same_v<typename std::tuple_element<num_args - 1, func_args_t>::type, output_terminals_type &>,
      "ttg::wrap(func, inedges, outedges): last argument of func must be std::tuple<output_terminals_type>&");

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

// pattern from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
#define TTG_UTIL_ALIAS_TEMPLATE_FUNCTION(aliasname,funcname)\
template<typename... Args> \
inline auto aliasname(Args&&... args) \
{ \
    return funcname(std::forward<Args>(args)...); \
}

TTG_UTIL_ALIAS_TEMPLATE_FUNCTION(make_op, wrap);

#endif  // CXXAPI_WRAP_H
