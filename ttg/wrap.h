// to be #include'd within runtime::ttg namespace

#ifndef CXXAPI_WRAP_H
#define CXXAPI_WRAP_H

// Class to wrap a callable with signature
//
// void op(const input_keyT&, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
//
template<typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
class WrapOp : public Op<keyT, output_terminalsT, WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>,
                         input_valuesT...> {
  using baseT =
  Op<keyT, output_terminalsT, WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>, input_valuesT...>;
  funcT func;

  template<typename Key, typename Tuple, std::size_t... S>
  void call_func(Key &&key, Tuple &&args,
              output_terminalsT &out, std::index_sequence<S...>) {
    // this is the tuple of values
    using func_args_t = std::remove_reference_t<std::tuple_element_t<1,boost::callable_traits::args_t<funcT>>>;
    // NB cannot use std::forward_as_tuple since that makes a tuple of refs!
    func_args_t unwrapped_args(baseT::template get<S, std::tuple_element_t<S,func_args_t>>(std::forward<Tuple>(args))...);
    func(std::forward<Key>(key),
         std::move(unwrapped_args),
         out);
  }

 public:
  template<typename funcT_>
  WrapOp(funcT_ &&f, const typename baseT::input_edges_type &inedges,
         const typename baseT::output_edges_type &outedges, const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(std::forward<funcT_>(f)) {}

  void op(const keyT &key, typename baseT::input_values_tuple_type &&args, output_terminalsT &out) {
    call_func(key, std::forward<typename baseT::input_values_tuple_type>(args), out,
           std::make_index_sequence<std::tuple_size<typename baseT::input_values_tuple_type>::value>{});
    //func(key, std::forward<typename baseT::input_values_tuple_type>(args), out);
  }
};

template<typename funcT, typename keyT, typename output_terminalsT, typename input_values_tupleT>
struct WrapOpUnwrapTuple;

template<typename funcT, typename keyT, typename output_terminalsT, typename ... input_valuesT>
struct WrapOpUnwrapTuple<funcT, keyT, output_terminalsT, std::tuple<input_valuesT...>> {
using type = WrapOp<funcT, keyT, output_terminalsT, std::remove_reference_t<input_valuesT>...>;
};

template <typename> struct type_printer;

// Class to wrap a callable with signature
//
// void op(const input_keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)
//
template<typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
class WrapOpArgs : public Op<keyT, output_terminalsT, WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>,
                             input_valuesT...> {
  using baseT =
  Op<keyT, output_terminalsT, WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>, input_valuesT...>;
  funcT func;

  template<typename Key, typename Tuple, std::size_t... S>
  void call_func(Key &&key, Tuple &&args,
              output_terminalsT &out, std::index_sequence<S...>) {
    using func_args_t = boost::callable_traits::args_t<funcT>;
    func(std::forward<Key>(key),
         baseT::template get<S, std::tuple_element_t<S+1,func_args_t>>(std::forward<Tuple>(args))...,
         out);
  }

 public:
  template<typename funcT_>
  WrapOpArgs(funcT_ &&f, const typename baseT::input_edges_type &inedges,
             const typename baseT::output_edges_type &outedges, const std::string &name,
             const std::vector<std::string> &innames, const std::vector<std::string> &outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template<typename Key>
  void op(Key &&key, typename baseT::input_values_tuple_type &&args, output_terminalsT &out) {
    call_func(
        std::forward<Key>(key), std::forward<typename baseT::input_values_tuple_type>(args), out,
        std::make_index_sequence<std::tuple_size<typename baseT::input_values_tuple_type>::value>{});
  };
};

template<typename funcT, typename keyT, typename output_terminalsT, typename input_values_tupleT>
struct WrapOpArgsUnwrapTuple;

template<typename funcT, typename keyT, typename output_terminalsT, typename ... input_valuesT>
struct WrapOpArgsUnwrapTuple<funcT, keyT, output_terminalsT, std::tuple<input_valuesT...>> {
  using type = WrapOpArgs<funcT, keyT, output_terminalsT, std::remove_reference_t<input_valuesT>...>;
};

// Factory function to assist in wrapping a callable with signature
//
// void op(const input_keyT&, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
template<typename keyT, typename funcT, typename... input_valuesT, typename... output_edgesT>
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
  static_assert(num_args == 3, "ttg::wrapt(func, ...): func must take 3 arguments");
  // 2. input_args_t = {input_valuesT&&...}
  using input_args_t = std::decay_t<typename std::tuple_element<1,func_args_t>::type>;
  using decayed_input_args_t = ::ttg::meta::decayed_tuple_t<input_args_t>;
  using wrapT = typename WrapOpUnwrapTuple<funcT, keyT, output_terminals_type, input_args_t>::type;
  // not sure if we need this level of type checking ...
  // TODO determine the generic signature of func
  static_assert(std::is_same<typename std::tuple_element<0,func_args_t>::type, const keyT&>::value, "ttg::wrapt(func, inedges, outedges): first argument of func must be const keyT&");
  static_assert(std::is_same<decayed_input_args_t, std::tuple<input_valuesT...>>::value, "ttg::wrapt(func, inedges, outedges): inedges value types do not match argument types of func");
  static_assert(std::is_same<typename std::tuple_element<num_args-1,func_args_t>::type, output_terminals_type&>::value, "ttg::wrapt(func, inedges, outedges): last argument of func must be std::tuple<output_terminals_type>&");

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

// Factory function to assist in wrapping a callable with signature
//
// void op(const input_keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)
template<typename keyT, typename funcT, typename... input_edge_valuesT, typename... output_edgesT>
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
  static_assert(num_args == sizeof...(input_edge_valuesT) + 2, "ttg::wrap(func, inedges): func's # of args != # of inedges");
  // 2. input_args_t = {input_valuesT&&...}
  using input_args_t = typename ::ttg::meta::take_first_n<typename ::ttg::meta::drop_first_n<func_args_t, std::size_t(1)>::type, std::tuple_size<func_args_t>::value-2>::type;
  using decayed_input_args_t = ::ttg::meta::decayed_tuple_t<input_args_t>;
  using wrapT = typename WrapOpArgsUnwrapTuple<funcT, keyT, output_terminals_type, input_args_t>::type;
  // not sure if we need this level of type checking ...
  // TODO determine the generic signature of func
  static_assert(std::is_same<typename std::tuple_element<0,func_args_t>::type, const keyT&>::value, "ttg::wrap(func, inedges, outedges): first argument of func must be const keyT&");
  static_assert(std::is_same<decayed_input_args_t, std::tuple<input_edge_valuesT...>>::value, "ttg::wrap(func, inedges, outedges): inedges value types do not match argument types of func");
  static_assert(std::is_same<typename std::tuple_element<num_args-1,func_args_t>::type, output_terminals_type&>::value, "ttg::wrap(func, inedges, outedges): last argument of func must be std::tuple<output_terminals_type>&");

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

#endif //CXXAPI_WRAP_H
