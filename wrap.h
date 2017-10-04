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
    func(std::forward<Key>(key),
         std::forward_as_tuple(baseT::template get<S, std::tuple_element_t<S,func_args_t>>(std::forward<Tuple>(args))...),
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
  using callable_type =
  std::function<void(const keyT &, std::tuple<input_valuesT...> &&, output_terminals_type &)>;
  callable_type f(std::forward<funcT>(func));  // primarily to check types
  using wrapT = WrapOp<funcT, keyT, output_terminals_type, input_valuesT...>;

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

// Factory function to assist in wrapping a callable with signature
//
// void op(const input_keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)
template<typename keyT, typename funcT, typename... input_valuesT, typename... output_edgesT>
auto wrap(funcT &&func, const std::tuple<::ttg::Edge<keyT, input_valuesT>...> &inedges,
          const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
          const std::vector<std::string> &innames = std::vector<std::string>(
              std::tuple_size<std::tuple<::ttg::Edge<keyT, input_valuesT>...>>::value, "input"),
          const std::vector<std::string> &outnames =
          std::vector<std::string>(std::tuple_size<std::tuple<output_edgesT...>>::value, "output")) {
  using output_terminals_type = typename ::ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;
  using wrapT = WrapOpArgs<funcT, keyT, output_terminals_type, input_valuesT...>;

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

#endif //CXXAPI_WRAP_H
