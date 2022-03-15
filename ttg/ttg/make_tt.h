// to be #include'd within runtime::ttg namespace

#ifndef TTG_MAKE_TT_H
#define TTG_MAKE_TT_H

namespace detail {

  template <typename... FromEdgeTypesT, std::size_t... I>
  inline auto edge_base_tuple(const std::tuple<FromEdgeTypesT...> &edges, std::index_sequence<I...>) {
    return std::make_tuple(std::get<I>(edges).edge()...);
  }

  template <typename... EdgeTs>
  inline decltype(auto) edge_base_tuple(const std::tuple<EdgeTs...> &edges) {
    /* avoid copying edges if there is no wrapper that forces us to create copies */
    if constexpr ((EdgeTs::is_wrapper_edge || ...)) {
      return edge_base_tuple(edges, std::make_index_sequence<sizeof...(EdgeTs)>{});
    } else {
      return edges;
    }
  }

  inline auto edge_base_tuple(const std::tuple<> &empty) { return empty; }
}  // namespace detail

// Class to wrap a callable with signature
//
// case 1 (keyT != void): void op(auto&& key, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
//
template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
class CallableWrapTT
    : public TT<keyT, output_terminalsT, CallableWrapTT<funcT, keyT, output_terminalsT, input_valuesT...>,
                ttg::typelist<input_valuesT...>> {
  using baseT = typename CallableWrapTT::ttT;

  using input_values_tuple_type = typename baseT::input_values_tuple_type;
  using input_refs_tuple_type = typename baseT::input_refs_tuple_type;
  using input_edges_type = typename baseT::input_edges_type;
  using output_edges_type = typename baseT::output_edges_type;

  using noref_funcT = std::remove_reference_t<funcT>;
  std::conditional_t<std::is_function_v<noref_funcT>, std::add_pointer_t<noref_funcT>, noref_funcT> func;

  template <typename Key, typename Tuple>
  void call_func(Key &&key, Tuple &&args, output_terminalsT &out) {
    func(std::forward<Key>(key), std::forward<Tuple>(args), out);
  }

  template <typename TupleOrKey>
  void call_func(TupleOrKey &&args, output_terminalsT &out) {
    func(std::forward<TupleOrKey>(args), out);
  }

  void call_func(output_terminalsT &out) { func(std::tuple<>(), out); }

 public:
  template <typename funcT_>
  CallableWrapTT(funcT_ &&f, const input_edges_type &inedges, const output_edges_type &outedges,
                 const std::string &name, const std::vector<std::string> &innames,
                 const std::vector<std::string> &outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template <typename funcT_>
  CallableWrapTT(funcT_ &&f, const std::string &name, const std::vector<std::string> &innames,
                 const std::vector<std::string> &outnames)
      : baseT(name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template <typename Key, typename ArgsTuple>
  std::enable_if_t<std::is_same_v<ArgsTuple, input_refs_tuple_type> && !ttg::meta::is_empty_tuple_v<ArgsTuple> &&
                       !ttg::meta::is_void_v<Key>,
                   void>
  op(Key &&key, ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<Key>(key), std::forward<ArgsTuple>(args_tuple), out);
  }

  template <typename ArgsTuple, typename Key = keyT>
  std::enable_if_t<std::is_same_v<ArgsTuple, input_refs_tuple_type> && !ttg::meta::is_empty_tuple_v<ArgsTuple> &&
                       ttg::meta::is_void_v<Key>,
                   void>
  op(ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<ArgsTuple>(args_tuple), out);
  }

  template <typename Key, typename ArgsTuple = input_values_tuple_type>
  std::enable_if_t<ttg::meta::is_empty_tuple_v<ArgsTuple> && !ttg::meta::is_void_v<Key>, void> op(
      Key &&key, output_terminalsT &out) {
    call_func(std::forward<Key>(key), out);
  }

  template <typename Key = keyT, typename ArgsTuple = input_values_tuple_type>
  std::enable_if_t<ttg::meta::is_empty_tuple_v<ArgsTuple> && ttg::meta::is_void_v<Key>, void> op(
      output_terminalsT &out) {
    call_func(out);
  }
};

template <typename funcT, typename keyT, typename output_terminalsT, typename input_values_tupleT>
struct CallableWrapTTUnwrapTuple;

template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
struct CallableWrapTTUnwrapTuple<funcT, keyT, output_terminalsT, std::tuple<input_valuesT...>> {
  using type = CallableWrapTT<funcT, keyT, output_terminalsT, std::remove_reference_t<input_valuesT>...>;
};

// Class to wrap a callable with signature
//
// case 1 (keyT != void): void op(auto&& key, input_valuesT&&..., std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(input_valuesT&&..., std::tuple<output_terminalsT...>&)
//
template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
class CallableWrapTTArgs
    : public TT<keyT, output_terminalsT, CallableWrapTTArgs<funcT, keyT, output_terminalsT, input_valuesT...>,
                ttg::typelist<input_valuesT...>> {
  using baseT = typename CallableWrapTTArgs::ttT;

  using input_values_tuple_type = typename baseT::input_values_tuple_type;
  using input_refs_tuple_type = typename baseT::input_refs_tuple_type;
  using input_edges_type = typename baseT::input_edges_type;
  using output_edges_type = typename baseT::output_edges_type;

  using noref_funcT = std::remove_reference_t<funcT>;
  std::conditional_t<std::is_function_v<noref_funcT>, std::add_pointer_t<noref_funcT>, noref_funcT> func;

  template <typename Key, typename Tuple, std::size_t... S>
  void call_func(Key &&key, Tuple &&args_tuple, output_terminalsT &out, std::index_sequence<S...>) {
    using func_args_t = ttg::meta::tuple_concat_t<std::tuple<const Key &>, input_refs_tuple_type, output_edges_type>;
    func(std::forward<Key>(key),
         baseT::template get<S, std::tuple_element_t<S + 1, func_args_t>>(std::forward<Tuple>(args_tuple))..., out);
  }

  template <typename Tuple, std::size_t... S>
  void call_func(Tuple &&args_tuple, output_terminalsT &out, std::index_sequence<S...>) {
    using func_args_t = ttg::meta::tuple_concat_t<input_refs_tuple_type, output_edges_type>;
    func(baseT::template get<S, std::tuple_element_t<S, func_args_t>>(std::forward<Tuple>(args_tuple))..., out);
  }

  template <typename Key>
  void call_func(Key &&key, output_terminalsT &out) {
    func(std::forward<Key>(key), out);
  }

  template <typename OutputTerminals>
  void call_func(OutputTerminals &out) {
    func(out);
  }

 public:
  template <typename funcT_>
  CallableWrapTTArgs(funcT_ &&f, const input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
                     const std::string &name, const std::vector<std::string> &innames,
                     const std::vector<std::string> &outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template <typename funcT_>
  CallableWrapTTArgs(funcT_ &&f, const std::string &name, const std::vector<std::string> &innames,
                     const std::vector<std::string> &outnames)
      : baseT(name, innames, outnames), func(std::forward<funcT_>(f)) {}

  template <typename Key, typename ArgsTuple>
  std::enable_if_t<std::is_same_v<ArgsTuple, input_refs_tuple_type> &&
                       !ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && !ttg::meta::is_void_v<Key>,
                   void>
  op(Key &&key, ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<Key>(key), std::forward<ArgsTuple>(args_tuple), out,
              std::make_index_sequence<std::tuple_size_v<ArgsTuple>>{});
  };

  template <typename ArgsTuple, typename Key = keyT>
  std::enable_if_t<std::is_same_v<ArgsTuple, input_refs_tuple_type> &&
                       !ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && ttg::meta::is_void_v<Key>,
                   void>
  op(ArgsTuple &&args_tuple, output_terminalsT &out) {
    call_func(std::forward<ArgsTuple>(args_tuple), out, std::make_index_sequence<std::tuple_size_v<ArgsTuple>>{});
  };

  template <typename Key, typename ArgsTuple = input_refs_tuple_type>
  std::enable_if_t<ttg::meta::is_empty_tuple_v<ArgsTuple> && !ttg::meta::is_void_v<Key>, void> op(
      Key &&key, output_terminalsT &out) {
    call_func(std::forward<Key>(key), out);
  };

  template <typename Key = keyT, typename ArgsTuple = input_refs_tuple_type>
  std::enable_if_t<ttg::meta::is_empty_tuple_v<ArgsTuple> && ttg::meta::is_void_v<Key>, void> op(
      output_terminalsT &out) {
    call_func(out);
  };
};

template <typename funcT, typename keyT, typename output_terminalsT, typename input_values_tupleT>
struct CallableWrapTTArgsUnwrapTuple;

template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
struct CallableWrapTTArgsUnwrapTuple<funcT, keyT, output_terminalsT, std::tuple<input_valuesT...>> {
  using type = CallableWrapTTArgs<funcT, keyT, output_terminalsT, std::remove_reference_t<input_valuesT>...>;
};

// Factory function to assist in wrapping a callable with signature
//
// If the callable is not generic, its arguments are inspected and the constness is used
// to determine mutable data. Otherwise, mutability information is taken from the input edge types.
// See \c ttg::make_const.
//
// case 1 (keyT != void): void op(const input_keyT&, std::tuple<input_valuesT&...>&&, std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(std::tuple<input_valuesT&...>&&, std::tuple<output_terminalsT...>&)
template <typename keyT, typename funcT, typename... input_edge_valuesT, typename... output_edgesT>
auto make_tt_tpl(funcT &&func, const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges,
                 const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
                 const std::vector<std::string> &innames = std::vector<std::string>(sizeof...(input_edge_valuesT),
                                                                                    "input"),
                 const std::vector<std::string> &outnames = std::vector<std::string>(sizeof...(output_edgesT),
                                                                                     "output")) {
  // ensure input types do not contain Void
  static_assert(ttg::meta::is_none_Void_v<input_edge_valuesT...>, "ttg::Void is for internal use only, do not use it");
  using output_terminals_type = typename ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;

  constexpr auto void_key = ttg::meta::is_void_v<keyT>;

  // list of base datum types (T or const T)
  using base_input_data_t = ttg::meta::typelist<typename ttg::Edge<keyT, input_edge_valuesT>::value_type...>;

  // gross list of candidate argument types
  using gross_candidate_func_args_t = ttg::meta::typelist<
      ttg::meta::candidate_argument_bindings_t<std::add_const_t<keyT>>,
      ttg::meta::candidate_argument_bindings_t<
          std::tuple<std::add_lvalue_reference_t<typename ttg::Edge<keyT, input_edge_valuesT>::value_type>...>>,
      ttg::meta::typelist<output_terminals_type &>>;

  // net list of candidate argument types excludes the empty typelists for void arguments
  using candidate_func_args_t = ttg::meta::filter_t<gross_candidate_func_args_t, ttg::meta::typelist_is_not_empty>;

  // list argument types with which func can be invoked
  using func_args_t = ttg::meta::typelist_to_tuple_t<decltype(ttg::meta::compute_arg_binding_types_r<void>(
      func, candidate_func_args_t{}))>;

  static_assert(!std::is_same_v<func_args_t, std::tuple<>>,
                "ttd::make_tt(func, inedges, ...): could not detect how to invoke func, either the signature of func "
                "is faulty, or inedges does match the expected list of types, or both");

  constexpr auto num_args = std::tuple_size_v<func_args_t>;

  static_assert(num_args == (void_key ? 2 : 3),
                "ttg::make_tt_tpl(func, ...): func must take 3 arguments (or 2, if keyT=void)");

  // 2. input_args_t = {input_valuesT&&...}
  using input_args_t = std::decay_t<std::tuple_element_t<void_key ? 0 : 1, func_args_t>>;
  using decayed_input_args_t = ttg::meta::decayed_tuple_t<input_args_t>;
  using wrapT = typename CallableWrapTTUnwrapTuple<funcT, keyT, output_terminals_type, input_args_t>::type;
  static_assert(std::is_same_v<decayed_input_args_t, std::tuple<input_edge_valuesT...>>,
                "ttg::make_tt_tpl(func, inedges, outedges): inedges value types do not match argument types of func");

  auto input_edges = detail::edge_base_tuple(inedges);

  return std::make_unique<wrapT>(std::forward<funcT>(func), input_edges, outedges, name, innames, outnames);
}

/// @brief Factory function to assist in wrapping a callable with signature
///
/// @tparam keyT a task ID type
/// @tparam funcT a callable type
/// @tparam input_edge_valuesT a pack of types of input data
/// @tparam output_edgesT a pack of types of output edges
/// @param[in] func a callable object; if `ttg::meta::is_void_v<keyT>==true`, the signature
///         must be `void(input_valuesT&&..., std::tuple<output_terminalsT...>&)`,
///         else `void(const keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)`
/// @param[in] inedges a tuple of input edges
/// @param[in] outedges a tuple of output edges
/// @param[in] name a string label for the resulting TT
/// @param[in] name a string label for the resulting TT
/// @param[in] innames string labels for the respective input terminals of the resulting TT
/// @param[in] outnames string labels for the respective output terminals of the resulting TT
///
/// @internal To be able to handle generic callables the input edges are used to determine the trial set of
/// argument types.
template <typename keyT, typename funcT, typename... input_edge_valuesT, typename... output_edgesT>
auto make_tt(funcT &&func, const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges,
             const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
             const std::vector<std::string> &innames = std::vector<std::string>(sizeof...(input_edge_valuesT), "input"),
             const std::vector<std::string> &outnames = std::vector<std::string>(sizeof...(output_edgesT), "output")) {
  // ensure input types do not contain Void
  static_assert(ttg::meta::is_none_Void_v<input_edge_valuesT...>, "ttg::Void is for internal use only, do not use it");

  using output_terminals_type = typename ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;

  constexpr auto void_key = ttg::meta::is_void_v<keyT>;

  // list of base datum types (T or const T)
  using base_input_data_t = ttg::meta::typelist<typename ttg::Edge<keyT, input_edge_valuesT>::value_type...>;

  // gross list of candidate argument types
  using gross_candidate_func_args_t = ttg::meta::typelist<
      ttg::meta::candidate_argument_bindings_t<std::add_const_t<keyT>>,
      ttg::meta::candidate_argument_bindings_t<typename ttg::Edge<keyT, input_edge_valuesT>::value_type>...,
      ttg::meta::typelist<output_terminals_type &>>;

  // net list of candidate argument types excludes the empty typelists for void arguments
  using candidate_func_args_t = ttg::meta::filter_t<gross_candidate_func_args_t, ttg::meta::typelist_is_not_empty>;

  // list argument types with which func can be invoked
  using func_args_t = ttg::meta::typelist_to_tuple_t<decltype(ttg::meta::compute_arg_binding_types_r<void>(
      func, candidate_func_args_t{}))>;

  constexpr auto DETECTED_HOW_TO_INVOKE_FUNC = !std::is_same_v<func_args_t, std::tuple<>>;
  static_assert(DETECTED_HOW_TO_INVOKE_FUNC,
                "ttd::make_tt(func, inedges, ...): could not detect how to invoke func, either the signature of func "
                "is faulty, or inedges does match the expected list of types, or both");

  constexpr auto num_args = std::tuple_size_v<func_args_t>;

  // TT needs actual types of arguments to func ... extract them and pass to CallableWrapTTArgs
  using input_values_full_tuple_type =
      ttg::meta::decayed_tuple_t<typename std::tuple<typename ttg::Edge<keyT, input_edge_valuesT>::value_type...>>;
  // input_args_t = {input_valuesT&&...}
  using input_args_t = typename ttg::meta::take_first_n<
      typename ttg::meta::drop_first_n<func_args_t, std::size_t(void_key ? 0 : 1)>::type,
      std::tuple_size_v<func_args_t> - (void_key ? 1 : 2)>::type;
  using decayed_input_args_t = ttg::meta::decayed_tuple_t<input_args_t>;
  // 3. full_input_args_t = edge-types with non-void types replaced by input_args_t
  using full_input_args_t = ttg::meta::replace_nonvoid_t<input_values_full_tuple_type, input_args_t>;
  using wrapT = typename CallableWrapTTArgsUnwrapTuple<funcT, keyT, output_terminals_type, full_input_args_t>::type;

  auto input_edges = detail::edge_base_tuple(inedges);

  return std::make_unique<wrapT>(std::forward<funcT>(func), input_edges, outedges, name, innames, outnames);
}

template <typename keyT, typename funcT, typename... input_valuesT, typename... output_edgesT>
[[deprecated("use make_tt_tpl instead")]] inline auto wrapt(
    funcT &&func, const std::tuple<ttg::Edge<keyT, input_valuesT>...> &inedges,
    const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
    const std::vector<std::string> &innames = std::vector<std::string>(sizeof...(input_valuesT), "input"),
    const std::vector<std::string> &outnames = std::vector<std::string>(sizeof...(output_edgesT), "output")) {
  return make_tt_tpl<keyT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

template <typename keyT, typename funcT, typename... input_edge_valuesT, typename... output_edgesT>
[[deprecated("use make_tt instead")]] auto wrap(
    funcT &&func, const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges,
    const std::tuple<output_edgesT...> &outedges, const std::string &name = "wrapper",
    const std::vector<std::string> &innames = std::vector<std::string>(sizeof...(input_edge_valuesT), "input"),
    const std::vector<std::string> &outnames = std::vector<std::string>(sizeof...(output_edgesT), "output")) {
  return make_tt<keyT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

#endif  // TTG_MAKE_TT_H
