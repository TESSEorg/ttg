// to be #include'd within runtime::ttg namespace

#ifndef TTG_MAKE_TT_H
#define TTG_MAKE_TT_H

// Class to wrap a callable with signature
//
// case 1 (keyT != void): void op(auto&& key, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
//
template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
class CallableWrapTT
    : public TT<keyT, output_terminalsT,
                CallableWrapTT<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT, input_valuesT...>,
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
    if constexpr (funcT_receives_outterm_tuple)
      func(std::forward<Key>(key), std::forward<Tuple>(args), out);
    else {
      auto old_output_tls_ptr = this->outputs_tls_ptr_accessor();
      this->set_outputs_tls_ptr();
      func(std::forward<Key>(key), std::forward<Tuple>(args));
      this->set_outputs_tls_ptr(old_output_tls_ptr);
    }
  }

  template <typename TupleOrKey>
  void call_func(TupleOrKey &&args, output_terminalsT &out) {
    if constexpr (funcT_receives_outterm_tuple)
      func(std::forward<TupleOrKey>(args), out);
    else {
      auto old_output_tls_ptr = this->outputs_tls_ptr_accessor();
      this->set_outputs_tls_ptr();
      func(std::forward<TupleOrKey>(args));
      this->set_outputs_tls_ptr(old_output_tls_ptr);
    }
  }

  void call_func(output_terminalsT &out) {
    if constexpr (funcT_receives_outterm_tuple)
      func(std::tuple<>(), out);
    else {
      auto old_output_tls_ptr = this->outputs_tls_ptr_accessor();
      this->set_outputs_tls_ptr();
      func(std::tuple<>());
      this->set_outputs_tls_ptr(old_output_tls_ptr);
    }
  }

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

template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename input_values_tupleT>
struct CallableWrapTTUnwrapTypelist;

template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
struct CallableWrapTTUnwrapTypelist<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                                    std::tuple<input_valuesT...>> {
  using type = CallableWrapTT<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                              std::remove_reference_t<input_valuesT>...>;
};

template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
struct CallableWrapTTUnwrapTypelist<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                                    ttg::meta::typelist<input_valuesT...>> {
  using type = CallableWrapTT<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                              std::remove_reference_t<input_valuesT>...>;
};

// Class to wrap a callable with signature
//
// case 1 (keyT != void): void op(auto&& key, input_valuesT&&..., std::tuple<output_terminalsT...>&)
// case 2 (keyT == void): void op(input_valuesT&&..., std::tuple<output_terminalsT...>&)
//
template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
class CallableWrapTTArgs
    : public TT<keyT, output_terminalsT,
                CallableWrapTTArgs<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT, input_valuesT...>,
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
    if constexpr (funcT_receives_outterm_tuple)
      func(std::forward<Key>(key),
           baseT::template get<S, std::tuple_element_t<S + 1, func_args_t>>(std::forward<Tuple>(args_tuple))..., out);
    else {
      auto old_output_tls_ptr = this->outputs_tls_ptr_accessor();
      this->set_outputs_tls_ptr();
      func(std::forward<Key>(key),
           baseT::template get<S, std::tuple_element_t<S + 1, func_args_t>>(std::forward<Tuple>(args_tuple))...);
      this->set_outputs_tls_ptr(old_output_tls_ptr);
    }
  }

  template <typename Tuple, std::size_t... S>
  void call_func(Tuple &&args_tuple, output_terminalsT &out, std::index_sequence<S...>) {
    using func_args_t = ttg::meta::tuple_concat_t<input_refs_tuple_type, output_edges_type>;
    if constexpr (funcT_receives_outterm_tuple)
      func(baseT::template get<S, std::tuple_element_t<S, func_args_t>>(std::forward<Tuple>(args_tuple))..., out);
    else {
      auto old_output_tls_ptr = this->outputs_tls_ptr_accessor();
      this->set_outputs_tls_ptr();
      func(baseT::template get<S, std::tuple_element_t<S, func_args_t>>(std::forward<Tuple>(args_tuple))...);
      this->set_outputs_tls_ptr(old_output_tls_ptr);
    }
  }

  template <typename Key>
  void call_func(Key &&key, output_terminalsT &out) {
    if constexpr (funcT_receives_outterm_tuple)
      func(std::forward<Key>(key), out);
    else {
      auto old_output_tls_ptr = this->outputs_tls_ptr_accessor();
      this->set_outputs_tls_ptr();
      func(std::forward<Key>(key));
      this->set_outputs_tls_ptr(old_output_tls_ptr);
    }
  }

  template <typename OutputTerminals>
  void call_func(OutputTerminals &out) {
    if constexpr (funcT_receives_outterm_tuple)
      func(out);
    else {
      auto old_output_tls_ptr = this->outputs_tls_ptr_accessor();
      this->set_outputs_tls_ptr();
      func();
      this->set_outputs_tls_ptr(old_output_tls_ptr);
    }
  }

  template <typename Tuple, std::size_t... I>
  static auto make_output_terminal_ptrs(const Tuple &output_terminals, std::index_sequence<I...>) {
    return std::array<ttg::TerminalBase *, sizeof...(I)>{
        {static_cast<ttg::TerminalBase *>(&std::get<I>(output_terminals))...}};
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
    assert(&out == &baseT::get_output_terminals());
    call_func(std::forward<Key>(key), std::forward<ArgsTuple>(args_tuple), out,
              std::make_index_sequence<std::tuple_size_v<ArgsTuple>>{});
  };

  template <typename ArgsTuple, typename Key = keyT>
  std::enable_if_t<std::is_same_v<ArgsTuple, input_refs_tuple_type> &&
                       !ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && ttg::meta::is_void_v<Key>,
                   void>
  op(ArgsTuple &&args_tuple, output_terminalsT &out) {
    assert(&out == &baseT::get_output_terminals());
    call_func(std::forward<ArgsTuple>(args_tuple), out, std::make_index_sequence<std::tuple_size_v<ArgsTuple>>{});
  };

  template <typename Key, typename ArgsTuple = input_refs_tuple_type>
  std::enable_if_t<ttg::meta::is_empty_tuple_v<ArgsTuple> && !ttg::meta::is_void_v<Key>, void> op(
      Key &&key, output_terminalsT &out) {
    assert(&out == &baseT::get_output_terminals());
    call_func(std::forward<Key>(key), out);
  };

  template <typename Key = keyT, typename ArgsTuple = input_refs_tuple_type>
  std::enable_if_t<ttg::meta::is_empty_tuple_v<ArgsTuple> && ttg::meta::is_void_v<Key>, void> op(
      output_terminalsT &out) {
    assert(&out == &baseT::get_output_terminals());
    call_func(out);
  };
};

template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename input_values_typelistT>
struct CallableWrapTTArgsAsTypelist;

template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
struct CallableWrapTTArgsAsTypelist<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                                    std::tuple<input_valuesT...>> {
  using type = CallableWrapTTArgs<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                                  std::remove_reference_t<input_valuesT>...>;
};

template <typename funcT, bool funcT_receives_outterm_tuple, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
struct CallableWrapTTArgsAsTypelist<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                                    ttg::meta::typelist<input_valuesT...>> {
  using type = CallableWrapTTArgs<funcT, funcT_receives_outterm_tuple, keyT, output_terminalsT,
                                  std::remove_reference_t<input_valuesT>...>;
};

// clang-format off
/// @brief Factory function to assist in wrapping a callable with signature
///
/// @tparam keyT a task ID type
/// @tparam funcT a callable type
/// @tparam input_edge_valuesT a pack of types of input data
/// @tparam output_edgesT a pack of types of output edges
/// @param[in] func a callable object; it can be _generic_ (e.g., a template function, a generic lambda, etc.; see
///            below) or _nongeneric_ (with concrete types for its arguments). In either case its signature
///            must match the following:
///         - if `ttg::meta::is_void_v<keyT>==true`:
///           - `void(const std::tuple<input_valuesT&...>&, std::tuple<output_terminalsT...>&)`: full form, with the explicitly-passed
///             output terminals ensuring compile-time type-checking of the dataflow into the output terminals (see ttg::send);
///           - `void(const std::tuple<input_valuesT&...>&)`: simplified form, with no type-checking of the dataflow into the output terminals;
///         - if `ttg::meta::is_void_v<keyT>==false`:
///           - `void(const keyT&, const std::tuple<input_valuesT&...>&, std::tuple<output_terminalsT...>&)`: full form, with the explicitly-passed
/////             output terminals ensuring compile-time type-checking of the dataflow into the output terminals (see ttg::send);
///           - `void(const keyT&, const std::tuple<input_valuesT&...>&)`: simplified form, with no type-checking of the dataflow into the output terminals.
/// @param[in] inedges a tuple of input edges
/// @param[in] outedges a tuple of output edges
/// @param[in] name a string label for the resulting TT
/// @param[in] name a string label for the resulting TT
/// @param[in] innames string labels for the respective input terminals of the resulting TT
/// @param[in] outnames string labels for the respective output terminals of the resulting TT
///
/// @note Handling of generic @p func is described in the documentation of make_tt()
// clang-format on
template <typename keyT = void, typename funcT, typename... input_edge_valuesT, typename... output_edgesT>
auto make_tt_tpl(funcT &&func, const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges = std::tuple<>{},
                 const std::tuple<output_edgesT...> &outedges = std::tuple<>{}, const std::string &name = "wrapper",
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
      ttg::meta::typelist<output_terminals_type &, void>>;

  // net list of candidate argument types excludes the empty typelists for void arguments
  using candidate_func_args_t = ttg::meta::filter_t<gross_candidate_func_args_t, ttg::meta::typelist_is_not_empty>;

  // compute list of argument types with which func can be invoked
  constexpr static auto func_is_generic = ttg::meta::is_generic_callable_v<funcT>;
  using gross_func_args_t = decltype(ttg::meta::compute_arg_binding_types_r<void>(func, candidate_func_args_t{}));
  constexpr auto DETECTED_HOW_TO_INVOKE_GENERIC_FUNC =
      func_is_generic ? !std::is_same_v<gross_func_args_t, ttg::typelist<>> : true;
  static_assert(DETECTED_HOW_TO_INVOKE_GENERIC_FUNC,
                "ttd::make_tt_tpl(func, inedges, ...): could not detect how to invoke generic callable func, either "
                "the signature of func "
                "is faulty, or inedges does match the expected list of types, or both");

  // net argument typelist
  using func_args_t = ttg::meta::drop_void_t<gross_func_args_t>;
  constexpr auto num_args = std::tuple_size_v<func_args_t>;

  // if given task id, make sure it's passed via const lvalue ref
  constexpr bool TASK_ID_PASSED_AS_CONST_LVALUE_REF =
      !void_key ? ttg::meta::probe_first_v<ttg::meta::is_const_lvalue_reference, true, func_args_t> : true;
  static_assert(TASK_ID_PASSED_AS_CONST_LVALUE_REF,
                "ttg::make_tt_tpl(func, ...): if given to func, the task id must be passed by const lvalue ref");

  // if given out-terminal tuple, make sure it's passed via nonconst lvalue ref
  constexpr bool have_outterm_tuple =
      func_is_generic ? !ttg::meta::is_last_void_v<gross_func_args_t>
                      : ttg::meta::probe_last_v<ttg::meta::is_nonconst_lvalue_reference_to_output_terminal_tuple, true,
                                                gross_func_args_t>;
  constexpr bool OUTTERM_TUPLE_PASSED_AS_NONCONST_LVALUE_REF =
      have_outterm_tuple ? ttg::meta::probe_last_v<ttg::meta::is_nonconst_lvalue_reference, true, func_args_t> : true;
  static_assert(
      OUTTERM_TUPLE_PASSED_AS_NONCONST_LVALUE_REF,
      "ttd::make_tt_tpl(func, ...): if given to func, the output terminal tuple must be passed by nonconst lvalue ref");

  static_assert(num_args == 3 - (void_key ? 1 : 0) - (have_outterm_tuple ? 0 : 1),
                "ttg::make_tt_tpl(func, ...): func takes wrong number of arguments (2, or 1, if keyT=void + optional "
                "tuple of output terminals)");

  // 2. input_args_t = {input_valuesT&&...}
  using nondecayed_input_args_t = std::tuple_element_t<void_key ? 0 : 1, func_args_t>;
  constexpr auto NO_ARGUMENTS_PASSED_AS_NONCONST_LVALUE_REF =
      !ttg::meta::is_any_nonconst_lvalue_reference_v<nondecayed_input_args_t>;
  static_assert(
      NO_ARGUMENTS_PASSED_AS_NONCONST_LVALUE_REF,
      "ttg::make_tt_tpl(func, inedges, outedges): one or more arguments to func can only be passed by nonconst lvalue "
      "ref; this is illegal, should only pass arguments as const lvalue ref or (nonconst) rvalue ref");
  using input_args_t = std::decay_t<nondecayed_input_args_t>;
  using decayed_input_args_t = ttg::meta::decayed_typelist_t<input_args_t>;
  using wrapT =
      typename CallableWrapTTUnwrapTypelist<funcT, have_outterm_tuple, keyT, output_terminals_type, input_args_t>::type;
  static_assert(std::is_same_v<decayed_input_args_t, std::tuple<input_edge_valuesT...>>,
                "ttg::make_tt_tpl(func, inedges, outedges): inedges value types do not match argument types of func");

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
}

// clang-format off
/// @brief Factory function to assist in wrapping a callable with signature
///
/// @tparam keyT a task ID type
/// @tparam funcT a callable type
/// @tparam input_edge_valuesT a pack of types of input data
/// @tparam output_edgesT a pack of types of output edges
/// @param[in] func a callable object; it can be _generic_ (e.g., a template function, a generic lambda, etc.; see
///            below) or _nongeneric_ (with concrete types for its arguments). In either case its signature
///            must match the following:
///         - if `ttg::meta::is_void_v<keyT>==true`:
///           - `void(input_valuesT&&..., std::tuple<output_terminalsT...>&)`: full form, with the explicitly-passed
///             output terminals ensuring compile-time type-checking of the dataflow into the output terminals (see ttg::send);
///           - `void(input_valuesT&&...)`: simplified form, with no type-checking of the dataflow into the output terminals;
///         - if `ttg::meta::is_void_v<keyT>==false`:
///           - `void(const keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)`: full form, with the explicitly-passed
///             output terminals ensuring compile-time type-checking of the dataflow into the output terminals (see ttg::send);
///           - `void(const keyT&, input_valuesT&&...)`: simplified form, with no type-checking of the dataflow into the output terminals.
/// @param[in] inedges a tuple of input edges
/// @param[in] outedges a tuple of output edges
/// @param[in] name a string label for the resulting TT
/// @param[in] name a string label for the resulting TT
/// @param[in] innames string labels for the respective input terminals of the resulting TT
/// @param[in] outnames string labels for the respective output terminals of the resulting TT
///
/// @warning You MUST NOT use generic callables that use concrete types for some data arguments, i.e. make either
///          ALL data types or NONE of them generic. This warning only applies to the data arguments and
///          does not apply to task ID (key) and optional out-terminal arguments.
///
/// @note For generic callables the arguments that are used read-only should be declared as `U&` (where `U` is the corresponding template parameter)
///       or `auto&` (in contexts such as generic lambdas where template arguments are implicit). The arguments that are
///       to be consumed (e.g. mutated, moved, etc.) should be declared as `U&&` or `auto&&` (i.e., as universal references).
///       For example, in
///       @code
///          make_tt([](auto& key, auto& datum1, auto&& datum2) { ... }, ...);
///       @endcode
///       the task id (`key`) and the first datum will be passed by const lvalue reference (i.e. no copy will be created by the runtime),
///       whereas the second datum will be passed by an rvalue reference, which may cause copying.
///       The corresponding free function analog of the above lambda is:
///       @code
///          template <typename K, typename D1, typename D2>
///          void func (K& key, D1& datum1, D2&& datum2) { ... }
///       @endcode
///
/// @warning Although generic arguments annotated by `const auto&` are also permitted, their use is discouraged to avoid confusion;
///          namely, `const auto&` denotes a _consumable_ argument, NOT read-only, despite the `const`.
// clang-format on
template <typename keyT = void, typename funcT, typename... input_edge_valuesT, typename... output_edgesT>
auto make_tt(funcT &&func, const std::tuple<ttg::Edge<keyT, input_edge_valuesT>...> &inedges = std::tuple<>{},
             const std::tuple<output_edgesT...> &outedges = std::tuple<>{}, const std::string &name = "wrapper",
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
      ttg::meta::typelist<output_terminals_type &, void>>;

  // net list of candidate argument types excludes the empty typelists for void arguments
  using candidate_func_args_t = ttg::meta::filter_t<gross_candidate_func_args_t, ttg::meta::typelist_is_not_empty>;

  // gross argument typelist for invoking func, can include void for optional args
  constexpr static auto func_is_generic = ttg::meta::is_generic_callable_v<funcT>;
  using gross_func_args_t = decltype(ttg::meta::compute_arg_binding_types_r<void>(func, candidate_func_args_t{}));
  constexpr auto DETECTED_HOW_TO_INVOKE_GENERIC_FUNC =
      func_is_generic ? !std::is_same_v<gross_func_args_t, ttg::typelist<>> : true;
  static_assert(DETECTED_HOW_TO_INVOKE_GENERIC_FUNC,
                "ttd::make_tt(func, inedges, ...): could not detect how to invoke generic callable func, either the "
                "signature of func "
                "is faulty, or inedges does match the expected list of types, or both");

  // net argument typelist
  using func_args_t = ttg::meta::drop_void_t<gross_func_args_t>;
  constexpr auto num_args = std::tuple_size_v<func_args_t>;

  // if given task id, make sure it's passed via const lvalue ref
  constexpr bool TASK_ID_PASSED_AS_CONST_LVALUE_REF =
      !void_key ? ttg::meta::probe_first_v<ttg::meta::is_const_lvalue_reference, true, func_args_t> : true;
  static_assert(TASK_ID_PASSED_AS_CONST_LVALUE_REF,
                "ttg::make_tt(func, ...): if given to func, the task id must be passed by const lvalue ref");

  // if given out-terminal tuple, make sure it's passed via nonconst lvalue ref
  constexpr bool have_outterm_tuple =
      func_is_generic ? !ttg::meta::is_last_void_v<gross_func_args_t>
                      : ttg::meta::probe_last_v<ttg::meta::decays_to_output_terminal_tuple, false, gross_func_args_t>;
  constexpr bool OUTTERM_TUPLE_PASSED_AS_NONCONST_LVALUE_REF =
      have_outterm_tuple ? ttg::meta::probe_last_v<ttg::meta::is_nonconst_lvalue_reference, false, func_args_t> : true;
  static_assert(
      OUTTERM_TUPLE_PASSED_AS_NONCONST_LVALUE_REF,
      "ttg::make_tt(func, ...): if given to func, the output terminal tuple must be passed by nonconst lvalue ref");

  // TT needs actual types of arguments to func ... extract them and pass to CallableWrapTTArgs
  using input_edge_value_types = ttg::meta::typelist<std::decay_t<input_edge_valuesT>...>;
  // input_args_t = {input_valuesT&&...}
  using input_args_t = typename ttg::meta::take_first_n<
      typename ttg::meta::drop_first_n<func_args_t, std::size_t(void_key ? 0 : 1)>::type,
      std::tuple_size_v<func_args_t> - (void_key ? 0 : 1) - (have_outterm_tuple ? 1 : 0)>::type;
  constexpr auto NO_ARGUMENTS_PASSED_AS_NONCONST_LVALUE_REF =
      !ttg::meta::is_any_nonconst_lvalue_reference_v<input_args_t>;
  static_assert(
      NO_ARGUMENTS_PASSED_AS_NONCONST_LVALUE_REF,
      "ttg::make_tt(func, inedges, outedges): one or more arguments to func can only be passed by nonconst lvalue "
      "ref; this is illegal, should only pass arguments as const lvalue ref or (nonconst) rvalue ref");
  using decayed_input_args_t = ttg::meta::decayed_typelist_t<input_args_t>;
  // 3. full_input_args_t = edge-types with non-void types replaced by input_args_t
  using full_input_args_t = ttg::meta::replace_nonvoid_t<input_edge_value_types, input_args_t>;
  using wrapT = typename CallableWrapTTArgsAsTypelist<funcT, have_outterm_tuple, keyT, output_terminals_type,
                                                      full_input_args_t>::type;

  return std::make_unique<wrapT>(std::forward<funcT>(func), inedges, outedges, name, innames, outnames);
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

#include "ttg/make_device_tt.h"

#endif  // TTG_MAKE_TT_H
