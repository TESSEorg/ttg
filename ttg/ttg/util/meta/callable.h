// SPDX-License-Identifier: BSD-3-Clause
//
// Created by Eduard Valeyev on 3/1/22.
//

#ifndef TTG_META_CALLABLE_H
#define TTG_META_CALLABLE_H

#include <type_traits>

#include "ttg/util/meta.h"
#include "ttg/util/typelist.h"

namespace ttg::meta {

  //////////////////////////////////////
  // callable signature introspection
  //////////////////////////////////////
  namespace detail {
    template <typename R, typename... Args>
    struct callable_signature {
      using return_type = R;
      using args = ttg::typelist<Args...>;
    };

    /// primary template: not a callable with a unique signature (e.g. generic or overloaded callables)
    template <typename T, typename Enabler = void>
    struct callable_traits_impl {};

    // functions and function pointers; C-style varargs are not part of the argument list
    template <typename R, typename... Args>
    struct callable_traits_impl<R(Args...)> : callable_signature<R, Args...> {};
    template <typename R, typename... Args>
    struct callable_traits_impl<R(Args...) noexcept> : callable_signature<R, Args...> {};
    template <typename R, typename... Args>
    struct callable_traits_impl<R(Args..., ...)> : callable_signature<R, Args...> {};
    template <typename R, typename... Args>
    struct callable_traits_impl<R(Args..., ...) noexcept> : callable_signature<R, Args...> {};
    template <typename F>
    struct callable_traits_impl<F*, std::enable_if_t<std::is_function_v<F>>> : callable_traits_impl<F> {};

    // member function pointers: the (reference to the) object is the first argument, args_sans_object omits it.
    // The macros below expand to one partial specialization per combination of the PMF's qualifiers:
    //   4 cv (none, const, volatile, const volatile) x 3 ref (none, &, &&) x 2 noexcept x 2 C-style varargs = 48.
    // OBJREF is the reference type of the object argument: & for unqualified and &-qualified PMFs, && for
    // &&-qualified ones (e.g. `void (C::*)(int) const` -> args = typelist<const C&, int>).
    // Abominable function types (e.g. `void(int) const`) are not callables and are deliberately not handled.
#define TTG_CALLABLE_TRAITS_PMF_IMPL(CV, REF, OBJREF, NOEXCEPT, VARARGS)                                       \
  template <typename R, typename C, typename... Args>                                                         \
  struct callable_traits_impl<R (C::*)(Args... VARARGS) CV REF NOEXCEPT>                                      \
      : callable_signature<R, CV C OBJREF, Args...> {                                                         \
    using args_sans_object = ttg::typelist<Args...>;                                                          \
  };
#define TTG_CALLABLE_TRAITS_COMMA_ELLIPSIS , ...
#define TTG_CALLABLE_TRAITS_PMF_VARARGS(CV, REF, OBJREF, NOEXCEPT) \
  TTG_CALLABLE_TRAITS_PMF_IMPL(CV, REF, OBJREF, NOEXCEPT, )        \
  TTG_CALLABLE_TRAITS_PMF_IMPL(CV, REF, OBJREF, NOEXCEPT, TTG_CALLABLE_TRAITS_COMMA_ELLIPSIS)
#define TTG_CALLABLE_TRAITS_PMF_NOEXCEPT(CV, REF, OBJREF) \
  TTG_CALLABLE_TRAITS_PMF_VARARGS(CV, REF, OBJREF, )      \
  TTG_CALLABLE_TRAITS_PMF_VARARGS(CV, REF, OBJREF, noexcept)
#define TTG_CALLABLE_TRAITS_PMF_REF(CV)    \
  TTG_CALLABLE_TRAITS_PMF_NOEXCEPT(CV, , &) \
  TTG_CALLABLE_TRAITS_PMF_NOEXCEPT(CV, &, &) \
  TTG_CALLABLE_TRAITS_PMF_NOEXCEPT(CV, &&, &&)

    TTG_CALLABLE_TRAITS_PMF_REF()
    TTG_CALLABLE_TRAITS_PMF_REF(const)
    TTG_CALLABLE_TRAITS_PMF_REF(volatile)
    TTG_CALLABLE_TRAITS_PMF_REF(const volatile)

#undef TTG_CALLABLE_TRAITS_PMF_REF
#undef TTG_CALLABLE_TRAITS_PMF_NOEXCEPT
#undef TTG_CALLABLE_TRAITS_PMF_VARARGS
#undef TTG_CALLABLE_TRAITS_COMMA_ELLIPSIS
#undef TTG_CALLABLE_TRAITS_PMF_IMPL

    /// traits of a function object given the type of `&T::operator()`; static operator() (C++23) yields a function
    /// pointer, whose arguments are used as is
    template <typename CallOpPtr, typename Enabler = void>
    struct function_object_traits : callable_traits_impl<CallOpPtr> {};
    // nonstatic operator(): drop the object argument
    template <typename CallOpPtr>
    struct function_object_traits<CallOpPtr, std::enable_if_t<std::is_member_function_pointer_v<CallOpPtr>>> {
      using return_type = typename callable_traits_impl<CallOpPtr>::return_type;
      using args = typename callable_traits_impl<CallOpPtr>::args_sans_object;
    };

    // function objects with a unique (non-template, non-overloaded) operator()
    template <typename T>
    struct callable_traits_impl<T, std::void_t<decltype(&T::operator())>>
        : function_object_traits<decltype(&T::operator())> {};
  }  // namespace detail

  /// introspects the signature of a nongeneric callable; for generic or overloaded callables it has no members
  /// \tparam Callable a function, (member) function pointer, function object, or reference to one of these
  template <typename Callable>
  using callable_traits = detail::callable_traits_impl<std::remove_cvref_t<Callable>>;

  /// argument types of a nongeneric \p Callable as a ttg::typelist; for member function pointers the first type
  /// is the reference to the object; SFINAE-friendly
  template <typename Callable>
  using callable_args_t = typename callable_traits<Callable>::args;

  /// return type of a nongeneric \p Callable; SFINAE-friendly
  template <typename Callable>
  using callable_return_t = typename callable_traits<Callable>::return_type;

  //////////////////////////////////////
  // nongeneric callables
  //////////////////////////////////////
  // to detect whether a callable is generic or not detect existence of callable_args_t
  template <typename Callable, typename = void>
  struct is_generic_callable : std::true_type {};

  template <typename Callable>
  struct is_generic_callable<Callable, ttg::meta::void_t<callable_args_t<Callable>>> : std::false_type {};

  template <typename Callable>
  constexpr inline bool is_generic_callable_v = is_generic_callable<Callable>::value;

  /// callable_args<Callable> detects whether `Callable` is generic or not, and in the latter case
  /// detects (using callable_traits) its return and argument types.
  /// callable_args<Callable> is a constexpr value of type
  /// `std::pair<bool,ttg::typelist<ttg::typelist<ReturnType>,ttg::typelist<ArgTypes>>>`
  /// where:
  /// - the boolean indicates whether `Callable` is generic (true) or not (false),
  /// - `ReturnType` is the return type of the `Callable` if it's generic, empty otherwise
  /// - `ArgTypes`  is the argument types of the `Callable` if it's generic, empty otherwise
  /// \tparam Callable a callable type
  template <typename Callable, typename Enabler = void>
  constexpr std::pair<bool, std::pair<ttg::typelist<>, ttg::typelist<>>> callable_args = {true, {}};

  template <typename Callable>
  constexpr auto callable_args<Callable, ttg::meta::void_t<callable_args_t<Callable>>> =
      std::pair<bool, std::pair<ttg::typelist<callable_return_t<Callable>>, callable_args_t<Callable>>>{false, {}};

  //////////////////////////////////////
  // generic callables
  //////////////////////////////////////

  //// Andrey's solution

  /// converts an ordinal to N-index
  /// @param ordinal the ordinal value
  /// @param extents extent of each mode
  template <std::size_t N>
  constexpr auto ordinal2index(std::size_t ordinal, std::array<std::size_t, N> extents) {
    std::array<std::size_t, N> idx = {};
    for (size_t d = 0; d < N; ++d) {
      idx[d] = ordinal % extents[d];
      ordinal /= extents[d];
    }
    return idx;
  }

  /// detects argument and return types of a generic callable (\p func)
  /// by trying each combination of types (\p argument_type_lists) for the respective arguments starting with
  /// the combination corresponding to the given \p Ordinal
  /// \tparam Ordinal a nonnegative integer specifying the ordinal of the type combination from \p Typelists to try;
  ///                 maximum value is `(std::tuple_size_v<Typelists> * ...)`
  /// \tparam Func a generic callable type
  /// \tparam Typelists a pack of ttg::typelist's each of which specifies candidate types for the respective
  ///         argument of \p Func
  /// \param func a generic callable
  /// \param argument_type_lists a ttg::typelist<Typelists...>
  /// @note iterates over \p argument_type_lists in "row-major" order (i.e. last list in \p argument_type_lists
  ///       is iterated first, etc.; the maxim
  /// @return an object of type `ttg::typelist<ttg::typelist<ReturnType>,ttg::typelist<ArgTypes>>`
  template <std::size_t Ordinal, typename Func, typename... Typelists, std::size_t... ArgIdx>
  auto compute_arg_binding_types_impl(Func& func, typelist<Typelists...> argument_type_lists,
                                      std::index_sequence<ArgIdx...> arg_idx = {}) {
    using arg_typelists_t = typelist<Typelists...>;
    constexpr auto Order = sizeof...(Typelists);
    constexpr std::array<std::size_t, Order> extents = {
        std::tuple_size_v<std::tuple_element_t<ArgIdx, arg_typelists_t>>...};
    constexpr auto tensor_size = (extents[ArgIdx] * ...);
    static_assert(tensor_size >= Ordinal);
    if constexpr (tensor_size == Ordinal) {
      return typelist<typelist<>, typelist<>>{};
    } else {
      constexpr auto idx = ordinal2index(Ordinal, extents);
      auto args = typelist<std::tuple_element_t<idx[ArgIdx], std::tuple_element_t<ArgIdx, arg_typelists_t>>...>{};
      using args_sans_void_t = drop_void_t<decltype(args)>;
      if constexpr (is_invocable_typelist_v<Func, args_sans_void_t>) {
        using return_type = invoke_result_typelist_t<Func, args_sans_void_t>;
        return ttg::typelist<ttg::typelist<return_type>, decltype(args)>{};
      } else {
        return compute_arg_binding_types_impl<Ordinal + 1>(func, argument_type_lists, arg_idx);
      }
    }
  }

  /// detects argument types of a generic callable (\p func)
  /// by trying each combination of types (\p argument_type_lists) for the respective arguments starting with
  /// the combination corresponding to the given \p Ordinal . The callable is expected to return \p ReturnType
  /// \tparam Ordinal a nonnegative integer specifying the ordinal of the type combination from \p Typelists to try;
  ///                 maximum value is `(std::tuple_size_v<Typelists> * ...)`
  /// \tparam ReturnType the expected return type of \p Func
  /// \tparam Func a generic callable type
  /// \tparam Typelists a pack of ttg::typelist's each of which specifies candidate types for the respective
  ///         argument of \p Func
  /// \param func a generic callable
  /// \param argument_type_lists a ttg::typelist<Typelists...>
  /// @note iterates over \p argument_type_lists in "row-major" order (i.e. last list in \p argument_type_lists
  ///       is iterated first, etc.; the maxim
  /// @return an object of type `ttg::typelist<ArgTypes>`
  template <std::size_t Ordinal, typename ReturnType, typename Func, typename... Typelists, std::size_t... ArgIdx>
  auto compute_arg_binding_types_r_impl(Func& func, typelist<Typelists...> argument_type_lists,
                                        std::index_sequence<ArgIdx...> arg_idx = {}) {
    using arg_typelists_t = typelist<Typelists...>;
    constexpr auto Order = sizeof...(Typelists);
    constexpr std::array<std::size_t, Order> extents = {
        std::tuple_size_v<std::tuple_element_t<ArgIdx, arg_typelists_t>>...};
    constexpr auto tensor_size = (extents[ArgIdx] * ...);
    static_assert(tensor_size >= Ordinal);
    if constexpr (tensor_size == Ordinal) {
      return typelist<>{};
    } else {
      constexpr auto idx = ordinal2index(Ordinal, extents);
      auto args = typelist<std::tuple_element_t<idx[ArgIdx], std::tuple_element_t<ArgIdx, arg_typelists_t>>...>{};
      if constexpr (is_invocable_typelist_r_v<ReturnType, Func, drop_void_t<decltype(args)>>) {
        return args;
      } else {
        return compute_arg_binding_types_r_impl<Ordinal + 1, ReturnType>(func, argument_type_lists, arg_idx);
      }
    }
  }

  /// @tparam Func a callable type
  /// @tparam Typelists a pack of typelists encoding how each argument can be invoked
  /// @param func a reference to callable of type @p Func
  /// @param argument_type_lists a list of possible types to try for each argument; can contain `void`
  /// @return a `ttg::typelist<ttg::typelist<ReturnType>,ttg::typelist<ArgsTypes>>` where
  ///          `ReturnType` is the return type of \p func and
  ///         `ArgsTypes` encodes:
  ///         - the exact argument bindings used by `Func`, if @p func is a nongeneric callable;
  ///         - the first invocable combination of argument types discovered by row-major iteration, if @p func is a
  ///         generic callable
  template <typename Func, typename... Typelists>
  auto compute_arg_binding_types(Func& func, typelist<Typelists...> argument_type_lists) {
    constexpr auto is_generic__args = callable_args<Func&>;
    constexpr bool is_generic = is_generic__args.first;
    if constexpr (is_generic) {
      return compute_arg_binding_types_impl<0>(func, argument_type_lists,
                                               std::make_index_sequence<sizeof...(Typelists)>{});
    } else {
      return is_generic__args.second;
    }
  }

  /// @tparam ReturnType a type expected to be returned by @p Func
  /// @tparam Func a callable type
  /// @tparam Typelists a pack of typelists encoding how each argument can be invoked
  /// @param func a reference to callable of type @p Func
  /// @param argument_type_lists a list of possible types to try for each argument; can contain `void`
  /// @return a ttg::typelist encoding:
  ///         - the exact argument bindings used by `Func`, if @p func is a nongeneric callable;
  ///         - the first invocable combination of argument types discovered by row-major iteration, if @p func is a
  ///         generic callable
  template <typename ReturnType, typename Func, typename... Typelists>
  auto compute_arg_binding_types_r(Func& func, typelist<Typelists...> argument_type_lists) {
    constexpr auto is_generic__args = callable_args<Func&>;
    constexpr bool is_generic = is_generic__args.first;
    if constexpr (is_generic) {
      return compute_arg_binding_types_r_impl<0, ReturnType>(func, argument_type_lists,
                                                             std::make_index_sequence<sizeof...(Typelists)>{});
    } else {
      return is_generic__args.second.second;
    }
  }

  /// @tparam T a non-reference type
  /// metafunction converts T into a list of types via which T can be bound to a callable
  template <typename T, typename = void>
  struct candidate_argument_bindings;

  template <typename T>
  struct candidate_argument_bindings<T, std::enable_if_t<!std::is_reference_v<T> && !std::is_void_v<T>>> {
    using type = std::conditional_t<std::is_const_v<T>, typelist<const T&>,
                                    typelist<
                                        // RATIONALE for this order of binding detection tries:
                                        // - to be able to distinguish arguments declared as auto& vs auto&& should try
                                        //   binding to T&& first since auto& won't bind to it
                                        // - HOWEVER argument declared as const T& will bind to either T&& or const T&,
                                        //   so this order will detect such argument as binding to T&&, which will
                                        //   indicate to the runtime that the argument is CONSUMABLE and may cause
                                        //   creation of extra copies. Thus you should not try to use nongeneric
                                        //   data arguments in generic task functions; for purely nongeneric functions
                                        //   a different introspection mechanism (callable_traits) is used
                                        T&&, const T&
                                        // - no need to check T& since auto& and auto&& both bind to it
                                        //, T&
                                        >>;
  };

  template <>
  struct candidate_argument_bindings<void, void> {
    using type = typelist<>;
  };

  template <>
  struct candidate_argument_bindings<const void, void> {
    using type = typelist<>;
  };

  template <typename T>
  using candidate_argument_bindings_t = typename candidate_argument_bindings<T>::type;
}  // namespace ttg::meta

#endif  // TTG_META_CALLABLE_H
