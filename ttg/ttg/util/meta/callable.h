//
// Created by Eduard Valeyev on 3/1/22.
//

#ifndef TTG_META_CALLABLE_H
#define TTG_META_CALLABLE_H

#include "ttg/util/meta.h"
#include "ttg/util/typelist.h"

namespace ttg::meta {

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
      return typelist<>{};
    } else {
      constexpr auto idx = ordinal2index(Ordinal, extents);
      auto args = typelist<std::tuple_element_t<idx[ArgIdx], std::tuple_element_t<ArgIdx, arg_typelists_t>>...>{};
      if constexpr (is_invocable_typelist_v<Func, drop_void_t<decltype(args)>>) {
        return args;
      } else {
        return compute_arg_binding_types_impl<Ordinal + 1>(func, argument_type_lists, arg_idx);
      }
    }
  }

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
  /// @return a ttg::typelist encoding the first invocable combination of argument types discovered by
  ///         row-major iteration
  template <typename Func, typename... Typelists>
  auto compute_arg_binding_types(Func& func, typelist<Typelists...> argument_type_lists) {
    return compute_arg_binding_types_impl<0>(func, argument_type_lists,
                                             std::make_index_sequence<sizeof...(Typelists)>{});
  }

  /// @tparam ReturnType a type expected to be returned by @p Func
  /// @tparam Func a callable type
  /// @tparam Typelists a pack of typelists encoding how each argument can be invoked
  /// @param func a reference to callable of type @p Func
  /// @param argument_type_lists a list of possible types to try for each argument; can contain `void`
  /// @return a ttg::typelist encoding the first invocable combination of argument types discovered by
  ///         row-major iteration
  template <typename ReturnType, typename Func, typename... Typelists>
  auto compute_arg_binding_types_r(Func& func, typelist<Typelists...> argument_type_lists) {
    return compute_arg_binding_types_r_impl<0, ReturnType>(func, argument_type_lists,
                                                           std::make_index_sequence<sizeof...(Typelists)>{});
  }

  /// @tparam T a non-reference type
  /// metafunction converts T into a list of types via which T can be bound to a callable
  template <typename T, typename = void>
  struct candidate_argument_bindings;

  template <typename T>
  struct candidate_argument_bindings<T, std::enable_if_t<!std::is_reference_v<T> && !std::is_void_v<T>>> {
    using type = std::conditional_t<std::is_const_v<T>, typelist<const T&>,
                                    typelist<const T&, T&&,
                                             // check for T& to be able to detect (erroneous) passing non-const lvalue&
                                             T&>>;
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
