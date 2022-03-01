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
  auto compute_arg_binding_types_impl(const Func& func, typelist<Typelists...> argument_type_lists,
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
      if constexpr (is_invocable_typelist_v<Func, decltype(args)>) {
        return args;
      } else {
        return compute_arg_binding_types_impl<Ordinal + 1>(func, argument_type_lists, arg_idx);
      }
    }
  }

  template <typename Func, typename... Typelists>
  auto compute_arg_binding_types(const Func& func, typelist<Typelists...> argument_type_lists) {
    return compute_arg_binding_types_impl<0>(func, argument_type_lists,
                                             std::make_index_sequence<sizeof...(Typelists)>{});
  }

}  // namespace ttg::meta

#endif  // TTG_META_CALLABLE_H
