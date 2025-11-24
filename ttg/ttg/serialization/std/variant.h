// SPDX-License-Identifier: BSD-3-Clause
//
// Created by Eduard Valeyev on 6/22/21.
//

#ifndef TTG_SERIALIZATION_STD_VARIANT_H
#define TTG_SERIALIZATION_STD_VARIANT_H

#include "ttg/serialization/traits.h"

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
// MADNESS does not supports std::variant serialization
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST

namespace boost {
  namespace serialization {

    namespace detail {

      template <typename Archive, typename... Ts, std::size_t I0, std::size_t... Is>
      Archive& variant_load_impl(Archive& ar, std::variant<Ts...>& v, std::size_t which,
                                 std::index_sequence<I0, Is...>) {
        constexpr bool writing = ttg::detail::is_output_archive_v<Archive>;
        static_assert(!writing);
        if (which == I0) {
          using type = std::variant_alternative_t<I0, std::variant<Ts...>>;
          if (!std::is_same_v<type, std::monostate>) {
            type value;
            ar& value;
            v.template emplace<I0>(std::move(value));
          }
        } else {
          if constexpr (sizeof...(Is) == 0)
            throw std::logic_error(
                "boost::serialization::detail::variant_load_impl(ar,v,idx,idxs): idx is not present in idxs");
          else
            return variant_load_impl(ar, v, which, std::index_sequence<Is...>{});
        }
        return ar;
      }

    }  // namespace detail

    template <typename Archive, typename... Ts>
    Archive& serialize(Archive& ar, std::variant<Ts...>& t, const unsigned int version) {
      constexpr bool writing = ttg::detail::is_output_archive_v<Archive>;
      const auto index = t.index();
      ar& index;
      // to write visit the current alternative
      if constexpr (writing) {
        std::visit(
            [&ar](const auto& v) {
              if constexpr (!std::is_same_v<std::decay_t<decltype(v)>, std::monostate>) ar& v;
            },
            t);
      } else  // reading by recursive traversal until found index
        detail::variant_load_impl(ar, t, index, std::make_index_sequence<sizeof...(Ts)>{});
      return ar;
    }

  }  // namespace serialization
}  // namespace boost

namespace ttg::detail {
  template <typename Archive, typename... Ts>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, std::variant<Ts...>> =
      (is_boost_serializable_v<Archive, Ts> && ...);
  template <typename Archive, typename... Ts>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, const std::variant<Ts...>> =
      (is_boost_serializable_v<Archive, const Ts> && ...);
}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#endif  // TTG_SERIALIZATION_STD_TUPLE_H
