//
// Created by Eduard Valeyev on 5/11/21.
//

#ifndef TTG_SERIALIZATION_STD_TUPLE_H
#define TTG_SERIALIZATION_STD_TUPLE_H

#include "ttg/serialization/traits.h"

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
// MADNESS supports std::tuple serialization by default
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST

namespace boost {
  namespace serialization {

    namespace detail {

      template <typename Archive, typename... Ts, std::size_t... Is>
      Archive& tuple_serialize_impl(Archive& ar, std::tuple<Ts...>& t, std::index_sequence<Is...>) {
        ((ar & std::get<Is>(t)), ...);
        return ar;
      }

    }  // namespace detail

    template <typename Archive, typename... Ts>
    Archive& serialize(Archive& ar, std::tuple<Ts...>& t, const unsigned int version) {
      detail::tuple_serialize_impl(ar, t, std::make_index_sequence<sizeof...(Ts)>{});
      return ar;
    }

  }  // namespace serialization
}  // namespace boost

namespace ttg::detail {
  template <typename Archive, typename... Ts>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, std::tuple<Ts...>> =
      (is_boost_serializable_v<Archive, Ts> && ...);
  template <typename Archive, typename... Ts>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, const std::tuple<Ts...>> =
      (is_boost_serializable_v<Archive, const Ts> && ...);
}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
#include <cereal/types/tuple.hpp>

namespace ttg::detail {
  template <typename Archive, typename... Ts>
  inline static constexpr bool is_stlcontainer_cereal_serializable_v<Archive, std::tuple<Ts...>> =
      (is_cereal_serializable_v<Archive, Ts> && ...);
  template <typename Archive, typename... Ts>
  inline static constexpr bool is_stlcontainer_cereal_serializable_v<Archive, const std::tuple<Ts...>> =
      (is_cereal_serializable_v<Archive, const Ts> && ...);
}  // namespace ttg::detail

#endif

#endif  // TTG_SERIALIZATION_STD_TUPLE_H
