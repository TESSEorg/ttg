//
// Created by Eduard Valeyev on 5/11/21.
//

#ifndef TTG_SERIALIZATION_STD_PAIR_H
#define TTG_SERIALIZATION_STD_PAIR_H

#include "ttg/serialization/traits.h"

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
// MADNESS supports std::pair serialization by default
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
#include <boost/serialization/utility.hpp>

namespace ttg::detail {
  template <typename Archive, typename T1, typename T2>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, std::pair<T1, T2>> =
      is_boost_serializable_v<Archive, T1>&& is_boost_serializable_v<Archive, T2>;
  template <typename Archive, typename T1, typename T2>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, const std::pair<T1, T2>> =
      is_boost_serializable_v<Archive, const T1>&& is_boost_serializable_v<Archive, const T2>;
}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#endif  // TTG_SERIALIZATION_STD_PAIR_H
