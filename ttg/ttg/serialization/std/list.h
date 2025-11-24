// SPDX-License-Identifier: BSD-3-Clause
//
// Created by Eduard Valeyev on 5/11/21.
//

#ifndef TTG_SERIALIZATION_STD_LIST_H
#define TTG_SERIALIZATION_STD_LIST_H

#include "ttg/serialization/std/allocator.h"
#include "ttg/serialization/traits.h"

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
// MADNESS supports std::list serialization by default
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
#include <boost/serialization/list.hpp>

namespace ttg::detail {
  template <typename Archive, typename T, typename A>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, std::list<T, A>> =
      is_boost_serializable_v<Archive, T>&& is_boost_serializable_v<Archive, A>;
  template <typename Archive, typename T, std::size_t N>
  inline static constexpr bool is_stlcontainer_boost_serializable_v<Archive, const std::list<T, A>> =
      is_boost_serializable_v<Archive, const T>&& is_boost_serializable_v<Archive, const A>;
}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#endif  // TTG_SERIALIZATION_STD_LIST_H
