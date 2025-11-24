// SPDX-License-Identifier: BSD-3-Clause
//
// Created by Eduard Valeyev on 5/11/21.
//

#ifndef TTG_SERIALIZATION_STD_ALLOCATOR_H
#define TTG_SERIALIZATION_STD_ALLOCATOR_H

#include "ttg/serialization/traits.h"

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST

namespace ttg::detail {
  template <typename Archive, typename T>
  inline static constexpr bool is_boost_serializable_v<Archive, std::allocator<T>> = is_boost_archive_v<Archive>;
  template <typename Archive, typename T>
  inline static constexpr bool is_boost_serializable_v<Archive, const std::allocator<T>> = is_boost_archive_v<Archive>;
}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#endif  // TTG_SERIALIZATION_STD_ALLOCATOR_H
