//
// Created by Eduard Valeyev on 5/3/21.
//

#ifndef TTG_SERIALIZATION_MADNESS_H
#define TTG_SERIALIZATION_MADNESS_H

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
#include <madness/world/archive.h>
#include <madness/world/type_traits.h>
#endif

namespace ttg::detail {

  //////// is_madness_serializable

  template <typename Archive, typename T, class = void>
  struct is_madness_output_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive, typename T>
  struct is_madness_output_serializable<
      Archive, T,
      std::void_t<decltype(madness::archive::ArchiveStoreImpl<Archive, T>::store(std::declval<Archive&>(),
                                                                                 std::declval<const T&>())),
                  std::enable_if<madness::archive::is_output_archive<Archive>::value>>> : std::true_type {};
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

  template <typename Archive, typename T>
  inline static constexpr bool is_madness_output_serializable_v = is_madness_output_serializable<Archive, T>::value;

  template <typename Archive, typename T, class = void>
  struct is_madness_input_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive, typename T>
  struct is_madness_input_serializable<Archive, T,
                                       std::void_t<decltype(madness::archive::ArchiveLoadImpl<Archive, T>::load(
                                                       std::declval<Archive&>(), std::declval<T&>())),
                                                   std::enable_if<madness::archive::is_input_archive<Archive>::value>>>
      : std::true_type {};
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

  template <typename Archive, typename T>
  inline static constexpr bool is_madness_input_serializable_v = is_madness_input_serializable<Archive, T>::value;

  template <typename Archive, typename T>
  inline static constexpr bool is_madness_serializable_v =
      is_madness_input_serializable_v<Archive, T> || is_madness_output_serializable_v<Archive, T>;

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_MADNESS_H
