//
// Created by Eduard Valeyev on 5/3/21.
//

#ifndef TTG_SERIALIZATION_MADNESS_H
#define TTG_SERIALIZATION_MADNESS_H

#include <type_traits>

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
#include <madness/world/archive.h>
#include <madness/world/type_traits.h>
#endif

namespace ttg::detail {

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  //////// is_archive_v for madness archives
  template <typename T>
  inline constexpr bool is_archive_v<T, std::enable_if_t<madness::is_archive_v<T>>> = true;
  template <typename T>
  inline constexpr bool is_input_archive_v<T, std::enable_if_t<madness::is_input_archive_v<T>>> = true;
  template <typename T>
  inline constexpr bool is_output_archive_v<T, std::enable_if_t<madness::is_output_archive_v<T>>> = true;
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

  //////// is_madness_serializable

  template <typename Archive, typename T, class = void>
  struct is_madness_output_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive, typename T>
  struct is_madness_output_serializable<
      Archive, T, std::enable_if_t<madness::is_output_archive_v<Archive> && madness::is_serializable_v<Archive, T>>>
      : std::true_type {};
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

  template <typename Archive, typename T>
  inline static constexpr bool is_madness_output_serializable_v = is_madness_output_serializable<Archive, T>::value;

  template <typename Archive, typename T, class = void>
  struct is_madness_input_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive, typename T>
  struct is_madness_input_serializable<
      Archive, T, std::enable_if_t<madness::is_input_archive_v<Archive> && madness::is_serializable_v<Archive, T>>>
      : std::true_type {};
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

  template <typename Archive, typename T>
  inline static constexpr bool is_madness_input_serializable_v = is_madness_input_serializable<Archive, T>::value;

  template <typename Archive, typename T>
  inline static constexpr bool is_madness_serializable_v =
      is_madness_input_serializable_v<Archive, T> || is_madness_output_serializable_v<Archive, T>;

  template <typename T, class = void>
  struct is_madness_buffer_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename T>
  struct is_madness_buffer_serializable<
      T, std::enable_if_t<is_madness_input_serializable_v<madness::archive::BufferInputArchive, T> &&
                          is_madness_output_serializable_v<madness::archive::BufferOutputArchive, T>>>
      : std::true_type {};
#endif

  /// evaluates to true if can serialize @p T to/from buffer using MADNESS serialization
  template <typename T>
  inline constexpr bool is_madness_buffer_serializable_v = is_madness_buffer_serializable<T>::value;

  template <typename T, class = void>
  struct is_madness_user_buffer_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename T>
  struct is_madness_user_buffer_serializable<
      T, std::enable_if_t<madness::is_user_serializable_v<madness::archive::BufferInputArchive, T> &&
                          madness::is_user_serializable_v<madness::archive::BufferOutputArchive, T>>> : std::true_type {
  };
#endif

  /// evaluates to true if can serialize @p T to/from buffer using user-provided MADNESS serialization
  template <typename T>
  inline constexpr bool is_madness_user_buffer_serializable_v = is_madness_user_buffer_serializable<T>::value;

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_MADNESS_H
