//
// Created by Eduard Valeyev on 5/3/21.
//

#ifndef TTG_SERIALIZATION_CEREAL_H
#define TTG_SERIALIZATION_CEREAL_H

#include <type_traits>

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/details/helpers.hpp>
#include <cereal/details/traits.hpp>
#endif

namespace ttg::detail {

  //////// is_cereal_serializable

  template <typename Archive, typename T, class = void>
  struct is_cereal_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
  template <typename Archive, typename T>
  struct is_cereal_serializable<Archive, T,
                                std::enable_if_t<cereal::traits::is_output_serializable<T, Archive>::value ||
                                                 cereal::traits::is_input_serializable<T, Archive>::value>>
      : std::true_type {};
#endif  // TTG_SERIALIZATION_SUPPORTS_CEREAL

  template <typename Archive, typename T>
  inline static constexpr bool is_cereal_serializable_v = is_cereal_serializable<Archive, T>::value;

  template <typename T, class = void>
  struct is_cereal_buffer_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
  template <typename T>
  struct is_cereal_buffer_serializable<T, std::enable_if_t<is_cereal_serializable_v<cereal::BinaryInputArchive, T> &&
                                                           is_cereal_serializable_v<cereal::BinaryOutputArchive, T>>>
      : std::true_type {};
#endif  // TTG_SERIALIZATION_SUPPORTS_CEREAL

  /// evaluates to true if can serialize @p T to/from buffer using Cereal serialization
  template <typename T>
  inline constexpr bool is_cereal_buffer_serializable_v = is_cereal_buffer_serializable<T>::value;

  template <typename Archive, typename T, typename Enabler = void>
  struct is_cereal_array_serializable;

  template <typename Archive, typename T>
  struct is_cereal_array_serializable<Archive, T, std::enable_if_t<!std::is_array<T>::value>> : std::false_type {};

  template <typename Archive, typename T>
  struct is_cereal_array_serializable<Archive, T, std::enable_if_t<std::is_array<T>::value>>
      : std::bool_constant<is_cereal_serializable_v<Archive, std::remove_extent_t<T>>> {};

  template <typename Archive, typename T>
  inline static constexpr bool is_cereal_array_serializable_v = is_cereal_array_serializable<Archive, T>::value;

  template <typename Archive, typename T>
  inline static constexpr bool is_stlcontainer_cereal_serializable_v = false;

  template <typename Archive, typename T, class = void>
  struct is_cereal_user_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
  template <typename Archive, typename T>
  struct is_cereal_user_serializable<
      Archive, T,
      std::enable_if_t<(cereal::traits::detail::count_specializations<T, Archive>::value != 0) ||
                       ((cereal::traits::is_input_serializable<T, Archive>::value ||
                         cereal::traits::is_output_serializable<T, Archive>::value) &&
                        (!std::is_arithmetic_v<T> && !ttg::detail::is_cereal_array_serializable_v<Archive, T> &&
                         !is_stlcontainer_cereal_serializable_v<Archive, T>))>> : std::true_type {};
#endif

  template <typename Archive, typename T>
  inline constexpr bool is_cereal_user_serializable_v = is_cereal_user_serializable<Archive, T>::value;

  template <typename T, class = void>
  struct is_cereal_user_buffer_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
  template <typename T>
  struct is_cereal_user_buffer_serializable<
      T, std::enable_if_t<is_cereal_user_serializable_v<cereal::BinaryInputArchive, T> ||
                          is_cereal_user_serializable_v<cereal::BinaryOutputArchive, T>>> : std::true_type {};
#endif  // TTG_SERIALIZATION_SUPPORTS_CEREAL

  /// evaluates to true if can serialize @p T to/from buffer using user-provided Cereal serialization
  template <typename T>
  inline constexpr bool is_cereal_user_buffer_serializable_v = is_cereal_user_buffer_serializable<T>::value;

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_CEREAL_H
