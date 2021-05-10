//
// Created by Eduard Valeyev on 5/3/21.
//

#ifndef TTG_SERIALIZATION_CEREAL_H
#define TTG_SERIALIZATION_CEREAL_H

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {
  //! Saving types to binary
  template <class T>
  inline typename std::enable_if<!std::is_arithmetic<T>::value && std::is_trivially_copyable<T>::value, void>::type
  CEREAL_SAVE_FUNCTION_NAME(BinaryOutputArchive& ar, T const& t) {
    ar.saveBinary(std::addressof(t), sizeof(t));
  }

  //! Loading POD types from binary
  template <class T>
  inline typename std::enable_if<!std::is_arithmetic<T>::value && std::is_trivially_copyable<T>::value, void>::type
  CEREAL_LOAD_FUNCTION_NAME(BinaryInputArchive& ar, T& t) {
    ar.loadBinary(std::addressof(t), sizeof(t));
  }

  //! Saving static-sized array of serializable types
  template <class Archive, class T, std::size_t N>
  inline typename std::enable_if<cereal::traits::is_output_serializable<T, Archive>::value, void>::type
  CEREAL_SAVE_FUNCTION_NAME(Archive& ar, T const (&t)[N]) {
    for (std::size_t i = 0; i != N; ++i) ar(t[i]);
  }

  //! Loading for POD types from binary
  template <class Archive, class T, std::size_t N>
  inline typename std::enable_if<cereal::traits::is_input_serializable<T, Archive>::value, void>::type
  CEREAL_LOAD_FUNCTION_NAME(Archive& ar, T (&t)[N]) {
    for (std::size_t i = 0; i != N; ++i) ar(t[i]);
  }
}  // namespace cereal

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
  struct is_cereal_buffer_serializable : std::bool_constant<is_cereal_serializable_v<cereal::BinaryInputArchive, T> &&
                                                            is_cereal_serializable_v<cereal::BinaryOutputArchive, T>> {
  };

  /// evaluates to true if can serialize @p T to/from buffer using Cereal serialization
  template <typename T>
  inline constexpr bool is_cereal_buffer_serializable_v = is_cereal_buffer_serializable<T>::value;

  template <typename T, class = void>
  struct is_cereal_user_buffer_serializable : is_cereal_buffer_serializable<T> {};

  /// evaluates to true if can serialize @p T to/from buffer using user-provided Cereal serialization
  template <typename T>
  inline constexpr bool is_cereal_user_buffer_serializable_v =
      is_cereal_user_buffer_serializable<T>::value && !std::is_fundamental_v<T>;

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_SUPPORTS_CEREAL

#endif  // TTG_SERIALIZATION_CEREAL_H
