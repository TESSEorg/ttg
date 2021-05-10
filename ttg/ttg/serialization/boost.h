//
// Created by Eduard Valeyev on 5/3/21.
//

#ifndef TTG_SERIALIZATION_BOOST_H
#define TTG_SERIALIZATION_BOOST_H

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

namespace ttg::detail {

  //////// is_boost_serializable

  template <typename Archive, typename T, class = void>
  struct is_boost_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename Archive, typename T>
  struct is_boost_serializable<
      Archive, T,
      std::void_t<decltype(std::declval<Archive&>() & std::declval<T&>()),
                  std::enable_if<std::is_base_of_v<boost::archive::detail::basic_iarchive, Archive> ||
                                 std::is_base_of_v<boost::archive::detail::basic_oarchive, Archive>>>>
      : std::true_type {};
#endif  //  TTG_SERIALIZATION_SUPPORTS_BOOST

  template <typename Archive, typename T>
  inline static constexpr bool is_boost_serializable_v = is_boost_serializable<Archive, T>::value;

  template <typename T, class = void>
  struct is_boost_buffer_serializable
      : std::bool_constant<is_boost_serializable_v<boost::archive::binary_iarchive, T> &&
                           is_boost_serializable_v<boost::archive::binary_oarchive, T>> {};

  /// evaluates to true if can serialize @p T to/from buffer using Boost serialization
  template <typename T>
  inline constexpr bool is_boost_buffer_serializable_v = is_boost_buffer_serializable<T>::value;

  template <typename T, class = void>
  struct is_boost_user_buffer_serializable : is_boost_buffer_serializable<T> {};

  /// evaluates to true if can serialize @p T to/from buffer using user-provided Boost serialization
  template <typename T>
  inline constexpr bool is_boost_user_buffer_serializable_v =
      is_boost_user_buffer_serializable<T>::value && !std::is_fundamental_v<T>;

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_BOOST_H
