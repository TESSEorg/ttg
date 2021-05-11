//
// Created by Eduard Valeyev on 5/3/21.
//

#ifndef TTG_SERIALIZATION_BOOST_H
#define TTG_SERIALIZATION_BOOST_H

#include <type_traits>

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/level.hpp>
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#include "ttg/serialization/traits.h"

namespace ttg::detail {

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename T>
  inline constexpr bool is_boost_archive_v = std::is_base_of_v<boost::archive::detail::basic_iarchive, T> ||
                                             std::is_base_of_v<boost::archive::detail::basic_oarchive, T>;
  template <typename T>
  inline constexpr bool is_boost_input_archive_v = std::is_base_of_v<boost::archive::detail::basic_iarchive, T>;

  template <typename T>
  inline constexpr bool is_boost_output_archive_v = std::is_base_of_v<boost::archive::detail::basic_oarchive, T>;

  //////// is_archive_v for boost archives
  template <typename T>
  inline constexpr bool is_archive_v<T, std::enable_if_t<is_boost_archive_v<T>>> = true;
  template <typename T>
  inline constexpr bool is_input_archive_v<T, std::enable_if_t<is_boost_input_archive_v<T>>> = true;
  template <typename T>
  inline constexpr bool is_output_archive_v<T, std::enable_if_t<is_boost_output_archive_v<T>>> = true;

#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

  //////// is_boost_serializable

  template <typename Archive, typename T, typename Enabler = void>
  inline static constexpr bool is_boost_serializable_v = false;

  template <typename Archive, typename T, typename Enabler = void>
  struct is_boost_array_serializable;

  template <typename Archive, typename T>
  struct is_boost_array_serializable<Archive, T, std::enable_if_t<!boost::is_array<T>::value>> : std::false_type {};

  template <typename Archive, typename T>
  struct is_boost_array_serializable<Archive, T, std::enable_if_t<boost::is_array<T>::value>>
      : std::bool_constant<is_boost_serializable_v<Archive, boost::remove_extent_t<T>>> {};

  template <typename Archive, typename T>
  inline static constexpr bool is_boost_array_serializable_v = is_boost_array_serializable<Archive, T>::value;

  template <typename Archive, typename T>
  inline static constexpr bool is_stlcontainer_boost_serializable_v = false;

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  /// @warning this will report user-defined `T` as not serializable if it has a free-standing serialize function in
  ///          namespace `boost::serialization` because Boost.Serialization library defines such a function without any
  ///          concept checks. To work around, either use intrusive serialization or sure that your
  ///          free-standing serialize function is discoverable by ADL.
  template <typename Archive, typename T>
  inline static constexpr bool is_boost_serializable_v<
      Archive, T,
      std::enable_if_t<
          // Archive is a boost archive
          is_boost_archive_v<Archive>
          // T is not not_serializable
          && !std::is_same_v<typename boost::serialization::implementation_level<T>::type,
                             boost::mpl::int_<boost::serialization::level_type::not_serializable>>
          // T is primitive or T is an array of serializables or else T has serialize methods
          && (std::is_same_v<typename boost::serialization::implementation_level<T>::type,
                             boost::mpl::int_<boost::serialization::level_type::primitive_type>> ||
              is_boost_array_serializable_v<Archive, T> ||
              (!std::is_same_v<typename boost::serialization::implementation_level<T>::type,
                               boost::mpl::int_<boost::serialization::level_type::primitive_type>> &&
               (ttg::detail::has_freestanding_serialize_with_version_v<ttg::meta::remove_cvr_t<T>, Archive> ||
                (ttg::detail::is_stlcontainer_boost_serializable_v<Archive, T> &&
                 ttg::detail::has_freestanding_boost_serialize_with_version_v<ttg::meta::remove_cvr_t<T>, Archive>) ||
                ttg::detail::has_member_serialize_with_version_v<ttg::meta::remove_cvr_t<T>, Archive> ||
                (ttg::detail::has_member_load_with_version_v<T, Archive> &&
                 ttg::detail::has_member_store_with_version_v<T, Archive>))))>> = true;
#endif  //  TTG_SERIALIZATION_SUPPORTS_BOOST

  template <typename Archive, typename T>
  struct is_boost_serializable : std::bool_constant<is_boost_serializable_v<Archive, T>> {};

  template <typename Archive, typename T, class = void>
  struct is_boost_default_serializable : std::bool_constant<std::is_trivially_copyable_v<T>> {};

  template <typename Archive, typename T>
  inline static constexpr bool is_boost_default_serializable_v = is_boost_default_serializable<Archive, T>::value;

  template <typename T, class = void>
  struct is_boost_buffer_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename T>
  struct is_boost_buffer_serializable<T, std::enable_if_t<is_boost_serializable_v<boost::archive::binary_iarchive, T> &&
                                                          is_boost_serializable_v<boost::archive::binary_oarchive, T>>>
      : std::true_type {};
#endif  //  TTG_SERIALIZATION_SUPPORTS_BOOST

  /// evaluates to true if can serialize @p T to/from buffer using Boost serialization
  template <typename T>
  inline constexpr bool is_boost_buffer_serializable_v = is_boost_buffer_serializable<T>::value;

  template <typename T, class = void>
  struct is_boost_default_buffer_serializable : std::false_type {};

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename T>
  struct is_boost_default_buffer_serializable<
      T, std::enable_if_t<is_boost_default_serializable_v<boost::archive::binary_iarchive, T> &&
                          is_boost_default_serializable_v<boost::archive::binary_oarchive, T>>> : std::true_type {};
#endif  //  TTG_SERIALIZATION_SUPPORTS_BOOST

  /// evaluates to true if can serialize @p T to/from buffer using default Boost serialization
  template <typename T>
  inline constexpr bool is_boost_default_buffer_serializable_v = is_boost_default_buffer_serializable<T>::value;

  /// evaluates to true if can serialize @p T to/from buffer using user-provided Boost serialization
  template <typename T>
  inline constexpr bool is_boost_user_buffer_serializable_v =
      is_boost_buffer_serializable<T>::value && !is_boost_default_buffer_serializable_v<T>;

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_BOOST_H
