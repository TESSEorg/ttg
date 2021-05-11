#ifndef TTG_SERIALIZATION_TRAITS_H
#define TTG_SERIALIZATION_TRAITS_H

#include "ttg/util/meta.h"

#include <type_traits>

#include <iostream>

namespace boost {
  namespace archive {}
  namespace serialization {
    struct unsigned_int {
      unsigned int v;
      unsigned_int(unsigned int v) : v(v) {}
      operator unsigned int() const { return v; }
    };
    const unsigned_int version_as_adl_tag = 0u;
  }  // namespace serialization
}  // namespace boost

namespace ttg::detail {

  /// helps to detect that `T` has a member serialization method that
  /// accepts single argument of type `Archive`
  /// @note use in combination with `ttg::meta`::is_detected_v
  template <typename T, typename Archive>
  using has_member_serialize_t = decltype(std::declval<T&>().serialize(std::declval<Archive&>()));

  /// helps to detect that `T` has a member serialization method that
  /// accepts one argument of type `Archive` and an unsigned version
  /// @note use in combination with ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_member_serialize_with_version_t = decltype(std::declval<T&>().serialize(std::declval<Archive&>(), 0u));

  /// helps to detect that `T` has a member serialization method that
  /// accepts single argument of type `Archive`
  /// @note use in combination with ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_member_load_t = decltype(std::declval<T&>().load(std::declval<Archive&>()));

  /// helps to detect that `T` has a member serialization method that
  /// accepts one argument of type `Archive` and an unsigned version
  /// @note use in combination with ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_member_load_with_version_t = decltype(std::declval<T&>().load(std::declval<Archive&>(), 0u));

  /// helps to detect that `T` has a member serialization method that
  /// accepts single argument of type `Archive`
  /// @note use in combination with ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_member_store_t = decltype(std::declval<T&>().store(std::declval<Archive&>()));

  /// helps to detect that `T` has a member serialization method that
  /// accepts one argument of type `Archive` and an unsigned version
  /// @note use in combination with ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_member_store_with_version_t = decltype(std::declval<T&>().store(std::declval<Archive&>(), 0u));

  /// helps to detect that `T` supports freestanding `serialize` function discoverable by ADL
  /// @note use in combination with std::is_detected_v or ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_freestanding_serialize_t = decltype(serialize(std::declval<Archive&>(), std::declval<T&>()));

  /// helps to detect that `T` supports freestanding `serialize` function discoverable by ADL that accepts version
  /// @note use in combination with ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_freestanding_serialize_with_version_t =
      decltype(serialize(std::declval<Archive&>(), std::declval<T&>(), 0u));

  /// helps to detect that `T` supports freestanding `boost::serialization::serialize` function that accepts version
  /// @note use in combination with ttg::meta::is_detected_v
  template <typename T, typename Archive>
  using has_freestanding_boost_serialize_with_version_t =
      decltype(serialize(std::declval<Archive&>(), std::declval<T&>(), boost::serialization::version_as_adl_tag));

  /// true if this is well-formed:
  /// \code
  ///   // T t; Archive ar;
  ///   t.serialize(ar);
  /// \endcode
  template <typename T, typename Archive>
  inline constexpr bool has_member_serialize_v = ttg::meta::is_detected_v<has_member_serialize_t, T, Archive>;

  /// true if this is well-formed:
  /// \code
  ///   // T t; Archive ar;
  ///   t.serialize(ar, 0u);
  /// \endcode
  template <typename T, typename Archive>
  inline constexpr bool has_member_serialize_with_version_v =
      ttg::meta::is_detected_v<has_member_serialize_with_version_t, T, Archive>;

  /// true if this is well-formed:
  /// \code
  ///   // T t; Archive ar;
  ///   t.load(ar, 0u);
  /// \endcode
  template <typename T, typename Archive>
  inline constexpr bool has_member_load_with_version_v =
      ttg::meta::is_detected_v<has_member_load_with_version_t, T, Archive>;

  /// true if this is well-formed:
  /// \code
  ///   // T t; Archive ar;
  ///   t.store(ar, 0u);
  /// \endcode
  template <typename T, typename Archive>
  inline constexpr bool has_member_store_with_version_v =
      ttg::meta::is_detected_v<has_member_store_with_version_t, T, Archive>;

  /// true if this is well-formed:
  /// \code
  ///   // T t; Archive ar;
  ///   serialize(ar, t);
  /// \endcode
  template <typename T, typename Archive>
  inline constexpr bool has_freestanding_serialize_v =
      ttg::meta::is_detected_v<has_freestanding_serialize_t, T, Archive>;

  /// true if this is well-formed:
  /// \code
  ///   // T t; Archive ar;
  ///   serialize(ar, t, 0u);
  /// \endcode
  template <typename T, typename Archive>
  inline constexpr bool has_freestanding_serialize_with_version_v =
      ttg::meta::is_detected_v<has_freestanding_serialize_with_version_t, T, Archive>;

  /// true if this is well-formed:
  /// \code
  ///   // T t; Archive ar;
  ///   boost::serialization::serialize(ar, t, 0u);
  /// \endcode
  template <typename T, typename Archive>
  inline constexpr bool has_freestanding_boost_serialize_with_version_v =
      ttg::meta::is_detected_v<has_freestanding_boost_serialize_with_version_t, T, Archive>;

  //////// is_{,input,output}_archive

  template <typename T, typename Enabler = void>
  inline constexpr bool is_archive_v = false;

  template <typename T, typename Enabler>
  inline constexpr bool is_input_archive_v = false;

  template <typename T, typename Enabler = void>
  inline constexpr bool is_output_archive_v = false;

  //////// is_printable

  template <class, class = void>
  struct is_printable : std::false_type {};

  template <class T>
  struct is_printable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>> : std::true_type {};

  template <typename T, typename Enabler = void>
  struct printer_helper {
    static void print(const void* object) { std::cout << "[unprintable object]" << std::endl; }
  };

  template <typename T>
  struct printer_helper<T, std::enable_if_t<is_printable<T>::value>> {
    static void print(const void* object) { std::cout << *(static_cast<const T*>(object)) << std::endl; }
  };

}  // namespace ttg::detail

#include "ttg/serialization/backends.h"

namespace ttg::detail {
  /// is_user_buffer_serializable<T> evaluates to true if `T` can be serialized to a buffer using user-provided methods
  template <typename T, typename Enabler = void>
  struct is_user_buffer_serializable : std::false_type {};

  template <typename T>
  struct is_user_buffer_serializable<T, std::enable_if_t<is_madness_user_buffer_serializable_v<T>>> : std::true_type {};
  //  template <typename T>
  //  struct is_user_buffer_serializable<T, std::enable_if_t<is_madness_user_buffer_serializable_v<T> ||
  //  is_boost_user_buffer_serializable_v<T> || is_cereal_user_buffer_serializable_v<T>>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_user_buffer_serializable_v = is_user_buffer_serializable<T>::value;

}  // namespace ttg::detail

#endif  // TTG_SERIALIZATION_TRAITS_H
