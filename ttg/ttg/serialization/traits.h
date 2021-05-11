#ifndef TTG_SERIALIZATION_TRAITS_H
#define TTG_SERIALIZATION_TRAITS_H

#include "ttg/serialization/all.h"

#include <type_traits>

#include <iostream>

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

#endif  // TTG_SERIALIZATION_TRAITS_H
