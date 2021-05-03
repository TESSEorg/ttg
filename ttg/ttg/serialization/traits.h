#ifndef TTG_SERIALIZATION_TRAITS_H
#define TTG_SERIALIZATION_TRAITS_H

#include <type_traits>

#include <iostream>

namespace ttg::detail {

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
