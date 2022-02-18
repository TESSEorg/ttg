#ifndef TTG_UTIL_INPUTTYPES_H
#define TTG_UTIL_INPUTTYPES_H

#include <tuple>

namespace ttg {

  /**
   * \brief A container for types.
   *
   * We use this to work around ADL issues when templating ttg::TT with
   * std::tuple. This is a simple wrapper type holding type information.
   * A tuple containing the types can be extracted using the \c tuple_type
   * member type.
   */
  template<typename... Ts>
  struct typelist
  { };

  namespace detail {

    template<typename T>
    struct is_typelist : std::false_type
    { };

    template<typename... Ts>
    struct is_typelist<ttg::typelist<Ts...>> : std::true_type
    { };

    template<typename T>
    constexpr bool is_typelist_v = is_typelist<T>::value;

    template<typename T>
    struct typelist_to_tuple;

    template<typename... T>
    struct typelist_to_tuple<ttg::typelist<T...>> {
      using type = std::tuple<T...>;
    };

    template<typename T>
    using typelist_to_tuple_t = typename typelist_to_tuple<T>::type;

  } // namespace detail

} // namespace ttg

#endif // TTG_UTIL_INPUTTYPES_H
