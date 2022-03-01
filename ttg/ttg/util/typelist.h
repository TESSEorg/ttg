#ifndef TTG_UTIL_INPUTTYPES_H
#define TTG_UTIL_INPUTTYPES_H

#include <tuple>

namespace ttg {

  namespace meta {

    template <typename T>
    struct type_identity {
      using type = T;
    };

    /**
     * \brief A container for types.
     *
     * We use this to work around ADL issues when templating ttg::TT with
     * std::tuple. This is a simple wrapper type holding type information.
     * A tuple containing the types can be extracted using the \c tuple_type
     * member type.
     */
    template <typename... Ts>
    struct typelist {
      /// @return the size of typelist
      constexpr auto size() const { return sizeof...(Ts); }

      template <std::size_t I>
      constexpr auto get() {
        return type_identity<std::tuple_element_t<I, std::tuple<Ts...>>>{};
      }
    };

    template <typename T>
    struct is_typelist : std::false_type {};

    template <typename... Ts>
    struct is_typelist<typelist<Ts...>> : std::true_type {};

    template <typename T>
    constexpr bool is_typelist_v = is_typelist<T>::value;

    template <typename T>
    struct typelist_to_tuple;

    template <typename... T>
    struct typelist_to_tuple<typelist<T...>> {
      using type = std::tuple<T...>;
    };

    template <typename T>
    using typelist_to_tuple_t = typename typelist_to_tuple<T>::type;

    template <typename T>
    struct typelist_size;

    template <typename... Ts>
    struct typelist_size<typelist<Ts...>> {
      constexpr static std::size_t value = sizeof...(Ts);
    };

    template <typename T>
    constexpr std::size_t typelist_size_v = typelist_size<T>::value;

    template <typename T>
    constexpr bool typelist_is_empty_v = (typelist_size_v<T> == 0);

  }  // namespace meta

  // typelist is user-centric API
  template <typename... Ts>
  using typelist = meta::typelist<Ts...>;

}  // namespace ttg

namespace std {

  template <typename... Ts>
  struct tuple_size<ttg::meta::typelist<Ts...>> {
    static constexpr auto value = sizeof...(Ts);
  };

  template <std::size_t I, typename... Ts>
  struct tuple_element<I, ttg::meta::typelist<Ts...>> {
    using type = typename decltype(ttg::meta::typelist<Ts...>{}.template get<I>())::type;
  };

}  // namespace std

namespace ttg::meta {
  template <std::size_t I, typename T, typename... RestOfTs>
  constexpr auto get(typelist<T, RestOfTs...>) {
    if constexpr (I == 0)
      return type_identity<T>{};
    else
      return get<I - 1>(typelist<RestOfTs...>{});
  }
}  // namespace ttg::meta

#endif  // TTG_UTIL_INPUTTYPES_H
