#ifndef CXXAPI_META_H
#define CXXAPI_META_H

#include <type_traits>

namespace ttg {

  class Void;

  namespace meta {

#if __cplusplus >= 201703L
    using std::void_t;
#else
    template <class...>
    using void_t = void;
#endif

    template <typename Tuple, std::size_t N, typename Enabler = void>
    struct drop_first_n;

    template <typename... Ts>
    struct drop_first_n<std::tuple<Ts...>, std::size_t(0)> {
      using type = std::tuple<Ts...>;
    };

    template <typename T, typename... Ts, std::size_t N>
    struct drop_first_n<std::tuple<T, Ts...>, N, std::enable_if_t<N != 0>> {
      using type = typename drop_first_n<std::tuple<Ts...>, N - 1>::type;
    };

    template <typename ResultTuple, typename InputTuple, std::size_t N, typename Enabler = void>
    struct take_first_n_helper;

    template <typename... Ts, typename... Us>
    struct take_first_n_helper<std::tuple<Ts...>, std::tuple<Us...>, std::size_t(0)> {
      using type = std::tuple<Ts...>;
    };

    template <typename... Ts, typename U, typename... Us, std::size_t N>
    struct take_first_n_helper<std::tuple<Ts...>, std::tuple<U, Us...>, N, std::enable_if_t<N != 0>> {
      using type = typename take_first_n_helper<std::tuple<Ts..., U>, std::tuple<Us...>, N - 1>::type;
    };

    template <typename Tuple, std::size_t N>
    struct take_first_n {
      using type = typename take_first_n_helper<std::tuple<>, Tuple, N>::type;
    };

    // tuple<Ts...> -> tuple<std::remove_reference_t<Ts>...>
    template <typename T, typename Enabler = void>
    struct nonref_tuple;

    template <typename... Ts>
    struct nonref_tuple<std::tuple<Ts...>> {
      using type = std::tuple<typename std::remove_reference<Ts>::type...>;
    };

    template <typename Tuple>
    using nonref_tuple_t = typename nonref_tuple<Tuple>::type;

    // tuple<Ts...> -> tuple<std::decay_t<Ts>...>
    template <typename T, typename Enabler = void>
    struct decayed_tuple;

    template <typename... Ts>
    struct decayed_tuple<std::tuple<Ts...>> {
      using type = std::tuple<typename std::decay<Ts>::type...>;
    };

    template <typename Tuple>
    using decayed_tuple_t = typename decayed_tuple<Tuple>::type;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // is_empty_tuple
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // true if tuple contains empty types, e.g. is_empty_tuple<std::tuple<>> or is_empty_tuple<std::tuple<Void>>
    template <typename T, typename Enabler = void>
    struct is_empty_tuple : std::false_type {};

    template <typename... Ts>
    struct is_empty_tuple<std::tuple<Ts...>, std::enable_if_t<(std::is_empty<Ts>::value && ...)> > : std::true_type {
    };

    template <typename Tuple>
    inline constexpr bool is_empty_tuple_v = is_empty_tuple<Tuple>::value;

    static_assert(!is_empty_tuple_v<std::tuple<int>>, "ouch");

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // is_Void_v
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    constexpr bool is_Void_v = std::is_same_v<std::decay_t<T>,Void>;

}  // namespace meta

}  // namespace ttg

#endif  // CXXAPI_SERIALIZATION_H_H
