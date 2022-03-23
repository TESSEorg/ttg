//
// Created by Eduard Valeyev on 3/1/22.
//

#ifndef TTG_META_INDEX_SEQUENCE_H
#define TTG_META_INDEX_SEQUENCE_H

#include <array>

namespace ttg::meta {

  // metaprogramming with index sequences

  template <std::size_t K, std::size_t N, std::size_t I0, std::size_t... I>
  constexpr std::array<std::size_t, N> sequence_to_array_impl(std::array<std::size_t, N> arr,
                                                              std::index_sequence<I0, I...> seq) {
    arr[K] = I0;
    if constexpr (K + 1 < N) {
      return sequence_to_array_impl<K + 1>(arr, std::index_sequence<I...>{});
    } else
      return arr;
  }

  template <std::size_t... I>
  constexpr std::array<std::size_t, sizeof...(I)> sequence_to_array(std::index_sequence<I...> seq) {
    if constexpr (sizeof...(I) > 0)
      return sequence_to_array_impl<0>(std::array<std::size_t, sizeof...(I)>{}, seq);
    else
      return std::array<std::size_t, 0>{};
  }

  // thanks to
  // https://stackoverflow.com/questions/56799396/is-it-possible-to-turn-a-constexpr-stdarray-into-a-stdinteger-sequence

  template <size_t N, typename ArrayAccessor, size_t... indexes>
  constexpr auto array_to_sequence_impl(ArrayAccessor accessor, std::index_sequence<indexes...> is) {
    return std::index_sequence<std::get<indexes>(accessor())...>{};
  }

  // only works for if accessor() returns reference to array with static storage duration
  template <typename ArrayAccessor>
  constexpr auto array_to_sequence(ArrayAccessor accessor) {
    constexpr size_t N = accessor().size();
    using indexes = std::make_index_sequence<N>;
    return array_to_sequence_impl<N>(accessor, indexes{});
  };

  template <std::size_t N, std::size_t... I>
  constexpr auto make_zero_index_sequence_impl(std::index_sequence<I...> prefix) {
    if constexpr (N == 0)
      return prefix;
    else
      return make_zero_index_sequence_impl<N - 1>(std::index_sequence<I..., 0>{});
  }

  template <std::size_t N>
  constexpr auto make_zero_index_sequence() {
    if constexpr (N == 0)
      return std::index_sequence<>{};
    else
      return make_zero_index_sequence_impl<N - 1>(std::index_sequence<0>{});
  }

  template <typename T, T... Is>
  constexpr T get(std::integer_sequence<T, Is...>, std::size_t I) {
    constexpr T arr[] = {Is...};
    return arr[I];
  }

  template <typename T, T... I1, T... I2>
  constexpr std::integer_sequence<T, I1..., I2...> concat(std::integer_sequence<T, I1...>,
                                                          std::integer_sequence<T, I2...>) {
    return std::integer_sequence<T, I1..., I2...>{};
  }

  template <std::size_t I, typename T, T... Ints>
  struct increment_at_helper;

  template <std::size_t I, typename T>
  struct increment_at_helper<I, T> {
    using result = std::integer_sequence<T>;
  };

  template <std::size_t I, typename T, T Int0, T... Ints>
  struct increment_at_helper<I, T, Int0, Ints...> {
    using start_of_result =
        std::conditional_t<I == 0, std::integer_sequence<T, Int0 + 1>, std::integer_sequence<T, Int0>>;
    static constexpr auto nrest = sizeof...(Ints);
    using rest_of_result = std::conditional_t<nrest == 0, std::integer_sequence<T>,
                                              typename increment_at_helper<I - 1, T, Ints...>::result>;
    using result = decltype(concat(start_of_result{}, rest_of_result{}));
  };

  template <std::size_t I, typename T, T... Ints>
  constexpr auto increment_at(std::integer_sequence<T, Ints...> seq) {
    static_assert(sizeof...(Ints) > 0);
    return typename increment_at_helper<I, T, Ints...>::result{};
  }

  template <std::size_t I, typename T, T... Ints>
  struct reset_prev_to_zero_helper;

  template <std::size_t I, typename T>
  struct reset_prev_to_zero_helper<I, T> {
    using result = std::integer_sequence<T>;
  };

  template <std::size_t I, typename T, T Int0, T... Ints>
  struct reset_prev_to_zero_helper<I, T, Int0, Ints...> {
    using start_of_result = std::conditional_t < I<1, std::integer_sequence<T, Int0>, std::integer_sequence<T, 0>>;
    static constexpr auto nrest = sizeof...(Ints);
    using rest_of_result =
        std::conditional_t<nrest == 0, std::integer_sequence<T>,
                           typename reset_prev_to_zero_helper<std::min(I, I - 1), T, Ints...>::result>;
    using result = decltype(concat(start_of_result{}, rest_of_result{}));
  };

  template <std::size_t I, typename T, T... Ints>
  constexpr auto reset_prev_to_zero(std::integer_sequence<T, Ints...> seq) {
    static_assert(sizeof...(Ints) > 0);
    return typename reset_prev_to_zero_helper<I, T, Ints...>::result{};
  }

}  // namespace ttg::meta

#endif  // TTG_META_INDEX_SEQUENCE_H
