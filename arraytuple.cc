#include <array>
#include <iostream>
#include <tuple>
#include "demangle.h"

template <std::size_t Begin, typename... T, std::size_t... I>
auto subtuple_(const std::tuple<T...>& t, std::index_sequence<I...>) {
  // return std::make_tuple(std::get<I+Begin>(t)...);// copies elements
  // return std::make_tuple(&std::get<I+Begin>(t)...);// takes pointers to elements
  return std::tie(std::get<I + Begin>(t)...);  // takes references to elements
}

template <std::size_t Begin, std::size_t End, typename... T>
auto subtuple(const std::tuple<T...>& t) {
  return subtuple_<Begin>(t, std::make_index_sequence<End - Begin>());
}

template <typename T, std::size_t N, typename... Ts>
auto make_array_crude(const Ts&... t) {
  return std::array<T, N>{{t...}};
}

// will fail for zero length subtuple
template <std::size_t Begin, std::size_t End, typename... Ts, std::size_t... I>
auto subtuple_to_array_of_ptrs_(std::tuple<Ts...>& t, std::index_sequence<I...>) {
  using arrayT = typename std::tuple_element<Begin, std::tuple<Ts*...>>::type;
  return make_array_crude<arrayT, End - Begin>(&std::get<I + Begin>(t)...);  // make_array in c++20??
}

template <std::size_t Begin, std::size_t End, typename... T>
auto subtuple_to_array_of_ptrs(std::tuple<T...>& t) {
  return subtuple_to_array_of_ptrs_<Begin, End>(t, std::make_index_sequence<End - Begin>());
}

template <typename T>
const T& value(const T& t) {
  return t;
}
template <typename T>
const T& value(const T* t) {
  return *t;
}
template <typename T>
const T& value(T* t) {
  return *t;
}

template <typename... T, std::size_t... I>
void printtuple_(const std::tuple<T...>& t, std::index_sequence<I...>) {
  int junk[] = {0, (std::cout << I << " " << value(std::get<I>(t)) << std::endl, 0)...};
  junk[0]++;
}

template <typename... T>
void printtuple(const std::tuple<T...>& t) {
  printtuple_(t, std::index_sequence_for<T...>{});
}

int main() {
  auto a = std::make_tuple(1, 'a', 3.14, 99L, "qwerty");
  auto b = subtuple<1, 4>(a);
  std::cout << demangled_type_name<decltype(b)>() << std::endl;
  printtuple(b);

  auto c = std::make_tuple("a", 0, 1, 2, 3.14);
  auto d = subtuple_to_array_of_ptrs<1, 4>(c);
  std::cout << demangled_type_name<decltype(c)>() << std::endl;
  std::cout << d[0] << " " << d[1] << " " << d[2] << std::endl;
  std::cout << *(d[0]) << " " << *(d[1]) << " " << *(d[2]) << std::endl;

  return 0;
}
