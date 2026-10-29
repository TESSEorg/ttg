// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include <functional>
#include <utility>

#include "ttg.h"

namespace callable_test {

  using ttg::typelist;
  using ttg::meta::callable_args_t;
  using ttg::meta::callable_return_t;
  using ttg::meta::is_generic_callable_v;

  template <typename T, typename = void>
  constexpr bool has_signature_v = false;
  template <typename T>
  constexpr bool has_signature_v<T, std::void_t<callable_args_t<T>, callable_return_t<T>>> = true;

  // functions, function pointers and references, incl. noexcept and C-style varargs
  int f(int, const double &);
  int f_noexcept(int) noexcept;
  int f_varargs(int, ...);
  int f_varargs_noexcept(int, ...) noexcept;
  static_assert(std::is_same_v<callable_args_t<decltype(f)>, typelist<int, const double &>>);
  static_assert(std::is_same_v<callable_return_t<decltype(f)>, int>);
  static_assert(std::is_same_v<callable_args_t<decltype(&f)>, typelist<int, const double &>>);
  static_assert(std::is_same_v<callable_args_t<decltype(f) &>, typelist<int, const double &>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&f) const &>, typelist<int, const double &>>);
  static_assert(std::is_same_v<callable_args_t<decltype(f_noexcept)>, typelist<int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&f_noexcept)>, typelist<int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(f_varargs)>, typelist<int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&f_varargs_noexcept)>, typelist<int>>);

  // member function pointers: object reference reflects cv- and ref-qualifiers
  struct X {
    void m(int);
    void m_c(int) const;
    void m_v(int) volatile;
    void m_cv(int) const volatile;
    void m_l(int) &;
    void m_cl(int) const &;
    void m_r(int) &&;
    void m_cr(int) const &&;
    void m_cvr_noexcept(int) const volatile && noexcept;
    void m_varargs(int, ...);
    void m_cl_varargs_noexcept(int, ...) const & noexcept;
    double m_ret() const;
    template <typename T>
    void m_template(T);
    void m_overloaded(int);
    void m_overloaded(double);
    int data;
  };
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m)>, typelist<X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_c)>, typelist<const X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_v)>, typelist<volatile X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_cv)>, typelist<const volatile X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_l)>, typelist<X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_cl)>, typelist<const X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_r)>, typelist<X &&, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_cr)>, typelist<const X &&, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_cvr_noexcept)>, typelist<const volatile X &&, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_varargs)>, typelist<X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_cl_varargs_noexcept)>, typelist<const X &, int>>);
  static_assert(std::is_same_v<callable_return_t<decltype(&X::m_ret)>, double>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_template<int>)>, typelist<X &, int>>);
  static_assert(std::is_same_v<callable_args_t<decltype(&X::m_c) const>, typelist<const X &, int>>);
  // pointers to data members are not introspected
  static_assert(!has_signature_v<decltype(&X::data)>);

  // function objects: object argument is dropped
  struct F {
    long operator()(int, double &) const;
  };
  struct F_mutable_noexcept {
    void operator()(int &&) noexcept;
  };
  struct F_derived : F {};
  struct F_overloaded {
    void operator()(int);
    void operator()(double);
  };
  struct F_template {
    template <typename T>
    void operator()(T);
  };
  static_assert(std::is_same_v<callable_args_t<F>, typelist<int, double &>>);
  static_assert(std::is_same_v<callable_return_t<F>, long>);
  static_assert(std::is_same_v<callable_args_t<const F &>, typelist<int, double &>>);
  static_assert(std::is_same_v<callable_args_t<F &&>, typelist<int, double &>>);
  static_assert(std::is_same_v<callable_args_t<F_mutable_noexcept>, typelist<int &&>>);
  static_assert(std::is_same_v<callable_args_t<F_derived>, typelist<int, double &>>);
  static_assert(std::is_same_v<callable_args_t<std::function<int(char)>>, typelist<char>>);
  static_assert(!has_signature_v<F_overloaded>);
  static_assert(!has_signature_v<F_template>);
  static_assert(!has_signature_v<int>);
  static_assert(!has_signature_v<int *>);
  static_assert(!has_signature_v<void>);

#if defined(__cpp_static_call_operator) && __cpp_static_call_operator >= 202207L
  struct F_static {
    static int operator()(int, const double &);
  };
  static_assert(std::is_same_v<callable_args_t<F_static>, typelist<int, const double &>>);
  static_assert(std::is_same_v<callable_return_t<F_static>, int>);
#endif

  // lambdas
  inline auto l = [](int &, const double &) { return 1; };
  inline auto l_mutable = [](int) mutable {};
  inline auto l_generic = [](auto &) {};
  inline auto l_partially_generic = [](int, auto) {};
  static_assert(std::is_same_v<callable_args_t<decltype(l)>, typelist<int &, const double &>>);
  static_assert(std::is_same_v<callable_return_t<decltype(l) &>, int>);
  static_assert(std::is_same_v<callable_args_t<decltype(l_mutable)>, typelist<int>>);
  static_assert(!is_generic_callable_v<decltype(l)>);
  static_assert(is_generic_callable_v<decltype(l_generic)>);
  static_assert(is_generic_callable_v<decltype(l_partially_generic)>);

  // callable_args packs return and argument types
  static_assert(std::is_same_v<std::remove_const_t<decltype(ttg::meta::callable_args<decltype(l)>)>,
                               std::pair<bool, std::pair<typelist<int>, typelist<int &, const double &>>>>);
  static_assert(!ttg::meta::callable_args<decltype(l)>.first);
  static_assert(ttg::meta::callable_args<decltype(l_generic)>.first);

  // SequencedKeysConstraint deduction guides take the key and ordinal types from the mapper's signature
  inline auto mapper = [](const std::pair<int, int> &k) -> long { return k.second; };
  using deduced_t = decltype(ttg::SequencedKeysConstraint(mapper));
  static_assert(std::is_same_v<typename deduced_t::key_type, std::pair<int, int>>);
  static_assert(std::is_same_v<typename deduced_t::ordinal_type, long>);
  static_assert(std::is_same_v<decltype(ttg::SequencedKeysConstraint(mapper, true)), deduced_t>);

}  // namespace callable_test

TEST_CASE("callable_traits", "[core][meta]") {
  // all checks are compile-time
  CHECK(!ttg::meta::is_generic_callable_v<decltype(callable_test::l)>);
}
