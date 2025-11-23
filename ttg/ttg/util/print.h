// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_PRINT_H
#define TTG_PRINT_H

#include <iostream>
#include <mutex>
#include <complex>
#include <vector>
#include <array>
#include <utility>

namespace ttg {
  constexpr char print_separator = ' ';
  constexpr char print_seq_separator = ',';
  constexpr char print_seq_begin = '{';
  constexpr char print_seq_end = '}';

  namespace iostream {

    /// @name I/O operators for standard types
    /// @{

    /// default printing of std::complex

    /// \tparam T The "real" type of the complex number.
    /// \param[in,out] s The output stream.
    /// \param[in] c The complex number.
    /// \return The output stream (for chaining).
    template <typename T>
    std::ostream &operator<<(std::ostream &s, const std::complex<T> &c) {
      s << c.real() << (c.imag() >= 0 ? "+" : "") << c.imag() << "j";
      return s;
    }

    /// default printing of std::pair

    /// \tparam T Type 1 of the pair.
    /// \tparam U Type 2 of the pair.
    /// \param[in,out] s The output stream.
    /// \param[in] p The pair.
    /// \return The output stream (for chaining).
    template <typename T, typename U>
    std::ostream &operator<<(std::ostream &s, const std::pair<T, U> &p) {
      s << print_seq_begin << p.first << print_seq_separator << p.second << print_seq_end;
      return s;
    }

    /// default printing of std::vector

    /// \tparam T Type stored in the vector.
    /// \param[in,out] s The output stream.
    /// \param[in] c The vector.
    /// \return The output stream (for chaining).
    template <typename T>
    std::ostream &operator<<(std::ostream &s, const std::vector<T> &c) {
      s << print_seq_begin;
      typename std::vector<T>::const_iterator it = c.begin();
      while (it != c.end()) {
        s << *it;
        ++it;
        if (it != c.end())
          s << print_seq_separator;
      };
      s << print_seq_end;
      return s;
    }

    /// default printing of std::array

    /// STL I/O already does char (thus the \c enable_if business).
    /// \tparam T Type of data in the array.
    /// \tparam N Size of the array.
    /// \param[in,out] s The output stream.
    /// \param[in] v The array.
    /// \return The output stream (for chaining).
    template <typename T, std::size_t N>
    typename std::enable_if<!std::is_same<T, char>::value, std::ostream &>::type
    operator<<(std::ostream &s, const std::array<T,N>& v) {
      s << print_seq_begin;
      for (std::size_t i = 0; i < N; ++i) {
        s << v[i];
        if (i != (N - 1))
          s << print_seq_separator;
      }
      s << print_seq_end;
      return s;
    }

    /// default printing of fixed dimension arrays.

    /// STL I/O already does char (thus the \c enable_if business).
    /// \tparam T Type of data in the array.
    /// \tparam N Size of the array.
    /// \param[in,out] s The output stream.
    /// \param[in] v The array.
    /// \return The output stream (for chaining).
    template <typename T, std::size_t N>
    typename std::enable_if<!std::is_same<T, char>::value, std::ostream &>::type
    operator<<(std::ostream &s, const T (&v)[N]) {
      s << print_seq_begin;
      for (std::size_t i = 0; i < N; ++i) {
        s << v[i];
        if (i != (N - 1))
          s << print_seq_separator;
      }
      s << print_seq_end;
      return s;
    }

  }  // namespace operators

  namespace detail {
    inline std::ostream &print_helper(std::ostream &out) { return out; }
    template <typename T, typename... Ts>
    inline std::ostream &print_helper(std::ostream &out, const T &t, const Ts &... ts) {
      using ttg::iostream::operator<<;
      out << print_separator << t;
      return print_helper(out, ts...);
    }
    //
    enum class StdOstreamTag { Cout, Cerr };
    template <StdOstreamTag>
    inline std::mutex &print_mutex_accessor() {
      static std::mutex mutex;
      return mutex;
    }
  }  // namespace detail

  /// atomically prints to std::cout a sequence of items (separated by ttg::print_separator) followed by std::endl
  template <typename T, typename... Ts>
  void print(const T &t, const Ts &... ts) {
    std::lock_guard<std::mutex> lock(detail::print_mutex_accessor<detail::StdOstreamTag::Cout>());
    std::cout << t;
    detail::print_helper(std::cout, ts...) << std::endl;
  }

  /// atomically prints to std::cerr a sequence of items (separated by ttg::print_separator) followed by std::endl
  template <typename T, typename... Ts>
  void print_error(const T &t, const Ts &... ts) {
    std::lock_guard<std::mutex> lock(detail::print_mutex_accessor<detail::StdOstreamTag::Cout>());
    std::cerr << t;
    detail::print_helper(std::cerr, ts...) << std::endl;
  }

  /// atomically prints to std::clog a sequence of items (separated by ttg::print_separator) followed by std::endl
  template <typename T, typename... Ts>
  void log(const T &t, const Ts &... ts) {
    std::lock_guard<std::mutex> lock(detail::print_mutex_accessor<detail::StdOstreamTag::Cout>());
    std::clog << t;
    detail::print_helper(std::clog, ts...) << std::endl;
  }
} // namespace ttg

#endif // TTG_PRINT_H
