#ifndef TTG_PRINT_H
#define TTG_PRINT_H

#include <iostream>
#include <mutex>

namespace ttg {
  constexpr char print_separator = ' ';
  namespace detail {
    inline std::ostream &print_helper(std::ostream &out) { return out; }
    template <typename T, typename... Ts>
    inline std::ostream &print_helper(std::ostream &out, const T &t, const Ts &... ts) {
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
} // namespace ttg

#endif // TTG_PRINT_H
