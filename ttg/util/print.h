#ifndef TTG_PRINT_H
#define TTG_PRINT_H

#include <iostream>
#include <mutex>

namespace ttg {
  namespace detail {
    inline std::ostream &print_helper(std::ostream &out) { return out; }
    template <typename T, typename... Ts>
    inline std::ostream &print_helper(std::ostream &out, const T &t, const Ts &... ts) {
      out << ' ' << t;
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

  template <typename T, typename... Ts>
  void print(const T &t, const Ts &... ts) {
    std::lock_guard<std::mutex> lock(detail::print_mutex_accessor<detail::StdOstreamTag::Cout>());
    std::cout << t;
    detail::print_helper(std::cout, ts...) << std::endl;
  }

  template <typename T, typename... Ts>
  void print_error(const T &t, const Ts &... ts) {
    std::lock_guard<std::mutex> lock(detail::print_mutex_accessor<detail::StdOstreamTag::Cout>()); // don't mix cerr and cout
    std::cerr << t;
    detail::print_helper(std::cerr, ts...) << std::endl;
  }
} // namespace ttg

#endif // TTG_PRINT_H
