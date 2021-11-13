#ifndef TTG_TRACE_H
#define TTG_TRACE_H

#include "ttg/util/print.h"

namespace ttg {
  namespace detail {
    inline bool &trace_accessor() {
      static bool trace = false;
      return trace;
    }
  }  // namespace detail

  /// \brief returns whether tracing was enabled at configure time
  inline constexpr bool trace_enabled() {
#ifdef TTG_ENABLE_TRACE
    return true;
#else
    return false;
#endif
  }

  /// \brief returns whether tracing is enabled

  /// To enable tracing invoke trace_on(). To disable tracing
  /// \return false, if `trace_enabled()==false`, otherwise returns true if the most recent call to `trace_on()`
  /// has not been followed by `trace_off()`
  inline bool tracing() {
    if constexpr (trace_enabled())
      return detail::trace_accessor();
    else
      return false;
  }

  /// \brief enables tracing; if `trace_enabled()==true` this has no effect
  inline void trace_on() { if constexpr (trace_enabled()) detail::trace_accessor() = true; }
  /// \brief disables tracing; if `trace_enabled()==true` this has no effect
  inline void trace_off() { if constexpr (trace_enabled()) detail::trace_accessor() = false; }

  /// if `trace_enabled()==true` and `tracing()==true` atomically prints to std::clog a sequence of items (separated by
  /// ttg::print_separator) followed by std::endl
  template <typename T, typename... Ts>
  inline void trace(const T &t, const Ts &... ts) {
    if constexpr (trace_enabled()) {
      if (tracing()) {
        log(t, ts...);
      }
    }
  }

} // namespace ttg

#endif // TTG_TRACE_H
