#ifndef TTG_TRACE_H
#define TTG_TRACE_H

namespace ttg {
  namespace detail {
    inline bool &trace_accessor() {
      static bool trace = false;
      return trace;
    }
  }  // namespace detail

  inline bool tracing() { return detail::trace_accessor(); }
  inline void trace_on() { detail::trace_accessor() = true; }
  inline void trace_off() { detail::trace_accessor() = false; }

} // namespace ttg

#endif // TTG_TRACE_H
