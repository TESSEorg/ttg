#ifndef TTG_PROFILE_H
#define TTG_PROFILE_H

namespace ttg {
  namespace detail {
    inline bool &profile_accessor() {
      static bool profile = false;
      return profile;
    }
  }  // namespace detail

  /// \brief returns whether profiling is enabled
  inline constexpr bool profile_enabled() {
#if defined(TTG_USE_PARSEC)
#if defined(PARSEC_PROF_TRACE)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
  }

  /// To enable profiling invoke profile_on(). To disable profiling invoke profile_off().
  /// \return false, if `profile_enabled()==false`, otherwise returns true if the most recent call to `profile_on()`
  /// has not been followed by `profile_off()`
  inline bool profiling() {
    if constexpr (profile_enabled())
      return detail::profile_accessor();
    else
      return false;
  }

  /// \brief enables profiling; if `profile_enabled()==true` this has no effect
  inline void profile_on() { if constexpr (profile_enabled()) detail::profile_accessor() = true; }
  /// \brief disables profiling; if `profile_enabled()==true` this has no effect
  inline void profile_off() { if constexpr (profile_enabled()) detail::profile_accessor() = false; }

} // namespace ttg

#endif // TTG_PROFILE_H
