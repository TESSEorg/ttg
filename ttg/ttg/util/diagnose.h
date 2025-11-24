// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_DIAGNOSE_H
#define TTG_DIAGNOSE_H

namespace ttg {
  namespace detail {
    inline bool &diagnose_accessor() {
      static bool diagnose = true;
      return diagnose;
    }
  }  // namespace detail

  inline bool diagnose() { return detail::diagnose_accessor(); }
  inline void diagnose_on() { detail::diagnose_accessor() = true; }
  inline void diagnose_off() { detail::diagnose_accessor() = false; }

}  // namespace ttg

#endif  // TTG_DIAGNOSE_H
