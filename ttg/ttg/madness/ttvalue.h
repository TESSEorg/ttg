// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_MADNESS_TTVALUE_H
#define TTG_MADNESS_TTVALUE_H

namespace ttg_madness {

  template<typename DerivedT>
  struct TTValue
  {
    /* empty */
  };

  template<typename ValueT>
  inline auto persistent(ValueT&& value) {
    return std::forward<ValueT>(value);
  }

} // namespace ttg_madness

  #endif // TTG_MADNESS_TTVALUE_H
