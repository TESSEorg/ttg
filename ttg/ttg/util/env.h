//
// Created by Eduard Valeyev on 11/5/21.
//

#ifndef TTG_UTIL_ENV_H
#define TTG_UTIL_ENV_H

namespace ttg {
  namespace detail {

    /// Determine the number of compute threads to use by TTG when not given to `ttg::initialize`

    /// The number of threads is queried from the environment variable `TTG_NUM_THREADS`; if not given,
    /// then `std::thread::hardware_concurrency` is used.
    /// @return the number of threads to use by TTG
    /// @post `num_threads()>0`
    int num_threads();

  }  // namespace detail
}  // namespace ttg

#endif  // TTG_UTIL_ENV_H
