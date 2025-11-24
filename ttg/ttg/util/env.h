// SPDX-License-Identifier: BSD-3-Clause
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

    /// Override whether TTG should attempt to communicate to and from device buffers.
    /// TTG will attempt to query device support from the underlying MPI implementation (e.g.,
    /// using the unofficial extension MPIX_Query_cuda_support). However, since not all MPI implementations
    /// support this extension, users can force the use of device buffers in communication by setting
    /// `TTG_FORCE_DEVICE_COMM` to a non-negative number.
    /// @return true if the user wants to force the use of device-side buffers in communication.
    bool force_device_comm();

  }  // namespace detail
}  // namespace ttg

#endif  // TTG_UTIL_ENV_H
