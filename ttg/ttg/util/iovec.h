// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_UTIL_IOVEC_H_
#define TTG_UTIL_IOVEC_H_

#include <cstdint>

namespace ttg {

  /**
   * Used to describe transfer payload in types using the \sa SplitMetadataDescriptor.
   */
  struct iovec {
    /// The number of bytes to read from / write to the memory location given by `data`.
    std::size_t num_bytes;
    /// Pointer to the data to be read from / written to.
    void* data;
  };

} // ttg

#endif // TTG_UTIL_IOVEC_H_
