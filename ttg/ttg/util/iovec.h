#ifndef TTG_UTIL_IOVEC_H_
#define TTG_UTIL_IOVEC_H_

#include <cstdint>

namespace ttg {

  /**
   * Used to describe transfer payload in types using the \sa SplitMetadataDescriptor.
   * @member data Pointer to the data to be read from / written to.
   * @member num_bytes The number of bytes to read from / write to the memory location
   *                   \sa data.
   */
  struct iovec {
    std::size_t num_bytes;
    void* data;
  };

} // ttg

#endif // TTG_UTIL_IOVEC_H_
