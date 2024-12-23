#ifndef TTG_MAD_DEVICEFUNC_H
#define TTG_MAD_DEVICEFUNC_H

#include "ttg/madness/buffer.h"

namespace ttg_madness {

  template<typename T, typename A>
  auto buffer_data(const Buffer<T, A>& buffer) {
    /* for now return the internal pointer, should be adapted if ever relevant for madness */
    return buffer.current_device_ptr();
  }

  template<typename... Views>
  inline bool register_device_memory(std::tuple<Views&...> &views)
  {
    /* nothing to do here */
    return true;
  }

  template<typename T, std::size_t N>
  inline bool register_device_memory(const ttg::span<T, N>& span)
  {
    /* nothing to do here */
    return true;
  }

} // namespace ttg_madness

#endif // TTG_MAD_DEVICEFUNC_H