// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_PARSEC_PARSEC_DATA_H
#define TTG_PARSEC_PARSEC_DATA_H

#include "ttg/parsec/buffer.h"
#include "ttg/buffer.h"

namespace ttg_parsec::detail {
  template<typename Value, typename Fn>
  void foreach_parsec_data(Value&& value, Fn&& fn) {
    /* protect for non-serializable types, allowed if the TT has no device op */
    if constexpr (ttg::detail::has_buffer_apply_v<Value>) {
      ttg::detail::buffer_apply(value, [&]<typename B>(B&& b){
        parsec_data_t *data = detail::get_parsec_data(b);
        if (nullptr != data) {
          fn(data);
        }
      });
    }
  }

  /**
   * Find the latest data copy on a device. Falls back to the host copy if
   * it's the latest or if there are no device copies.
   * Will add a reader to the copy that has to be removed later.
   */
  inline std::tuple<int, parsec_data_copy_t*> find_device_copy(parsec_data_t* data) {
    parsec_atomic_lock(&data->lock);
    int version = data->device_copies[0]->version; // default to host
    int device = 0;
    parsec_data_copy_t* device_copy = nullptr;
    for (int i = 1; i < parsec_nb_devices; ++i) {
      if (data->device_copies[i] == nullptr) continue;
      if (data->device_copies[i]->version >= version) {
        device = i;
        version = data->device_copies[i]->version;
        device_copy = data->device_copies[i];
      }
    }
    if (device != 0) {
        /* add a reader to the device copy */
        parsec_atomic_fetch_add_int32(&device_copy->readers, 1);
    }
    parsec_atomic_unlock(&data->lock);
    return {device, device_copy};
  }
} // namespace ttg_parsec::detail

#endif // TTG_PARSEC_PARSEC_DATA_H