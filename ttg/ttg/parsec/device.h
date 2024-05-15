#ifndef TTG_PARSEC_DEVICE_H
#define TTG_PARSEC_DEVICE_H

#include "ttg/device/device.h"
#include <parsec/mca/device/device.h>

namespace ttg_parsec {

  namespace detail {

    // the first ID of an accelerator in the parsec ID-space
    inline int first_device_id = -1;

    /**
     * map from TTG ID-space to parsec ID-space
     */
    inline
    int ttg_device_to_parsec_device(const ttg::device::Device& device) {
      if (device.is_host()) {
        return 0;
      } else {
        return device.id() + first_device_id;
      }
    }

    /**
     * map from parsec ID-space to TTG ID-space
     */
    inline
    ttg::device::Device parsec_device_to_ttg_device(int parsec_id) {
      if (parsec_id < first_device_id) {
        return ttg::device::Device(parsec_id, ttg::ExecutionSpace::Host);
      }
      return ttg::device::Device(parsec_id - first_device_id,
                                ttg::device::available_execution_space);
    }
  } // namespace detail


  inline
  int num_devices() {
    return parsec_nb_devices - detail::first_device_id;
  }

} // namespace ttg_parsec

#endif // TTG_PARSEC_DEVICE_H