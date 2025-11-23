// SPDX-License-Identifier: BSD-3-Clause
#ifndef TTG_MADNESS_DEVICE_H
#define TTG_MADNESS_DEVICE_H

namespace ttg_madness {
  /* no device support in MADNESS */
  inline int num_devices() { return 0; }
}

#endif // TTG_MADNESS_DEVICE_H