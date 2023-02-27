#ifndef TTG_DEVICESCOPE_H
#define TTG_DEVICESCOPE_H

namespace ttg {
  enum class scope {
    Allocate     = 0x0,  //< memory allocated as scratch, but not moved in or out
    SyncIn       = 0x2,  //< data will be allocated on and transferred to device
                         //< if latest version resides on the device (no previous sync-out) the data will
                         //< not be transferred again
    SyncOut      = 0x4,  //< value will be transferred from device to host after kernel completes
    SyncInOut    = 0x8,  //< data will be moved in and synchronized back out after the kernel completes
  };
} // namespace ttg

#endif // TTG_DEVICESCOPE_H