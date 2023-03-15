#ifndef TTG_DEVICESCOPE_H
#define TTG_DEVICESCOPE_H

namespace ttg {
  enum class scope {
    Allocate     = 0x0,  //< memory allocated as scratch, but not moved in or out
    SyncIn       = 0x2,  //< memory allocated as scratch and data transferred to device
  };
} // namespace ttg

#endif // TTG_DEVICESCOPE_H