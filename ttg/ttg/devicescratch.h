#ifndef TTG_DEVICESCRATCH_H
#define TTG_DEVICESCRATCH_H

#include "ttg/devicescope.h"
#include "ttg/impl_selector.h"

namespace ttg {

template<typename T>
using devicescratch = TTG_IMPL_NS::devicescratch<T>;

template<typename T>
auto make_scratch(T* val, ttg::scope scope, std::size_t count = 1) {
  return devicescratch<T>(val, scope, 1);
}

} // namespace ttg

#endif // TTG_DEVICESCRATCH_H