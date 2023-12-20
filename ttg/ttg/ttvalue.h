#ifndef TTG_TTVALUE_H
#define TTG_TTVALUE_H

#include "ttg/fwd.h"

namespace ttg {

  template<typename T>
  using TTValue = TTG_IMPL_NS::TTValue<T, Allocator>;

} // namespace ttg

#endif // TTG_TTVALUE_H