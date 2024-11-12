#ifndef TTG_BUFFER_H
#define TTG_BUFFER_H

#include <memory>

#include "ttg/fwd.h"

namespace ttg {

template<typename T, typename Allocator = std::allocator<std::decay_t<T>>>
using Buffer = TTG_IMPL_NS::Buffer<T, Allocator>;

} // namespace ttg

#endif // TTG_buffer_H