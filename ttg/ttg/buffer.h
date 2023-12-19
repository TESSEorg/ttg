#ifndef TTG_BUFFER_H
#define TTG_BUFFER_H

#include <memory>
#include "ttg/impl_selector.h"

#if defined(TTG_IMPL_DEVICE_SUPPORT)

namespace ttg {

template<typename T, typename Allocator = std::allocator<T>>
using Buffer = TTG_IMPL_NS::Buffer<T, Allocator>;

} // namespace ttg

#endif // TTG_IMPL_DEVICE_SUPPORT
#endif // TTG_buffer_H