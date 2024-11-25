#ifndef TTG_BUFFER_H
#define TTG_BUFFER_H

#include <memory>

#include "ttg/fwd.h"
#include "ttg/util/meta.h"

namespace ttg {

template<typename T, typename Allocator = std::allocator<std::decay_t<T>>>
using Buffer = TTG_IMPL_NS::Buffer<T, Allocator>;

namespace meta {

  /* Specialize some traits */

  template<typename T, typename A>
  struct is_buffer<ttg::Buffer<T, A>> : std::true_type
  { };

  template<typename T, typename A>
  struct is_buffer<const ttg::Buffer<T, A>> : std::true_type
  { };

  /* buffers are const if their value types are const */
  template<typename T, typename A>
  struct is_const<ttg::Buffer<T, A>> : std::is_const<T>
  { };

} // namespace meta

} // namespace ttg

#endif // TTG_buffer_H