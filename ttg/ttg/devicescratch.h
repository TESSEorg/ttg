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

namespace detail {

  template<typename T>
  struct is_devicescratch : std::false_type
  { };

  template<typename T>
  struct is_devicescratch<ttg::devicescratch<T>> : std::true_type
  { };

  template<typename T>
  constexpr bool is_devicescratch_v = is_devicescratch<T>::value;

} // namespace detail

} // namespace ttg

#endif // TTG_DEVICESCRATCH_H