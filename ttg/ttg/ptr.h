#ifndef TTG_PTR_H
#define TTG_PTR_H

#include "ttg/impl_selector.h"

namespace ttg {

template<typename T>
using ptr = TTG_IMPL_NS::ptr<T>;

template<typename T>
ptr<T> get_ptr(const T& obj) {
  return TTG_IMPL_NS::get_ptr(obj);
}

template<typename T, typename... Args>
ptr<T> make_ptr(Args&&... args) {
  return TTG_IMPL_NS::make_ptr(std::forward<Args>(args)...);
}

} // namespace ttg

#endif // TTG_PTR_H