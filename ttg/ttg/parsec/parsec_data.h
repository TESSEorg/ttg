#ifndef TTG_PARSEC_PARSEC_DATA_H
#define TTG_PARSEC_PARSEC_DATA_H

#include "ttg/parsec/buffer.h"
#include "ttg/buffer.h"

namespace ttg_parsec::detail {
  template<typename Value, typename Fn>
  void foreach_parsec_data(Value&& value, Fn&& fn) {
    /* protect for non-serializable types, allowed if the TT has no device op */
    if constexpr (ttg::detail::has_buffer_apply_v<Value>) {
      ttg::detail::buffer_apply(value, [&]<typename B>(B&& b){
        fn(detail::get_parsec_data(b));
      });
    }
  }
} // namespace ttg_parsec::detail

#endif // TTG_PARSEC_PARSEC_DATA_H