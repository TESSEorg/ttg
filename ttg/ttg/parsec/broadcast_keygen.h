#ifndef TTG_PARSEC_BROADCAST_KEYGEN_H
#define TTG_PARSEC_BROADCAST_KEYGEN_H

#include "ttg/parsec/thread_local.h"
#include "ttg/parsec/task.h"
#include "ttg/parsec/fwd.h"

namespace ttg_parsec {

    template <typename keyT, typename valueT>
    inline void broadcast_keygen(const keyT& key, const valueT& value) {
      /* type punning to void*; the TT on the other end knows the types too */
      detail::parsec_ttg_caller->broadcast_keygen(&key, &value);
    }

    template <typename keyT, typename valueT>
    inline void prepare_keygen(const keyT& key, const valueT& value) {
      /* type punning to void*; the TT on the other end knows the types too */
      detail::parsec_ttg_caller->prepare_keygen(&key, &value);
    }

}  // namespace ttg_parsec

#endif  // TTG_PARSEC_BROADCAST_KEYGEN_H