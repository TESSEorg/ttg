#ifndef TTG_PARSEC_IMPORT_H
#define TTG_PARSEC_IMPORT_H

#include "ttg/runtimes.h"

#if defined(TTG_SELECTED_DEFAULT_IMPL)
#error "A default TTG implementation has already been selected"
#endif  // defined(TTG_SELECTED_DEFAULT_IMPL)

#define TTG_SELECTED_DEFAULT_IMPL parsec
#define TTG_PARSEC_IMPORTED 1
#define TTG_IMPL_NS ttg_parsec

constexpr const ttg::Runtime ttg_runtime = ttg::Runtime::PaRSEC;

namespace ttg {

  /* Mark the ttg_parsec namespace as the default */
  using namespace ttg_parsec;

}  // namespace ttg

#endif  // TTG_PARSEC_IMPORT_H
