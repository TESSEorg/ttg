#ifndef TTG_MADNESS_IMPORT_H
#define TTG_MADNESS_IMPORT_H

#include "ttg/runtimes.h"

#if defined(TTG_SELECTED_DEFAULT_IMPL)
#error "A default TTG implementation has already been selected"
#endif  // defined(TTG_SELECTED_DEFAULT_IMPL)

#define TTG_SELECTED_DEFAULT_IMPL madness
#define TTG_MADNESS_IMPORTED 1
#define TTG_IMPL_NS ttg_madness

namespace ttg_madness {};

namespace ttg {

  using namespace ::ttg_madness;

  constexpr const ttg::Runtime ttg_runtime = ttg::Runtime::MADWorld;

}

#endif  // TTG_MADNESS_IMPORT_H
