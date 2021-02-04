#ifndef TTG_PARSEC_IMPORT_H
#define TTG_PARSEC_IMPORT_H

#if defined(TTG_SELECTED_DEFAULT_IMPL)
#error "A default TTG implementation has already been selected"
#endif // defined(TTG_SELECTED_DEFAULT_IMPL)

#define TTG_SELECTED_DEFAULT_IMPL parsec

#define TTG_PARSEC_IMPORTED 1

namespace ttg {

  /* Mark the ttg_parsec namespace as the default */
  inline namespace ttg_parsec { }

}

/* Now include the implementation header file */
#include "ttg/parsec/ttg.h"

#endif // TTG_PARSEC_IMPORT_H
