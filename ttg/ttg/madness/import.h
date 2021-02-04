#ifndef TTG_MADNESS_IMPORT_H
#define TTG_MADNESS_IMPORT_H


#if defined(TTG_SELECTED_DEFAULT_IMPL)
#error "A default TTG implementation has already been selected"
#endif // defined(TTG_SELECTED_DEFAULT_IMPL)

#define TTG_SELECTED_DEFAULT_IMPL madness

#define TTG_MADNESS_IMPORTED 1

namespace ttg {

  /* Mark the ttg_madness namespace as the default */
  inline namespace ttg_madness { }

}

/* Now include the implementation header file */
#include "ttg/madness/ttg.h"

#endif // TTG_MADNESS_IMPORT_H

