#ifndef TTG_IMPL_SELECTOR_H
#define TTG_IMPL_SELECTOR_H

/**
 * Select a default backend implementation if none was specified.
 * This file should be included /first/ by any header relying on these macros.
 */

/* Error if both backends were selected */
#if (defined(TTG_USE_MADNESS) && defined(TTG_USE_PARSEC)) || !(defined(TTG_USE_MADNESS) || defined(TTG_USE_PARSEC))
#error \
    "One default implementation must be selected! "\
       "Please select either the MADNESS backend (TTG_USE_MADNESS) or PaRSEC backend (TTG_USE_PARSEC)"
#endif

#if defined(TTG_USE_PARSEC)
#include "parsec/import.h"
#endif  // TTG_USE_PARSEC

#if defined(TTG_USE_MADNESS)
#include "madness/import.h"
#endif  // TTG_USE_MADNESS

#endif // TTG_IMPL_SELECTOR_H
