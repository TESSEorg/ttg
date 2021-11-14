#ifndef TTG_IMPL_SELECTOR_H
#define TTG_IMPL_SELECTOR_H

/**
 * Select a default backend implementation if none was specified.
 * This file should be included /first/ by any header relying on these macros.
 */

/* Error if >1 or 0 backends were selected */
#if (defined(TTG_USE_MADNESS) && defined(TTG_USE_PARSEC)) || !(defined(TTG_USE_MADNESS) || defined(TTG_USE_PARSEC))
#error \
    "One default implementation must be selected! "\
       "Please select either the PaRSEC backend (TTG_USE_PARSEC) or the MADNESS backend (TTG_USE_MADNESS)"
#endif

#if defined(TTG_USE_PARSEC)
#include "parsec/import.h"
#elif defined(TTG_USE_MADNESS)
#include "madness/import.h"
#endif  // TTG_USE_PARSEC|MADNESS

#endif  // TTG_IMPL_SELECTOR_H
