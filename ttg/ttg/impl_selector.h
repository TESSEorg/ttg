#ifndef TTG_UTIL_IMPL_SELECTOR_H
#define TTG_UTIL_IMPL_SELECTOR_H

/**
 * Select a default backend implementation if none was specified.
 * This file should be included /first/ by any header relying on these macros.
 */

/* Error out if both backends were selected */
#if defined(TTG_USE_MADNESS) && defined(TTG_USE_PARSEC)
#error "More than one implementation selected! "\
       "Please select either the MADNESS backend (TTG_USE_MADNESS) or PaRSEC backend (TTG_USE_PARSEC)"
#endif

/* Use the PaRSEC backend as default */
#if !(defined(TTG_USE_MADNESS) || defined(TTG_USE_PARSEC))
#define TTG_USE_PARSEC 1
#endif

#if defined(TTG_USE_PARSEC)
#define TTG_IMPL_NS ttg_parsec
#define TTG_IMPL_NAME parsec
#define TTG_IMPL_PARSEC_INLINE_NS inline
#else
#define TTG_IMPL_PARSEC_INLINE_NS
#endif // TTG_USE_PARSEC

#if defined(TTG_USE_MADNESS)
#define TTG_IMPL_NS ttg_madness
#define TTG_IMPL_NAME madness
#define TTG_IMPL_MADNESS_INLINE_NS inline
#else
#define TTG_IMPL_MADNESS_INLINE_NS
#endif // TTG_USE_MADNESS

#endif // TTG_UTIL_IMPL_SELECTOR_H
