#ifndef TTG_UTIL_IMPL_SELECTOR_H
#define TTG_UTIL_IMPL_SELECTOR_H

/**
 * Select a default backend implementation if none was specified.
 * This file should be included /first/ by any header relying on these macros.
 */


/* TTG's sub-namespace for madness, to avoid duplicated `madness` namespace */
#define TTG_MADNESS_NS madness_ttg

/* Error out if both backends were selected */
#if defined(TTG_USE_MADNESS) && defined(TTG_USE_PARSEC)
#error "More than one implementation selected! "\
       "Please select either the MADNESS backend (TTG_USE_MADNESS) or PaRSEC backend (TTG_USE_PARSEC)"
#endif

/* Use the PaRSEC backend as default */
#if !(defined(TTG_USE_MADNESS) || defined(TTG_USE_PARSEC))
#error "No implementation selected! "\
       "Please select either the MADNESS backend (TTG_USE_MADNESS) or PaRSEC backend (TTG_USE_PARSEC)"
#endif

#if defined(TTG_USE_PARSEC)
#define TTG_IMPL_NS parsec
#define TTG_IMPL_NAME parsec
#endif // TTG_USE_PARSEC

#if defined(TTG_USE_MADNESS)
#define TTG_IMPL_NS TTG_MADNESS_NS
#define TTG_IMPL_NAME madness
#endif // TTG_USE_MADNESS

#endif // TTG_UTIL_IMPL_SELECTOR_H
