# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 29a2bf3d3c2670c608b7bfdf2299d76fbc20e041)
set(TTG_TRACKED_PARSEC_TAG 7042a5cdbfe3e4900c6067a9747e5017ce7279ba)
set(TTG_TRACKED_BTAS_TAG f84c27b2d3196373145edb3c48049fa885fd219e)
