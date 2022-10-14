# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 851ef271858129c47a912a088cc833b085c382aa)
set(TTG_TRACKED_PARSEC_TAG 89fdb88c6bdfcee3904c013a0856403762a5479f)
set(TTG_TRACKED_BTAS_TAG ab866a760b72ff23053266bf46b87f19d3df44b7)
