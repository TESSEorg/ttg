# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 4f7d30b0a738621037b96bb5b820029835753667)
set(TTG_TRACKED_PARSEC_TAG 6fd959e4a8d2dad8b701d0b83cd85b372f1a306b)
set(TTG_TRACKED_BTAS_TAG d73153ad9bc41a177e441ef04eceff7fab0c766d)
