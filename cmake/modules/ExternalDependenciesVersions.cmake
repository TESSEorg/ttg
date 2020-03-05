# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_BOOST_PREVIOUS_VERSION 1.66)

set(TTG_TRACKED_MADNESS_TAG e8d13860ea853044cda0dc4208f29cf705000fed)
set(TTG_TRACKED_MADNESS_PREVIOUS_TAG e8d13860ea853044cda0dc4208f29cf705000fed)

set(TTG_TRACKED_PARSEC_TAG e1a0d5c11c7ae575b1b71096e660459084d600db)
set(TTG_TRACKED_PARSEC_PREVIOUS_TAG e1a0d5c11c7ae575b1b71096e660459084d600db)
