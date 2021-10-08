# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 0bc4af405bafc4ef57ef4e9b74844e6d4aa62826)
set(TTG_TRACKED_PARSEC_TAG 340d8498a0ff1e1956d550b51d539e64b27c8dbc)
set(TTG_TRACKED_BTAS_TAG 1b3b8ce5ca51a224aaa386b5c23ec4d742362e89)
