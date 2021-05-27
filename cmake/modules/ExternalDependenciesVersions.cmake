# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG e8bf851c6c830ec1287d1273e35db2cb2b94771c)
set(TTG_TRACKED_PARSEC_TAG a4b90a941669257421950b9fd706f93c6d29964e)
set(TTG_TRACKED_BTAS_TAG de4c7ff3893f9f2ad3b247f076374bb0cad07ead)
