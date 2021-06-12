# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG e8bf851c6c830ec1287d1273e35db2cb2b94771c)
set(TTG_TRACKED_PARSEC_TAG 6bfad5317ef24f5503bbcd8d52e7628d08d876e5)
set(TTG_TRACKED_BTAS_TAG d7794799e4510cf66844081dd8f1f5b648112d33)
