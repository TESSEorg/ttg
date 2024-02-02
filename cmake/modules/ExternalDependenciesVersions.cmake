# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 3.5.0)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG cb195817d7807c4aead10ba200cd20649036cbae)
#set(TTG_TRACKED_PARSEC_TAG 25d1931e863b6741e453112d2117d85ad32e7fba)
set(TTG_TRACKED_PARSEC_TAG 8ed7cdf02ad607051c130a6deeb8f12c84f2e4d3)
set(TTG_TRACKED_BTAS_TAG a02be0d29fb4a788ecef43de711dcd6d6f1cb6b8)
set(TTG_TRACKED_TILEDARRAY_TAG f0115e9e4a3f988224afbfb3c241e92171e916b8)
