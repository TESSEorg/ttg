# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
set(TTG_TRACKED_BOOST_VERSION 1.66)
set(TTG_TRACKED_CATCH2_VERSION 2.13.1)
set(TTG_TRACKED_CEREAL_VERSION 1.3.0)
set(TTG_TRACKED_MADNESS_TAG 313aa92f06e3c980a826f7cf020b2b79d5e6e16b)
set(TTG_TRACKED_PARSEC_TAG 13f3d84982ec7a62477fb380aae97aa4552b2ac1)
set(TTG_TRACKED_BTAS_TAG 8ac131460e05e9470779880e15acc5642a451e7a)
