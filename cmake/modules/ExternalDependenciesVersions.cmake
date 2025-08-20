# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

set(TTG_TRACKED_VG_CMAKE_KIT_TAG cda539db32be6e8171f5cbebdb1a7c38d5ab4b34)  # provides FindOrFetchLinalgPP and "real" FindOrFetchBoost
set(TTG_TRACKED_CATCH2_VERSION 3.5.0)
set(TTG_TRACKED_MADNESS_TAG 71d0936adf013ca96967cb517beae30bd15521d0)
set(TTG_TRACKED_PARSEC_TAG parsec-for-ttg)
set(TTG_TRACKED_BTAS_TAG c25b0a11d2a76190bfb13fa72f9e9dc3e57c3c2f)
set(TTG_TRACKED_TILEDARRAY_TAG 5944bdba3266a3fa19f1809c8e2accf3dad4d815)

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
# BUT if will be building examples, inherit the oldest version from the pickiest Boost consumer (TA and/or BSPMM)
if (TTG_EXAMPLES)
  set(TTG_OLDEST_BOOST_VERSION 1.81)
else()
  set(TTG_OLDEST_BOOST_VERSION 1.66)
endif()
