# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

set(TTG_TRACKED_VG_CMAKE_KIT_TAG 72bb8f049e68443e817ce7299f0d1dabfaf01b7e)  # provides FindOrFetchLinalgPP and "real" FindOrFetchBoost
set(TTG_TRACKED_CATCH2_VERSION 3.5.0)
set(TTG_TRACKED_MADNESS_TAG 93a9a5cec2a8fa87fba3afe8056607e6062a9058)
set(TTG_TRACKED_PARSEC_TAG 58f8f3089ecad2e8ee50e80a9586e05ce8873b1c)
set(TTG_TRACKED_BTAS_TAG c25b0a11d2a76190bfb13fa72f9e9dc3e57c3c2f)
set(TTG_TRACKED_TILEDARRAY_TAG 3b4baf7afed4ac7357290f3d90f5d4e3edbc09e0)

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
# BUT if will be building examples, inherit the oldest version from the pickiest Boost consumer (TA and/or BSPMM)
if (TTG_EXAMPLES)
  set(TTG_OLDEST_BOOST_VERSION 1.81)
else()
  set(TTG_OLDEST_BOOST_VERSION 1.66)
endif()
