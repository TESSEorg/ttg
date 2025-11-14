# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

set(TTG_TRACKED_VG_CMAKE_KIT_TAG 98d2702a86fe1d5b519048f5459b700edbcae161)  # provides FindOrFetchLinalgPP and "real" FindOrFetchBoost
set(TTG_TRACKED_CATCH2_VERSION 3.5.0)
set(TTG_TRACKED_MADNESS_TAG a50e8d440fc2a1718a1ba0140af6866f49813d14)
set(TTG_TRACKED_PARSEC_TAG parsec-for-ttg)
set(TTG_TRACKED_UMPIRE_CXX_ALLOCATOR_TAG a48ad360e20b9733263768b54aa24afe5894faa4)
set(TTG_TRACKED_BTAS_TAG c25b0a11d2a76190bfb13fa72f9e9dc3e57c3c2f)
set(TTG_TRACKED_TILEDARRAY_TAG 42e0d65df9400469d29ebeb929b045be1caaf3d6)

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
# BUT if will be building examples, inherit the oldest version from the pickiest Boost consumer (TA and/or BSPMM)
if (TTG_EXAMPLES)
  set(TTG_OLDEST_BOOST_VERSION 1.81)
else()
  set(TTG_OLDEST_BOOST_VERSION 1.66)
endif()
