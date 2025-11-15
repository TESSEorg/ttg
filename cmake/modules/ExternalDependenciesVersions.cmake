# for each dependency track both current and previous id (the variable for the latter must contain PREVIOUS)
# to be able to auto-update them

set(TTG_TRACKED_VG_CMAKE_KIT_TAG 94db3e9755ac55dbdb80f5a4cc7df1d3a29151b5)  # provides FindOrFetchLinalgPP and "real" FindOrFetchBoost
set(TTG_TRACKED_CATCH2_VERSION 3.5.0)
set(TTG_TRACKED_MADNESS_TAG a50e8d440fc2a1718a1ba0140af6866f49813d14)
set(TTG_TRACKED_PARSEC_TAG 996dda4c0ff3120bc65385f86e999befd4b3fe7a)
set(TTG_TRACKED_UMPIRE_CXX_ALLOCATOR_TAG a48ad360e20b9733263768b54aa24afe5894faa4)
set(TTG_TRACKED_BTAS_TAG c25b0a11d2a76190bfb13fa72f9e9dc3e57c3c2f)
set(TTG_TRACKED_TILEDARRAY_TAG 136ae0999ca0c750d77a55891b677fc3b2f6e00e)

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
# BUT if will be building examples, inherit the oldest version from the pickiest Boost consumer (TA and/or BSPMM)
if (TTG_EXAMPLES)
  set(TTG_OLDEST_BOOST_VERSION 1.81)
else()
  set(TTG_OLDEST_BOOST_VERSION 1.66)
endif()

# oldest Doxygen we can tolerate ... doxygen-awesome-css requires doxygen 1.9.1 - 1.9.4 or 1.9.6 - 1.14.0, so just use 1.9.6 as the minimum
set(TTG_OLDEST_DOXYGEN_VERSION 1.9.6)
set(TTG_TRACKED_DOXYGEN_AWESOME_CSS_VERSION v2.4.1)
