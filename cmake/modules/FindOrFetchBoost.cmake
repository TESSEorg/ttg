find_package(Boost ${TTG_TRACKED_BOOST_VERSION} QUIET)

if (NOT TARGET Boost::boost)

  FetchContent_Declare(
          BOOST
          GIT_REPOSITORY      https://github.com/Orphis/boost-cmake
  )
  FetchContent_MakeAvailable(BOOST)
  FetchContent_GetProperties(BOOST
          SOURCE_DIR BOOST_SOURCE_DIR
          BINARY_DIR BOOST_BINARY_DIR
          )

endif(NOT TARGET Boost::boost)

# postcond check
if (NOT TARGET Boost::boost)
  message(FATAL_ERROR "FindOrFetchBOOST could not make Boost::boost target available")
endif(NOT TARGET Boost::boost)
