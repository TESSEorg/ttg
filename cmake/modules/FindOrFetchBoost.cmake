if (NOT TARGET Boost::boost)
  find_package(Boost ${TTG_TRACKED_BOOST_VERSION} QUIET OPTIONAL_COMPONENTS serialization)
endif(NOT TARGET Boost::boost)

if (TARGET Boost::boost)
  set(_msg "Found Boost at ${Boost_CONFIG}")
  if (TARGET Boost::serialization)
    list(APPEND _msg " includes Boost::serialization")
  endif(TARGET Boost::serialization)
  message(STATUS "${_msg}")
else (TARGET Boost::boost)

  FetchContent_Declare(
          CMAKEBOOST
          GIT_REPOSITORY      https://github.com/Orphis/boost-cmake
  )
  FetchContent_MakeAvailable(CMAKEBOOST)
  FetchContent_GetProperties(CMAKEBOOST
          SOURCE_DIR CMAKEBOOST_SOURCE_DIR
          BINARY_DIR CMAKEBOOST_BINARY_DIR
          )

endif(TARGET Boost::boost)

# postcond check
if (NOT TARGET Boost::boost)
  message(FATAL_ERROR "FindOrFetchBoost could not make Boost::boost target available")
endif(NOT TARGET Boost::boost)
