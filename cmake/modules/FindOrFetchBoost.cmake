find_package(Boost ${TTG_TRACKED_BOOST_VERSION} QUIET)

if (NOT TARGET Boost::boost)

  include(DownloadProject)
  download_project(PROJ                BOOST
    GIT_REPOSITORY      https://github.com/Orphis/boost-cmake
    PREFIX              "${PROJECT_BINARY_DIR}/external"
    UPDATE_DISCONNECTED 1
    )

  add_subdirectory("${BOOST_SOURCE_DIR}" "${BOOST_BINARY_DIR}")

  if (NOT TARGET Boost::boost)
    message(FATAL_ERROR "Downloaded and configured boost, but Boost::boost is still not found. Please create an issue at ${PROJECT_HOMEPAGE_URL}")
  endif(NOT TARGET Boost::boost)

endif(NOT TARGET Boost::boost)

