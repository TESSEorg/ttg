find_package(Boost ${TTG_TRACKED_BOOST_VERSION} CONFIG QUIET HINTS ${Boost_DIR})

if (NOT TARGET Boost::boost)

  include(DownloadProject)
  download_project(PROJ                BOOST
    GIT_REPOSITORY      https://github.com/Orphis/boost-cmake
    PREFIX              ${PROJECT_BINARY_DIR}/external
    UPDATE_DISCONNECTED 1
    )

  add_subdirectory(${BOOST_SOURCE_DIR} ${BOOST_BINARY_DIR})

endif(NOT TARGET Boost::boost)
