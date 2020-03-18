find_package(MADNESS 0.10.1 CONFIG QUIET COMPONENTS world HINTS "${MADNESS_ROOT_DIR}")

if (NOT TARGET MADworld)

  include(DownloadProject)
  download_project(PROJ                MADNESS
    GIT_REPOSITORY      https://github.com/m-a-d-n-e-s-s/madness.git
    GIT_TAG             ${TTG_TRACKED_MADNESS_TAG}
    PREFIX              "${PROJECT_BINARY_DIR}/external"
    UPDATE_DISCONNECTED 1
    )

  add_subdirectory("${MADNESS_SOURCE_DIR}" "${MADNESS_BINARY_DIR}")

endif(NOT TARGET MADworld)
