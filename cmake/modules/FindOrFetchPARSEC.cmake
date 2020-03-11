find_package(PaRSEC CONFIG QUIET COMPONENTS parsec HINTS ${PARSEC_ROOT_DIR})

if (NOT TARGET PaRSEC::parsec)

  # configure PaRSEC
  set(SUPPORT_FORTRAN OFF CACHE BOOL "Disable Fortran support in PaRSEC")
  set(CMAKE_CROSSCOMPILING OFF)
  set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})
  set(PARSEC_WITH_DEVEL_HEADERS ON CACHE BOOL "Install PaRSEC headers")

  include(DownloadProject)
  download_project(PROJ                PARSEC
    GIT_REPOSITORY      https://bitbucket.org/icldistcomp/parsec.git
    GIT_TAG             ${TTG_TRACKED_PARSEC_TAG}
    PREFIX              ${PROJECT_BINARY_DIR}/external
    UPDATE_DISCONNECTED 1
    )

  add_subdirectory(${PARSEC_SOURCE_DIR} ${PARSEC_BINARY_DIR})

endif(NOT TARGET PaRSEC::parsec)
