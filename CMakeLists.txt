cmake_minimum_required (VERSION 3.14) # supports FetchContenta_MakeAvailable

# Set TTG version =======================================================

# see https://semver.org/
set(TTG_MAJOR_VERSION 0)
set(TTG_MINOR_VERSION 1)
set(TTG_MICRO_VERSION 0)
set(TTG_PRERELEASE_ID )

set(TTG_VERSION "${TTG_MAJOR_VERSION}.${TTG_MINOR_VERSION}.${TTG_MICRO_VERSION}")
if (TTG_PRERELEASE_ID)
  set(TTG_EXT_VERSION "${TTG_VERSION}-${TTG_PRERELEASE_ID}")
else(TTG_PRERELEASE_ID)
  set(TTG_EXT_VERSION "${TTG_VERSION}")
endif(TTG_PRERELEASE_ID)

# Declare TTG =======================================================
project(ttg
    VERSION ${TTG_VERSION}
    DESCRIPTION "Task Template Graph: flowgraph programming model"
    LANGUAGES CXX
    HOMEPAGE_URL "https://tesseorg.github.io/ttg/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")
include(GNUInstallDirs)
include(ExternalDependenciesVersions)
include(LoadFetchContent)
include(AddCustomTargetSubproject)

set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

##########################
#### prerequisites
##########################
# MRA code needs CBLAS
find_package(CBLAS)
if (CBLAS_FOUND)
  message(STATUS "** CBLAS_LIBRARIES = ${CBLAS_LIBRARIES}")
  message(STATUS "** CBLAS_INCLUDE_DIR = ${CBLAS_INCLUDE_DIR}")
  message(STATUS "** CBLAS_INCLUDE_FILE = ${CBLAS_INCLUDE_FILE}")
endif()

# sparse tensor algorithms need Eigen
find_package(Eigen3)
message (STATUS "EIGEN3_FOUND=${EIGEN3_FOUND}")
if (EIGEN3_FOUND)
  add_library(eigen3 INTERFACE IMPORTED)
  set_property(TARGET eigen3 PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
endif (EIGEN3_FOUND)
# Boost
include(FindOrFetchBoost)
# block-sparse tensor algorithms also need BTAS and (header-only) Boost.Iterator and Boost.Containers
include(FindOrFetchBTAS)

##########################
#### prerequisite runtimes
##########################
# MADNESS
include(FindOrFetchMADNESS)
if (TARGET MADworld)
  message(STATUS "MADNESS_FOUND=1")
endif(TARGET MADworld)
# PaRSEC
include(FindOrFetchPARSEC)
if (TARGET PaRSEC::parsec)
  message(STATUS "PARSEC_FOUND=1")
endif(TARGET PaRSEC::parsec)

##########################
#### Components
##########################
add_subdirectory(ttg)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(doc)


