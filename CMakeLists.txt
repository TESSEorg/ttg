cmake_minimum_required (VERSION 3.6)
project(ttg)

enable_language(CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")

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

##########################
#### prerequisites
##########################

# need Boost.CallableTraits (header only, part of Boost 1.66 released in Dec 2017) for wrap.h to work
find_package(Boost 1.66 REQUIRED)

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
# block-sparse tensor algorithms also need BTAS and (header-only) Boost.Iterator and Boost.Containers
if (Boost_FOUND)
  find_package(BTAS)
  message (STATUS "BTAS_FOUND=${BTAS_FOUND}")
  if (BTAS_FOUND)
    add_library(BTAS INTERFACE IMPORTED)
    set_property(TARGET BTAS PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES ${BTAS_INCLUDE_DIRS})
  endif(BTAS_FOUND)
endif()

##########################
#### prerequisite runtimes
##########################
# MADNESS
find_package(MADNESS 0.10.1 CONFIG QUIET COMPONENTS world HINTS ${MADNESS_ROOT_DIR})
message(STATUS "MADNESS_FOUND=${MADNESS_FOUND}")
# PaRSEC
find_package(PARSEC MODULE COMPONENTS MPI)
message(STATUS "PARSEC_FOUND=${PARSEC_FOUND}")
if (${PARSEC_FOUND})
    link_directories(${PARSEC_LIBRARY_DIRS})
endif(${PARSEC_FOUND})

##########################
#### Components
##########################
add_subdirectory(ttg)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(doc)

