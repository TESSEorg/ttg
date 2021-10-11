cmake_minimum_required (VERSION 3.14) # supports FetchContent_MakeAvailable

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

# augment GNUInstallDirs's dirs with CMAKEDIR
set(CMAKE_INSTALL_CMAKEDIR "lib/cmake/ttg"
        CACHE PATH "TTG CMAKE install directory")

########################################
#### user-defined configuration options
########################################
option(TTG_PARSEC_USE_BOOST_SERIALIZATION "Whether to select Boost serialization methods in PaRSEC backend" ON)
# See https://medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (FORCE_COLORED_OUTPUT)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif (FORCE_COLORED_OUTPUT)

##########################
#### discover runtimes
##########################
# PaRSEC
include(FindOrFetchPARSEC)
if (TARGET PaRSEC::parsec)
    message(STATUS "PARSEC_FOUND=1")
endif(TARGET PaRSEC::parsec)
# MADNESS
include(FindOrFetchMADNESS)
if (TARGET MADworld)
    message(STATUS "MADNESS_FOUND=1")
endif(TARGET MADworld)

##########################
#### other prerequisites
##########################
# Boost
include(FindOrFetchBoost)
# Cereal
include(FindOrFetchCereal)

##########################
#### prerequisites for examples
##########################
# bspmm can use TiledArray for benchmarking
# n.b. load TiledArray first since it provides Eigen, Boost, and BTAS
option(TTG_TILEDARRAY "Whether to build the TiledArray tensor framework (MADNESS runtime only)" OFF)
if (TARGET MADworld AND TTG_TILEDARRAY)
    include(FindOrFetchTiledArray)
endif(TARGET MADworld AND TTG_TILEDARRAY)
# sparse tensor algorithms need Eigen
find_package(Eigen3)
message (STATUS "EIGEN3_FOUND=${EIGEN3_FOUND}")
if (EIGEN3_FOUND)
  add_library(eigen3 INTERFACE IMPORTED)
  set_property(TARGET eigen3 PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
endif (EIGEN3_FOUND)
# BTAS brings in linear algebra (BLAS++/LAPACK++)
include(FindOrFetchBTAS)
# OpenMP may also be used by some examples
find_package(OpenMP COMPONENTS CXX)
# std::execution may also be used by some examples
find_package(CXXStdExecution)
# bspmm can use libint2 to provide real data
option(TTG_LIBINT "Whether to build the Libint2 gaussian integrals library" OFF)
if (TTG_LIBINT)
    include(FindOrFetchLibint2)
endif(TTG_LIBINT)
# for benchmarking potrf
include(FindDPLASMA)

##########################
#### Components
##########################
add_subdirectory(ttg)

enable_testing(true)
add_custom_target_subproject(ttg check USES_TERMINAL COMMAND ${CMAKE_CTEST_COMMAND} -V -R "ttg/test/" )
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(doc)

# Create the version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(ttg-config-version.cmake
        VERSION ${TTG_VERSION} COMPATIBILITY AnyNewerVersion)

# Create the targets file
export(EXPORT ttg
       FILE "${PROJECT_BINARY_DIR}/ttg-targets.cmake")

# Create the configure file
configure_package_config_file(cmake/ttg-config.cmake.in
        "${PROJECT_BINARY_DIR}/ttg-config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
        PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_BINDIR
        CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
        CMAKE_INSTALL_DOCDIR CMAKE_INSTALL_CMAKEDIR)

# Install config, version, and target files
install(EXPORT ttg
        FILE "ttg-targets.cmake"
        DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
        COMPONENT ttg-config)
install(FILES
        "${PROJECT_BINARY_DIR}/ttg-config.cmake"
        "${PROJECT_BINARY_DIR}/ttg-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
        COMPONENT ttg-config)
