cmake_minimum_required (VERSION 3.14) # supports FetchContent_MakeAvailable

# Set TTG version =======================================================

# see https://semver.org/
set(TTG_MAJOR_VERSION 1)
set(TTG_MINOR_VERSION 0)
set(TTG_MICRO_VERSION 0)
set(TTG_PRERELEASE_ID alpha)

set(TTG_VERSION "${TTG_MAJOR_VERSION}.${TTG_MINOR_VERSION}.${TTG_MICRO_VERSION}")
if (TTG_PRERELEASE_ID)
  set(TTG_EXT_VERSION "${TTG_VERSION}-${TTG_PRERELEASE_ID}")
else(TTG_PRERELEASE_ID)
  set(TTG_EXT_VERSION "${TTG_VERSION}")
endif(TTG_PRERELEASE_ID)

# CMake defaults to address key pain points =========================

# safety net for dev workflow: accidental install will not affect FindOrFetch*
if (NOT DEFINED CACHE{CMAKE_FIND_NO_INSTALL_PREFIX})
    set(CMAKE_FIND_NO_INSTALL_PREFIX ON CACHE BOOL "Whether find_* commands will search CMAKE_INSTALL_PREFIX and CMAKE_STAGING_PREFIX; see https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_NO_INSTALL_PREFIX.html#variable:CMAKE_FIND_NO_INSTALL_PREFIX")
endif()

# Declare TTG =======================================================
project(ttg
    VERSION ${TTG_VERSION}
    DESCRIPTION "Template Task Graph: flowgraph programming model for distributed-memory computing"
    LANGUAGES CXX
    HOMEPAGE_URL "https://tesseorg.github.io/ttg/")

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD "20" CACHE STRING "The C++ standard")
endif()
if (NOT CMAKE_CXX_EXTENSIONS)
  set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "The C++ standard extensions allowed?")
endif()

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
option(TTG_EXAMPLES "Whether to build examples" OFF)

option(TTG_FETCH_BOOST "Whether to fetch+build Boost, if missing" OFF)
option(TTG_IGNORE_BUNDLED_EXTERNALS "Whether to skip installation and use of bundled external depenedencies (Boost.CallableTraits)" OFF)
option(TTG_ENABLE_TRACE "Whether to enable ttg::trace() output" OFF)
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
#### prerequisites
##########################
include(CTest)
enable_testing()
if (BUILD_TESTING)
    include(FindOrFetchCatch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
    include(Catch)
endif (BUILD_TESTING)

###########################
#### optional prerequisites
###########################
# Boost
include(FindOrFetchBoost)
# Cereal
#include(FindOrFetchCereal)
# C++ coroutines
find_package(CXXStdCoroutine MODULE REQUIRED COMPONENTS Final Experimental)


##########################
#### CUDA: must come before PaRSEC
##########################
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif(CMAKE_CUDA_COMPILER)
set(TTG_HAVE_CUDA ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if TTG supports compiling .cu files")

find_package(CUDAToolkit)
if (TARGET CUDA::cudart)
  set(TTG_HAVE_CUDART True CACHE BOOL "True support for CUDART was found.")
endif(TARGET CUDA::cudart)


##########################
#### prerequisite runtimes
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
#### Examples
##########################
# N.B. discover last so that we do not affect core dependencies
if (TTG_EXAMPLES)
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
endif (TTG_EXAMPLES)


##########################
#### Components
##########################
add_subdirectory(ttg)

if (BUILD_TESTING OR TTG_EXAMPLES)
    add_custom_target_subproject(ttg check USES_TERMINAL COMMAND ${CMAKE_CTEST_COMMAND} -V -R "ttg/test/" )
else()
    add_custom_target_subproject(ttg check USES_TERMINAL COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red "check-ttg target disabled since neither BUILD_TESTING nor TTG_EXAMPLES is true" )
endif()
if (BUILD_TESTING)
  add_subdirectory(tests)
endif(BUILD_TESTING)
if (TTG_EXAMPLES)
  add_subdirectory(examples)
endif(TTG_EXAMPLES)
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

# Install some CMake modules
install(FILES
        "${PROJECT_SOURCE_DIR}/cmake/modules/AddTTGExecutable.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/modules/AddTTGTestExecutable.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/modules/FindCXXStdCoroutine.cmake"
        DESTINATION "${CMAKE_INSTALL_CMAKEDIR}/modules"
        COMPONENT ttg-config)

# Install utilities
install(PROGRAMS
        "${PROJECT_SOURCE_DIR}/bin/pbt_to_ctf.py"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/"
        COMPONENT ttg-utils)

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
