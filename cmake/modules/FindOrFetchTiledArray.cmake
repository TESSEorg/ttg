if (NOT TARGET tiledarray)
  find_package(TiledArray CONFIG QUIET COMPONENTS tiledarray)
endif (NOT TARGET tiledarray)

set(TA_PYTHON OFF)

if (TARGET tiledarray)
  message(STATUS "Found TiledArray CONFIG at ${TiledArray_CONFIG}")

else (TARGET tiledarray)

  # update CMake cache for TA
  if (DEFINED MADNESS_CMAKE_EXTRA_ARGS)
    set(MADNESS_CMAKE_EXTRA_ARGS "${MADNESS_CMAKE_EXTRA_ARGS};-DENABLE_DQ_PREBUF=OFF" CACHE STRING "Extra CMake arguments to MADNESS" FORCE)
  else(DEFINED MADNESS_CMAKE_EXTRA_ARGS)
    set(MADNESS_CMAKE_EXTRA_ARGS "-DENABLE_DQ_PREBUF=OFF" CACHE STRING "Extra CMake arguments to MADNESS")
  endif(DEFINED MADNESS_CMAKE_EXTRA_ARGS)
  if (NOT DEFINED TA_ASSUMES_ASLR_DISABLED)
    set(TA_ASSUMES_ASLR_DISABLED ${MPQC_ASSUMES_ASLR_DISABLED} CACHE BOOL "TA assumes the Address Space Layout Randomization (ASLR) to be disabled")
  endif(NOT DEFINED TA_ASSUMES_ASLR_DISABLED)
  if (NOT DEFINED TA_BUILD_UNITTEST)
    set(TA_BUILD_UNITTEST FALSE CACHE BOOL "Whether to build TA unit tests")
  endif (NOT DEFINED TA_BUILD_UNITTEST)
  set(TA_ASSERT_POLICY TA_ASSERT_THROW CACHE STRING "")

  include(FetchContent)
  FetchContent_Declare(
      TILEDARRAY
      GIT_REPOSITORY      https://github.com/ValeevGroup/tiledarray.git
      GIT_TAG             ec9b23fbc0489664ef96e3d4e9851d42979185a8
  )
  FetchContent_MakeAvailable(TILEDARRAY)
  FetchContent_GetProperties(TILEDARRAY
      SOURCE_DIR TILEDARRAY_SOURCE_DIR
      BINARY_DIR TILEDARRAY_BINARY_DIR
      )

  include("${TILEDARRAY_BINARY_DIR}/cmake/modules/ReimportTargets.cmake")
  if (NOT TARGET MADworld)
    message(FATAL_ERROR "did not find re-imported target MADworld")
  endif(NOT TARGET MADworld)

  # this is where tiledarray-config.cmake will end up
  # must be in sync with the "install(FILES ...tiledarray-config.cmake" statement in https://github.com/ValeevGroup/tiledarray/blob/${MPQC_TRACKED_TILEDARRAY_TAG}/CMakeLists.txt
  set(TiledArray_CONFIG "${CMAKE_INSTALL_PREFIX}/${TILEDARRAY_INSTALL_CMAKEDIR}" CACHE INTERNAL "The location of installed tiledarray-config.cmake file")
endif(TARGET tiledarray)
