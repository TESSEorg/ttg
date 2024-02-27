if (NOT TARGET tiledarray AND NOT MPQC_BUILD_DEPENDENCIES_FROM_SOURCE)
  if(TiledArray_INSTALL_DIR)
    set(TiledArray_DIR ${TiledArray_INSTALL_DIR}/lib/cmake/tiledarray)
  endif()
  find_package(TiledArray CONFIG QUIET COMPONENTS tiledarray)
endif ()

set(TA_PYTHON OFF)

if (TARGET tiledarray)
  message(STATUS "Found TiledArray CONFIG at ${TiledArray_CONFIG}")

  if ((NOT TA_ASSUMES_ASLR_DISABLED AND MPQC_ASSUMES_ASLR_DISABLED) OR (TA_ASSUMES_ASLR_DISABLED AND NOT MPQC_ASSUMES_ASLR_DISABLED))
    message(FATAL_ERROR "Found TiledArray configured with TA_ASSUMES_ASLR_DISABLED=${TA_ASSUMES_ASLR_DISABLED} but MPQC is configured with MPQC_ASSUMES_ASLR_DISABLED=${MPQC_ASSUMES_ASLR_DISABLED}; MPQC_ASSUMES_ASLR_DISABLED and TA_ASSUMES_ASLR_DISABLED should be the same")
  endif()

else (TARGET tiledarray)

  # enable CUDA if TTG has it
  set(ENABLE_CUDA ${TTG_HAVE_CUDA} CACHE BOOL "Enable CUDA")

  # update CMake cache for TA
  if (DEFINED MADNESS_CMAKE_EXTRA_ARGS)
    set(MADNESS_CMAKE_EXTRA_ARGS "${MADNESS_CMAKE_EXTRA_ARGS};-DENABLE_DQ_PREBUF=OFF" CACHE STRING "Extra CMake arguments to MADNESS" FORCE)
  else(DEFINED MADNESS_CMAKE_EXTRA_ARGS)
    set(MADNESS_CMAKE_EXTRA_ARGS "-DENABLE_DQ_PREBUF=OFF" CACHE STRING "Extra CMake arguments to MADNESS")
  endif(DEFINED MADNESS_CMAKE_EXTRA_ARGS)
  if (NOT DEFINED TA_ASSUMES_ASLR_DISABLED)
    set(TA_ASSUMES_ASLR_DISABLED ${MPQC_ASSUMES_ASLR_DISABLED} CACHE BOOL "TA assumes the Address Space Layout Randomization (ASLR) to be disabled")
  endif(NOT DEFINED TA_ASSUMES_ASLR_DISABLED)

  include(FetchContent)
  FetchContent_Declare(
      TILEDARRAY
      GIT_REPOSITORY      https://github.com/ValeevGroup/tiledarray.git
      GIT_TAG             ${TTG_TRACKED_TILEDARRAY_TAG}
  )
  FetchContent_MakeAvailable(TILEDARRAY)
  FetchContent_GetProperties(TILEDARRAY
      SOURCE_DIR TILEDARRAY_SOURCE_DIR
      BINARY_DIR TILEDARRAY_BINARY_DIR
      )
  # TA includes dependencies that are built manually, not using FetchContent, hence make sure we build them before building any MPQC code
  # add_dependencies(deps-mpqc External-tiledarray)

  set(TTG_DOWNLOADED_TILEDARRAY ON CACHE BOOL "Whether TTG downloaded TiledArray")

  include("${TILEDARRAY_BINARY_DIR}/cmake/modules/ReimportTargets.cmake")
  if (NOT TARGET MADworld)
    message(FATAL_ERROR "did not find re-imported target MADworld")
  endif(NOT TARGET MADworld)

  # this is where tiledarray-config.cmake will end up
  # must be in sync with the "install(FILES ...tiledarray-config.cmake" statement in https://github.com/ValeevGroup/tiledarray/blob/${MPQC_TRACKED_TILEDARRAY_TAG}/CMakeLists.txt
  set(TiledArray_CONFIG "${CMAKE_INSTALL_PREFIX}/${TILEDARRAY_INSTALL_CMAKEDIR}" CACHE INTERNAL "The location of installed tiledarray-config.cmake file")
endif(TARGET tiledarray)
