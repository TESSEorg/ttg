# SPDX-License-Identifier: BSD-3-Clause
if (NOT TARGET MADworld)
  find_package(MADNESS 0.10.1 CONFIG QUIET COMPONENTS world HINTS "${MADNESS_ROOT_DIR}")
  if (TARGET MADworld)
      message(STATUS "Found MADNESS: MADNESS_CONFIG=${MADNESS_CONFIG}")
  endif (TARGET MADworld)
endif (NOT TARGET MADworld)

if (NOT TARGET MADworld)
  set(MADNESS_BUILD_MADWORLD_ONLY ON CACHE BOOL "Whether to build MADNESS runtime only")
  set(MADNESS_TASK_BACKEND PaRSEC CACHE STRING "The task backend to use for MADNESS tasks")
  FetchContent_Declare(
          MADNESS
          GIT_REPOSITORY https://github.com/m-a-d-n-e-s-s/madness.git
          GIT_TAG ${TTG_TRACKED_MADNESS_TAG}
  )
  FetchContent_MakeAvailable(MADNESS)
  FetchContent_GetProperties(MADNESS
          SOURCE_DIR MADNESS_SOURCE_DIR
          BINARY_DIR MADNESS_BINARY_DIR
          )

  # set MADNESS_CONFIG to the install location so that we know where to find it
  set(MADNESS_CONFIG ${CMAKE_INSTALL_PREFIX}/${MADNESS_INSTALL_CMAKEDIR}/madness-config.cmake)

endif(NOT TARGET MADworld)

# postcond check
if (NOT TARGET MADworld)
  message(FATAL_ERROR "FindOrFetchMADNESS could not make MADworld target available")
endif(NOT TARGET MADworld)
