# SPDX-License-Identifier: BSD-3-Clause
# Bring ValeevGroup cmake toolkit, if not yet available
if (NOT DEFINED vg_cmake_kit_SOURCE_DIR)
    include(FetchContent)
    if (DEFINED PROJECT_BINARY_DIR)
        set(VG_CMAKE_KIT_PREFIX_DIR PROJECT_BINARY_DIR)
    else ()
        set(VG_CMAKE_KIT_PREFIX_DIR CMAKE_CURRENT_BINARY_DIR)
    endif()
    FetchContent_Declare(
            vg_cmake_kit
            QUIET
            GIT_REPOSITORY      https://github.com/ValeevGroup/kit-cmake.git
            GIT_TAG             ${TTG_TRACKED_VG_CMAKE_KIT_TAG}
            SOURCE_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg
            BINARY_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg-build
            SUBBUILD_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg-subbuild
    )
    FetchContent_MakeAvailable(vg_cmake_kit)
    list(APPEND CMAKE_MODULE_PATH "${vg_cmake_kit_SOURCE_DIR}/modules")
endif()