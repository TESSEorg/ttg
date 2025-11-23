# SPDX-License-Identifier: BSD-3-Clause
# - Try to find Libunwind
# Input variables:
#  LIBUNWIND_DIR          - The libunwind install directory;
#                           if not set the LIBUNWIND_DIR environment variable will be also queried
# Output variables:
#  LIBUNWIND_FOUND        - System has libunwind
#  LIBUNWIND_INCLUDE_DIR  - The libunwind include directories
#  LIBUNWIND_LIBRARIES    - The libraries needed to use libunwind
#  LIBUNWIND_HAS_UNW_INIT_LOCAL  - Whether Libunwind provides unw_init_local
#  LIBUNWIND_VERSION      - The version string for libunwind

include(FindPackageHandleStandardArgs)

if(NOT DEFINED LIBUNWIND_FOUND)

    # if not set already, set LIBUNWIND_ROOT_DIR from environment
    if (DEFINED ENV{LIBUNWIND_DIR} AND NOT DEFINED LIBUNWIND_DIR)
        set(LIBUNWIND_DIR $ENV{LIBUNWIND_DIR})
    endif()

    # Set default sarch paths for libunwind
    if(LIBUNWIND_DIR)
        set(LIBUNWIND_INCLUDE_PATH ${LIBUNWIND_DIR}/include CACHE PATH "The include directory for libunwind")
        if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(LIBUNWIND_LIBRARY_PATH ${LIBUNWIND_DIR}/lib64;${LIBUNWIND_DIR}/lib CACHE PATH "The library directory for libunwind")
        else()
            set(LIBUNWIND_LIBRARY_PATH ${LIBUNWIND_DIR}/lib CACHE PATH "The library directory for libunwind")
        endif()
    endif()

    find_path(LIBUNWIND_INCLUDE_DIR NAMES libunwind.h
            HINTS ${LIBUNWIND_INCLUDE_PATH})
    if(NOT EXISTS "${LIBUNWIND_INCLUDE_DIR}/unwind.h")
        MESSAGE("Found libunwind.h but corresponding unwind.h is absent!")
        SET(LIBUNWIND_INCLUDE_DIR "")
    endif()

    find_library(LIBUNWIND_LIBRARIES unwind
            HINTS ${LIBUNWIND_LIBRARY_PATH})

    # Get libunwind version
    if(EXISTS "${LIBUNWIND_INCLUDE_DIR}/libunwind-common.h")
        file(READ "${LIBUNWIND_INCLUDE_DIR}/libunwind-common.h" _libunwind_version_header)
        string(REGEX REPLACE ".*define[ \t]+UNW_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1"
                LIBUNWIND_MAJOR_VERSION "${_libunwind_version_header}")
        string(REGEX REPLACE ".*define[ \t]+UNW_VERSION_MINOR[ \t]+([0-9]+).*" "\\1"
                LIBUNWIND_MINOR_VERSION "${_libunwind_version_header}")
        string(REGEX REPLACE ".*define[ \t]+UNW_VERSION_EXTRA[ \t]+([0-9]+).*" "\\1"
                LIBUNWIND_MICRO_VERSION "${_libunwind_version_header}")
        if (LIBUNWIND_MICRO_VERSION)
          set(LIBUNWIND_VERSION "${LIBUNWIND_MAJOR_VERSION}.${LIBUNWIND_MINOR_VERSION}.${LIBUNWIND_MICRO_VERSION}")
        else()
            set(LIBUNWIND_VERSION "${LIBUNWIND_MAJOR_VERSION}.${LIBUNWIND_MINOR_VERSION}")
        endif()
        unset(_libunwind_version_header)
    endif()

    include(CMakePushCheckState)
    cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${LIBUNWIND_LIBRARIES})
    list(APPEND CMAKE_REQUIRED_INCLUDES ${LIBUNWIND_INCLUDE_DIR})
    set(CMAKE_REQUIRED_QUIET TRUE)
    include(CheckIncludeFileCXX)
    check_cxx_source_compiles("
        #define UNW_LOCAL_ONLY
        #include <libunwind.h>
        int main(int argc, char* argv[]) {
          int result = unw_init_local(nullptr, nullptr);
          return 0;
        }" LIBUNWIND_HAS_UNW_INIT_LOCAL)
    cmake_pop_check_state()

    # handle the QUIETLY and REQUIRED arguments and set LIBUNWIND_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(Libunwind
            FOUND_VAR LIBUNWIND_FOUND
            VERSION_VAR LIBUNWIND_VERSION
            REQUIRED_VARS LIBUNWIND_LIBRARIES LIBUNWIND_INCLUDE_DIR LIBUNWIND_HAS_UNW_INIT_LOCAL)

    mark_as_advanced(LIBUNWIND_INCLUDE_DIR LIBUNWIND_LIBRARIES LIBUNWIND_HAS_UNW_INIT_LOCAL)

endif()

if (NOT TARGET TTG_Libwunind)

    if (LIBUNWIND_FOUND AND LIBUNWIND_HAS_UNW_INIT_LOCAL)
        add_library(TTG_Libunwind INTERFACE)
        set_property(TARGET TTG_Libunwind PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES ${LIBUNWIND_INCLUDE_DIR})
        set_property(TARGET TTG_Libunwind PROPERTY
            INTERFACE_LINK_LIBRARIES ${LIBUNWIND_LIBRARIES})
        set_property(TARGET TTG_Libunwind PROPERTY
            INTERFACE_COMPILE_DEFINITIONS TTG_HAS_LIBUNWIND)
        install(TARGETS TTG_Libunwind EXPORT ttg COMPONENT ttg)
    endif()

endif(NOT TARGET TTG_Libwunind)