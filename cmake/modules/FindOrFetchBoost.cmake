# Boost can be discovered by every (sub)package but only the top package can *build* it ...
# in either case must declare the components used by TA
set(required_components
        headers
        callable_traits
)
if (TTG_PARSEC_USE_BOOST_SERIALIZATION)
    list(APPEND required_components
            serialization
            iostreams
    )
endif()
if (DEFINED Boost_REQUIRED_COMPONENTS)
    list(APPEND Boost_REQUIRED_COMPONENTS
            ${required_components})
    list(REMOVE_DUPLICATES Boost_REQUIRED_COMPONENTS)
else()
    set(Boost_REQUIRED_COMPONENTS "${required_components}" CACHE STRING "Components of Boost to discovered or built")
endif()
set(optional_components
)
if (DEFINED Boost_OPTIONAL_COMPONENTS)
    list(APPEND Boost_OPTIONAL_COMPONENTS
            ${optional_components}
    )
    list(REMOVE_DUPLICATES Boost_OPTIONAL_COMPONENTS)
else()
    set(Boost_OPTIONAL_COMPONENTS "${optional_components}" CACHE STRING "Optional components of Boost to discovered or built")
endif()

if (NOT DEFINED Boost_FETCH_IF_MISSING AND TTG_FETCH_BOOST)
    set(Boost_FETCH_IF_MISSING 1)
endif()

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
endif()
include(${vg_cmake_kit_SOURCE_DIR}/modules/FindOrFetchBoost.cmake)

