# update the Boost version that we can tolerate
if (NOT DEFINED Boost_OLDEST_BOOST_VERSION)
    set(Boost_OLDEST_BOOST_VERSION ${TTG_OLDEST_BOOST_VERSION})
else()
    if (${Boost_OLDEST_BOOST_VERSION} VERSION_LESS ${TTG_OLDEST_BOOST_VERSION})
        if (DEFINED CACHE{Boost_OLDEST_BOOST_VERSION})
            set(Boost_OLDEST_BOOST_VERSION "${TTG_OLDEST_BOOST_VERSION}" CACHE STRING "Oldest Boost version to use" FORCE)
        else()
            set(Boost_OLDEST_BOOST_VERSION ${TTG_OLDEST_BOOST_VERSION})
        endif()
    endif()
endif()

# Boost can be discovered by every (sub)package but only the top package can *build* it ...
# in either case must declare the components used by TTG
set(required_components
        headers
        callable_traits
)
set(optional_components
)
if (TTG_PARSEC_USE_BOOST_SERIALIZATION)
    list(APPEND optional_components
            serialization
            iostreams
    )
else()
    list(APPEND BOOST_EXCLUDE_LIBRARIES iostreams)  # install of this library fails unless it's already built
endif()
if (BUILD_EXAMPLES)
    list(APPEND optional_components
            bimap   # dependency of graph
            foreach # dependency of graph
            graph   # used for generation of inputs for spmm
            property_map # dependency of graph
            xpressive # dependency of graph
    )
endif()

# if not allowed to fetch Boost make all Boost optional
if (NOT DEFINED Boost_FETCH_IF_MISSING AND TTG_FETCH_BOOST)
    set(Boost_FETCH_IF_MISSING 1)
endif()
if (NOT Boost_FETCH_IF_MISSING)
    foreach(__component IN LISTS required_components)
    list(APPEND optional_components
            ${__component}
    )
    endforeach()
    set(required_components )
endif()

if (DEFINED Boost_REQUIRED_COMPONENTS)
    list(APPEND Boost_REQUIRED_COMPONENTS
            ${required_components})
    list(REMOVE_DUPLICATES Boost_REQUIRED_COMPONENTS)
else()
    set(Boost_REQUIRED_COMPONENTS "${required_components}" CACHE STRING "Components of Boost to discovered or built")
endif()
if (DEFINED Boost_OPTIONAL_COMPONENTS)
    list(APPEND Boost_OPTIONAL_COMPONENTS
            ${optional_components}
    )
    list(REMOVE_DUPLICATES Boost_OPTIONAL_COMPONENTS)
else()
    set(Boost_OPTIONAL_COMPONENTS "${optional_components}" CACHE STRING "Optional components of Boost to discovered or built")
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
if (Boost_BUILT_FROM_SOURCE)
    set(TTG_BUILT_BOOST_FROM_SOURCE 1)
endif()

if (TARGET Boost::headers)
    set(TTG_HAS_BOOST 1)
endif()
