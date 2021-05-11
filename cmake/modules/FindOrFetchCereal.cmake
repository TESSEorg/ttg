if (NOT TARGET cereal::cereal)
  # find_package(cereal ${TTG_TRACKED_CEREAL_VERSION} QUIET)
  # homebrew on macos provides cereal-config with version "unknown"
  #find_package(cereal)
  if (cereal_FOUND AND NOT TARGET cereal::cereal)
    if (TARGET cereal)
      add_library(cereal::cereal ALIAS cereal)
    else ()
      message(FATAL_ERROR "cereal_FOUND=TRUE but no cereal target")
    endif()
  endif()
endif(NOT TARGET cereal::cereal)

if (TARGET cereal::cereal)
  message(STATUS "Found cereal at ${cereal_CONFIG}")
else (TARGET cereal::cereal)
  # going hungry today
endif()

# fetchcontent is disabled for now
if (FALSE)
  FetchContent_Declare(
          cereal
          GIT_REPOSITORY      https://github.com/USCiLab/cereal
          GIT_TAG v${TTG_TRACKED_CEREAL_VERSION})
  FetchContent_MakeAvailable(cereal)
  FetchContent_GetProperties(cereal
          SOURCE_DIR CEREAL_SOURCE_DIR
          BINARY_DIR CEREAL_BINARY_DIR
          )

endif(FALSE)
