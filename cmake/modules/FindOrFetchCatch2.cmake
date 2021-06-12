find_package(Catch2 ${TTG_TRACKED_CATCH2_VERSION} QUIET)

if (NOT TARGET Catch2::Catch2)

  FetchContent_Declare(
          Catch2
          GIT_REPOSITORY https://github.com/catchorg/Catch2.git
          GIT_TAG v${TTG_TRACKED_CATCH2_VERSION})

  FetchContent_MakeAvailable(Catch2)
  FetchContent_GetProperties(Catch2
          SOURCE_DIR Catch2_SOURCE_DIR
          BINARY_DIR Catch2_BINARY_DIR
          )

endif (NOT TARGET Catch2::Catch2)

# postcond check
if (NOT TARGET Catch2::Catch2)
  message(FATAL_ERROR "FindOrFetchCatch2 could not make Catch2::Catch2 target available")
endif (NOT TARGET Catch2::Catch2)
