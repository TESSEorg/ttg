# SPDX-License-Identifier: BSD-3-Clause
#-- Add an Option to toggle the generation of the API documentation

find_package(Doxygen ${TTG_OLDEST_DOXYGEN_VERSION})

if(DOXYGEN_FOUND)
  # Fetch Doxygen Awesome CSS theme
  include(FetchContent)
  FetchContent_Declare(
    doxygen-awesome-css
    GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG ${TTG_TRACKED_DOXYGEN_AWESOME_CSS_VERSION}
  )
  FetchContent_MakeAvailable(doxygen-awesome-css)
  # Save the location the files were cloned into
  # This allows us to get the path to doxygen-awesome.css
  FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

  configure_file(config/Doxyfile.in config/Doxyfile @ONLY IMMEDIATE)
  # uses standard GNU target name for building html documentation, "html"
  add_custom_target_subproject (ttg html USES_TERMINAL
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/dox/config/Doxyfile
    SOURCES ${PROJECT_BINARY_DIR}/doc/dox/config/Doxyfile)
  add_dependencies(doc-ttg html-ttg)
endif()

if (TTG_EXAMPLES)
  add_subdirectory(user/examples)
endif()
