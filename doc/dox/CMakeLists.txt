#-- Add an Option to toggle the generation of the API documentation

find_package(Doxygen 1.8.12)
if(DOXYGEN_FOUND)
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
