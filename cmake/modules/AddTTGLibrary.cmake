# Copyright 2021 Eduard F Valeyev
# Distributed under the OSI-approved BSD 3-Clause License.
# See https://opensource.org/licenses/BSD-3-Clause for details.

#
# add_ttg_library(X sources_list [PUBLIC_HEADER header_list] [LINK_LIBRARIES link_libraries] [COMPILE_DEFINITIONS compile_definitions])
#
# creates library ttg-X
#
# example: add_ttg_library(ttg-test "test/test1.cc;test/test2.cc" PUBLIC_HEADER "test/test.h" LINK_LIBRARIES ttg-serialization INCLUDE_DIRECTORIES "external/boost" )
#          creates library ttg-test that depends on ttg-serialization
#

macro(add_ttg_library)

    set(optionArgs )
    set(multiValueArgs PUBLIC_HEADER LINK_LIBRARIES COMPILE_DEFINITIONS INCLUDE_DIRECTORIES)
    cmake_parse_arguments(ADD_TTG_LIBRARY "${optionArgs}" ""
            "${multiValueArgs}" ${ARGN})

    list(LENGTH ADD_TTG_LIBRARY_UNPARSED_ARGUMENTS _num_unparsed_args)
    if (${_num_unparsed_args} LESS 2)
        message(FATAL_ERROR "wrong number of arguments to add_ttg_library: must provide library name and list of source files")
    endif()

    list(POP_FRONT ADD_TTG_LIBRARY_UNPARSED_ARGUMENTS _library)
    set(_sources_list "${ADD_TTG_LIBRARY_UNPARSED_ARGUMENTS}")

    set(_header_only TRUE)
    foreach(file ${_sources_list})
        if (file MATCHES ".*\\.cpp$" OR file MATCHES ".*\\.cc$" OR file MATCHES ".*\\.cu")
            set(_header_only FALSE)
        endif()
    endforeach()

    if (NOT _header_only)
      add_library(${_library} "${_sources_list}")

      foreach(_dep ${ADD_TTG_LIBRARY_LINK_LIBRARIES})
        if(TARGET ${_dep})
            target_link_libraries(${_library} PUBLIC "${_dep}")
        else(TARGET ${_dep})
            message(FATAL_ERROR "TTG library ${_library} is declared to depend on ${_dep}, but the target is missing")
        endif(TARGET ${_dep})
      endforeach(_dep)

      target_compile_definitions(${_library} PUBLIC "${ADD_TTG_LIBRARY_COMPILE_DEFINITIONS}")
      target_compile_features(${_library} PUBLIC cxx_std_17)

      target_include_directories(${_library} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>  # look in binary dir also for files preprocessed by configure_file
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            )

      foreach(_inc ${ADD_TTG_LIBRARY_INCLUDE_DIRECTORIES})
          if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_inc}")
              target_include_directories(${_library} PUBLIC
                      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${_inc}>
                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${_inc}>
                      )
          elseif(EXISTS "${_inc}")
              target_include_directories(${_library} PUBLIC ${_inc})
          elseif("${_inc}" MATCHES "_INTERFACE")
              target_include_directories(${_library} PUBLIC ${_inc})
          endif()
      endforeach(_inc)

    else (NOT _header_only)
        add_library(${_library} INTERFACE)

        foreach(_dep ${ADD_TTG_LIBRARY_LINK_LIBRARIES})
            if(TARGET ${_dep})
                target_link_libraries(${_library} INTERFACE "${_dep}")
            else(TARGET ${_dep})
                message(FATAL_ERROR "TTG library ${_library} is declared to depend on ${_dep}, but the target is missing")
            endif(TARGET ${_dep})
        endforeach(_dep)

        target_compile_definitions(${_library} INTERFACE "${ADD_TTG_LIBRARY_COMPILE_DEFINITIONS}")
        target_compile_features(${_library} INTERFACE cxx_std_17)

        # Use current CMAKE_CXX_FLAGS to compile targets dependent on this library
        string (REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
        target_compile_options(${_library} INTERFACE $<INSTALL_INTERFACE:${CMAKE_CXX_FLAG_LIST}>)

        foreach(_inc ${ADD_TTG_LIBRARY_INCLUDE_DIRECTORIES})
            message(STATUS "_inc=${CMAKE_CURRENT_SOURCE_DIR}/${_inc}")
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_inc}")
                target_include_directories(${_library} INTERFACE
                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${_inc}>
                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${_inc}>
                        )
            elseif(EXISTS "${_inc}")
                target_include_directories(${_library} INTERFACE ${_inc})
            elseif("${_inc}" MATCHES "_INTERFACE")
                target_include_directories(${_library} INTERFACE ${_inc})
            endif()
        endforeach(_inc)

    endif(NOT _header_only)

    # this does not work with hierarchies of header files
    # see here for possible workaround if frameworks are really needed:
    #     http://cmake.3232098.n2.nabble.com/Install-header-directory-hierarchy-td5638507.html
    # set_target_properties(${_library} PROPERTIES PUBLIC_HEADER "${ADD_TTG_LIBRARY_PUBLIC_HEADER}")
    # install manually
    foreach ( file ${ADD_TTG_LIBRARY_PUBLIC_HEADER} )
        # N.B. some files are in the build tree
        if ("${file}" MATCHES "^${PROJECT_SOURCE_DIR}/ttg")
            file(RELATIVE_PATH _rel_file_path "${PROJECT_SOURCE_DIR}/ttg" "${file}")
        elseif("${file}" MATCHES "^${PROJECT_BINARY_DIR}/ttg")
            file(RELATIVE_PATH _rel_file_path "${PROJECT_BINARY_DIR}/ttg" "${file}")
        else()
            message(FATAL_ERROR "AddTTGLibrary: could not deduce install location for public header ${file} of component ${_library}")
        endif()
        get_filename_component( dir "${_rel_file_path}" DIRECTORY )
        install( FILES ${file} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${dir}" COMPONENT ${_library})
    endforeach()

    # Add library to the list of installed components
    install(TARGETS ${_library} EXPORT ttg
            COMPONENT ${_library}
            PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

endmacro()
