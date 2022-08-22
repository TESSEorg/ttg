# Copyright 2020 Eduard F Valeyev
# Distributed under the OSI-approved BSD 3-Clause License.
# See https://opensource.org/licenses/BSD-3-Clause for details.

#
# add_ttg_executable(X sources_list [RUNTIMES runtime_list] [LINK_LIBRARIES link_libraries] [COMPILE_DEFINITIONS compile_definitions] [COMPILE_FEATURES compile_features] [SINGLERANKONLY] [TEST_CMDARGS test_cmdline_args])
#
# creates executable X-r for every r in runtime_list:
# * if RUNTIMES is omitted, will use all known runtimes, otherwise use the specified runtimes
#
# example: add_ttg_executable(test "test/test1.cc;test/test2.cc" RUNTIMES "mad" LINK_LIBRARIES "BTAS" COMPILE_DEFINITIONG "COOL_DEFINE=1" COMPILE_FEATURES "cxx_std_20" NOT_EXCLUDE_FROM_ALL SINGLERANKONLY)
#

include(AddTTGTestExecutable)

macro(add_ttg_executable)

    set(optionArgs SINGLERANKONLY NOT_EXCLUDE_FROM_ALL)
    set(multiValueArgs RUNTIMES LINK_LIBRARIES COMPILE_DEFINITIONS COMPILE_FEATURES TEST_CMDARGS)
    cmake_parse_arguments(ADD_TTG_EXECUTABLE "${optionArgs}" ""
            "${multiValueArgs}" ${ARGN})

    list(LENGTH ADD_TTG_EXECUTABLE_UNPARSED_ARGUMENTS _num_unparsed_args)
    if (${_num_unparsed_args} LESS 2)
        message(FATAL_ERROR "wrong number of arguments to add_ttg_executable: must provide executable name and list of source files")
    endif()

    list(POP_FRONT ADD_TTG_EXECUTABLE_UNPARSED_ARGUMENTS _executable)
    set(_sources_list "${ADD_TTG_EXECUTABLE_UNPARSED_ARGUMENTS}")

    if (NOT DEFINED ADD_TTG_EXECUTABLE_RUNTIMES)
        set(ADD_TTG_EXECUTABLE_RUNTIMES )
        if (TARGET MADworld)
            list(APPEND ADD_TTG_EXECUTABLE_RUNTIMES "mad")
        endif()
        if (TARGET PaRSEC::parsec)
            list(APPEND ADD_TTG_EXECUTABLE_RUNTIMES "parsec")
        endif()
    endif()

    foreach(r ${ADD_TTG_EXECUTABLE_RUNTIMES})
        if (r STREQUAL "mad" AND NOT TARGET MADworld)
            message(FATAL_ERROR "add_ttg_executable: MADNESS runtime is not available, but requested")
        elseif(r STREQUAL "parsec" AND NOT TARGET PaRSEC::parsec)
            message(FATAL_ERROR "add_ttg_executable: PaRSEC runtime is not available, but requested")
        endif()
    endforeach()

    foreach(r ${ADD_TTG_EXECUTABLE_RUNTIMES})

        set(_compile_definitions "TTG_EXECUTABLE=1")
        if (r STREQUAL "mad")
            list(APPEND _compile_definitions "TTG_USE_MADNESS=1")
        elseif(r STREQUAL "parsec")
            list(APPEND _compile_definitions "TTG_USE_PARSEC=1")
        endif()
        if (DEFINED ADD_TTG_EXECUTABLE_COMPILE_DEFINITIONS)
            list(APPEND _compile_definitions "${ADD_TTG_EXECUTABLE_COMPILE_DEFINITIONS}")
        endif()

        set(_link_libraries ttg-${r})
        if (DEFINED ADD_TTG_EXECUTABLE_LINK_LIBRARIES)
            list(APPEND _link_libraries "${ADD_TTG_EXECUTABLE_LINK_LIBRARIES}")
        endif()

        set(_compile_features )
        if (DEFINED ADD_TTG_EXECUTABLE_COMPILE_FEATURES)
            list(APPEND _compile_features "${ADD_TTG_EXECUTABLE_COMPILE_FEATURES}")
        endif ()

        if (NOT ADD_TTG_EXECUTABLE_NOT_EXCLUDE_FROM_ALL)
            add_executable(${_executable}-${r} EXCLUDE_FROM_ALL "${_sources_list}")
        else()
            add_executable(${_executable}-${r} "${_sources_list}")
        endif()
        target_compile_definitions(${_executable}-${r} PRIVATE "${_compile_definitions}")
        target_link_libraries(${_executable}-${r} PRIVATE "${_link_libraries}")
        if (_compile_features)
            target_compile_features(${_executable}-${r} PRIVATE "${_compile_features}")
        endif (_compile_features)

        set(_ranksrange 1)
        if (ADD_TTG_EXECUTABLE_SINGLERANKONLY)
            list(APPEND _ranksrange 1)
        else ()
            list(APPEND _ranksrange 2)
        endif ()

        add_ttg_test_executable(${_executable}-${r} "${_ranksrange}" "${ADD_TTG_EXECUTABLE_TEST_CMDARGS}")
    endforeach()

endmacro()
