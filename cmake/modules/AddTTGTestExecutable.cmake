# Copyright 2020 Eduard F Valeyev
# Distributed under the OSI-approved BSD 3-Clause License.
# See https://opensource.org/licenses/BSD-3-Clause for details.

#
# add_ttg_test_executable(X) defines up to 3 tests:
# * ttg/X/build
# * ttg/X/run-np-1
# * ttg/X/run-np-2
#
# example: add_ttg_test_executable(X)
#

macro(add_ttg_test_executable _executable)
    add_test(ttg/test/${_executable}/build "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${_executable})
    set_tests_properties(ttg/test/${_executable}/build PROPERTIES FIXTURES_SETUP TTG_TEST_${_executable}_FIXTURE)
    foreach(p RANGE 1 2)
        # add run tests only if have MPIEXEC_EXECUTABLE
        if (MPIEXEC_EXECUTABLE)
            set(_ttg_test_${_executable}_run_cmd_${p} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${p} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${_executable}> ${MPIEXEC_POSTFLAGS})
    else (MPIEXEC_EXECUTABLE)
        set(_ttg_test_${_executable}_run_cmd_${p} echo "skipped TTG run test for executable ${_executable}: MPIEXEC_EXECUTABLE not found/given, see documentation for CMake's FindMPI module for instructions on how to help find it")
    endif (MPIEXEC_EXECUTABLE)
    add_test(NAME ttg/test/${_executable}/run-np-${p}
            COMMAND ${_ttg_test_${_executable}_run_cmd_${p}})
    set_tests_properties(ttg/test/${_executable}/run-np-${p}
            PROPERTIES FIXTURES_REQUIRED TTG_TEST_${_executable}_FIXTURE
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            ENVIRONMENT MAD_NUM_THREADS=2)
    endforeach(p)
endmacro()