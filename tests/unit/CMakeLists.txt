# SPDX-License-Identifier: BSD-3-Clause
include(AddTTGExecutable)

# TT unit test: core TTG ops
set(ut_src
        fibonacci.cc
        ranges.cc
        broadcast_keygen.cc
        broadcast_algorithms.cc
        tt.cc
        unit_main.cpp
		streams.cc
    )
set(ut_libs Catch2::Catch2)

# coroutine tests
# we definitely have TARGET std::coroutine
if (CXXStdCoroutine_FOUND)
    list(APPEND ut_src fibonacci-coro.cc device_broadcast.cc)
    if (TTG_HAVE_CUDA)
        list(APPEND ut_src device_coro.cc cuda_kernel.cu)
    endif(TTG_HAVE_CUDA)
    list(APPEND ut_libs std::coroutine)
endif(CXXStdCoroutine_FOUND)

list(APPEND ut_src constraints.cc)

add_ttg_executable(core-unittests-ttg "${ut_src}" LINK_LIBRARIES "${ut_libs}" COMPILE_DEFINITIONS "CATCH_CONFIG_NO_POSIX_SIGNALS=1" )

# the keygen broadcast tests above use the default (binary tree) broadcast, run them again with the other broadcasts
if (TARGET core-unittests-ttg-parsec AND MPIEXEC_EXECUTABLE)
    foreach (bcast star pipe)
        foreach (p 1 2)
            add_test(NAME ttg/test/core-unittests-ttg-parsec/run-keygen-${bcast}-np-${p}
                     COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${p} ${MPIEXEC_PREFLAGS}
                             $<TARGET_FILE:core-unittests-ttg-parsec> ${MPIEXEC_POSTFLAGS} "[keygen]")
            set_tests_properties(ttg/test/core-unittests-ttg-parsec/run-keygen-${bcast}-np-${p}
                    PROPERTIES FIXTURES_REQUIRED TTG_TEST_core-unittests-ttg-parsec_FIXTURE
                    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                    ENVIRONMENT "TTG_NUM_THREADS=2;TTG_BCAST_TYPE=${bcast}")
        endforeach(p)
    endforeach(bcast)
endif ()

# serialization test: probes serialization via all supported serialization methods (MADNESS, Boost::serialization) that are available
add_ttg_executable(serialization serialization.cc unit_main.cpp
                   LINK_LIBRARIES Catch2::Catch2 ttg-serialization  $<TARGET_NAME_IF_EXISTS:BTAS::BTAS>
                   COMPILE_DEFINITIONS $<$<TARGET_EXISTS:BTAS::BTAS>:TTG_HAS_BTAS=1>)

# Boost serialization test: checks low-level codegen
if (TTG_PARSEC_USE_BOOST_SERIALIZATION)
    add_ttg_executable(serialization_boost serialization_boost.cc
                        LINK_LIBRARIES ttg-serialization-boost RUNTIMES "parsec")
endif(TTG_PARSEC_USE_BOOST_SERIALIZATION)

# TODO: convert into unit test
#if (TARGET MADworld)
#add_executable(splitmd_serialization splitmd_serialization.cc unit_main.cpp)
#endif(TARGET MADworld)


catch_discover_tests(serialization-parsec TEST_PREFIX "ttg/test/unit/")
catch_discover_tests(serialization-mad TEST_PREFIX "ttg/test/unit/")
