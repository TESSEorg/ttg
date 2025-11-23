# SPDX-License-Identifier: BSD-3-Clause
include(AddTTGExecutable)

# TT unit test: core TTG ops
set(ut_src
        fibonacci.cc
        ranges.cc
        tt.cc
        unit_main.cpp
		streams.cc
    )
set(ut_libs Catch2::Catch2)

# coroutine tests
# we definitely have TARGET std::coroutine
if (CXXStdCoroutine_FOUND)
    list(APPEND ut_src fibonacci-coro.cc)
    list(APPEND ut_src device_coro.cc)
    if (TTG_HAVE_CUDA)
        list(APPEND ut_src cuda_kernel.cu)
    endif(TTG_HAVE_CUDA)
    list(APPEND ut_libs std::coroutine)
endif(CXXStdCoroutine_FOUND)

list(APPEND ut_src constraints.cc)

add_ttg_executable(core-unittests-ttg "${ut_src}" LINK_LIBRARIES "${ut_libs}" COMPILE_DEFINITIONS "CATCH_CONFIG_NO_POSIX_SIGNALS=1" )

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
