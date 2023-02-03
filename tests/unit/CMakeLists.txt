include(AddTTGExecutable)

# TT unit test: core TTG ops
set(ut_src
        #fibonacci.cc
        #ranges.cc
        #tt.cc
        unit_main.cpp
    )
set(ut_libs Catch2::Catch2)
#if (TARGET std::coroutine)
    #list(APPEND ut_src fibonacci-coro.cc)
    list(APPEND ut_src device_coro.cc)
    list(APPEND ut_libs std::coroutine)
#endif()
add_ttg_executable(core-unittests-ttg "${ut_src}" LINK_LIBRARIES "${ut_libs}")


# serialization test: probes serialization via all supported serialization methods (MADNESS, Boost::serialization, cereal) that are available
add_executable(serialization "serialization.cc;unit_main.cpp")
target_link_libraries(serialization "Catch2::Catch2;ttg-serialization")
if (TARGET BTAS::BTAS)
    target_link_libraries(serialization BTAS::BTAS)
    target_compile_definitions(serialization PRIVATE TTG_HAS_BTAS=1)
endif (TARGET BTAS::BTAS)

# TODO: convert into unit test
#if (TARGET MADworld)
#add_executable(splitmd_serialization splitmd_serialization.cc unit_main.cpp)
#endif(TARGET MADworld)


catch_discover_tests(serialization TEST_PREFIX "ttg/test/unit/")
