include(FindOrFetchCatch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)

# serialization test: probes serialization via all supported serialization methods (MADNESS, Boost::serialization, cereal) that are available
add_executable(serialization serialization.cc unit_main.cpp)
target_link_libraries(serialization "Catch2::Catch2;ttg-serialization")

catch_discover_tests(serialization)
