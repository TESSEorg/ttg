cmake_minimum_required(VERSION 3.14)
project(test)

find_package(ttg REQUIRED)

add_executable(test-mad test.cpp)
target_compile_features(test-mad PRIVATE cxx_std_17)
target_compile_definitions(test-mad PRIVATE TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
target_link_libraries(test-mad ttg-mad)

# parsec is not usable from install tree due to Boost's install borken (component targets are not available)
#add_executable(test-parsec test.cpp)
#target_compile_features(test-parsec PRIVATE cxx_std_17)
#target_compile_definitions(test-parsec PRIVATE TTG_USE_PARSEC=1)
#target_link_libraries(test-parsec ttg-parsec)
