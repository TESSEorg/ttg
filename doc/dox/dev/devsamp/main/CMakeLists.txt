cmake_minimum_required(VERSION 3.14)
project(test)

find_package(ttg REQUIRED)

add_executable(test test.cc)
target_compile_features(test PRIVATE cxx_std_17)
target_compile_definitions(test PRIVATE TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
target_link_libraries(test ttg-mad)
