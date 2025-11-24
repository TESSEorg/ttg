# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.14)
project(ttg-devsample-helloworld)

find_package(ttg REQUIRED)
if (NOT TARGET ttg-parsec) # else build from source
    include(FetchContent)
    FetchContent_Declare(ttg GIT_REPOSITORY https://github.com/TESSEorg/ttg.git)
    FetchContent_MakeAvailable( ttg )
endif()

add_ttg_executable(helloworld helloworld.cpp NOT_EXCLUDE_FROM_ALL)
