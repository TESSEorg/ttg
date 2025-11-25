# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.14)
project(ttg-devsample-fibonacci)

find_package(ttg REQUIRED)
if (NOT TARGET ttg-parsec) # else build from source
    include(FetchContent)
    FetchContent_Declare(ttg GIT_REPOSITORY https://github.com/TESSEorg/ttg.git)
    FetchContent_MakeAvailable( ttg )
endif()

add_ttg_executable(fibonacci fibonacci.cc NOT_EXCLUDE_FROM_ALL)
# Fib device test
message(WARNING "TTG_HAVE_CUDA=${TTG_HAVE_CUDA}")
if (TTG_HAVE_CUDA)
    add_ttg_executable(fibonacci_cuda
            fibonacci_device.cc
            fibonacci_cuda_kernel.h
            fibonacci_cuda_kernel.cu
            LINK_LIBRARIES std::coroutine RUNTIMES "parsec" NOT_EXCLUDE_FROM_ALL)
endif()
