cmake_minimum_required(VERSION 3.14)
project(test)

find_package(ttg REQUIRED)

add_ttg_executable(test test.cpp NOT_EXCLUDE_FROM_ALL)

add_ttg_executable(fibonacci fibonacci/fibonacci.cc NOT_EXCLUDE_FROM_ALL)
# Fib device test
if (TTG_HAVE_CUDA)
    add_ttg_executable(fibonacci/fibonacci_device
            fibonacci/fibonacci_device.cc
            fibonacci/fibonacci_cuda_kernel.h
            fibonacci/fibonacci_cuda_kernel.cu
            LINK_LIBRARIES std::coroutine RUNTIMES "parsec" NOT_EXCLUDE_FROM_ALL)
endif()