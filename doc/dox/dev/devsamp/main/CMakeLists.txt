cmake_minimum_required(VERSION 3.14)
project(test)

find_package(ttg REQUIRED)

add_ttg_executable(test test.cpp NOT_EXCLUDE_FROM_ALL)
