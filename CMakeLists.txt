cmake_minimum_required (VERSION 3.0)
project(cxxapi)

enable_language(CXX)
include(GNUInstallDirs)

set(flow_headers flow.h)
add_library(flow INTERFACE)
target_include_directories(flow INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_compile_features(flow INTERFACE cxx_return_type_deduction)
install(FILES ${flow_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT flow)
# Use current CMAKE_CXX_FLAGS to compile targets dependent on this library
string (REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
target_compile_options(flow INTERFACE $<INSTALL_INTERFACE:${CMAKE_CXX_FLAG_LIST}>)

add_executable (t9 t9.cc)
target_link_libraries(t9 LINK_PUBLIC flow)

