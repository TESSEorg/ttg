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

# tree algorithms
add_executable (t9 t9.cc)
target_link_libraries(t9 LINK_PUBLIC flow)

# dense summa
add_executable (mxm_simple_suma mxm_simple_suma.cc)
target_link_libraries(mxm_simple_suma LINK_PUBLIC flow)

# sparse tensor algorithms
find_package(Eigen3)
message (STATUS "EIGEN3_FOUND=${EIGEN3_FOUND}")
if (EIGEN3_FOUND)
  add_library(eigen3 INTERFACE IMPORTED)
  set_property(TARGET eigen3 PROPERTY
               INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
  
  add_executable (spmm spmm.cc)
  target_link_libraries(spmm LINK_PUBLIC eigen3 flow)
endif(EIGEN3_FOUND)

