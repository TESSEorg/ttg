cmake_minimum_required (VERSION 3.0)
project(cxxapi)

enable_language(CXX)
include(GNUInstallDirs)

# sparse tensor algorithms need Eigen
find_package(Eigen3)
message (STATUS "EIGEN3_FOUND=${EIGEN3_FOUND}")
if (EIGEN3_FOUND)
  add_library(eigen3 INTERFACE IMPORTED)
  set_property(TARGET eigen3 PROPERTY
               INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
endif (EIGEN3_FOUND)

################################## "flow" API
set(flow_headers flow.h edge.h)
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

# helloworlds
add_executable (helloworld helloworld.cc)
target_link_libraries(helloworld LINK_PUBLIC flow)
add_executable (helloworld_wrap helloworld_wrap.cc)
target_link_libraries(helloworld_wrap LINK_PUBLIC flow)

# tree algorithms
add_executable (t9 t9.cc)
target_link_libraries(t9 LINK_PUBLIC flow)
add_executable (t9-wrap t9-wrap.cc)
target_link_libraries(t9-wrap LINK_PUBLIC flow)

# dense summa
add_executable (mxm_simple_suma mxm_simple_suma.cc)
target_link_libraries(mxm_simple_suma LINK_PUBLIC flow)

# sparse matmul
if (EIGEN3_FOUND)
  add_executable (spmm spmm.cc)
  target_link_libraries(spmm LINK_PUBLIC eigen3 flow)
endif(EIGEN3_FOUND)

################################## TTG API
set(ttg_headers ttg.h ttg_base.h demangle.h)
add_library(ttg INTERFACE)
target_include_directories(ttg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_compile_features(ttg INTERFACE cxx_return_type_deduction)
install(FILES ${ttg_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ttg)
# Use current CMAKE_CXX_FLAGS to compile targets dependent on this library
string (REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
target_compile_options(ttg INTERFACE $<INSTALL_INTERFACE:${CMAKE_CXX_FLAG_LIST}>)
# MADNESS is the only prerequisite
find_package(MADNESS REQUIRED 0.10.1 CONFIG QUIET COMPONENTS world HINTS ${MADNESS_ROOT_DIR})
target_link_libraries(ttg INTERFACE MADworld)

# sanity check
add_executable (ttgtest ttg.cc)
target_link_libraries(ttgtest LINK_PUBLIC ttg)

# T9 example
add_executable (t9-wrap-ttg t9-wrap-ttg.cc)
target_link_libraries(t9-wrap-ttg LINK_PUBLIC ttg)

# dense matmul
add_executable (mxm_simple_suma_ttg mxm_simple_suma_ttg.cc)
target_link_libraries(mxm_simple_suma_ttg LINK_PUBLIC ttg)

# sparse matmul
if (EIGEN3_FOUND)
  add_executable (spmm-ttg spmm-ttg.cc)
  target_link_libraries(spmm-ttg LINK_PUBLIC eigen3 ttg)
endif(EIGEN3_FOUND)
