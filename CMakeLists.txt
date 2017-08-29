cmake_minimum_required (VERSION 3.0)
project(cxxapi)

enable_language(CXX)
include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")

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

################
####### MADNESS
################
set(ttg-mad_headers madness/ttg.h ttg.h demangle.h)
add_library(ttg-mad INTERFACE)
target_include_directories(ttg-mad INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_compile_features(ttg-mad INTERFACE cxx_return_type_deduction)
install(FILES ${ttg-mad_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ttg-mad)
# Use current CMAKE_CXX_FLAGS to compile targets dependent on this library
string (REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
target_compile_options(ttg-mad INTERFACE $<INSTALL_INTERFACE:${CMAKE_CXX_FLAG_LIST}>)
# MADNESS is the only prerequisite
find_package(MADNESS 0.10.1 CONFIG QUIET COMPONENTS world HINTS ${MADNESS_ROOT_DIR})
message(STATUS "MADNESS_FOUND=${MADNESS_FOUND}")
target_link_libraries(ttg-mad INTERFACE MADworld ${MADNESS_DISABLEPIE_LINKER_FLAG})

# sanity check
add_executable (ttgtest-mad madness/ttg.cc)
target_link_libraries(ttgtest-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

# T9 example
add_executable (t9-wrap-mad madness/t9-wrap-ttg.cc)
target_link_libraries(t9-wrap-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

# dense matmul
add_executable (mxm-summa-mad madness/mxm-summa.cc)
target_link_libraries(mxm-summa-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

# sparse matmul
if (EIGEN3_FOUND)
  add_executable (spmm-mad madness/spmm.cc)
  target_link_libraries(spmm-mad LINK_PUBLIC ttg-mad eigen3 ${MADNESS_DISABLEPIE_LINKER_FLAG})
endif()

################
####### PaRSEC
################
set(ttg-parsec_headers parsec/ttg.h ttg.h demangle.h)
add_library(ttg-parsec INTERFACE)
target_include_directories(ttg-parsec INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_compile_features(ttg-parsec INTERFACE cxx_return_type_deduction)
install(FILES ${ttg-parsec_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ttg-parsec)
# Use current CMAKE_CXX_FLAGS to compile targets dependent on this library
string (REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
target_compile_options(ttg-parsec INTERFACE $<INSTALL_INTERFACE:${CMAKE_CXX_FLAG_LIST}>)
# PaRSEC is the only explicit prerequisite
find_package(PARSEC MODULE COMPONENTS MPI)
link_directories(${PARSEC_LIBRARY_DIRS})
target_include_directories(ttg-parsec INTERFACE ${PARSEC_INCLUDE_DIRS})
target_link_libraries(ttg-parsec INTERFACE parsec ${PARSEC_LIBRARIES} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY})

# sanity check
add_executable (ttgtest-parsec parsec/ttg.cc)
target_link_libraries(ttgtest-parsec LINK_PUBLIC ttg-parsec)
