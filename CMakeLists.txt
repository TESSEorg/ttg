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
# block-sparse tensor algorithms also need BTAS and (header-only) Boost.Iterator and Boost.Containers
find_package(Boost)
message (STATUS "Boost_FOUND=${Boost_FOUND}")
if (Boost_FOUND)
  find_package(BTAS)
  message (STATUS "BTAS_FOUND=${BTAS_FOUND}")
  if (BTAS_FOUND)
    add_library(BTAS INTERFACE IMPORTED)
    set_property(TARGET BTAS PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES ${BTAS_INCLUDE_DIRS})
  endif(BTAS_FOUND)
endif()

################################## TTG API

########################
####### Runtime-neutral
########################
add_library(ttg INTERFACE)
target_include_directories(ttg INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
target_compile_features(ttg INTERFACE cxx_return_type_deduction)
set(ttg-headers
        ${CMAKE_CURRENT_SOURCE_DIR}/ttg.h
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization.h
        ${CMAKE_CURRENT_SOURCE_DIR}/demangle.h
        )
target_sources(ttg INTERFACE ${ttg-headers})
install(FILES ${ttg-headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ttg)

################
####### MADNESS
################
add_library(ttg-mad INTERFACE)
target_link_libraries(ttg-mad INTERFACE ttg)
target_include_directories(ttg-mad INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_compile_features(ttg-mad INTERFACE cxx_return_type_deduction)
set(ttg-mad_headers
        ${CMAKE_CURRENT_SOURCE_DIR}/madness/ttg.h
        )
target_sources(ttg-mad INTERFACE ${ttg-mad_headers})
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
add_executable (test-mad madness/test.cc)
target_link_libraries(test-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

# madness-1d example (not yet runtime-agnostic)
add_executable(madness-1d madness/madness-1d/madness-1d.cc)
target_link_libraries(madness-1d LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

# T9 example
add_executable (t9-mad madness/t9.cc t9.impl.h)
target_link_libraries(t9-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

# sparse matmul
if (EIGEN3_FOUND)
  add_executable (spmm-mad madness/spmm.cc spmm.impl.h)
  target_link_libraries(spmm-mad LINK_PUBLIC ttg-mad eigen3 ${MADNESS_DISABLEPIE_LINKER_FLAG})
  # block-sparse needs BTAS
  if (BTAS_FOUND)
    add_executable (bspmm-mad madness/spmm.cc spmm.impl.h)
    target_compile_definitions(bspmm-mad PUBLIC -DBLOCK_SPARSE_GEMM=1)
    target_link_libraries(bspmm-mad LINK_PUBLIC ttg-mad eigen3 BTAS Boost::boost ${MADNESS_DISABLEPIE_LINKER_FLAG})
  endif (BTAS_FOUND)
endif()

# serialization
add_executable (serialization serialization.cc)
target_link_libraries(serialization LINK_PUBLIC ttg MADworld ${MADNESS_DISABLEPIE_LINKER_FLAG})

################
####### PaRSEC
################
add_library(ttg-parsec INTERFACE)
target_link_libraries(ttg-parsec INTERFACE ttg)
target_include_directories(ttg-parsec INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_compile_features(ttg-parsec INTERFACE cxx_return_type_deduction)
set(ttg-parsec_headers
        ${CMAKE_CURRENT_SOURCE_DIR}/parsec/ttg.h
        )
target_sources(ttg-parsec INTERFACE ${ttg-parsec_headers})
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
target_link_libraries(ttg-parsec INTERFACE parsec ${PARSEC_LIBRARIES} ${MPI_C_LIBRARIES})

# sanity check
add_executable (test-parsec parsec/test.cc)
target_link_libraries(test-parsec LINK_PUBLIC ttg-parsec)

# T9 example
add_executable (t9-parsec parsec/t9.cc t9.impl.h)
target_link_libraries(t9-parsec LINK_PUBLIC ttg-parsec)

# sparse matmul
if (EIGEN3_FOUND)
  add_executable (spmm-parsec parsec/spmm.cc spmm.impl.h)
  target_link_libraries(spmm-parsec LINK_PUBLIC ttg-parsec eigen3)
  # block-sparse needs BTAS and MADNESS (for serialization only)
  if (BTAS_FOUND AND MADNESS_FOUND)
    add_executable (bspmm-parsec parsec/spmm.cc spmm.impl.h)
    target_compile_definitions(bspmm-parsec PUBLIC -DBLOCK_SPARSE_GEMM=1)
    target_link_libraries(bspmm-parsec LINK_PUBLIC ttg-parsec eigen3 BTAS Boost::boost MADworld)
  endif (BTAS_FOUND AND MADNESS_FOUND)
endif()
