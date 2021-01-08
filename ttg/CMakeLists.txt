########################
####### Runtime-neutral
########################
set(ttg-sources
        ${CMAKE_CURRENT_SOURCE_DIR}/ttg.h
        ${CMAKE_CURRENT_SOURCE_DIR}/wrap.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/backtrace.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/backtrace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util/broadcast.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/bug.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/bug.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/util/demangle.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/future.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/meta.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/reduce.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/serialization.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/tree.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/version.h
        ${CMAKE_CURRENT_SOURCE_DIR}/util/version.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/util/world.cpp
    )
add_library(ttg ${ttg-sources})
target_include_directories(ttg PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
target_link_libraries(ttg PUBLIC Boost::boost)
if (TARGET TTG_Libunwind)
  target_link_libraries(ttg PUBLIC TTG_Libunwind)
endif(TARGET TTG_Libunwind)
install(FILES ${ttg-sources}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ttg)

#########################
####### MADNESS-specific
#########################
if (TARGET MADworld)
  add_library(ttg-mad INTERFACE)
  target_link_libraries(ttg-mad INTERFACE ttg Boost::boost)
  target_include_directories(ttg-mad INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  set(ttg-mad_headers madness/ttg.h)
  target_sources(ttg-mad INTERFACE ${ttg-mad_headers})
  install(FILES ${ttg-mad_headers}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          COMPONENT ttg-mad)
  # Use current CMAKE_CXX_FLAGS to compile targets dependent on this library
  string (REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
  target_compile_options(ttg-mad INTERFACE $<INSTALL_INTERFACE:${CMAKE_CXX_FLAG_LIST}>)
  target_link_libraries(ttg-mad INTERFACE MADworld)
endif(TARGET MADworld)

########################
####### PaRSEC-specific
########################
if (TARGET PaRSEC::parsec)
  add_library(ttg-parsec INTERFACE)
  find_package(MPI)
  target_link_libraries(ttg-parsec INTERFACE ttg Boost::boost MPI::MPI_CXX)
  target_include_directories(ttg-parsec INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  set(ttg-parsec_headers parsec/ttg.h)
  target_sources(ttg-parsec INTERFACE ${ttg-parsec_headers})
  install(FILES ${ttg-parsec_headers}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          COMPONENT ttg-parsec)
  # Use current CMAKE_CXX_FLAGS to compile targets dependent on this library
  string (REPLACE " " ";" CMAKE_CXX_FLAG_LIST "${CMAKE_CXX_FLAGS}")
  target_compile_options(ttg-parsec INTERFACE $<INSTALL_INTERFACE:${CMAKE_CXX_FLAG_LIST}>)
  target_link_libraries(ttg-parsec INTERFACE PaRSEC::parsec)
endif(TARGET PaRSEC::parsec)
