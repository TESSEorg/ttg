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
    )
add_library(ttg ${ttg-sources})
target_include_directories(ttg PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
target_link_libraries(ttg PUBLIC Boost::boost)
install(FILES ${ttg-sources}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ttg)

#########################
####### MADNESS-specific
#########################
if (${MADNESS_FOUND})
  add_library(ttg-mad INTERFACE)
  target_link_libraries(ttg-mad INTERFACE ttg Boost::boost)
  target_include_directories(ttg-mad INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
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
  target_link_libraries(ttg-mad INTERFACE MADworld ${MADNESS_DISABLEPIE_LINKER_FLAG})
endif(${MADNESS_FOUND})

########################
####### PaRSEC-specific
########################
if (${PARSEC_FOUND})
  add_library(ttg-parsec INTERFACE)
  target_link_libraries(ttg-parsec INTERFACE ttg Boost::boost)
  target_include_directories(ttg-parsec INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
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
  link_directories(${PARSEC_LIBRARY_DIRS})
  target_include_directories(ttg-parsec INTERFACE ${PARSEC_INCLUDE_DIRS})
  target_link_libraries(ttg-parsec INTERFACE parsec ${PARSEC_LIBRARIES} ${MPI_C_LIBRARIES})
endif(${PARSEC_FOUND})
