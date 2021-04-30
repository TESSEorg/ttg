include(FindOrFetchCatch2)
include(FindOrFetchCereal)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)

include(AddTTGExecutable)

# serialization (tests MADNESS serialization and, optionally, Boost::serialization and cereal)
if (TARGET MADworld)
  set(serialization_link_libraries "Catch2::Catch2")
  set(serialization_compile_definitions "")
  if (TARGET cereal::cereal)
    list(APPEND serialization_link_libraries "cereal::cereal")
    list(APPEND serialization_compile_definitions "SERIALIZATION_TEST_HAS_CEREAL=1")
  endif()
  if (TARGET Boost::serialization)
    list(APPEND serialization_link_libraries "Boost::serialization")
    list(APPEND serialization_compile_definitions "SERIALIZATION_TEST_HAS_BOOST=1")
  endif()
  add_ttg_executable(serialization serialization.cc unit_main.cpp RUNTIMES "mad" SINGLERANKONLY
          LINK_LIBRARIES "${serialization_link_libraries}"
          COMPILE_DEFINITIONS "${serialization_compile_definitions}")
endif (TARGET MADworld)

catch_discover_tests(serialization-mad)
