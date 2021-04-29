include(FindOrFetchCatch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)

include(AddTTGExecutable)

# serialization (needs MADNESS)
if (TARGET MADworld)
  add_ttg_executable(serialization serialization.cc unit_main.cpp RUNTIMES "mad" SINGLERANKONLY LINK_LIBRARIES "Catch2::Catch2")
endif (TARGET MADworld)

catch_discover_tests(serialization-mad)
