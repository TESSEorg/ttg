include(AddTTGExecutable)

# serialization (needs MADNESS)
if (TARGET MADworld)
  add_ttg_executable(serialization serialization.cc RUNTIMES "mad")
endif(TARGET MADworld)
