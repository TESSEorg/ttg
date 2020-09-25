if (TARGET MADworld)
  # sanity check
  add_executable (test-mad madness/test.cc test.impl.h)
  target_link_libraries(test-mad LINK_PUBLIC ttg-mad)

  # madness-1d example (not yet runtime-agnostic)
  add_executable(madness-1d madness/madness-1d/madness-1d.cc)
  target_link_libraries(madness-1d LINK_PUBLIC ttg-mad)

  # reciprocal solver example (not yet runtime-agnostic)
  add_executable (reciprocal-mad madness/reciprocal.cc)
  target_link_libraries(reciprocal-mad LINK_PUBLIC ttg-mad)

  # T9 example
  add_executable (t9-mad madness/t9.cc t9.impl.h)
  target_link_libraries(t9-mad LINK_PUBLIC ttg-mad)

	#Wavefront example (workflow with shared data)
	add_executable (wavefront-wf-mad madness/wavefront-wf.cc wavefront-wf.impl.h)
	target_link_libraries(wavefront-wf-mad LINK_PUBLIC ttg-mad)

  # Wavefront example (workflow with classes)
  add_executable (wavefront-wf2-mad madness/wavefront-wf2.cc wavefront-wf2.impl.h)
  target_link_libraries(wavefront-wf2-mad LINK_PUBLIC ttg-mad)

	# Wavefront example (distributed)
	add_executable (wavefront-df-mad madness/wavefront-df.cc wavefront-df.impl.h)
	target_link_libraries(wavefront-df-mad LINK_PUBLIC ttg-mad)

  # Floyd Warshall's all pairs shortest paths
  add_executable (fw-apsp-mad madness/floyd_warshall.cc floyd_warshall.impl.h)
  target_link_libraries(fw-apsp-mad LINK_PUBLIC ttg-mad)

  # Floyd Warshall's all pairs shortest paths (distributed)
  if (OpenMP_CXX_FOUND AND TARGET std::execution)
    add_executable (fw-apsp-df-mad madness/floyd_warshall_df.cc floyd_warshall_df.impl.h)
    target_link_libraries(fw-apsp-df-mad LINK_PUBLIC ttg-mad OpenMP::OpenMP_CXX std::execution)
  endif()

  # Guassian Elimination Example
  add_executable (ge-mad madness/ge.cc ge.impl.h)
  target_link_libraries(ge-mad LINK_PUBLIC ttg-mad)

  # Guassian Elimination (dataflow) Example
  add_executable (ge-df-mad madness/ge_df.cc ge_df.impl.h)
  target_link_libraries(ge-df-mad LINK_PUBLIC ttg-mad)

  # Smith Waterman Algorithm
  add_executable (sw-mad madness/sw.cc sw.impl.h)
  target_link_libraries(sw-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

  # sparse matmul
  if (EIGEN3_FOUND)
    add_executable (spmm-mad madness/spmm.cc spmm.impl.h)
    target_link_libraries(spmm-mad LINK_PUBLIC ttg-mad eigen3)
    # block-sparse needs BTAS
    if (TARGET BTAS::BTAS)
      add_executable (bspmm-mad madness/spmm.cc spmm.impl.h)
      target_compile_definitions(bspmm-mad PUBLIC -DBLOCK_SPARSE_GEMM=1)
      target_link_libraries(bspmm-mad LINK_PUBLIC ttg-mad eigen3 BTAS Boost::boost)
    endif (TARGET BTAS::BTAS)
  endif(EIGEN3_FOUND)

  # MRA on ttg (not yet runtime-agnostic)
  if (CBLAS_FOUND AND MKL_FOUND)
    add_executable (mrattg-mad madness/mrattg.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h)
    target_link_libraries(mrattg-mad LINK_PUBLIC ttg-mad ${CBLAS_LIBRARIES} -lpthread -lm -ldl)
    target_include_directories(mrattg-mad PUBLIC . ${CBLAS_INCLUDE_DIR})
    target_compile_definitions(mrattg-mad PUBLIC -DCBLAS_HEADER="${CBLAS_INCLUDE_FILE}")
  endif()
endif(TARGET MADworld)

# "new" PaRSEC is not usable yet
if (TARGET PaRSEC::parsec)
  # sanity check
  add_executable (test-parsec parsec/test.cc test.impl.h)
  target_link_libraries(test-parsec LINK_PUBLIC ttg-parsec)

  if (TRUE)
	  # T9 example
  	add_executable (t9-parsec parsec/t9.cc t9.impl.h)
  	target_link_libraries(t9-parsec LINK_PUBLIC ttg-parsec)

  	# sparse matmul need MADNESS for serialization of ttg::matrix::Shape (part of next-gen prototype)
  	if (EIGEN3_FOUND AND TARGET MADworld)
    	add_executable (spmm-parsec parsec/spmm.cc spmm.impl.h)
    	target_link_libraries(spmm-parsec LINK_PUBLIC ttg-parsec eigen3 MADworld)
    	# block-sparse needs BTAS
    	if (TARGET BTAS::BTAS)
      	add_executable (bspmm-parsec parsec/spmm.cc spmm.impl.h)
      	target_compile_definitions(bspmm-parsec PUBLIC -DBLOCK_SPARSE_GEMM=1)
      	target_link_libraries(bspmm-parsec LINK_PUBLIC ttg-parsec eigen3 BTAS Boost::boost MADworld)
    	endif (TARGET BTAS::BTAS)
  	endif(EIGEN3_FOUND AND TARGET MADworld)
  endif (TRUE)
endif(TARGET PaRSEC::parsec)
