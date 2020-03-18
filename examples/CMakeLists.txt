if (TARGET MADworld)
  # sanity check
  add_executable (test-mad madness/test.cc test.impl.h)
  target_link_libraries(test-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

  # madness-1d example (not yet runtime-agnostic)
  add_executable(madness-1d madness/madness-1d/madness-1d.cc)
  target_link_libraries(madness-1d LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

  # reciprocal solver example (not yet runtime-agnostic)
  add_executable (reciprocal-mad madness/reciprocal.cc)
  target_link_libraries(reciprocal-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

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
      target_link_libraries(bspmm-mad LINK_PUBLIC ttg-mad eigen3 BTAS ${MADNESS_DISABLEPIE_LINKER_FLAG})
    endif (BTAS_FOUND)
  endif(EIGEN3_FOUND)

  # MRA on ttg (not yet runtime-agnostic)
  if (CBLAS_FOUND AND MKL_FOUND)
    add_executable (mrattg-mad madness/mrattg.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h)
    target_link_libraries(mrattg-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG} ${CBLAS_LIBRARIES} -lpthread -lm -ldl)
    target_include_directories(mrattg-mad PUBLIC . ${CBLAS_INCLUDE_DIR})
    target_compile_definitions(mrattg-mad PUBLIC -DCBLAS_HEADER="${CBLAS_INCLUDE_FILE}")
  endif()
endif(TARGET MADworld)

# "new" PaRSEC is not usable yet
if (TARGET PaRSEC::parsec)
  # sanity check
  add_executable (test-parsec parsec/test.cc test.impl.h)
  target_link_libraries(test-parsec LINK_PUBLIC ttg-parsec)

  # T9 example
  add_executable (t9-parsec parsec/t9.cc t9.impl.h)
  target_link_libraries(t9-parsec LINK_PUBLIC ttg-parsec)

  # sparse matmul need MADNESS for serialization of ttg::matrix::Shape (part of next-gen prototype)
  if (EIGEN3_FOUND AND TARGET MADworld)
    add_executable (spmm-parsec parsec/spmm.cc spmm.impl.h)
    target_link_libraries(spmm-parsec LINK_PUBLIC ttg-parsec eigen3 MADworld ${MADNESS_DISABLEPIE_LINKER_FLAG})
    # block-sparse needs BTAS
    if (BTAS_FOUND)
      add_executable (bspmm-parsec parsec/spmm.cc spmm.impl.h)
      target_compile_definitions(bspmm-parsec PUBLIC -DBLOCK_SPARSE_GEMM=1)
      target_link_libraries(bspmm-parsec LINK_PUBLIC ttg-parsec eigen3 BTAS Boost::boost MADworld ${MADNESS_DISABLEPIE_LINKER_FLAG})
    endif (BTAS_FOUND)
  endif(EIGEN3_FOUND AND TARGET MADworld)
endif(TARGET PaRSEC::parsec)
