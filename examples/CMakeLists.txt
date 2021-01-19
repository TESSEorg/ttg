if (TARGET MADworld)
  # sanity check
  add_executable (test-mad test/test.cc)
  target_compile_definitions(test-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(test-mad LINK_PUBLIC ttg-mad)

  # madness-1d example (not yet runtime-agnostic)
  add_executable(madness-1d madness/madness-1d/madness-1d.cc)
  target_link_libraries(madness-1d LINK_PUBLIC ttg-mad)

  # reciprocal solver example (not yet runtime-agnostic)
  add_executable (reciprocal-mad madness/reciprocal.cc)
  target_link_libraries(reciprocal-mad LINK_PUBLIC ttg-mad)

  # T9 example
  add_executable (t9-mad t9/t9.cc)
  target_compile_definitions(t9-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(t9-mad LINK_PUBLIC ttg-mad)

  #Wavefront example (workflow with shared data)
  add_executable (wavefront-wf-mad wavefront/wavefront-wf.cc)
  target_compile_definitions(wavefront-wf-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(wavefront-wf-mad LINK_PUBLIC ttg-mad)

  # Wavefront example (workflow with classes)
  add_executable (wavefront-wf2-mad wavefront/wavefront-wf2.cc)
  target_compile_definitions(wavefront-wf2-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(wavefront-wf2-mad LINK_PUBLIC ttg-mad)

  # Wavefront example (distributed)
  add_executable (wavefront-df-mad wavefront/wavefront-df.cc)
  target_compile_definitions(wavefront-df-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(wavefront-df-mad LINK_PUBLIC ttg-mad)

  # Floyd Warshall's all pairs shortest paths
  add_executable (fw-apsp-mad floyd-warshall/floyd_warshall.cc)
  target_compile_definitions(fw-apsp-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(fw-apsp-mad LINK_PUBLIC ttg-mad)

  # Floyd Warshall's all pairs shortest paths (distributed)
  if (OpenMP_CXX_FOUND AND TARGET std::execution)
    add_executable (fw-apsp-df-mad floyd-warshall/floyd_warshall_df.cc)
    target_compile_definitions(fw-apsp-df-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
    target_link_libraries(fw-apsp-df-mad LINK_PUBLIC ttg-mad OpenMP::OpenMP_CXX std::execution)
  endif()

  # Guassian Elimination Example
  add_executable (ge-mad ge/ge.cc)
  target_compile_definitions(ge-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(ge-mad LINK_PUBLIC ttg-mad)

  # Guassian Elimination (dataflow) Example
  if (TARGET std::execution)
    add_executable (ge-df-mad ge/ge_df.cc)
    target_compile_definitions(ge-df-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
    target_link_libraries(ge-df-mad LINK_PUBLIC ttg-mad std::execution)
  endif(TARGET std::execution)

  # Smith Waterman Algorithm
  add_executable (sw-mad sw/sw.cc)
  target_compile_definitions(sw-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
  target_link_libraries(sw-mad LINK_PUBLIC ttg-mad ${MADNESS_DISABLEPIE_LINKER_FLAG})

  # sparse matmul
  if (EIGEN3_FOUND)
    add_executable (spmm-mad spmm/spmm.cc)
    target_compile_definitions(spmm-mad PUBLIC TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
    target_link_libraries(spmm-mad LINK_PUBLIC ttg-mad eigen3)
    # block-sparse needs BTAS
    if (TARGET BTAS::BTAS)
      add_executable (bspmm-mad spmm/spmm.cc)
      target_compile_definitions(bspmm-mad PUBLIC BLOCK_SPARSE_GEMM=1;TTG_USE_MADNESS=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
      target_link_libraries(bspmm-mad LINK_PUBLIC ttg-mad eigen3 BTAS Boost::boost -ldl)
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
  add_executable (test-parsec test/test.cc)
  target_compile_definitions(test-parsec PUBLIC TTG_USE_PARSEC=1)
  target_link_libraries(test-parsec LINK_PUBLIC ttg-parsec)

  if (TRUE)
      # T9 example
      #add_executable (t9-parsec parsec/t9.cc t9.impl.h)
      add_executable (t9-parsec t9/t9.cc)
      target_compile_definitions(t9-parsec PUBLIC TTG_USE_PARSEC=1)
      target_link_libraries(t9-parsec LINK_PUBLIC ttg-parsec)

      add_executable (sw-parsec sw/sw.cc)
      target_compile_definitions(sw-parsec PUBLIC TTG_USE_PARSEC=1)
      target_compile_options(sw-parsec PUBLIC -g)
      target_link_libraries(sw-parsec LINK_PUBLIC ttg-parsec MADworld)

      # Wavefront example (distributed)
      add_executable (wavefront-df-parsec wavefront/wavefront-df.cc)
      target_compile_definitions(wavefront-df-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
      target_link_libraries(wavefront-df-parsec LINK_PUBLIC ttg-parsec MADworld)

      #Wavefront example (workflow with shared data)
      add_executable (wavefront-wf-parsec wavefront/wavefront-wf.cc)
      target_compile_definitions(wavefront-wf-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
      target_link_libraries(wavefront-wf-parsec LINK_PUBLIC ttg-parsec MADworld)

      # Wavefront example (workflow with classes)
      add_executable (wavefront-wf2-parsec wavefront/wavefront-wf2.cc)
      target_compile_definitions(wavefront-wf2-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
      target_link_libraries(wavefront-wf2-parsec LINK_PUBLIC ttg-parsec MADworld)

      # Guassian Elimination Example
      add_executable (ge-parsec ge/ge.cc)
      target_compile_definitions(ge-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
      target_link_libraries(ge-parsec LINK_PUBLIC ttg-parsec MADworld)

      # Guassian Elimination (dataflow) Example
      if (TARGET std::execution)
        add_executable (ge-df-parsec ge/ge_df.cc)
        target_compile_definitions(ge-df-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
        target_link_libraries(ge-df-parsec LINK_PUBLIC ttg-parsec MADworld std::execution)
      endif(TARGET std::execution)


      # Floyd Warshall's all pairs shortest paths
      add_executable (fw-apsp-parsec floyd-warshall/floyd_warshall.cc)
      target_compile_definitions(fw-apsp-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
      target_link_libraries(fw-apsp-parsec LINK_PUBLIC ttg-parsec MADworld)

      # Floyd Warshall's all pairs shortest paths (distributed)
      if (OpenMP_CXX_FOUND AND TARGET std::execution)
        add_executable (fw-apsp-df-parsec floyd-warshall/floyd_warshall_df.cc)
        target_compile_definitions(fw-apsp-df-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
        target_link_libraries(fw-apsp-df-parsec LINK_PUBLIC ttg-parsec MADworld OpenMP::OpenMP_CXX std::execution)
      endif()



      # sparse matmul need MADNESS for serialization of ttg::matrix::Shape (part of next-gen prototype)
      if (EIGEN3_FOUND AND TARGET MADworld)
        add_executable (spmm-parsec spmm/spmm.cc)
        target_compile_definitions(spmm-parsec PUBLIC TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
        target_link_libraries(spmm-parsec LINK_PUBLIC ttg-parsec eigen3 MADworld)
        # block-sparse needs BTAS
        if (TARGET BTAS::BTAS)
          add_executable (bspmm-parsec spmm/spmm.cc)
          target_compile_definitions(bspmm-parsec PUBLIC BLOCK_SPARSE_GEMM=1;TTG_USE_PARSEC=1;WORLD_INSTANTIATE_STATIC_TEMPLATES=1)
          target_link_libraries(bspmm-parsec LINK_PUBLIC ttg-parsec eigen3 BTAS Boost::boost MADworld -ldl)
        endif (TARGET BTAS::BTAS)
      endif(EIGEN3_FOUND AND TARGET MADworld)
  endif (TRUE)
endif(TARGET PaRSEC::parsec)
