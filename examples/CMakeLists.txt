include(AddTTGExecutable)

add_ttg_executable(test test/test.cc)
add_ttg_executable(t9 t9/t9.cc)

# sparse matmul
if (TARGET eigen3)
    # MADworld used for MADNESS serialization
    add_ttg_executable(spmm spmm/spmm.cc LINK_LIBRARIES eigen3 MADworld)
    # block-sparse needs BTAS
    if (TARGET BTAS::BTAS)
        # MADworld used for MADNESS serialization
        add_ttg_executable(bspmm spmm/spmm.cc LINK_LIBRARIES eigen3 BTAS Boost::boost MADworld COMPILE_DEFINITIONS BLOCK_SPARSE_GEMM=1)
    endif (TARGET BTAS::BTAS)
endif(TARGET eigen3)

if (TARGET MADworld)
    add_ttg_executable(madness-1d madness/madness-1d/madness-1d.cc RUNTIMES "mad")
    add_ttg_executable(reciprocal-mad madness/reciprocal.cc RUNTIMES "mad")
    if (CBLAS_FOUND AND MKL_FOUND)
        add_executable (mrattg-mad madness/mrattg.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h)
        target_link_libraries(mrattg-mad LINK_PUBLIC ttg-mad ${CBLAS_LIBRARIES} -lpthread -lm -ldl)
        target_include_directories(mrattg-mad PUBLIC . ${CBLAS_INCLUDE_DIR})
        target_compile_definitions(mrattg-mad PUBLIC -DCBLAS_HEADER="${CBLAS_INCLUDE_FILE}")
    endif()
endif(TARGET MADworld)

add_ttg_executable(wavefront-wf wavefront/wavefront-wf.cc LINK_LIBRARIES MADworld)
add_ttg_executable(wavefront-wf2 wavefront/wavefront-wf2.cc LINK_LIBRARIES MADworld)
add_ttg_executable(wavefront-df wavefront/wavefront-df.cc LINK_LIBRARIES MADworld)
add_ttg_executable(fw-apsp floyd-warshall/floyd_warshall.cc LINK_LIBRARIES MADworld)
if (OpenMP_CXX_FOUND AND TARGET std::execution)
    add_ttg_executable(fw-apsp-df floyd-warshall/floyd_warshall_df.cc LINK_LIBRARIES OpenMP::OpenMP_CXX std::execution)
endif ()
add_ttg_executable(ge ge/ge.cc LINK_LIBRARIES MADworld)
if (TARGET std::execution)
    add_ttg_executable(ge-df ge/ge_df.cc LINK_LIBRARIES std::execution)
endif(TARGET std::execution)
add_ttg_executable(sw sw/sw.cc LINK_LIBRARIES MADworld)

# RandomAccess HPCC Benchmark
# TODO convert to new example format
