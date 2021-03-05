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
    #add_ttg_executable(reciprocal-mad madness/reciprocal.cc RUNTIMES "mad")
    add_ttg_executable(mrattg madness/mra/mrattg.cc madness/mra/mragl.cc madness/mra/mratwoscale.cc madness/mra/mradomain.h madness/mra/mrafunctiondata.h madness/mra/mrafunctionfunctor.h madness/mra/mrafunctionnode.h madness/mra/mragl.h madness/mra/mrahash.h madness/mra/mrakey.h madness/mra/mramisc.h madness/mra/mramxm.h madness/mra/mrarange.h madness/mra/mrasimpletensor.h madness/mra/mratwoscale.h madness/mra/mratypes.h LINK_LIBRARIES blaspp RUNTIMES "mad")
endif (TARGET MADworld)

#add_ttg_executable(wavefront-wf wavefront/wavefront-wf.cc LINK_LIBRARIES MADworld SINGLERANKONLY)
#add_ttg_executable(wavefront-wf2 wavefront/wavefront-wf2.cc LINK_LIBRARIES MADworld SINGLERANKONLY)
add_ttg_executable(wavefront-df wavefront/wavefront-df.cc LINK_LIBRARIES MADworld SINGLERANKONLY)
#add_ttg_executable(fw-apsp floyd-warshall/floyd_warshall.cc LINK_LIBRARIES MADworld)
if (OpenMP_CXX_FOUND AND TARGET std::execution)
    #    add_ttg_executable(fw-apsp-df floyd-warshall/floyd_warshall_df.cc LINK_LIBRARIES OpenMP::OpenMP_CXX std::execution)
endif ()
#add_ttg_executable(ge ge/ge.cc LINK_LIBRARIES MADworld)
if (TARGET std::execution)
    #    add_ttg_executable(ge-df ge/ge_df.cc LINK_LIBRARIES std::execution SINGLERANKONLY)
endif (TARGET std::execution)
#add_ttg_executable(sw sw/sw.cc LINK_LIBRARIES MADworld)

# RandomAccess HPCC Benchmark
# TODO convert to new example format
