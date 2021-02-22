include(AddTTGExecutable)

add_ttg_executable(test test/test.cc)
add_ttg_executable(t9 t9/t9.cc)

# sparse matmul
if (TARGET eigen3)
    # MADworld used for MADNESS serialization
    add_ttg_executable(spmm spmm/spmm.cc LINK_LIBRARIES eigen3)
    # block-sparse needs BTAS
    if (TARGET BTAS::BTAS)
        # MADworld used for MADNESS serialization
        add_ttg_executable(bspmm spmm/spmm.cc LINK_LIBRARIES eigen3 BTAS Boost::boost COMPILE_DEFINITIONS BLOCK_SPARSE_GEMM=1)
    endif (TARGET BTAS::BTAS)
endif(TARGET eigen3)

if (TARGET MADworld)
    add_ttg_executable(madness-1d madness/madness-1d/madness-1d.cc RUNTIMES "mad")
    #add_ttg_executable(reciprocal-mad madness/reciprocal.cc RUNTIMES "mad")
    if (CBLAS_FOUND AND MKL_FOUND)
        add_executable(mrattg-mad madness/mrattg.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h)
        target_link_libraries(mrattg-mad LINK_PUBLIC ttg-mad ${CBLAS_LIBRARIES} -lpthread -lm -ldl)
        target_include_directories(mrattg-mad PUBLIC . ${CBLAS_INCLUDE_DIR})
        target_compile_definitions(mrattg-mad PUBLIC -DCBLAS_HEADER="${CBLAS_INCLUDE_FILE}")
    endif ()
endif (TARGET MADworld)

add_ttg_executable(wavefront-wf wavefront/wavefront-wf.cc SINGLERANKONLY)
add_ttg_executable(wavefront-wf2 wavefront/wavefront-wf2.cc SINGLERANKONLY)
add_ttg_executable(wavefront-df wavefront/wavefront-df.cc)
add_ttg_executable(fw-apsp floyd-warshall/floyd_warshall.cc LINK_LIBRARIES MADworld SINGLERANKONLY)

#add_ttg_executable(potrf potrf/potrf.cc LINK_LIBRARIES MADworld)
add_executable(potrf-parsec potrf/potrf.cc)
# TODO: how to properly import the LAPACKPP sub project?
target_link_libraries(potrf-parsec LINK_PUBLIC ttg-parsec -L${CMAKE_CURRENT_BINARY_DIR}/../_deps/lapackpp-build/ -llapackpp -L${CMAKE_CURRENT_BINARY_DIR}/../_deps/blaspp-build/ -lblaspp)
target_link_libraries(potrf-parsec LINK_PUBLIC ${blaspp_libraries})
target_include_directories(potrf-parsec PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../_deps/lapackpp-src/include/ ${CMAKE_CURRENT_BINARY_DIR}/../_deps/blaspp-src/include/ ${CMAKE_CURRENT_BINARY_DIR}/../_deps/blaspp-build/include/)

if (OpenMP_CXX_FOUND AND TARGET std::execution)
        add_ttg_executable(fw-apsp-df floyd-warshall/floyd_warshall_df.cc LINK_LIBRARIES OpenMP::OpenMP_CXX std::execution MADworld)
endif ()
add_ttg_executable(ge ge/ge.cc SINGLERANKONLY)
if (TARGET std::execution)
        add_ttg_executable(ge-df ge/ge_df.cc LINK_LIBRARIES std::execution MADworld)
endif (TARGET std::execution)
add_ttg_executable(sw sw/sw.cc)

# RandomAccess HPCC Benchmark
if (TARGET MADworld)
  add_ttg_executable(randomaccess randomaccess/randomaccess.cc RUNTIMES "mad")
endif (TARGET MADworld)
