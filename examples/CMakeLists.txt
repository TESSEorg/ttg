include(AddTTGExecutable)

add_ttg_executable(test test/test.cc)
add_ttg_executable(t9 t9/t9.cc)
add_ttg_executable(t9-streaming t9/t9_streaming.cc)

# sparse matmul need Eigen ... it's always provided by TA
if (TARGET tiledarray)
    # MADworld used for MADNESS serialization
    add_ttg_executable(spmm spmm/spmm.cc LINK_LIBRARIES TiledArray_Eigen $<TARGET_NAME_IF_EXISTS:Boost::graph>
                                         COMPILE_DEFINITIONS $<$<TARGET_EXISTS:Boost::graph>:HAVE_BOOST_GRAPH=1>)
    # block-sparse needs BTAS ... it's always provided by TA
    # since only need to use matrices, limit BTAS_TARGET_MAX_INDEX_RANK to 2
    add_ttg_executable(bspmm spmm/spmm.cc LINK_LIBRARIES tiledarray TiledArray_Eigen BTAS
                                          COMPILE_DEFINITIONS BLOCK_SPARSE_GEMM=1;BTAS_TARGET_MAX_INDEX_RANK=2)

    add_ttg_executable(testing_dpotrf potrf/testing_dpotrf.cc LINK_LIBRARIES tiledarray lapackpp)
    add_ttg_executable(testing_dtrtri potrf/testing_dtrtri.cc LINK_LIBRARIES tiledarray lapackpp)
    add_ttg_executable(testing_dlauum potrf/testing_dlauum.cc LINK_LIBRARIES tiledarray lapackpp)
    add_ttg_executable(testing_dpoinv potrf/testing_dpoinv.cc LINK_LIBRARIES tiledarray lapackpp)

    if (TARGET CUDA::cublas)
      add_ttg_executable(bspmm-cuda spmm/spmm.cc
                         LINK_LIBRARIES tiledarray TiledArray_Eigen BTAS CUDA::cublas
                         COMPILE_DEFINITIONS BLOCK_SPARSE_GEMM=1;BTAS_TARGET_MAX_INDEX_RANK=2;TTG_ENABLE_CUDA=1
                         RUNTIMES "parsec")

      if (TARGET CUDA::cusolver)
        add_ttg_executable(testing_dpotrf_cuda potrf/testing_dpotrf.cc
                          LINK_LIBRARIES lapackpp tiledarray CUDA::cublas CUDA::cusolver
                          COMPILE_DEFINITIONS TTG_ENABLE_CUDA=1 #;DEBUG_TILES_VALUES=1
                          RUNTIMES "parsec")
      endif(TARGET CUDA::cusolver)
    elseif (TARGET roc::hipblas)
      add_ttg_executable(bspmm-hip spmm/spmm.cc
                          LINK_LIBRARIES tiledarray TiledArray_Eigen roc::hipblas
                          COMPILE_DEFINITIONS BLOCK_SPARSE_GEMM=1;BTAS_TARGET_MAX_INDEX_RANK=2;TTG_ENABLE_HIP=1
                          RUNTIMES "parsec")
      if (TARGET roc::hipsolver)
        add_ttg_executable(testing_dpotrf_hip potrf/testing_dpotrf.cc
                           LINK_LIBRARIES lapackpp tiledarray roc::hipblas roc::hipsolver
                           COMPILE_DEFINITIONS TTG_ENABLE_HIP=1 #;DEBUG_TILES_VALUES=1
                           RUNTIMES "parsec")
      endif(TARGET roc::hipsolver)
    elseif (TARGET MKL::MKL_DPCPP)
      add_ttg_executable(bspmm-lz spmm/spmm.cc
                          LINK_LIBRARIES tiledarray TiledArray_Eigen BTAS MKL::MKL_DPCPP level_zero::ze_loader m
                          COMPILE_DEFINITIONS BLOCK_SPARSE_GEMM=1;BTAS_TARGET_MAX_INDEX_RANK=2;TTG_ENABLE_LEVEL_ZERO=1
                          RUNTIMES "parsec")
    endif()

    if (TTG_HAVE_CUDA)
      add_ttg_executable(chain-ttg-cuda task-benchmarks/chain-ttg-dev.cc LINK_LIBRARIES tiledarray RUNTIMES "parsec")
    endif(TTG_HAVE_CUDA)

    if (TTG_HAVE_HIP)
      add_ttg_executable(chain-ttg-hip task-benchmarks/chain-ttg-dev.cc LINK_LIBRARIES tiledarray RUNTIMES "parsec")
    endif(TTG_HAVE_HIP)
endif()

if (TARGET MADworld)
  add_ttg_executable(madness-1d madness/madness-1d/madness-1d.cc RUNTIMES "mad")
  if (TARGET blaspp) #(CBLAS_FOUND AND MKL_FOUND)
    add_ttg_executable(mrattg madness/mrattg.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h LINK_LIBRARIES blaspp MADworld)

    add_ttg_executable(mrattg-streaming madness/mrattg_streaming.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h LINK_LIBRARIES blaspp MADworld)
  endif ()
endif (TARGET MADworld)

add_ttg_executable(wavefront-wf wavefront/wavefront-wf.cc SINGLERANKONLY)
add_ttg_executable(wavefront-wf2 wavefront/wavefront-wf2.cc SINGLERANKONLY)
add_ttg_executable(wavefront-df wavefront/wavefront-df.cc)
add_ttg_executable(wavefront-pull wavefront/wavefront-pull.cc LINK_LIBRARIES MADworld)
add_ttg_executable(fw-apsp floyd-warshall/floyd_warshall.cc LINK_LIBRARIES MADworld SINGLERANKONLY)
add_ttg_executable(helloworld helloworld/helloworld.cpp)
add_ttg_executable(simplegenerator simplegenerator/simplegenerator.cc RUNTIMES "mad")

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

