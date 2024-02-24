//
// Created by Eduard Valeyev on 10/31/22.
//

#ifndef TTG_CONFIG_IN_H
#define TTG_CONFIG_IN_H

/** the C++ header containing the coroutine API */
#define TTG_CXX_COROUTINE_HEADER <@CXX_COROUTINE_HEADER@>

/** the C++ namespace containing the coroutine API */
#define TTG_CXX_COROUTINE_NAMESPACE @CXX_COROUTINE_NAMESPACE@

/** whether TTG has CUDA language support */
#cmakedefine TTG_HAVE_CUDA

/** whether TTG has CUDA runtime support */
#cmakedefine TTG_HAVE_CUDART

/** whether TTG has HIP support */
#cmakedefine TTG_HAVE_HIP

/** whether TTG has HIP BLAS library */
#cmakedefine TTG_HAVE_HIPBLAS

/** whether TTG has Intel Level Zero support */
#cmakedefine TTG_HAVE_LEVEL_ZERO

/** whether TTG has any device programming model (CUDA/HIP/LEVEL_ZERO) support */
#cmakedefine TTG_HAVE_DEVICE

/** whether TTG has MPI library */
#cmakedefine TTG_HAVE_MPI

/** whether TTG has the mpi-ext.h header */
#cmakedefine TTG_HAVE_MPIEXT

#endif  // TTG_CONFIG_IN_H
