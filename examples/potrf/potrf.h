#pragma once

#include <ttg.h>
#include "lapack.hh"
#include "pmw.h"

#undef DEBUG_TILES_VALUES

#if defined(TTG_HAS_CUDART)
#define ES ttg::ExecutionSpace::CUDA
#define TASKRET -> ttg::device_task
#elif defined(TTG_HAS_HIP)
#define ES ttg::ExecutionSpace::HIP
#define TASKRET -> ttg::device_task
#else
#define ES ttg::ExecutionSpace::Host
#define TASKRET -> void
#endif



namespace potrf {

  /* FLOP macros taken from DPLASMA */
  inline double FMULS_POTRF(double __n) { return (__n * (((1. / 6.) * __n + 0.5) * __n + (1. / 3.))); }
  inline double FADDS_POTRF(double __n) { return (__n * (((1. / 6.) * __n) * __n - (1. / 6.))); }
  inline double FLOPS_DPOTRF(double __n) { return FMULS_POTRF(__n) + FADDS_POTRF(__n); }

  template <typename MatrixT>
  auto make_potrf(MatrixT& A,
                  ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& input_disp,  // from the dispatcher
                  ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& input,
                  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_trsm,
                  ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_result) {
      using T = typename MatrixT::element_type;
#if defined(TTG_HAS_CUDART) || defined(TTG_HAS_HIP)
    static int device_potrf_workspace_size(blk_t &A) {
      int Lwork;
      #if defined(TTG_HAVE_CUDA)
        cusolverDnDpotrf_bufferSize(ttg::detail::cublas_get_handle(),
                                    CUBLAS_FILL_MODE_LOWER, A.extent(1),
                                    nullptr, A.extent(0),
                                    &Lwork);
        return Lwork;
      #elif defined(TTG_HAVE_HIPBLAS)
        #error TBCoded
      #else
        return 0;
      #endif
    }

    static void device_potrf(blk_t &A, double *workspace, int Lwork, int *devInfo) {
      int device = A.b.get_current_device();
      assert(device != 0);
    #if defined(TTG_HAVE_CUDA)
      cusolverDnDpotrf(ttg::detail::cublas_get_handle(),
                       CUBLAS_FILL_MODE_LOWER, A.extent(1),
                       A.b.device_ptr_on(device), A.extent(0),
                       workspace, Lwork,
                       devInfo);
    #elif defined(TTG_HAVE_HIPBLAS)
      hipsolverDpotrf(ttg::detail::hipblas_get_handle(),
                       HIPSOLVER_FILL_MODE_LOWER, A.extent(1),
                       A.b.device_ptr_on(device), A.extent(0),
                       workspace, Lwork,
                       devInfo);
    #endif
    }

    auto f_dev = [=](const Key1& key, MatrixTile<T>&& A,
                     std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>>& out) TASKRET {
      const auto K = key[0];

      /* compute successors before submitting the kernel running
       * TODO: this is parsec specific since this code is still executing on the worker threads
       */
      std::vector<Key2> keylist;
      keylist.reserve(A.rows() - K);
      /* TODO: reverse order of arrays */
      for (int m = K + 1; m < A.rows(); ++m) {
        /* send tile to trsm */
        keylist.push_back(Key2(m, K));
      }

      /* pull the matrix onto the device, as computing the workspace size might in theory depend on the data */
      //TODO: extend MatrixTile<T> to be heterogeneous-aware. Look at spmm-cuda.cc 50-253
      //       Need to include a ttg::buffer<T> _data instead of a shared_ptr;
      //       Check pmw.h: when we generate the MatrixTile
      //       Also check pinned allocator at the end of DeviceTensor (250-253)

      int Lwork = device_potrf_workspace_size(A);

      // Instead of using scratch here, we should have hostWS and hostInfo globals and use to_device
      // this would reduce the number of I/O operations to devices
      double hostWS[Lwork];
      ttg::devicescratch<double> devWS = ttg::make_scratch(hostWS, ttg::scope::Allocate);
      int hostInfo = -1;
      ttg::devicescratch<int> devInfo = ttg::make_scratch(&hostInfo, ttg::scope::Allocate);

      /* the workspace and the devInfo must be device-level pointers */
      co_await ttg::to_device(A._data, devWS, devInfo);

      /* everything is on the device, call the POTRF */
      device_potrf(A, devWS, Lwork, devInfo);

      /* wait for the kernel to complete */
      co_await ttg::wait_kernel(devInfo);

      if( hostInfo == 0 ) {
        co_await ttg::device::forward(ttg::device::broadcast<0, 1>(std::make_tuple(Key2(K, K), std::move(keylist)), std::move(A), out));
        // Anything after this co_await is never executed
        // co_return would look better, but co_return would destroy keylist before the runtime can handle it
      } else {
        // Well... Here we should interrupt the DAG of tasks, there is an error. Raise?
        std::cerr << "Factorization is SUSPICIOUS (the matrix might not be diagonally dominant)" << std::endl;
        ttg::abort();
      }
    }
    return ttg::make_tt<ES>(f_dev, ttg::edges(ttg::fuse(input, input_disp)), ttg::edges(output_result, output_trsm), "POTRF",
                        {"tile_kk/dispatcher"}, {"output_result", "output_trsm"});
#else /* defined(TTG_HAS_CUDART) || defined(TTG_HAS_HIP) */
    auto f = [=](const Key1& key, MatrixTile<T>&& tile_kk,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>>& out) {
      const int K = key[0];

      if (ttg::tracing()) ttg::print("POTRF(", key, ")");
#if defined(DEBUG_TILES_VALUES)
      std::cout << "Before POTRF(" << key << "), A(" << K << ", " << K << ") is " << tile_kk;
#endif

      auto info = lapack::potrf(lapack::Uplo::Lower, tile_kk.rows(), tile_kk.data(), tile_kk.lda());
      assert(info == 0);

#if defined(DEBUG_TILES_VALUES)
      std::cout << "After POTRF(" << key << "), A(" << K << ", " << K << ") is " << tile_kk << std::endl;
#endif

      /* send the tile to outputs */
      std::vector<Key2> keylist;
      keylist.reserve(A.rows() - K);
      /* TODO: reverse order of arrays */
      for (int m = K + 1; m < A.rows(); ++m) {
        /* send tile to trsm */
        if (ttg::tracing()) ttg::print("POTRF(", key, "): sending output to TRSM(", Key2{m, K}, ")");
        keylist.push_back(Key2(m, K));
      }
      ttg::broadcast<0, 1>(std::make_tuple(Key2(K, K), keylist), std::move(tile_kk), out);
    };
    return ttg::make_tt(f, ttg::edges(ttg::fuse(input, input_disp)), ttg::edges(output_result, output_trsm), "POTRF",
                        {"tile_kk/dispatcher"}, {"output_result", "output_trsm"});
#endif
  }

  template <typename MatrixT>
  auto make_trsm(MatrixT& A,
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_disp,   // from the dispatcher
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_kk,     // from POTRF
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_mk,     // from previous GEMM
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_diag,  // to SYRK
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& output_row,   // to GEMM
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& output_col,   // to GEMM
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_result) {
    using T = typename MatrixT::element_type;
#if defined(TTG_HAS_CUDART) || defined(TTG_HAS_HIP)
    auto f = [=](const Key2& key, const MatrixTile<T>& tile_kk, MatrixTile<T>&& tile_mk,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key3, MatrixTile<T>>,
                            ttg::Out<Key3, MatrixTile<T>>>& out) TASKRET {
      const int M = key[0];
      const int K = key[1];  // the column equals the outer most look K (same as PO)

      auto mb = tile_mk.rows();
      auto nb = tile_mk.cols();

      /* in trsm, tile_mk is mb x nb, and tile_kk needs to be lda x nb because side = Right */
      assert(nb == tile_kk.rows());

      if (ttg::tracing()) ttg::print("TRSM(", key, ")");

      /* populate successor keys while we're on the worker thread */
      std::vector<Key3> keylist_row;
      keylist_row.reserve(M - K);
      std::vector<Key3> keylist_col;
      keylist_col.reserve(A.rows() - M - 1);

      /* send tile to syrk on diagonal */
      if (ttg::tracing()) ttg::print("TRSM(", key, "): sending output to syrk(", Key2{K, M}, ")");

      /* send the tile to all gemms across in row i */
      for (int n = K + 1; n < M; ++n) {
        if (ttg::tracing()) ttg::print("TRSM(", key, "): sending output to gemm( ", Key3{M, n, K}, ")");
        keylist_row.push_back(Key3(M, n, K));
      }

      /* send the tile to all gemms down in column i */
      for (int m = M + 1; m < A.rows(); ++m) {
        if (ttg::tracing()) ttg::print("TRSM(", key, "): sending output to gemm( ", Key3{m, M, K}, ")");
        keylist_col.push_back(Key3(m, M, K));
      }


      co_await ttg::to_device(tile_kk.b, tile_mk.b);
      int device = tile_kk.b.get_current_device();
      double alpha = 1.0;
#if defined(TTG_HAVE_CUDA)
      cublasDtrsm(ttg::detail::cublas_get_handle(),
                  CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                  mb, nb, &alpha,
                  tile_kk.b.device_ptr_on(device), tile_kk.lda(),
                  tile_mk.b.device_ptr_on(device), tile_mk.lda());
#elif defined(TTG_HAVE_HIPBLAS)
      hipblasDtrsm(ttg::detail:hipblas_get_handle(),
                   HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_LOWER,
                   HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT,
                   mb, nb, &alpha,
                   tile_kk.b.device_ptr_on(device), tile_kk.lda(),
                   tile_mk.b.device_ptr_on(device), tile_mk.lda());
#endif

      co_await ttg::device::forward(ttg::device::broadcast<0, 1, 2, 3>(std::make_tuple(key, Key2(K, M), keylist_row, keylist_col),
                                                                       std::move(tile_mk), out));
    };
    return ttg::make_tt<ES>(f, ttg::edges(input_kk, ttg::fuse(input_mk, input_disp)),
                            ttg::edges(output_result, output_diag, output_row, output_col), "TRSM",
                            {"tile_kk", "tile_mk/dispatcher"}, {"output_result", "tile_mk", "output_row", "output_col"});
#else
    auto f = [=](const Key2& key, const MatrixTile<T>& tile_kk, MatrixTile<T>&& tile_mk,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key3, MatrixTile<T>>,
                            ttg::Out<Key3, MatrixTile<T>>>& out) {
      const int M = key[0];
      const int K = key[1];  // the column equals the outer most look K (same as PO)

      auto mb = tile_mk.rows();
      auto nb = tile_mk.cols();

      /* in trsm, tile_mk is mb x nb, and tile_kk needs to be lda x nb because side = Right */
      assert(nb == tile_kk.rows());

      if (ttg::tracing()) ttg::print("TRSM(", key, ")");
#if defined(DEBUG_TILES_VALUES)
      std::cout << "Before TRSM(" << key << "), A(" << K << ", " << K << ") is " << tile_kk << " and A(" << M << ", "
                << K << ") is " << tile_mk;
#endif

      blas::trsm(blas::Layout::ColMajor, blas::Side::Right, lapack::Uplo::Lower, blas::Op::Trans, blas::Diag::NonUnit,
                 mb, nb, 1.0, tile_kk.data(), tile_kk.lda(), tile_mk.data(), tile_mk.lda());

#if defined(DEBUG_TILES_VALUES)
      std::cout << "After TRSM(" << key << "), A(" << K << ", " << K << ") is " << tile_mk << std::endl;
#endif

      std::vector<Key3> keylist_row;
      keylist_row.reserve(M - K);
      std::vector<Key3> keylist_col;
      keylist_col.reserve(A.rows() - M - 1);

      /* send tile to syrk on diagonal */
      if (ttg::tracing()) ttg::print("TRSM(", key, "): sending output to syrk(", Key2{K, M}, ")");

      /* send the tile to all gemms across in row i */
      for (int n = K + 1; n < M; ++n) {
        if (ttg::tracing()) ttg::print("TRSM(", key, "): sending output to gemm( ", Key3{M, n, K}, ")");
        keylist_row.push_back(Key3(M, n, K));
      }

      /* send the tile to all gemms down in column i */
      for (int m = M + 1; m < A.rows(); ++m) {
        if (ttg::tracing()) ttg::print("TRSM(", key, "): sending output to gemm( ", Key3{m, M, K}, ")");
        keylist_col.push_back(Key3(m, M, K));
      }

      ttg::broadcast<0, 1, 2, 3>(std::make_tuple(key, Key2(K, M), keylist_row, keylist_col), std::move(tile_mk), out);
    };
    return ttg::make_tt(f, ttg::edges(input_kk, ttg::fuse(input_mk, input_disp)),
                        ttg::edges(output_result, output_diag, output_row, output_col), "TRSM",
                        {"tile_kk", "tile_mk/dispatcher"}, {"output_result", "tile_mk", "output_row", "output_col"});
#endif
  }

  template <typename MatrixT>
  auto make_syrk(MatrixT& A,
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_disp,    // from the dispatcher
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_mk,      // from TRSM
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_kk,      // from SYRK
                 ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& output_potrf,  // to POTRF
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_syrk) {
    using T = typename MatrixT::element_type;
#if defined(TTG_HAS_CUDART) || defined(TTG_HAS_HIP)
    auto f = [=](const Key2& key, const MatrixTile<T>& tile_mk, MatrixTile<T>&& tile_kk,
                 std::tuple<ttg::Out<Key1, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>>& out) TASKRET {
      const int K = key[0];
      const int M = key[1];

      /* tile_kk is mb x mb and tile_mk is mb x nb */
      assert(tile_kk.rows() == tile_kk.cols());
      assert(tile_mk.rows() == tile_kk.rows());

      auto mb = tile_mk.rows();
      auto nb = tile_mk.cols();

      if (ttg::tracing()) ttg::print("SYRK(", key, ")");

      co_await ttg::to_device(tile_kk.b, tile_mk.b);

      double alpha = -1.0;
      double beta  =  1.0;
#if defined(TTG_HAVE_CUDA)
      cublasDsyrk(ttg::detail::cublas_get_handle(),
                  CUBLAS_FILL_MODE_LOWER,
                  CUBLAS_OP_N,
                  mb, nb, &alpha,
                  tile_nk.b.device_ptr_on(device), tile_mk.lda(), &beta,
                  tile_kk.b.device_ptr_on(device), tile_kk.lda());
#elif defined(TTG_HAVE_HIPBLAS)
      hipblasDsyrk(ttg::detail:hipblas_get_handle(),
                   HIPBLAS_FILL_MODE_LOWER,
                   HIPBLAS_OP_N,
                   mb, nb, &alpha,
                   tile_kk.b.device_ptr_on(device), tile_kk.lda(), &beta,
                   tile_mk.b.device_ptr_on(device), tile_mk.lda());
#endif

      if (M == K + 1) {
        /* send the tile to potrf */
        if (ttg::tracing()) ttg::print("SYRK(", key, "): sending output to POTRF(", Key1{K + 1}, ")");
        co_await ttg::device::send<0>(Key1(K + 1), std::move(tile_kk), out);
      } else {
        /* send output to next syrk */
        if (ttg::tracing()) ttg::print("SYRK(", key, "): sending output to SYRK(", Key2{K + 1, M}, ")");
        co_await ttg::device::send<1>(Key2(K + 1, M), std::move(tile_kk), out);
      }
    };
    return ttg::make_tt<ES>(f, ttg::edges(input_mk, ttg::fuse(input_kk, input_disp)), ttg::edges(output_potrf, output_syrk),
                            "SYRK", {"tile_mk", "tile_kk/dispatcher"}, {"output_potrf", "output_syrk"});
#else
    auto f = [=](const Key2& key, const MatrixTile<T>& tile_mk, MatrixTile<T>&& tile_kk,
                 std::tuple<ttg::Out<Key1, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>>& out) {
      const int K = key[0];
      const int M = key[1];

      /* tile_kk is mb x mb and tile_mk is mb x nb */
      assert(tile_kk.rows() == tile_kk.cols());
      assert(tile_mk.rows() == tile_kk.rows());

      auto mb = tile_mk.rows();
      auto nb = tile_mk.cols();

      if (ttg::tracing()) ttg::print("SYRK(", key, ")");
#if defined(DEBUG_TILES_VALUES)
      std::cout << "Before SYRK(" << key << "), A(" << M << ", " << K << ") is " << tile_mk << " and A(" << K << ", "
                << K << ") is " << tile_kk;
#endif

      blas::syrk(blas::Layout::ColMajor, lapack::Uplo::Lower, blas::Op::NoTrans, mb, nb, -1.0, tile_mk.data(),
                 tile_mk.lda(), 1.0, tile_kk.data(), tile_kk.lda());

#if defined(DEBUG_TILES_VALUES)
      std::cout << "After SYRK(" << key << "), A(" << K << ", " << K << ") is " << tile_kk << std::endl;
#endif

      if (M == K + 1) {
        /* send the tile to potrf */
        if (ttg::tracing()) ttg::print("SYRK(", key, "): sending output to POTRF(", Key1{K + 1}, ")");
        ttg::send<0>(Key1(K + 1), std::move(tile_kk), out);
      } else {
        /* send output to next syrk */
        if (ttg::tracing()) ttg::print("SYRK(", key, "): sending output to SYRK(", Key2{K + 1, M}, ")");
        ttg::send<1>(Key2(K + 1, M), std::move(tile_kk), out);
      }
    };
    return ttg::make_tt(f, ttg::edges(input_mk, ttg::fuse(input_kk, input_disp)), ttg::edges(output_potrf, output_syrk),
                        "SYRK", {"tile_mk", "tile_kk/dispatcher"}, {"output_potrf", "output_syrk"});
#endif
  }

  template <typename MatrixT>
  auto make_gemm(MatrixT& A,
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_disp,   // From the dispatcher
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_mk,     // from TRSM
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_nk,     // from TRSM
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_mn,     // from TRSM
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_trsm,  // to TRSM
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& output_gemm) {
    using T = typename MatrixT::element_type;
#if defined(TTG_HAS_CUDART) || defined(TTG_HAS_HIP)
    auto f = [=](const Key3& key, const MatrixTile<T>& tile_mk, const MatrixTile<T>& tile_nk, MatrixTile<T>&& tile_mn,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key3, MatrixTile<T>>>& out) TASKRET {
      const int M = key[0];
      const int N = key[1];
      const int K = key[2];
      assert(M != N && M > K && N > K);

      assert(tile_mk.cols() == tile_nk.cols());
      assert(tile_mk.rows() == tile_mn.rows());
      assert(tile_nk.rows() == tile_mn.cols());

      if (ttg::tracing()) ttg::print("GEMM(", key, ")");
#if defined(DEBUG_TILES_VALUES)
      std::cout << "Before GEMM(" << key << "), A(" << M << ", " << K << ") is " << tile_mk << " and A(" << K << ", "
                << N << ") is " << tile_nk << " and A(" << M << ", " << N << ") is " << tile_mn;
#endif

      co_await ttg::to_device(tile_mk.b, tile_nk.b, tile_mn.b);

      double alpha = -1.0;
      double beta  =  1.0;
#if defined(TTG_HAVE_CUDA)
      cublasDgemm(ttg::detail:cublas_get_handle(),
                  CUBLAS_OP_N, CUBLAS_OP_T,
                  tile_mk.rows(), tile_nk.rows(),
                  tile_nk.cols(), &alpha,
                  tile_mk.data(), tile_mk.lda(),
                  tile_nk.data(), tile_nk.lda(), &beta,
                  tile_mn.data(), tile_mn.lda());
#elif defined(TTG_HAVE_HIPBLAS)
      hipblasDgemm(ttg::detail:hipblas_get_handle(),
                   HIPBLAS_OP_N, HIPBLAS_OP_T,
                   tile_mk.rows(), tile_nk.rows(),
                   tile_nk.cols(), &alpha,
                   tile_mk.data(), tile_mk.lda(),
                   tile_nk.data(), tile_nk.lda(), &beta,
                   tile_mn.data(), tile_mn.lda());
#endif

#if defined(DEBUG_TILES_VALUES)
      std::cout << "After GEMM(" << key << "), A(" << M << ", " << N << ") is " << tile_mn << std::endl;
#endif

      if (N == K + 1) {
        /* send the tile to trsm */
        if (ttg::tracing()) ttg::print("GEMM(", key, "): sending output to TRSM(", Key2{M, N}, ")");
        co_await ttg::device::send<0>(Key2(M, N), std::move(tile_mn), out);
      } else {
        /* send the tile to the next gemm */
        if (ttg::tracing()) ttg::print("GEMM(", key, "): sending output to GEMM(", Key3{M, N, K + 1}, ")");
        co_await ttg::device::send<1>(Key3(M, N, K + 1), std::move(tile_mn), out);
      }
    };
    return ttg::make_tt<ES>(f, ttg::edges(input_mk, input_nk, ttg::fuse(input_disp, input_mn)),
                            ttg::edges(output_trsm, output_gemm), "GEMM", {"input_mk", "input_kn", "input_mn/dispatcher"},
                            {"output_trsm", "outout_gemm"});
#else
    auto f = [=](const Key3& key, const MatrixTile<T>& tile_mk, const MatrixTile<T>& tile_nk, MatrixTile<T>&& tile_mn,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key3, MatrixTile<T>>>& out) {
      const int M = key[0];
      const int N = key[1];
      const int K = key[2];
      assert(M != N && M > K && N > K);

      assert(tile_mk.cols() == tile_nk.cols());
      assert(tile_mk.rows() == tile_mn.rows());
      assert(tile_nk.rows() == tile_mn.cols());

      if (ttg::tracing()) ttg::print("GEMM(", key, ")");
#if defined(DEBUG_TILES_VALUES)
      std::cout << "Before GEMM(" << key << "), A(" << M << ", " << K << ") is " << tile_mk << " and A(" << K << ", "
                << N << ") is " << tile_nk << " and A(" << M << ", " << N << ") is " << tile_mn;
#endif

      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, tile_mk.rows(), tile_nk.rows(),
                 tile_nk.cols(), -1.0, tile_mk.data(), tile_mk.lda(), tile_nk.data(), tile_nk.lda(), 1.0,
                 tile_mn.data(), tile_mn.lda());

#if defined(DEBUG_TILES_VALUES)
      std::cout << "After GEMM(" << key << "), A(" << M << ", " << N << ") is " << tile_mn << std::endl;
#endif

      if (N == K + 1) {
        /* send the tile to trsm */
        if (ttg::tracing()) ttg::print("GEMM(", key, "): sending output to TRSM(", Key2{M, N}, ")");
        ttg::send<0>(Key2(M, N), std::move(tile_mn), out);
      } else {
        /* send the tile to the next gemm */
        if (ttg::tracing()) ttg::print("GEMM(", key, "): sending output to GEMM(", Key3{M, N, K + 1}, ")");
        ttg::send<1>(Key3(M, N, K + 1), std::move(tile_mn), out);
      }
    };
    return ttg::make_tt(f, ttg::edges(input_mk, input_nk, ttg::fuse(input_disp, input_mn)),
                        ttg::edges(output_trsm, output_gemm), "GEMM", {"input_mk", "input_kn", "input_mn/dispatcher"},
                        {"output_trsm", "outout_gemm"});
#endif
  }

  template <typename T>
  auto make_dispatcher(ttg::Edge<Key2, MatrixTile<T>>& input, ttg::Edge<Key1, MatrixTile<T>>& to_potrf,
                       ttg::Edge<Key2, MatrixTile<T>>& to_trsm, ttg::Edge<Key2, MatrixTile<T>>& to_syrk,
                       ttg::Edge<Key3, MatrixTile<T>>& to_gemm) {
    auto f = [=](const Key2& key, MatrixTile<T>&& tile,
                 std::tuple<ttg::Out<Key1, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>,
                            ttg::Out<Key3, MatrixTile<T>>>& out) {
      if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ")");
      if (0 == key[0] && 0 == key[1]) {
        // First element goes to POTRF
        if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to POTRF(", Key1{key[0]}, ")");
        ttg::send<0>(Key1{key[0]}, std::move(tile), out);
        return;
      }
      if (key[0] == key[1]) {
        // Other diagonal elements go to SYRK
        if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to SYRK(", Key2{0, key[0]}, ")");
        ttg::send<2>(Key2{0, key[0]}, std::move(tile), out);
        return;
      }
      // We only consider the lower triangular
      assert(key[0] > key[1]);
      if (0 == key[1]) {
        // First column goes to TRSM
        if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to TRSM(", key, ")");
        ttg::send<1>(key, std::move(tile), out);
        return;
      }
      // Rest goes to GEMM
      if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to GEMM(", Key3{key[0], key[1], 0}, ")");
      ttg::send<3>(Key3{key[0], key[1], 0}, std::move(tile), out);
    };

    return ttg::make_tt(f, ttg::edges(input), ttg::edges(to_potrf, to_trsm, to_syrk, to_gemm), "POTRF Dispatch",
                        {"Input"}, {"POTRF", "TRSM", "SYRK", "GEMM"});
  }

  template <typename MatrixT>
  auto make_potrf_ttg(MatrixT& A, ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input,
                      ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output, bool defer_write) {
    using T = typename MatrixT::element_type;
    auto keymap1 = [&](const Key1& key) { return A.rank_of(key[0], key[0]); };

    auto keymap2a = [&](const Key2& key) { return A.rank_of(key[0], key[1]); };
    auto keymap2b = [&](const Key2& key) { return A.rank_of(key[0], key[0]); };

    auto keymap3 = [&](const Key3& key) { return A.rank_of(key[0], key[1]); };

    ttg::Edge<Key1, MatrixTile<T>> syrk_potrf("syrk_potrf"), disp_potrf("disp_potrf");

    ttg::Edge<Key2, MatrixTile<T>> potrf_trsm("potrf_trsm"), trsm_syrk("trsm_syrk"), gemm_trsm("gemm_trsm"),
        syrk_syrk("syrk_syrk"), disp_trsm("disp_trsm"), disp_syrk("disp_syrk");
    ttg::Edge<Key3, MatrixTile<T>> gemm_gemm("gemm_gemm"), trsm_gemm_row("trsm_gemm_row"),
        trsm_gemm_col("trsm_gemm_col"), disp_gemm("disp_gemm");

    auto tt_dispatch = make_dispatcher(input, disp_potrf, disp_trsm, disp_syrk, disp_gemm);
    tt_dispatch->set_keymap(keymap2a);
    tt_dispatch->set_defer_writer(defer_write);

    auto tt_potrf = make_potrf(A, disp_potrf, syrk_potrf, potrf_trsm, output);
    tt_potrf->set_keymap(keymap1);
    tt_potrf->set_defer_writer(defer_write);

    auto tt_trsm = make_trsm(A, disp_trsm, potrf_trsm, gemm_trsm, trsm_syrk, trsm_gemm_row, trsm_gemm_col, output);
    tt_trsm->set_keymap(keymap2a);
    tt_trsm->set_defer_writer(defer_write);

    auto tt_syrk = make_syrk(A, disp_syrk, trsm_syrk, syrk_syrk, syrk_potrf, syrk_syrk);
    tt_syrk->set_keymap(keymap2b);
    tt_syrk->set_defer_writer(defer_write);

    auto tt_gemm = make_gemm(A, disp_gemm, trsm_gemm_row, trsm_gemm_col, gemm_gemm, gemm_trsm, gemm_gemm);
    tt_gemm->set_keymap(keymap3);
    tt_gemm->set_defer_writer(defer_write);

    /* Priorities taken from DPLASMA */
    auto nt = A.cols();
    tt_potrf->set_priomap([nt](const Key1& key) { return ((nt - key[0]) * (nt - key[0]) * (nt - key[0])); });
    tt_trsm->set_priomap([nt](const Key2& key) {
      return ((nt - key[0]) * (nt - key[0]) * (nt - key[0]) + 3 * ((2 * nt) - key[1] - key[0] - 1) * (key[0] - key[1]));
    });
    tt_syrk->set_priomap(
        [nt](const Key2& key) { return ((nt - key[0]) * (nt - key[0]) * (nt - key[0]) + 3 * (key[0] - key[1])); });
    tt_gemm->set_priomap([nt](const Key3& key) {
      return ((nt - key[0]) * (nt - key[0]) * (nt - key[0]) + 3 * ((2 * nt) - key[0] - key[1] - 3) * (key[0] - key[1]) +
              6 * (key[0] - key[2]));
    });

    auto ins = std::make_tuple(tt_dispatch->template in<0>());
    auto outs = std::make_tuple(tt_potrf->template out<0>());
    std::vector<std::unique_ptr<ttg::TTBase>> ops(5);
    ops[0] = std::move(tt_dispatch);
    ops[1] = std::move(tt_potrf);
    ops[2] = std::move(tt_syrk);
    ops[3] = std::move(tt_trsm);
    ops[4] = std::move(tt_gemm);

    return make_ttg(std::move(ops), ins, outs, "POTRF TTG");
  }

};  // namespace potrf
