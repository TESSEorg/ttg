#pragma once

#include <ttg.h>
#include "pmw.h"
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>
#include "lapack.hh"

#undef DEBUG_TILES_VALUES

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
    auto f = [=](const Key1& key, MatrixTile<T>&& tile_kk,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>>& out) {
      const int K = key.K;

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
    auto f = [=](const Key2& key, const MatrixTile<T>& tile_kk, MatrixTile<T>&& tile_mk,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key3, MatrixTile<T>>,
                            ttg::Out<Key3, MatrixTile<T>>>& out) {
      const int M = key.I;
      const int K = key.J;  // the column equals the outer most look K (same as PO)

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
  }

  template <typename MatrixT>
  auto make_syrk(MatrixT& A,
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_disp,    // from the dispatcher
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_mk,      // from TRSM
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_kk,      // from SYRK
                 ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& output_potrf,  // to POTRF
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_syrk) {
    using T = typename MatrixT::element_type;
    auto f = [=](const Key2& key, const MatrixTile<T>& tile_mk, MatrixTile<T>&& tile_kk,
                 std::tuple<ttg::Out<Key1, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>>& out) {
      const int K = key.I;
      const int M = key.J;

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
  }

  template <typename MatrixT>
  auto make_gemm(MatrixT& A,
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_disp,   // From the dispatcher
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_kn,     // from TRSM
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_mk,     // from TRSM
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_mn,     // from TRSM
                 ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_trsm,  // to TRSM
                 ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& output_gemm) {
    using T = typename MatrixT::element_type;
    auto f = [=](const Key3& key, const MatrixTile<T>& tile_kn, const MatrixTile<T>& tile_mk, MatrixTile<T>&& tile_mn,
                 std::tuple<ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key3, MatrixTile<T>>>& out) {
      const int M = key.I;
      const int N = key.J;
      const int K = key.K;
      assert(M != N && M > K && N > K);

      assert(tile_mk.cols() == tile_kn.rows());
      assert(tile_mk.rows() == tile_mn.rows());
      assert(tile_kn.cols() == tile_mn.cols());

      if (ttg::tracing()) ttg::print("GEMM(", key, ")");
#if defined(DEBUG_TILES_VALUES)
      std::cout << "Before GEMM(" << key << "), A(" << M << ", " << K << ") is " << tile_mk << " and A(" << K << ", "
                << N << ") is " << tile_kn << " and A(" << M << ", " << N << ") is " << tile_mn;
#endif

      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, tile_mk.rows(), tile_mn.cols(),
                 tile_kn.rows(), -1.0, tile_mk.data(), tile_mk.lda(), tile_kn.data(), tile_kn.lda(), 1.0,
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
    return ttg::make_tt(f, ttg::edges(input_mk, input_kn, ttg::fuse(input_disp, input_mn)),
                        ttg::edges(output_trsm, output_gemm), "GEMM", {"input_mk", "input_kn", "input_mn/dispatcher"},
                        {"output_trsm", "outout_gemm"});
  }

  template <typename T>
  auto make_dispatcher(ttg::Edge<Key2, MatrixTile<T>>& input, ttg::Edge<Key1, MatrixTile<T>>& to_potrf,
                       ttg::Edge<Key2, MatrixTile<T>>& to_trsm, ttg::Edge<Key2, MatrixTile<T>>& to_syrk,
                       ttg::Edge<Key3, MatrixTile<T>>& to_gemm) {
    auto f = [=](const Key2& key, MatrixTile<T>&& tile,
                 std::tuple<ttg::Out<Key1, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>, ttg::Out<Key2, MatrixTile<T>>,
                            ttg::Out<Key3, MatrixTile<T>>>& out) {
      if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ")");
      if (0 == key.I && 0 == key.J) {
        // First element goes to POTRF
        if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to POTRF(", Key1{key.I}, ")");
        ttg::send<0>(Key1{key.I}, std::move(tile), out);
        return;
      }
      if (key.I == key.J) {
        // Other diagonal elements go to SYRK
        if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to SYRK(", Key2{0, key.I}, ")");
        ttg::send<2>(Key2{0, key.I}, std::move(tile), out);
        return;
      }
      // We only consider the lower triangular
      assert(key.I > key.J);
      if (0 == key.J) {
        // First column goes to TRSM
        if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to TRSM(", key, ")");
        ttg::send<1>(key, std::move(tile), out);
        return;
      }
      // Rest goes to GEMM
      if (ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to GEMM(", Key3{key.I, key.J, 0}, ")");
      ttg::send<3>(Key3{key.I, key.J, 0}, std::move(tile), out);
    };

    return ttg::make_tt(f, ttg::edges(input), ttg::edges(to_potrf, to_trsm, to_syrk, to_gemm), "POTRF Dispatch",
                        {"Input"}, {"POTRF", "TRSM", "SYRK", "GEMM"});
  }

  template <typename MatrixT>
  auto make_potrf_ttg(MatrixT& A, ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input,
                      ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output, bool defer_write) {
    using T = typename MatrixT::element_type;
    auto keymap1 = [&](const Key1& key) { return A.rank_of(key.K, key.K); };

    auto keymap2a = [&](const Key2& key) { return A.rank_of(key.I, key.J); };
    auto keymap2b = [&](const Key2& key) { return A.rank_of(key.I, key.I); };

    auto keymap3 = [&](const Key3& key) { return A.rank_of(key.I, key.J); };

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
    tt_potrf->set_priomap([&](const Key1& key) { return ((nt - key.K) * (nt - key.K) * (nt - key.K)); });
    tt_trsm->set_priomap([&](const Key2& key) {
      return ((nt - key.I) * (nt - key.I) * (nt - key.I) + 3 * ((2 * nt) - key.J - key.I - 1) * (key.I - key.J));
    });
    tt_syrk->set_priomap(
        [&](const Key2& key) { return ((nt - key.I) * (nt - key.I) * (nt - key.I) + 3 * (key.I - key.J)); });
    tt_gemm->set_priomap([&](const Key3& key) {
      return ((nt - key.I) * (nt - key.I) * (nt - key.I) + 3 * ((2 * nt) - key.I - key.J - 3) * (key.I - key.J) +
              6 * (key.I - key.K));
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
