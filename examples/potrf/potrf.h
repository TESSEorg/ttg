#pragma once

#include <ttg.h>
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>
#include "pmw.h"
#include "lapack.hh"

namespace potrf {

/* FLOP macros taken from DPLASMA */
#define FMULS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) )

template <typename T>
auto make_potrf(MatrixT<T>& A,
                ttg::Edge<Key1, MatrixTile<T>>& input_disp, // from the dispatcher
                ttg::Edge<Key1, MatrixTile<T>>& input,
                ttg::Edge<Key2, MatrixTile<T>>& output_trsm,
                ttg::Edge<Key2, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key1& key,
               MatrixTile<T>&& tile_kk,
               std::tuple<ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>>& out){
    const int K = key.K;

    if(ttg::tracing()) ttg::print("POTRF(", key, ")");

    lapack::potrf(lapack::Uplo::Lower, tile_kk.rows(), tile_kk.data(), tile_kk.rows());

    /* send the tile to outputs */
    std::vector<Key2> keylist;
    keylist.reserve(A.rows() - K);
    /* TODO: reverse order of arrays */
    for (int m = K+1; m < A.rows(); ++m) {
      /* send tile to trsm */
      if(ttg::tracing()) ttg::print("POTRF(", key, "): sending output to TRSM(", Key2{m, K}, ")");
      keylist.push_back(Key2(m, K));
    }
    ttg::broadcast<0, 1>(std::make_tuple(Key2(K, K), keylist), std::move(tile_kk), out);
  };
  return ttg::make_tt(f, ttg::edges(ttg::fuse(input, input_disp)), ttg::edges(output_result, output_trsm), "POTRF", 
                      {"tile_kk/dispatcher"}, {"output_result", "output_trsm"});
}

template <typename T>
auto make_trsm(MatrixT<T>& A,
               ttg::Edge<Key2, MatrixTile<T>>& input_disp,   // from the dispatcher
               ttg::Edge<Key2, MatrixTile<T>>& input_kk,     // from POTRF
               ttg::Edge<Key2, MatrixTile<T>>& input_mk,     // from previous GEMM
               ttg::Edge<Key2, MatrixTile<T>>& output_diag,  // to SYRK
               ttg::Edge<Key3, MatrixTile<T>>& output_row,   // to GEMM
               ttg::Edge<Key3, MatrixTile<T>>& output_col,   // to GEMM
               ttg::Edge<Key2, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key2& key,
               const MatrixTile<T>&  tile_kk,
                     MatrixTile<T>&& tile_mk,
                     std::tuple<ttg::Out<Key2, MatrixTile<T>>,
                                ttg::Out<Key2, MatrixTile<T>>,
                                ttg::Out<Key3, MatrixTile<T>>,
                                ttg::Out<Key3, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.J; // the column equals the outer most look K (same as PO)

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_kk.rows());
    assert(tile_mk.cols() == tile_kk.cols());

    auto m = tile_mk.rows();

    if(ttg::tracing()) ttg::print("TRSM(", key, ")");

    blas::trsm(blas::Layout::ColMajor,
               blas::Side::Right,
               lapack::Uplo::Lower,
               blas::Op::Trans,
               blas::Diag::NonUnit,
               tile_kk.rows(), m, 1.0,
               tile_kk.data(), m,
               tile_mk.data(), m);

    std::vector<Key3> keylist_row;
    keylist_row.reserve(I-J-1);
    std::vector<Key3> keylist_col;
    keylist_col.reserve(A.rows()-I-1);

    /* send tile to syrk on diagonal */
    if(ttg::tracing()) ttg::print("TRSM(", key, "): sending output to syrk(", Key2{I, K}, ")");

    /* send the tile to all gemms across in row i */
    for (int n = J+1; n < I; ++n) {
      if(ttg::tracing()) ttg::print("TRSM(", key, "): sending output to gemm( ", Key3{I, n, K}, ")");
      keylist_row.push_back(Key3(I, n, K));
    }

    /* send the tile to all gemms down in column i */
    for (int m = I+1; m < A.rows(); ++m) {
      if(ttg::tracing()) ttg::print("TRSM(", key, "): sending output to gemm( ", Key3{m, I, K}, ")");
      keylist_col.push_back(Key3(m, I, K));
    }

    ttg::broadcast<0, 1, 2, 3>(std::make_tuple(key,
                                               Key2(I, K),
                                               keylist_row, keylist_col),
                               std::move(tile_mk), out);
  };
  return ttg::make_tt(f, ttg::edges(input_kk, ttg::fuse(input_mk, input_disp)), 
                      ttg::edges(output_result, output_diag, output_row, output_col),
                      "TRSM", {"tile_kk", "tile_mk/dispatcher"}, 
                      {"output_result", "output_diag", "output_row", "output_col"});
}


template <typename T>
auto make_syrk(MatrixT<T>& A,
               ttg::Edge<Key2, MatrixTile<T>>& input_disp,  // from the dispatcher
               ttg::Edge<Key2, MatrixTile<T>>& input_mk,    // from TRSM
               ttg::Edge<Key2, MatrixTile<T>>& input_mm,    // from SYRK
               ttg::Edge<Key1, MatrixTile<T>>& output_potrf,// to POTRF
               ttg::Edge<Key2, MatrixTile<T>>& output_syrk)
{
  auto f = [=](const Key2& key,
               const MatrixTile<T>&  tile_mk,
                     MatrixTile<T>&& tile_mm,
                     std::tuple<ttg::Out<Key1, MatrixTile<T>>,
                                ttg::Out<Key2, MatrixTile<T>>>& out){
    const int I = key.I;
    const int K = key.J;

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_mm.rows());
    assert(tile_mk.cols() == tile_mm.cols());

    auto m = tile_mk.rows();

    if(ttg::tracing()) ttg::print("SYRK(", key, ")");

    blas::syrk(blas::Layout::ColMajor,
               lapack::Uplo::Lower,
               blas::Op::NoTrans,
               tile_mk.rows(), m, -1.0,
               tile_mk.data(), m, 1.0,
               tile_mm.data(), m);

    if (I == K+1) {
      /* send the tile to potrf */
      if(ttg::tracing()) ttg::print("SYRK(", key, "): sending output to POTRF(", Key1{K+1}, ")");
      ttg::send<0>(Key1(K+1), std::move(tile_mm), out);
    } else {
      /* send output to next syrk */
      if(ttg::tracing()) ttg::print("SYRK(", key, "): sending output to SYRK(", Key2{I, K+1}, ")");
      ttg::send<1>(Key2(I, K+1), std::move(tile_mm), out);
    }

  };
  return ttg::make_tt(f,
                      ttg::edges(input_mk, ttg::fuse(input_mm, input_disp)),
                      ttg::edges(output_potrf, output_syrk), "SYRK",
                      {"tile_mk", "tile_mm/dispatcher"}, {"output_potrf", "output_syrk"});
}


template <typename T>
auto make_gemm(MatrixT<T>& A,
               ttg::Edge<Key3, MatrixTile<T>>& input_disp,  // From the dispatcher
               ttg::Edge<Key3, MatrixTile<T>>& input_nk,    // from TRSM
               ttg::Edge<Key3, MatrixTile<T>>& input_mk,    // from TRSM
               ttg::Edge<Key3, MatrixTile<T>>& input_nm,    // from TRSM
               ttg::Edge<Key2, MatrixTile<T>>& output_trsm, // to TRSM
               ttg::Edge<Key3, MatrixTile<T>>& output_gemm)
{
  auto f = [=](const Key3& key,
              const MatrixTile<T>& tile_nk,
              const MatrixTile<T>& tile_mk,
                    MatrixTile<T>&& tile_nm,
                    std::tuple<ttg::Out<Key2, MatrixTile<T>>,
                               ttg::Out<Key3, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I != J && I > K && J > K);

    /* No support for different tile sizes yet */
    assert(tile_nk.rows() == tile_mk.rows() && tile_nk.rows() == tile_nm.rows());
    assert(tile_nk.cols() == tile_mk.cols() && tile_nk.cols() == tile_nm.cols());

    auto m = tile_nk.rows();

    if(ttg::tracing()) ttg::print("GEMM(", key, ")");

    blas::gemm(blas::Layout::ColMajor,
               blas::Op::NoTrans,
               blas::Op::Trans,
               m, m, m, -1.0,
               tile_nk.data(), m,
               tile_mk.data(), m, 1.0,
               tile_nm.data(), m);

    if (J == K+1) {
      /* send the tile to trsm */
      if(ttg::tracing()) ttg::print("GEMM(", key, "): sending output to TRSM(", Key2{I, J}, ")");
      ttg::send<0>(Key2(I, J), std::move(tile_nm), out);
    } else {
      /* send the tile to the next gemm */
      if(ttg::tracing()) ttg::print("GEMM(", key, "): sending output to GEMM(", Key3{I, J, K+1}, ")");
      ttg::send<1>(Key3(I, J, K+1), std::move(tile_nm), out);
    }
  };
  return ttg::make_tt(f,
                      ttg::edges(input_nk, input_mk, ttg::fuse(input_disp, input_nm)),
                      ttg::edges(output_trsm, output_gemm), "GEMM",
                      {"input_nk", "input_mk", "input_nm/dispatcher"},
                      {"output_trsm", "outout_gemm"});
}

template<typename T>
auto make_dispatcher(ttg::Edge<Key2, MatrixTile<T>>& input,
                     ttg::Edge<Key1, MatrixTile<T>>& to_potrf,
                     ttg::Edge<Key2, MatrixTile<T>>& to_trsm,
                     ttg::Edge<Key2, MatrixTile<T>>& to_syrk,
                     ttg::Edge<Key3, MatrixTile<T>>& to_gemm)
{
  auto f = [=](const Key2& key,
               MatrixTile<T>&&tile,
               std::tuple<ttg::Out<Key1, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key3, MatrixTile<T>>>& out){
    if(ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ")");
    if(0 == key.I && 0 == key.J) {
      // First element goes to POTRF
      if(ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to POTRF(", Key1{key.I}, ")");
      ttg::send<0>(Key1{key.I}, std::move(tile), out);
      return;
    }
    if(key.I == key.J) {
      // Other diagonal elements go to SYRK
      if(ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to SYRK(", Key2{key.I, 0}, ")");
      ttg::send<2>(Key2{key.I, 0}, std::move(tile), out);
      return;
    }
    // We only consider the lower triangular
    assert(key.I > key.J);
    if(0 == key.J) {
      // First column goes to TRSM
      if(ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to TRSM(", key, ")");
      ttg::send<1>(key, std::move(tile), out);
      return;
    }
    // Rest goes to GEMM
    if(ttg::tracing()) ttg::print("POTRF_Dispatch(", key, ") sending to GEMM(", Key3{key.I, key.J, 0}, ")");
    ttg::send<3>(Key3{key.I, key.J, 0}, std::move(tile), out);
  };

  return ttg::make_tt(f, ttg::edges(input), ttg::edges(to_potrf, to_trsm, to_syrk, to_gemm), "POTRF Dispatch", {"Input"}, {"POTRF", "TRSM", "SYRK", "GEMM"});
}

auto make_potrf_ttg(MatrixT<double> &A, ttg::Edge<Key2, MatrixTile<double>>&input, ttg::Edge<Key2, MatrixTile<double>>&output ) {
  auto keymap1 = [&](const Key1& key) {
    return A.rank_of(key.K, key.K);
  };

  auto keymap2 = [&](const Key2& key) {
    return A.rank_of(key.I, key.J);
  };

  auto keymap3 = [&](const Key3& key) {
    return A.rank_of(key.I, key.J);
  };

  ttg::Edge<Key1, MatrixTile<double>> syrk_potrf("syrk_potrf"), 
                                      disp_potrf("disp_potrf");

  ttg::Edge<Key2, MatrixTile<double>> potrf_trsm("potrf_trsm"),
                                      trsm_syrk("trsm_syrk"),
                                      gemm_trsm("gemm_trsm"),
                                      syrk_syrk("syrk_syrk"),
                                      disp_trsm("disp_trsm"),
                                      disp_syrk("disp_syrk");
  ttg::Edge<Key3, MatrixTile<double>> gemm_gemm("gemm_gemm"),
                                      trsm_gemm_row("trsm_gemm_row"),
                                      trsm_gemm_col("trsm_gemm_col"),
                                      disp_gemm("disp_gemm");

  auto tt_dispatch = make_dispatcher(input, disp_potrf, disp_trsm, disp_syrk, disp_gemm);
  tt_dispatch->set_keymap(keymap2);

  auto tt_potrf = make_potrf(A, disp_potrf, syrk_potrf, potrf_trsm, output);
  tt_potrf->set_keymap(keymap1);
  auto tt_trsm  = make_trsm(A,
                            disp_trsm, potrf_trsm, gemm_trsm,
                            trsm_syrk, trsm_gemm_row, trsm_gemm_col, output);
  tt_trsm->set_keymap(keymap2);
  auto tt_syrk  = make_syrk(A, disp_syrk, trsm_syrk, syrk_syrk, syrk_potrf, syrk_syrk);
  tt_syrk->set_keymap(keymap2);
  auto tt_gemm  = make_gemm(A,
                            disp_gemm, trsm_gemm_row, trsm_gemm_col, gemm_gemm,
                            gemm_trsm, gemm_gemm);
  tt_gemm->set_keymap(keymap3);

  /* Priorities taken from DPLASMA */
  auto nt = A.cols();
  tt_potrf->set_priomap([&](const Key1& key){ return ((nt - key.K) * (nt - key.K) * (nt - key.K)); });
  tt_trsm->set_priomap([&](const Key2& key) { return ((nt - key.I) * (nt - key.I) * (nt - key.I)
                                                      + 3 * ((2 * nt) - key.J - key.I - 1) * (key.I - key.J)); });
  tt_syrk->set_priomap([&](const Key2& key) { return ((nt - key.I) * (nt - key.I) * (nt - key.I)
                                                      + 3 * (key.I - key.J)); });
  tt_gemm->set_priomap([&](const Key3& key) { return ((nt - key.I) * (nt - key.I) * (nt - key.I)
                                                      + 3 * ((2 * nt) - key.I - key.J - 3) * (key.I - key.J)
                                                      + 6 * (key.I - key.K)); });

  std::vector<std::unique_ptr<ttg::TTBase>> ops(5);
  ops[0] = std::move(tt_dispatch);
  ops[1] = std::move(tt_potrf);
  ops[2] = std::move(tt_syrk);
  ops[3] = std::move(tt_trsm);
  ops[4] = std::move(tt_gemm);
  auto ins = std::make_tuple(tt_dispatch->template in<0>());
  auto outs = std::make_tuple(tt_potrf->template out<0>());

  return make_ttg(std::move(ops), ins, outs, "POTRF TTG");
}

};