#pragma once

#include <ttg.h>
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>
#include "pmw.h"
#include "lapack.hh"

#include <ttg.h>
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>
#include "pmw.h"
#include "lapack.hh"

namespace trtri_UPPER {

/* FLOP macros taken from DPLASMA */
inline double FMULS_DTRTRI(double __n) { return __n * (__n * ( 1./6. * __n + 0.5 ) + 1./3.); }
inline double FADDS_DTRTRI(double __n) { return __n * (__n * ( 1./6. * __n - 0.5 ) + 1./3.); }
inline double FLOPS_DTRTRI(double __n) { return      FMULS_DTRTRI(__n) +       FADDS_DTRTRI(__n); }

template <typename MatrixT>
auto make_trtri(const MatrixT& A,
                lapack::Diag diag,
                ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& input_disp, // from the dispatcher
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_result)
{
  using T = typename MatrixT::element_type;
  auto f = [=](const Key1& key,
               MatrixTile<T>&& tile_kk,
               std::tuple<ttg::Out<Key2, MatrixTile<T>>>& out){
    const int K = key.K;

    if(ttg::tracing()) ttg::print("TRTRI(", key, ")");
    
    int info = lapack::trtri(lapack::Uplo::Upper, diag, tile_kk.cols(), tile_kk.data(), tile_kk.lda());
    assert(0 == info);

    /* send the tile to output */
    ttg::send<0>(Key2{K, K}, std::move(tile_kk), out);
  };
  return ttg::make_tt(f, ttg::edges(input_disp), ttg::edges(output_result), "TRTRI", 
                      {"tile_kk"}, {"output_result"});
}

template <typename MatrixT>
auto make_trsmr(const MatrixT& A,
                lapack::Diag diag,
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_kk, // will come from the dispatcher
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_mk,
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& output_result)
{
  using T = typename MatrixT::element_type;
  auto f = [=](const Key2& key,
               const MatrixTile<T>& tile_kk,
               MatrixTile<T>&& tile_mk,
               std::tuple<ttg::Out<Key2, MatrixTile<T>>>& out){
    const int K = key.I;
    const int M = key.J;

    /* tile_mk is mb x nb */
    auto mb = tile_mk.rows();
    auto nb = tile_mk.cols();
    assert(tile_kk.cols() == nb /* tile_kk must be tile_kk.rows() x nb if side == Right in TRSM */);

    if(ttg::tracing()) ttg::print("TRSMR(", key, ")");

    blas::trsm(blas::Layout::ColMajor,
               blas::Side::Right,
               lapack::Uplo::Upper,
               blas::Op::NoTrans,
               diag,
               mb, nb, 1.0,
               tile_kk.data(), tile_kk.lda(),
               tile_mk.data(), tile_mk.lda());

    /* send the tile to output */
    ttg::send<0>(Key2{M, K}, std::move(tile_mk), out);
  };
  return ttg::make_tt(f, ttg::edges(input_kk, input_mk), ttg::edges(output_result), "TRSMR", 
                      {"tile_kk", "tile_mk"}, {"output_result"});
}


template <typename MatrixT>
auto make_trsml(const MatrixT& A,
                lapack::Diag diag,
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_kk, // will from the dispatcher
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input_kn, // will also come from the dispatcher
                ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& to_gemm_A,
                ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& to_gemm_B,
                ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& to_gemm_C,
                ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_trsml_kn)
{
  using T = typename MatrixT::element_type;
  auto f = [=](const Key2& key,
               const MatrixTile<T>& tile_kk,
               MatrixTile<T>&& tile_kn,
               std::tuple<ttg::Out<Key3, MatrixTile<T>>, // gemm_A 
                          ttg::Out<Key3, MatrixTile<T>>, // gemm_B
                          ttg::Out<Key3, MatrixTile<T>>, // gemm_C
                          ttg::Out<Key2, MatrixTile<T>>  // trsmr_mk
                         >& out){
    const int K = key.I;
    const int N = key.J;

    auto mb = tile_kn.rows();
    auto nb = tile_kn.cols();
    assert(tile_kk.cols() == mb /* tile_kk must be tile_kk.rows() x mb if side == Left in TRSM */);

    if(ttg::tracing()) ttg::print("TRSML(", key, ")");

    blas::trsm(blas::Layout::ColMajor,
               blas::Side::Left,
               lapack::Uplo::Upper,
               blas::Op::NoTrans,
               diag,
               mb, nb, -1.0,
               tile_kk.data(), tile_kk.lda(),
               tile_kn.data(), tile_kn.lda());

    std::vector<Key3> keylist_A_gemm;
    std::vector<Key3> keylist_B_gemm;
    std::vector<Key3> keylist_C_gemm;
    std::vector<Key2> keylist_kn_trsmr;

    if(K > 0) {
        keylist_B_gemm.reserve(K-1);
        for(auto k = 0; k < K; k++)
            keylist_B_gemm.push_back(Key3{K, k, N});
    }

    if(N == K+1 && K < A.cols()-2) {
        keylist_A_gemm.reserve(A.rows() - N + 1);
        for(auto n = N+1; n < A.cols(); n++)
            keylist_A_gemm.push_back(Key3{K+1, K, n});
    }

    if(N > K+1) {
        keylist_C_gemm.reserve(1);
        keylist_C_gemm.push_back(Key3{K+1, K, N});
    } else if(N == K+1) {
        keylist_kn_trsmr.reserve(1);
        keylist_kn_trsmr.push_back(Key2{K+1, K});
    }

    ttg::broadcast<0, 1, 2, 3>(std::make_tuple(keylist_A_gemm, keylist_B_gemm, keylist_C_gemm, keylist_kn_trsmr),
                                  std::move(tile_kn), out);
  };
  return ttg::make_tt(f, ttg::edges(input_kk, input_kn), 
                         ttg::edges(to_gemm_A, to_gemm_B, to_gemm_C, to_trsml_kn), "TRSML", 
                      {"tile_kk", "tile_kn"}, {"GEMM_A", "GEMM_B", "GEMM_C", "TRSMR_kn"});
}

template <typename MatrixT>
auto make_gemm(const MatrixT& A,
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_A,
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_B,
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& input_C,
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& to_gemm_A,
               ttg::Edge<Key3, MatrixTile<typename MatrixT::element_type>>& to_gemm_C,
               ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_trsmr_mk)
{
  using T = typename MatrixT::element_type;
  auto f = [=](const Key3& key,
               const MatrixTile<T>& input_A,
               const MatrixTile<T>& input_B,
               MatrixTile<T>&& input_C,
               std::tuple<ttg::Out<Key3, MatrixTile<T>>, // gemm_A
                          ttg::Out<Key3, MatrixTile<T>>, // gemm_C
                          ttg::Out<Key2, MatrixTile<T>>  // trsmr_mk
                         >& out){
    const int K = key.I;
    const int M = key.J;
    const int N = key.K;

    assert(input_A.cols() == input_B.rows());
    assert(input_A.rows() == input_C.rows());
    assert(input_B.cols() == input_C.cols());

    if(ttg::tracing()) ttg::print("GEMM(", key, ")");

    blas::gemm(blas::Layout::ColMajor,
               blas::Op::NoTrans,
               blas::Op::NoTrans,
               input_A.rows(), input_B.cols(), input_A.cols(), 
               1.0, input_A.data(), input_A.lda(),
                    input_B.data(), input_B.lda(), 
               1.0, input_C.data(), input_C.lda());

    std::vector<Key3> keylist_A_gemm;
    std::vector<Key3> keylist_C_gemm;
    std::vector<Key2> keylist_mk_trsmr;

    if(N == K+1) {
        keylist_mk_trsmr.reserve(1);
        keylist_mk_trsmr.push_back(Key2{K+1, M});
    }

    if(N == K+1 && K < A.rows()-2) {
        keylist_A_gemm.reserve(A.rows() - N + 1);
        for(auto n = N+1; n < A.cols(); n++)
            keylist_A_gemm.push_back(Key3{K+1, M, n});
    }

    if(N > K+1) {
        keylist_C_gemm.reserve(1);
        keylist_C_gemm.push_back(Key3{K+1, M, N});
    }

    ttg::broadcast<0, 1, 2>(std::make_tuple(keylist_A_gemm, keylist_C_gemm, keylist_mk_trsmr),
                            std::move(input_C), out);
  };
  return ttg::make_tt(f, ttg::edges(input_A, input_B, input_C), 
                         ttg::edges(to_gemm_A, to_gemm_C, to_trsmr_mk), "GEMM", 
                      {"A", "B", "C"}, {"GEMM_A", "GEMM_C", "TRSMR_mk"});
}


template<typename MatrixT>
auto make_dispatcher(const MatrixT& A,
                     ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& input,
                     ttg::Edge<Key1, MatrixTile<typename MatrixT::element_type>>& to_trtri,
                     ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_trsmr_kk,
                     ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_trsml_kk,
                     ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>& to_trsml_kn)
{
  using T = typename MatrixT::element_type;
  auto f = [=](const Key2& key,
               MatrixTile<T>&&tile,
               std::tuple<ttg::Out<Key1, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>>& out){
    if(ttg::tracing()) ttg::print("TRTRI_Dispatch(", key, ")");
    if(key.I == key.J) {
      std::vector<Key2> keylist_trsml;
      std::vector<Key2> keylist_trsmr;
      if(ttg::tracing()) ttg::print("TRTRI_Dispatch(", key, ") sending to TRTRI(", Key1{key.I}, ")");
      
      if(key.I < A.rows()) {
          keylist_trsml.reserve(A.rows()-key.I+1);
          for(int k = key.I+1; k < A.rows(); k++) {
              if(ttg::tracing()) ttg::print("TRTRI_Dispatch(", key, ") sending to input_kk of TRSML(", Key2{key.I, k}, ")");
              keylist_trsml.push_back(Key2{key.I, k});
          }
      }
      if(key.I > 0) {
          keylist_trsmr.reserve(key.I-1);
          for(int k = 0; k < key.I; k++) {
              if(ttg::tracing()) ttg::print("TRTRI_Dispatch(", key, ") sending to input_kk of TRSMR(", Key2{key.I, k}, ")");
              keylist_trsmr.push_back(Key2{key.I, k});
          }
      }
      ttg::broadcast<0, 1, 2>(std::make_tuple(Key1{key.I}, std::move(keylist_trsmr), std::move(keylist_trsml)), std::move(tile), out);
      return;
    }

    if(ttg::tracing()) ttg::print("TRTRI_Dispatch(", key, ") sending to input_kn of TRSM_L(", Key2{key.I, key.J}, ")");
    ttg::send<3>(Key2{key.I, key.J}, std::move(tile), out);
  };

  return ttg::make_tt(f, ttg::edges(input), ttg::edges(to_trtri, to_trsmr_kk, to_trsml_kk, to_trsml_kn), "TRTRI Dispatch", {"Input"}, {"TRTRI", "TRSMR_kk", "TRSML_kk", "TRSML_kn"});
}

template <typename MatrixT>
auto make_trtri_ttg(const MatrixT &A, lapack::Diag diag, ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>&input, ttg::Edge<Key2, MatrixTile<typename MatrixT::element_type>>&output, bool defer_write ) {
  using T = typename MatrixT::element_type;
  auto keymap1 = [&](const Key1& key) {
    return A.rank_of(key.K, key.K);
  };

  auto keymap2a = [&](const Key2& key) {
    return A.rank_of(key.J, key.I);
  };
  auto keymap2b = [&](const Key2& key) {
    return A.rank_of(key.I, key.J);
  };

  auto keymap3 = [&](const Key3& key) {
    return A.rank_of(key.J, key.K);
  };

  ttg::Edge<Key1, MatrixTile<T>> disp_trtri("disp_trtri");

  ttg::Edge<Key2, MatrixTile<T>> disp_trsmr_kk("disp_trsmr_kk"),
                                      disp_trsml_kk("disp_trsml_kk"),
                                      disp_trsml_kn("disp_trsml_kn"),
                                      trsml_trsmr("trsml_trsmr"),
                                      gemm_trsmr("gemm_trsmr"),
                                      trsmr_km("trsmr_km");

  ttg::Edge<Key3, MatrixTile<T>> gemm_gemm_B("gemm_gemm_B"),
                                      gemm_gemm_C("gemm_gemm_C"),
                                      trsml_gemm_A("trsml_gemm_A"),
                                      trsml_gemm_B("trsml_gemm_B"),
                                      trsml_gemm_C("trsml_gemm_C"),
                                      gemm_B("gemm_B"),
                                      gemm_C("gemm_C");

  auto tt_dispatch = make_dispatcher(A, input, disp_trtri, disp_trsmr_kk, disp_trsml_kk, disp_trsml_kn);
  tt_dispatch->set_keymap(keymap2b);
  tt_dispatch->set_defer_writer(defer_write);

  auto tt_trtri = make_trtri(A, diag, disp_trtri, output);
  tt_trtri->set_keymap(keymap1);
  tt_trtri->set_defer_writer(defer_write);

  trsmr_km = ttg::fuse(gemm_trsmr, trsml_trsmr);
  auto tt_trsmr  = make_trsmr(A, diag, disp_trsmr_kk, trsmr_km, output);
  tt_trsmr->set_keymap(keymap2a);
  tt_trsmr->set_defer_writer(defer_write);

  auto tt_trsml = make_trsml(A, diag, disp_trsml_kk, disp_trsml_kn, 
                             trsml_gemm_A, trsml_gemm_B, trsml_gemm_C,
                             trsml_trsmr);
  tt_trsml->set_keymap(keymap2b);
  tt_trsml->set_defer_writer(defer_write);

  gemm_B = ttg::fuse(trsml_gemm_B, gemm_gemm_B);
  gemm_C = ttg::fuse(trsml_gemm_C, gemm_gemm_C);
  auto tt_gemm  = make_gemm(A,
                            trsml_gemm_A,
                            gemm_B,
                            gemm_C,
                            gemm_gemm_B,
                            gemm_gemm_C,
                            gemm_trsmr);
  tt_gemm->set_keymap(keymap3);
  tt_gemm->set_defer_writer(defer_write);

  auto ins = std::make_tuple(tt_dispatch->template in<0>());
  auto outs = std::make_tuple(tt_trtri->template out<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops(5);
  ops[0] = std::move(tt_dispatch);
  ops[1] = std::move(tt_trtri);
  ops[2] = std::move(tt_trsmr);
  ops[3] = std::move(tt_trsml);
  ops[4] = std::move(tt_gemm);

  return make_ttg(std::move(ops), ins, outs, "TRTRI TTG");
}

}