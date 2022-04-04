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

namespace trtri {

/* FLOP macros taken from DPLASMA */
double FMULS_DTRTRI(double __n) { return __n * (__n * ( 1./6. * __n + 0.5 ) + 1./3.); }
double FADDS_DTRTRI(double __n) { return __n * (__n * ( 1./6. * __n - 0.5 ) + 1./3.); }
double FLOPS_DTRTRI(double __n) { return      FMULS_DTRTRI(__n) +       FADDS_DTRTRI(__n); }

static int event_trtri_trsmr_startkey, event_trtri_trsmr_endkey;
static int event_trtri_gemm_startkey, event_trtri_gemm_endkey;
static int event_trtri_trsml_startkey, event_trtri_trsml_endkey;
static int event_trtri_trtri_startkey, event_trtri_trtri_endkey;
#define EVENT_TRTRI_INFO_CONVERTER "I{int}"

/**
 * Wrapper around parsec_profiling_trace_flags to enable/disable at will
 */
int trtri_parsec_profiling_trace_flags(parsec_profiling_stream_t* context, int key,
                                       uint64_t event_id, uint32_t taskpool_id,
                                       const void *info, uint16_t flags )
{
  int rc = 0;
#if USE_PARSEC_PROF_API
  if (profiling_enabled) {
    init_prof_thread();
    rc = parsec_profiling_trace_flags(context, key, event_id, taskpool_id, info, flags);
  }
#endif // USE_PARSEC_PROF_API
  return rc;
}

template <typename T>
auto make_trtri(const MatrixT<T>& A,
                lapack::Diag diag,
                ttg::Edge<Key1, MatrixTile<T>>& input_disp, // from the dispatcher
                ttg::Edge<Key2, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key1& key,
               MatrixTile<T>&& tile_kk,
               std::tuple<ttg::Out<Key2, MatrixTile<T>>>& out){
    const int K = key.K;

    std::cout << "TRTRI " << key << std::endl;
    trtri_parsec_profiling_trace_flags(prof, event_trtri_trtri_startkey, K, PROFILE_OBJECT_ID_NULL,
                                       &key, PARSEC_PROFILING_EVENT_HAS_INFO);

    
    int info = lapack::trtri(lapack::Uplo::Lower, diag, tile_kk.rows(), tile_kk.data(), tile_kk.rows());
    assert(0 == info);

    trtri_parsec_profiling_trace_flags(prof, event_trtri_trtri_endkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    /* send the tile to output */
    ttg::send<0>(Key2{K, K}, std::move(tile_kk), out);
  };
  return ttg::make_tt(f, ttg::edges(input_disp), ttg::edges(output_result), "TRTRI", 
                      {"tile_kk"}, {"output_result"});
}

template <typename T>
auto make_trsml(const MatrixT<T>& A,
                lapack::Diag diag,
                ttg::Edge<Key2, MatrixTile<T>>& input_kk, // will come from the dispatcher
                ttg::Edge<Key2, MatrixTile<T>>& input_kn,
                ttg::Edge<Key2, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key2& key,
               const MatrixTile<T>& tile_kk,
               MatrixTile<T>&& tile_kn,
               std::tuple<ttg::Out<Key2, MatrixTile<T>>>& out){
    const int K = key.I;
    const int N = key.J;

    auto mb = tile_kn.rows();
    assert(tile_kk.rows() == mb);

    std::cout << "TRSML " << key << std::endl;
    trtri_parsec_profiling_trace_flags(prof, event_trtri_trsml_startkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    blas::trsm(blas::Layout::ColMajor,
               blas::Side::Left,
               lapack::Uplo::Lower,
               blas::Op::NoTrans,
               diag,
               mb, mb, 1.0,
               tile_kk.data(), mb,
               tile_kn.data(), mb);

    trtri_parsec_profiling_trace_flags(prof, event_trtri_trsml_endkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    /* send the tile to output */
    ttg::send<0>(Key2{K, N}, std::move(tile_kn), out);
  };
  return ttg::make_tt(f, ttg::edges(input_kk, input_kn), ttg::edges(output_result), "TRSML", 
                      {"tile_kk", "tile_kn"}, {"output_result"});
}


template <typename T>
auto make_trsmr(const MatrixT<T>& A,
                lapack::Diag diag,
                ttg::Edge<Key2, MatrixTile<T>>& input_kk, // will from the dispatcher
                ttg::Edge<Key2, MatrixTile<T>>& input_mk, // will also come from the dispatcher
                ttg::Edge<Key3, MatrixTile<T>>& to_gemm_A,
                ttg::Edge<Key3, MatrixTile<T>>& to_gemm_B,
                ttg::Edge<Key3, MatrixTile<T>>& to_gemm_C,
                ttg::Edge<Key2, MatrixTile<T>>& to_trsml_kn)
{
  auto f = [=](const Key2& key,
               const MatrixTile<T>& tile_kk,
               MatrixTile<T>&& tile_mk,
               std::tuple<ttg::Out<Key3, MatrixTile<T>>, // gemm_A 
                          ttg::Out<Key3, MatrixTile<T>>, // gemm_B
                          ttg::Out<Key3, MatrixTile<T>>, // gemm_C
                          ttg::Out<Key2, MatrixTile<T>>  // trsml_kn
                         >& out){
    const int K = key.I;
    const int M = key.J;

    auto mb = tile_mk.rows();
    assert(tile_kk.rows() == mb);

    std::cout << "TRSMR " << key << std::endl;
    trtri_parsec_profiling_trace_flags(prof, event_trtri_trsmr_startkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    blas::trsm(blas::Layout::ColMajor,
               blas::Side::Right,
               lapack::Uplo::Lower,
               blas::Op::NoTrans,
               diag,
               mb, mb, -1.0,
               tile_kk.data(), mb,
               tile_mk.data(), mb);

    trtri_parsec_profiling_trace_flags(prof, event_trtri_trsmr_endkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    std::vector<Key3> keylist_A_gemm;
    std::vector<Key3> keylist_B_gemm;
    std::vector<Key3> keylist_C_gemm;
    std::vector<Key2> keylist_kn_trsml;

    if(K > 0) {
        keylist_A_gemm.reserve(K-1);
        for(auto k = 0; k < K; k++)
            keylist_A_gemm.push_back(Key3{K, M, k});
    }

    if(M == K+1 && K < A.rows()-2) {
        keylist_B_gemm.reserve(A.rows() - M + 1);
        for(auto m = M+1; m < A.rows(); m++)
            keylist_B_gemm.push_back(Key3{K+1, m, K});
    }

    if(M > K+1) {
        keylist_C_gemm.reserve(1);
        keylist_C_gemm.push_back(Key3{K+1, M, K});
    } else if(M == K+1) {
        keylist_kn_trsml.reserve(1);
        keylist_kn_trsml.push_back(Key2{K+1, K});
    }

    ttg::broadcast<0, 1, 2, 3>(std::make_tuple(keylist_A_gemm, keylist_B_gemm, keylist_C_gemm, keylist_kn_trsml),
                                  std::move(tile_mk), out);
  };
  return ttg::make_tt(f, ttg::edges(input_kk, input_mk), 
                         ttg::edges(to_gemm_A, to_gemm_B, to_gemm_C, to_trsml_kn), "TRSMR", 
                      {"tile_kk", "tile_mk"}, {"GEMM_A", "GEMM_B", "GEMM_C", "TRSML_kn"});
}

template <typename T>
auto make_gemm(const MatrixT<T>& A,
               ttg::Edge<Key3, MatrixTile<T>>& input_A,
               ttg::Edge<Key3, MatrixTile<T>>& input_B,
               ttg::Edge<Key3, MatrixTile<T>>& input_C,
               ttg::Edge<Key3, MatrixTile<T>>& to_gemm_B,
               ttg::Edge<Key3, MatrixTile<T>>& to_gemm_C,
               ttg::Edge<Key2, MatrixTile<T>>& to_trsml_kn)
{
  auto f = [=](const Key3& key,
               const MatrixTile<T>& input_A,
               const MatrixTile<T>& input_B,
               MatrixTile<T>&& input_C,
               std::tuple<ttg::Out<Key3, MatrixTile<T>>, // gemm_B
                          ttg::Out<Key3, MatrixTile<T>>, // gemm_C
                          ttg::Out<Key2, MatrixTile<T>> // trsml_kn
                         >& out){
    const int K = key.I;
    const int M = key.J;
    const int N = key.K;

    auto mb = input_A.rows();
    assert(input_B.rows() == mb);
    assert(input_C.rows() == mb);

    std::cout << "GEMM " << key << std::endl;
    trtri_parsec_profiling_trace_flags(prof, event_trtri_gemm_startkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    blas::gemm(blas::Layout::ColMajor,
               blas::Op::NoTrans,
               blas::Op::NoTrans,
               mb, mb, mb, 
               1.0, input_A.data(), mb,
                    input_B.data(), mb, 
               1.0, input_C.data(), mb);

    trtri_parsec_profiling_trace_flags(prof, event_trtri_gemm_endkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    std::vector<Key3> keylist_B_gemm;
    std::vector<Key3> keylist_C_gemm;
    std::vector<Key2> keylist_kn_trsml;

    if(M == K+1) {
        keylist_kn_trsml.reserve(1);
        keylist_kn_trsml.push_back(Key2{K+1, N});
    }

    if(M == K+1 && K < A.rows()-2) {
        keylist_B_gemm.reserve(A.rows() - M + 1);
        for(auto m = M+1; m < A.rows(); m++)
            keylist_B_gemm.push_back(Key3{K+1, m, N});
    }

    if(M > K+1) {
        keylist_C_gemm.reserve(1);
        keylist_C_gemm.push_back(Key3{K+1, M, N});
    }

    ttg::broadcast<0, 1, 2>(std::make_tuple(keylist_B_gemm, keylist_C_gemm, keylist_kn_trsml),
                            std::move(input_C), out);
  };
  return ttg::make_tt(f, ttg::edges(input_A, input_B, input_C), 
                         ttg::edges(to_gemm_B, to_gemm_C, to_trsml_kn), "GEMM", 
                      {"A", "B", "C"}, {"GEMM_B", "GEMM_C", "TRSML_kn"});
}


template<typename T>
auto make_dispatcher(const MatrixT<T>& A,
                     ttg::Edge<Key2, MatrixTile<T>>& input,
                     ttg::Edge<Key1, MatrixTile<T>>& to_trtri,
                     ttg::Edge<Key2, MatrixTile<T>>& to_trsml_kk,
                     ttg::Edge<Key2, MatrixTile<T>>& to_trsmr_kk,
                     ttg::Edge<Key2, MatrixTile<T>>& to_trsmr_mk)
{
  auto f = [=](const Key2& key,
               MatrixTile<T>&&tile,
               std::tuple<ttg::Out<Key1, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>>& out){
    std::cout << "TRTRI_Dispatch called with " << key << std::endl;
    if(key.I == key.J) {
      std::vector<Key2> keylist_trsml;
      std::vector<Key2> keylist_trsmr;
      std::cout << "TRTRI_Dispatch(" << key << ") sending to TRTRI(" << Key1{key.I} << ")" << std::endl;
      
      if(key.I < A.rows()) {
          keylist_trsmr.reserve(A.rows()-key.I+1);
          for(int k = key.I+1; k < A.rows(); k++) {
              std::cout << "TRTRI_Dispatch(" << key << ") sending to input_kk of TRSMR(" << Key2{key.I, k} << ")" << std::endl;
              keylist_trsmr.push_back(Key2{key.I, k});
          }
      }
      if(key.I > 0) {
          keylist_trsml.reserve(key.I-1);
          for(int k = 0; k < key.I; k++) {
              std::cout << "TRTRI_Dispatch(" << key << ") sending to input_kk of TRSML(" << Key2{key.I, k} << ")" << std::endl;
              keylist_trsml.push_back(Key2{key.I, k});
          }
      }
      ttg::broadcast<0, 1, 2>(std::make_tuple(Key1{key.I}, keylist_trsml, keylist_trsmr), std::move(tile), out);
      return;
    }

    std::cout << "TRTRI_Dispatch(" << key << ") sending to input_mk of TRSM_R(" << Key2{key.J, key.I} << ")" << std::endl;
    ttg::send<3>(Key2{key.J, key.I}, std::move(tile), out);
  };

  return ttg::make_tt(f, ttg::edges(input), ttg::edges(to_trtri, to_trsml_kk, to_trsmr_kk, to_trsmr_mk), "TRTRI Dispatch", {"Input"}, {"TRTRI", "TRSML_kk", "TRSMR_kk", "TRSMR_mk"});
}

auto make_trtri_ttg(MatrixT<double> &A, lapack::Diag diag, ttg::Edge<Key2, MatrixTile<double>>&input, ttg::Edge<Key2, MatrixTile<double>>&output ) {
  auto keymap1 = [&](const Key1& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.I, key.J) << std::endl;
    return A.rank_of(key.K, key.K);
  };

  auto keymap2a = [&](const Key2& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.I, key.J) << std::endl;
    return A.rank_of(key.I, key.J);
  };
  auto keymap2b = [&](const Key2& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.J, key.I) << std::endl;
    return A.rank_of(key.J, key.I);
  };

  auto keymap3 = [&](const Key3& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.J, key.K) << std::endl;
    return A.rank_of(key.J, key.K);
  };

  ttg::Edge<Key1, MatrixTile<double>> disp_trtri("disp_trtri");

  ttg::Edge<Key2, MatrixTile<double>> disp_trsml_kk("disp_trsml_kk"),
                                      disp_trsmr_kk("disp_trsmr_kk"),
                                      disp_trsmr_mk("disp_trsmr_mk"),
                                      trsmr_trsml("trsmr_trsml"),
                                      gemm_trsml("gemm_trsml"),
                                      trsml_nk("trsml_nk");

  ttg::Edge<Key3, MatrixTile<double>> gemm_gemm_B("gemm_gemm_B"),
                                      gemm_gemm_C("gemm_gemm_C"),
                                      trsmr_gemm_A("trsmr_gemm_A"),
                                      trsmr_gemm_B("trsmr_gemm_B"),
                                      trsmr_gemm_C("trsmr_gemm_C"),
                                      gemm_B("gemm_B"),
                                      gemm_C("gemm_C");

  auto tt_dispatch = make_dispatcher(A, input, disp_trtri, disp_trsml_kk, disp_trsmr_kk, disp_trsmr_mk);
  tt_dispatch->set_keymap(keymap2a);

  auto tt_trtri = make_trtri(A, diag, disp_trtri, output);
  tt_trtri->set_keymap(keymap1);

  trsml_nk = ttg::fuse(gemm_trsml, trsmr_trsml);
  auto tt_trsml  = make_trsml(A, diag, disp_trsml_kk, trsml_nk, output);
  tt_trsml->set_keymap(keymap2a);

  auto tt_trsmr = make_trsmr(A, diag, disp_trsmr_kk, disp_trsmr_mk, 
                             trsmr_gemm_A, trsmr_gemm_B, trsmr_gemm_C,
                             trsmr_trsml);
  tt_trsmr->set_keymap(keymap2b);

  gemm_B = ttg::fuse(trsmr_gemm_B, gemm_gemm_B);
  gemm_C = ttg::fuse(trsmr_gemm_C, gemm_gemm_C);
  auto tt_gemm  = make_gemm(A,
                            trsmr_gemm_A,
                            gemm_B,
                            gemm_C,
                            gemm_gemm_B,
                            gemm_gemm_C,
                            gemm_trsml);
  tt_gemm->set_keymap(keymap3);

  std::vector<std::unique_ptr<ttg::TTBase>> ops(5);
  ops[0] = std::move(tt_dispatch);
  ops[1] = std::move(tt_trtri);
  ops[2] = std::move(tt_trsml);
  ops[3] = std::move(tt_trsmr);
  ops[4] = std::move(tt_gemm);
  auto ins = std::make_tuple(tt_dispatch->template in<0>());
  auto outs = std::make_tuple(tt_trtri->template out<0>());

  return make_ttg(std::move(ops), ins, outs, "TRTRI TTG");
}

}