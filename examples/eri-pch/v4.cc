// SPDX-License-Identifier: BSD-3-Clause
//
// eri-pch v4 -- v3's block update plus row-parallel chunking.
//
// In v3, gather_pivot did everything serially:
//   1. assemble Q from slabs
//   2. R_AB = Q[(A,B), :] - L[(A,B), :] · L[(A,B), :]^T   (small)
//   3. pivoted Cholesky on R_AB                            (small)
//   4. L_AB_chosen extraction                              (small)
//   5. L_new = (Q[:, Π] - L · L_AB_chosen^T) · U^{-T}      (BIG dgemm + dtrsm)
//   6. write L_new into L; update d                        (proportional to BIG)
//
// Step 5+6 are row-disjoint: row p of L_new depends only on row p of L
// and row p of Q. v4 splits steps 5+6 across N_CHUNKS row-chunks fanned
// out as TTG tasks. Steps 1-4 (cheap, ~ms) stay in gather_pivot.
//
// Topology change from v3:
//   gather_pivot --> e_chunk_dispatch --> compute_l_chunk --> e_chunk_done --> gather_finalize --> e_loop_iter
//                                                              (streaming reducer barrier)
//   (When pivoted Cholesky finds W' = 0 columns, gather_pivot emits
//    iter+1 directly; no chunks are fanned out.)
//
// The shared L matrix and d vector are written by chunks in disjoint row
// ranges. Per-iteration block setup data (Q_chosen, U, L_AB_chosen)
// lives as metadata on the driver, populated by gather_pivot before
// chunk fan-out and read concurrently by chunks. The streaming reducer
// at gather_finalize ensures all chunks of iter N complete before iter
// N+1's gather_pivot may overwrite that metadata.
//
// Usage: eri-pch_v4-{mad,parsec} [<molecule.xyz> [<basis-name> [<tolerance>]]]

#include "common.h"

#include <ttg.h>
#include <ttg/util/dot.h>
#include <ttg/util/multiindex.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace eri_pch {

struct PivotSlab {
  std::size_t sp_idx;
  std::size_t pivot_sp_idx;
  std::size_t pair_begin;
  std::size_t npairs;
  std::size_t W;
  std::vector<double> rows;

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & sp_idx & pivot_sp_idx & pair_begin & npairs & W & rows;
  }
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & sp_idx & pivot_sp_idx & pair_begin & npairs & W & rows;
  }
};
using PivotSlabs = std::vector<PivotSlab>;

struct FlowProfile {
  std::atomic<long> ns_compute_diag{0};
  std::atomic<long> ns_compute_slab{0};
  std::atomic<long> ns_gather_setup{0};   // Q assembly + R_AB + pivoted-chol + L_AB_chosen
  std::atomic<long> ns_chunk_compute{0};  // BIG dgemm + dtrsm + d update, per chunk
  std::atomic<long> ns_iter_dispatch{0};
  std::atomic<int> n_compute_slab_calls{0};
  std::atomic<int> n_gather_calls{0};
  std::atomic<int> n_chunk_calls{0};
  std::atomic<int> n_new_columns{0};
  std::mutex tids_mu;
  std::set<std::thread::id> tids_compute_slab;
  std::set<std::thread::id> tids_chunk;
};
inline long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Per-iteration block-update inputs, populated by prepare_block_update and
// read by all compute_l_chunk tasks of that iteration.
struct BlockSetup {
  std::size_t M_old = 0;
  std::size_t Wp = 0;
  std::size_t pivot_sp_idx = 0;
  Eigen::MatrixXd Q_chosen;       // P × Wp     (already permuted)
  Eigen::MatrixXd L_AB_chosen;    // Wp × M_old (already permuted)
  Eigen::MatrixXd U_block;        // Wp × Wp lower triangular
};

// Choose a target chunk count for the gather row-fan-out. Per chunk we
// want enough work to amortize task spawn overhead (target >= a few ms
// of dgemm per chunk) but enough chunks to fill the box. 12 is fine for
// hardware_concurrency() up through 24; users can override with the env
// var TTG_ERI_PCH_CHUNKS.
inline std::size_t default_chunk_count() {
  if (const char* env = std::getenv("TTG_ERI_PCH_CHUNKS")) {
    if (auto v = std::atoi(env); v > 0) return static_cast<std::size_t>(v);
  }
  unsigned int hc = std::thread::hardware_concurrency();
  if (hc == 0) hc = 8;
  return static_cast<std::size_t>(std::min<unsigned int>(hc, 24));
}

class CholeskyDriver {
 public:
  IntegralData D;
  EnginePool pool;
  Eigen::VectorXd d;
  Eigen::MatrixXd L;
  std::size_t M_current = 0;
  std::size_t M_capacity = 0;
  std::size_t total_shellpair_recomputes = 0;
  std::size_t n_chunks;
  BlockSetup cur_block;

  bool verbose = false;
  bool trace_each_vector = false;

  CholeskyDriver(IntegralData data)
      : D(std::move(data)),
        pool(D.basis.max_nprim(), D.basis.max_l()),
        n_chunks(default_chunk_count()) {
    const auto P = static_cast<Eigen::Index>(D.npairs());
    d = Eigen::VectorXd::Zero(P);
    M_capacity = 64;
    L = Eigen::MatrixXd(P, M_capacity);
  }

  void compute_diagonal_for_shell_pair(std::size_t sp_idx) {
    compute_diag_block(D, pool, sp_idx, d.data());
  }

  PivotSlab compute_pivot_slab(std::size_t sp_idx, std::size_t pivot_sp_idx) {
    const auto& sp = D.shell_pairs[sp_idx];
    const auto& pivot = D.shell_pairs[pivot_sp_idx];
    PivotSlab slab;
    slab.sp_idx = sp_idx;
    slab.pivot_sp_idx = pivot_sp_idx;
    slab.pair_begin = sp.pair_begin;
    slab.npairs = sp.npairs;
    slab.W = pivot.npairs;
    slab.rows.assign(slab.npairs * slab.W, 0.0);
    compute_pivot_block(D, pool, sp_idx, pivot_sp_idx, slab.rows.data());
    return slab;
  }

  void reserve_columns(std::size_t need) {
    if (M_current + need <= M_capacity) return;
    std::size_t new_cap =
        std::max<std::size_t>(M_capacity * 2, M_current + need + 64);
    L.conservativeResize(L.rows(), new_cap);
    M_capacity = new_cap;
  }

  // The cheap part of the per-iter block update: Q assembly is done by the
  // caller (gather_pivot) into Q. This routine does R_AB construction +
  // pivoted Cholesky + L_AB_chosen extraction, then reserves and bumps
  // M_current by Wp. Returns Wp. After this returns successfully (Wp > 0)
  // the metadata in cur_block is stable for chunk tasks to read.
  std::size_t prepare_block_update(std::size_t pivot_sp_idx,
                                   const Eigen::MatrixXd& Q,
                                   double tol) {
    const auto& pivot_sp = D.shell_pairs[pivot_sp_idx];
    const auto pair_begin = static_cast<Eigen::Index>(pivot_sp.pair_begin);
    const auto W = static_cast<Eigen::Index>(pivot_sp.npairs);
    const auto M_old = static_cast<Eigen::Index>(M_current);

    Eigen::MatrixXd R_AB = Q.block(pair_begin, 0, W, W);
    if (M_old > 0) {
      auto L_AB = L.block(pair_begin, 0, W, M_old);
      R_AB.noalias() -= L_AB * L_AB.transpose();
    }
    R_AB = 0.5 * (R_AB + R_AB.transpose());

    const double X_max_now = d.maxCoeff();
    const double local_thr = X_max_now * 1e-3;

    std::vector<Eigen::Index> perm(W);
    std::iota(perm.begin(), perm.end(), 0);
    Eigen::VectorXd diag = R_AB.diagonal();
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(W, W);

    Eigen::Index Wp = 0;
    for (Eigen::Index k = 0; k < W; ++k) {
      Eigen::Index best = k;
      double best_val = diag(perm[k]);
      for (Eigen::Index i = k + 1; i < W; ++i) {
        if (diag(perm[i]) > best_val) {
          best_val = diag(perm[i]);
          best = i;
        }
      }
      if (best != k) {
        std::swap(perm[k], perm[best]);
        if (k > 0) U.row(k).head(k).swap(U.row(best).head(k));
      }
      if (best_val <= local_thr || best_val < tol) break;
      const double sqd = std::sqrt(best_val);
      U(k, k) = sqd;
      const Eigen::Index pk = perm[k];
      for (Eigen::Index i = k + 1; i < W; ++i) {
        const Eigen::Index pi = perm[i];
        double r = R_AB(pi, pk);
        for (Eigen::Index m = 0; m < k; ++m) r -= U(i, m) * U(k, m);
        U(i, k) = r / sqd;
        diag(pi) -= U(i, k) * U(i, k);
      }
      ++Wp;
    }
    if (Wp == 0) {
      cur_block.M_old = static_cast<std::size_t>(M_old);
      cur_block.Wp = 0;
      cur_block.pivot_sp_idx = pivot_sp_idx;
      return 0;
    }

    Eigen::MatrixXd Q_chosen(static_cast<Eigen::Index>(Q.rows()), Wp);
    for (Eigen::Index j = 0; j < Wp; ++j) Q_chosen.col(j) = Q.col(perm[j]);

    Eigen::MatrixXd L_AB_chosen(Wp, M_old);
    if (M_old > 0) {
      for (Eigen::Index j = 0; j < Wp; ++j) {
        L_AB_chosen.row(j) = L.block(pair_begin + perm[j], 0, 1, M_old);
      }
    }

    Eigen::MatrixXd U_block = U.topLeftCorner(Wp, Wp);

    // Reserve and bump M_current up-front so chunk tasks can write directly
    // into L's [M_old, M_old + Wp) column range. The chunks read
    // L.middleRows(p, ...).leftCols(M_old) for the dgemm, which is unchanged
    // by reserve_columns / increment.
    reserve_columns(static_cast<std::size_t>(Wp));
    cur_block.M_old = static_cast<std::size_t>(M_old);
    cur_block.Wp = static_cast<std::size_t>(Wp);
    cur_block.pivot_sp_idx = pivot_sp_idx;
    cur_block.Q_chosen = std::move(Q_chosen);
    cur_block.L_AB_chosen = std::move(L_AB_chosen);
    cur_block.U_block = std::move(U_block);
    M_current += static_cast<std::size_t>(Wp);
    return static_cast<std::size_t>(Wp);
  }

  // Per-chunk: compute and write rows [p_lo, p_hi) of L_new, update the
  // matching segment of d. Reads cur_block (set by prepare_block_update of
  // the same iteration). Disjoint row ranges across chunks => safe with no
  // synchronization on L or d.
  void apply_chunk(std::size_t p_lo, std::size_t p_hi) {
    const auto& cb = cur_block;
    if (cb.Wp == 0) return;
    if (p_lo >= p_hi) return;
    const auto pl = static_cast<Eigen::Index>(p_lo);
    const auto ph = static_cast<Eigen::Index>(p_hi);
    const auto P_chunk = ph - pl;
    const auto Wp = static_cast<Eigen::Index>(cb.Wp);
    const auto M_old = static_cast<Eigen::Index>(cb.M_old);

    Eigen::MatrixXd L_new_chunk = cb.Q_chosen.middleRows(pl, P_chunk);
    if (M_old > 0) {
      L_new_chunk.noalias() -= L.middleRows(pl, P_chunk).leftCols(M_old)
                               * cb.L_AB_chosen.transpose();
    }
    Eigen::MatrixXd L_new_chunk_T = L_new_chunk.transpose();
    cb.U_block.triangularView<Eigen::Lower>().solveInPlace(L_new_chunk_T);
    L_new_chunk = L_new_chunk_T.transpose();

    L.block(pl, static_cast<Eigen::Index>(cb.M_old), P_chunk, Wp) = L_new_chunk;

    auto d_chunk = d.segment(pl, P_chunk);
    d_chunk.noalias() -= L_new_chunk.cwiseAbs2().rowwise().sum();
    d_chunk = d_chunk.cwiseMax(0.0);
  }
};

struct CholeskyFlow {
  CholeskyDriver& drv;
  const double tol;
  FlowProfile prof;

  ttg::Edge<std::size_t, void> e_diag_in{"e_diag_in"};
  ttg::Edge<int, int> e_diag_done{"e_diag_done"};
  ttg::Edge<int, void> e_seed_iter{"e_seed_iter"};
  ttg::Edge<int, void> e_loop_iter{"e_loop_iter"};
  ttg::Edge<ttg::MultiIndex<2>, std::size_t> e_pivot_in{"e_pivot_in"};
  ttg::Edge<int, PivotSlabs> e_pivot_collect{"e_pivot_collect"};
  ttg::Edge<ttg::MultiIndex<2>, void> e_chunk_dispatch{"e_chunk_dispatch"};
  ttg::Edge<int, int> e_chunk_done{"e_chunk_done"};
  ttg::Edge<int, void> e_terminate{"e_terminate"};

  std::unique_ptr<ttg::TTBase> t_diag_drive;
  std::unique_ptr<ttg::TTBase> t_diag_compute;
  std::unique_ptr<ttg::TTBase> t_diag_barrier;
  std::unique_ptr<ttg::TTBase> t_iter_dispatch;
  std::unique_ptr<ttg::TTBase> t_compute_slab;
  std::unique_ptr<ttg::TTBase> t_gather_pivot;
  std::unique_ptr<ttg::TTBase> t_compute_l_chunk;
  std::unique_ptr<ttg::TTBase> t_gather_finalize;
  std::unique_ptr<ttg::TTBase> t_terminate;

  CholeskyFlow(CholeskyDriver& d, double tol_) : drv(d), tol(tol_) { build(); }

  void build() {
    const std::size_t nsp = drv.D.nshell_pairs();
    const std::size_t P = drv.D.npairs();
    const std::size_t n_chunks = drv.n_chunks;

    auto diag_drive = ttg::make_tt<void>(
        [&, nsp](std::tuple<ttg::Out<std::size_t, void>>& out) {
          for (std::size_t i = 0; i < nsp; ++i) ttg::sendk<0>(i, out);
        },
        ttg::edges(), ttg::edges(e_diag_in), "drive_diag", {}, {"to_diag"});

    auto diag_compute = ttg::make_tt(
        [&](const std::size_t& sp_idx,
            std::tuple<ttg::Out<int, int>>& out) {
          drv.compute_diagonal_for_shell_pair(sp_idx);
          ttg::send<0>(0, 1, out);
        },
        ttg::edges(e_diag_in), ttg::edges(e_diag_done),
        "compute_diag", {"sp_idx"}, {"done_token"});

    auto diag_barrier = ttg::make_tt(
        [&, P, tol_local = tol](
            const int& iter_key, const int& /*count*/,
            std::tuple<ttg::Out<int, void>>& out) {
          double X_max = drv.d.maxCoeff();
          const double prescreen_thr =
              (tol_local * tol_local) / std::max(X_max, 1e-300);
          drv.d = drv.d.cwiseMax(0.0);
          for (Eigen::Index p = 0; p < drv.d.size(); ++p)
            if (drv.d(p) < prescreen_thr) drv.d(p) = 0.0;
          if (drv.verbose) {
            std::cout << "[eri-pch] N = " << drv.D.nbf
                      << ", #shells = " << drv.D.nshell
                      << ", #shell pairs = " << drv.D.nshell_pairs()
                      << ", #AO pairs P = " << P
                      << ", X_max(diag) = " << X_max
                      << ", chunks = " << drv.n_chunks << std::endl;
          }
          ttg::sendk<0>(iter_key, out);
        },
        ttg::edges(e_diag_done), ttg::edges(e_seed_iter),
        "diag_barrier", {"tokens"}, {"to_iter"});

    auto iter_dispatch = ttg::make_tt(
        [&, nsp, tol_local = tol](
            const int& iter,
            std::tuple<ttg::Out<ttg::MultiIndex<2>, std::size_t>,
                       ttg::Out<int, void>>& out) {
          const long t0 = now_ns();
          Eigen::Index pivot_p_idx = 0;
          double pivot_val = drv.d.maxCoeff(&pivot_p_idx);
          const auto pivot_p = static_cast<std::size_t>(pivot_p_idx);
          if (pivot_val < tol_local) {
            ttg::sendk<1>(iter, out);
            prof.ns_iter_dispatch.fetch_add(now_ns() - t0,
                                            std::memory_order_relaxed);
            return;
          }
          const std::size_t pivot_sp_idx = drv.D.pairs[pivot_p].shell_pair_idx;
          ++drv.total_shellpair_recomputes;
          for (std::size_t sp = 0; sp < nsp; ++sp) {
            ttg::send<0>(ttg::MultiIndex<2>{static_cast<unsigned long>(iter), sp},
                         pivot_sp_idx, out);
          }
          prof.ns_iter_dispatch.fetch_add(now_ns() - t0,
                                          std::memory_order_relaxed);
        },
        ttg::edges(ttg::fuse(e_seed_iter, e_loop_iter)),
        ttg::edges(e_pivot_in, e_terminate),
        "iter_dispatch", {"iter"}, {"to_compute_slab", "to_terminate"});

    auto compute_slab = ttg::make_tt(
        [&](const ttg::MultiIndex<2>& key,
            const std::size_t& pivot_sp_idx,
            std::tuple<ttg::Out<int, PivotSlabs>>& out) {
          const long t0 = now_ns();
          const auto iter = static_cast<int>(key[0]);
          const auto sp_idx = static_cast<std::size_t>(key[1]);
          PivotSlabs one;
          one.push_back(drv.compute_pivot_slab(sp_idx, pivot_sp_idx));
          ttg::send<0>(iter, std::move(one), out);
          prof.ns_compute_slab.fetch_add(now_ns() - t0,
                                         std::memory_order_relaxed);
          prof.n_compute_slab_calls.fetch_add(1, std::memory_order_relaxed);
          {
            std::lock_guard<std::mutex> lk(prof.tids_mu);
            prof.tids_compute_slab.insert(std::this_thread::get_id());
          }
        },
        ttg::edges(e_pivot_in), ttg::edges(e_pivot_collect),
        "compute_slab", {"key=(iter,sp)/pivot_sp"}, {"slab"});

    // gather_pivot: assemble Q, run prepare_block_update (the cheap part),
    // then either fan out chunks or skip directly to iter+1 if Wp = 0.
    auto gather_pivot = ttg::make_tt(
        [&, P, n_chunks, tol_local = tol](
            const int& iter, PivotSlabs&& slabs,
            std::tuple<ttg::Out<ttg::MultiIndex<2>, void>,
                       ttg::Out<int, void>>& out) {
          prof.n_gather_calls.fetch_add(1, std::memory_order_relaxed);
          const long t_setup_0 = now_ns();
          if (slabs.empty()) {
            ttg::sendk<1>(iter + 1, out);
            return;
          }
          const std::size_t W = slabs.front().W;
          const std::size_t pivot_sp = slabs.front().pivot_sp_idx;
          Eigen::MatrixXd Q(static_cast<Eigen::Index>(P),
                            static_cast<Eigen::Index>(W));
          Q.setZero();
          for (auto& s : slabs) {
            for (std::size_t i = 0; i < s.npairs; ++i) {
              const std::size_t row = s.pair_begin + i;
              for (std::size_t j = 0; j < W; ++j) {
                Q(static_cast<Eigen::Index>(row),
                  static_cast<Eigen::Index>(j)) = s.rows[i * W + j];
              }
            }
          }
          const std::size_t Wp = drv.prepare_block_update(pivot_sp, Q, tol_local);
          prof.ns_gather_setup.fetch_add(now_ns() - t_setup_0,
                                         std::memory_order_relaxed);
          prof.n_new_columns.fetch_add(static_cast<int>(Wp),
                                       std::memory_order_relaxed);

          if (Wp == 0) {
            ttg::sendk<1>(iter + 1, out);
            return;
          }
          if (drv.trace_each_vector) {
            std::cout << "  iter " << std::setw(5) << iter
                      << "  rank=" << std::setw(5) << drv.M_current
                      << "  +" << std::setw(2) << Wp
                      << " from sp=" << pivot_sp << std::endl;
          }
          // Fan out n_chunks chunk tasks. Static argstream size on the
          // gather_finalize input matches.
          for (std::size_t c = 0; c < n_chunks; ++c) {
            ttg::sendk<0>(
                ttg::MultiIndex<2>{static_cast<unsigned long>(iter), c}, out);
          }
        },
        ttg::edges(e_pivot_collect),
        ttg::edges(e_chunk_dispatch, e_loop_iter),
        "gather_pivot", {"slabs"}, {"to_chunks", "to_iter"});

    // compute_l_chunk: each chunk handles a row range [p_lo, p_hi).
    auto compute_l_chunk = ttg::make_tt(
        [&, P, n_chunks](
            const ttg::MultiIndex<2>& key,
            std::tuple<ttg::Out<int, int>>& out) {
          const long t0 = now_ns();
          const auto iter = static_cast<int>(key[0]);
          const auto c_idx = static_cast<std::size_t>(key[1]);
          const std::size_t rows_per = (P + n_chunks - 1) / n_chunks;
          const std::size_t p_lo = c_idx * rows_per;
          const std::size_t p_hi = std::min(p_lo + rows_per, P);
          drv.apply_chunk(p_lo, p_hi);
          ttg::send<0>(iter, 1, out);
          prof.ns_chunk_compute.fetch_add(now_ns() - t0,
                                          std::memory_order_relaxed);
          prof.n_chunk_calls.fetch_add(1, std::memory_order_relaxed);
          {
            std::lock_guard<std::mutex> lk(prof.tids_mu);
            prof.tids_chunk.insert(std::this_thread::get_id());
          }
        },
        ttg::edges(e_chunk_dispatch), ttg::edges(e_chunk_done),
        "compute_l_chunk", {"key=(iter,c_idx)"}, {"done_token"});

    // gather_finalize: streaming barrier on n_chunks tokens per iter.
    // Fires iter+1 onto e_loop_iter.
    auto gather_finalize = ttg::make_tt(
        [&](const int& iter, const int& /*count*/,
            std::tuple<ttg::Out<int, void>>& out) {
          ttg::sendk<0>(iter + 1, out);
        },
        ttg::edges(e_chunk_done), ttg::edges(e_loop_iter),
        "gather_finalize", {"chunk_tokens"}, {"to_iter"});

    // streaming reducers
    diag_barrier->template set_input_reducer<0>(
        [](int& acc, const int& in) { acc += in; });
    diag_barrier->template set_static_argstream_size<0>(nsp);

    gather_pivot->template set_input_reducer<0>(
        [](PivotSlabs& acc, const PivotSlabs& in) {
          for (const auto& s : in) acc.push_back(s);
        });
    gather_pivot->template set_static_argstream_size<0>(nsp);

    gather_finalize->template set_input_reducer<0>(
        [](int& acc, const int& in) { acc += in; });
    gather_finalize->template set_static_argstream_size<0>(n_chunks);

    auto terminate = ttg::make_tt(
        [&](const int& iter) {
          if (drv.verbose) {
            std::cout << "[eri-pch] converged at iter " << iter
                      << ": rank = " << drv.M_current
                      << ", shell-pair recomputes = "
                      << drv.total_shellpair_recomputes
                      << " (out of " << drv.D.nshell_pairs() << " possible)"
                      << std::endl;
          }
        },
        ttg::edges(e_terminate), ttg::edges(),
        "terminate", {"iter"}, {});

    t_diag_drive = std::move(diag_drive);
    t_diag_compute = std::move(diag_compute);
    t_diag_barrier = std::move(diag_barrier);
    t_iter_dispatch = std::move(iter_dispatch);
    t_compute_slab = std::move(compute_slab);
    t_gather_pivot = std::move(gather_pivot);
    t_compute_l_chunk = std::move(compute_l_chunk);
    t_gather_finalize = std::move(gather_finalize);
    t_terminate = std::move(terminate);

    ttg::make_graph_executable(t_diag_drive.get());
    ttg::execute();
  }

  void run() {
    if (ttg::default_execution_context().rank() == 0) t_diag_drive->invoke();
    ttg::fence();
  }
};

Eigen::MatrixXd run_cholesky(CholeskyDriver& drv, double tol,
                             bool dump_dot_on_root) {
  CholeskyFlow flow(drv, tol);

  if (dump_dot_on_root && ttg::default_execution_context().rank() == 0) {
    ttg::Dot dot;
    auto graph_str = dot(flow.t_diag_drive.get());
    std::ofstream out("eri-pch_v4.dot");
    out << graph_str;
    std::cout << "[eri-pch] wrote TTG graph to eri-pch_v4.dot" << std::endl;
  }

  flow.run();

  if (drv.verbose) {
    const auto ms = [](long ns) { return ns / 1.0e6; };
    std::cout << "[eri-pch] profile (cumulative across worker threads):\n"
              << "  iter_dispatch    : " << std::fixed << std::setprecision(1)
              << ms(flow.prof.ns_iter_dispatch.load()) << " ms\n"
              << "  compute_slab     : "
              << ms(flow.prof.ns_compute_slab.load()) << " ms over "
              << flow.prof.n_compute_slab_calls.load() << " calls on "
              << flow.prof.tids_compute_slab.size() << " threads\n"
              << "  gather setup     : "
              << ms(flow.prof.ns_gather_setup.load()) << " ms over "
              << flow.prof.n_gather_calls.load() << " calls\n"
              << "  chunk compute    : "
              << ms(flow.prof.ns_chunk_compute.load()) << " ms over "
              << flow.prof.n_chunk_calls.load() << " calls on "
              << flow.prof.tids_chunk.size() << " threads ("
              << flow.prof.n_new_columns.load() << " columns)"
              << std::defaultfloat << std::endl;
  }

  return drv.L.leftCols(static_cast<Eigen::Index>(drv.M_current));
}

}  // namespace eri_pch

int main(int argc, char* argv[]) {
  ttg::initialize(argc, argv);
  libint2::initialize();

  auto args = eri_pch::parse_cli(argc, argv);
  std::vector<libint2::Atom> atoms = args.xyz_path.empty()
                                         ? eri_pch::default_water()
                                         : eri_pch::read_xyz(args.xyz_path);

  eri_pch::IntegralData D;
  D.build(std::move(atoms), args.basis_name);

  const bool root = (ttg::default_execution_context().rank() == 0);

  eri_pch::CholeskyDriver drv(std::move(D));
  drv.verbose = root;
  if (const char* trace = std::getenv("TTG_ERI_PCH_TRACE"); trace && *trace == '1') {
    drv.trace_each_vector = root;
  }

  if (root) eri_pch::print_run_header(args);

  auto t0 = std::chrono::steady_clock::now();
  Eigen::MatrixXd Lmat = eri_pch::run_cholesky(drv, args.tol, root);
  auto t1 = std::chrono::steady_clock::now();

  if (root) {
    std::cout << "[eri-pch] elapsed: "
              << std::chrono::duration<double>(t1 - t0).count() << " s"
              << std::endl;
    eri_pch::EnginePool refpool(drv.D.basis.max_nprim(), drv.D.basis.max_l());
    eri_pch::validate_decomposition(drv.D, refpool, Lmat, args.tol);
  }

  libint2::finalize();
  ttg::finalize();
  return 0;
}
