// SPDX-License-Identifier: BSD-3-Clause
//
// eri-pch v3 -- block-Cholesky-per-shell-pair.
//
// Same TTG dataflow topology as v2 (single connected graph with a feedback
// edge, streaming reducers for diag init and the per-iter gather), but the
// inner sub-decomposition in gather_pivot is replaced by a single block
// update that produces all W' new Cholesky vectors of the iteration's
// pivot shell pair (A,B) in one BLAS-3 sweep. The L matrix is stored as
// a contiguous Eigen::MatrixXd (P x M_current) so dgemm/dtrsm can be
// dispatched directly via Eigen's matrix operations.
//
// Algorithm reference: H. Koch, A. Sanchez de Meras, T. B. Pedersen,
//   J. Chem. Phys. 118, 9481 (2003).
//
// See v3.md for the design discussion and a comparison with v2.
//
// Usage: eri-pch_v3-{mad,parsec} [<molecule.xyz> [<basis-name> [<tolerance>]]]
//
// Side effects: writes the TTG flow graph to `eri-pch_v3.dot`.

#include "common.h"

#include <ttg.h>
#include <ttg/util/dot.h>
#include <ttg/util/multiindex.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

// One shell pair's contribution to Q for the iteration's pivot shell pair.
// Identical to v2 -- this is the unit of dataflow on e_pivot_collect.
struct PivotSlab {
  std::size_t sp_idx;
  std::size_t pivot_sp_idx;
  std::size_t pair_begin;
  std::size_t npairs;
  std::size_t W;
  std::vector<double> rows;  // npairs * W, row-major

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

// Per-phase wall-time and thread-fanout counters. Same as v2; useful to
// confirm the block update actually shrinks the serial fraction.
struct FlowProfile {
  std::atomic<long> ns_compute_diag{0};
  std::atomic<long> ns_compute_slab{0};
  std::atomic<long> ns_gather_assemble{0};
  std::atomic<long> ns_gather_block{0};  // renamed: block update
  std::atomic<long> ns_iter_dispatch{0};
  std::atomic<int> n_compute_slab_calls{0};
  std::atomic<int> n_gather_calls{0};
  std::atomic<int> n_block_updates{0};
  std::atomic<int> n_new_columns{0};
  std::mutex tids_mu;
  std::set<std::thread::id> tids_compute_slab;
};
inline long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Driver state. L is stored as a contiguous P x M_capacity Eigen::MatrixXd
// (column-major); only the first M_current columns are valid. d[] is an
// Eigen vector for cheap cwise updates.
class CholeskyDriver {
 public:
  IntegralData D;
  EnginePool pool;
  Eigen::VectorXd d;             // running diagonals
  Eigen::MatrixXd L;             // P x M_capacity; only first M_current cols valid
  std::size_t M_current = 0;
  std::size_t M_capacity = 0;
  std::size_t total_shellpair_recomputes = 0;

  bool verbose = false;
  bool trace_each_vector = false;

  CholeskyDriver(IntegralData data)
      : D(std::move(data)),
        pool(D.basis.max_nprim(), D.basis.max_l()) {
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

  // Grow L's column capacity (preserving existing data).
  void reserve_columns(std::size_t need) {
    if (M_current + need <= M_capacity) return;
    std::size_t new_cap =
        std::max<std::size_t>(M_capacity * 2, M_current + need + 64);
    L.conservativeResize(L.rows(), new_cap);
    M_capacity = new_cap;
  }

  // Block update: do *one* pivoted Cholesky on the W x W residual block of
  // the iteration's pivot shell pair, generating all W' new Cholesky vectors
  // (W' <= W) at once via BLAS-3 dgemm + dtrsm.
  //
  // Returns the number of new columns appended.
  //
  // Q is the full P x W row-block already assembled by gather_pivot.
  std::size_t apply_block_cholesky_update(std::size_t pivot_sp_idx,
                                          const Eigen::MatrixXd& Q,
                                          double tol) {
    const auto& pivot_sp = D.shell_pairs[pivot_sp_idx];
    const auto pair_begin = static_cast<Eigen::Index>(pivot_sp.pair_begin);
    const auto W = static_cast<Eigen::Index>(pivot_sp.npairs);
    const auto P = static_cast<Eigen::Index>(D.npairs());
    const auto M_old = static_cast<Eigen::Index>(M_current);

    // Step 1: residual W x W block
    //   R_AB = Q[(A,B), (A,B)] - L[(A,B), :M_old] * L[(A,B), :M_old]^T
    Eigen::MatrixXd R_AB = Q.block(pair_begin, 0, W, W);
    if (M_old > 0) {
      auto L_AB = L.block(pair_begin, 0, W, M_old);
      R_AB.noalias() -= L_AB * L_AB.transpose();
    }
    // Force symmetry to suppress accumulated round-off.
    R_AB = 0.5 * (R_AB + R_AB.transpose());

    // Step 2: pivoted Cholesky on R_AB (small, W x W). We accept sub-pivots
    // while their residual diagonal is above max(local_thr, tol), where
    // local_thr is X_max_now * 1e-3 per Koch et al.
    const double X_max_now = d.maxCoeff();
    const double local_thr = X_max_now * 1e-3;

    std::vector<Eigen::Index> perm(W);
    std::iota(perm.begin(), perm.end(), 0);
    Eigen::VectorXd diag = R_AB.diagonal();   // residual diagonals (W)
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(W, W);  // store pivoted-Cholesky factor

    Eigen::Index Wp = 0;
    for (Eigen::Index k = 0; k < W; ++k) {
      // Pick the largest residual diagonal among remaining slots.
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
        // Swap the partially-filled U rows so U[i, m] for m < k continues
        // to track the row currently sitting at position i in perm.
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

    if (Wp == 0) return 0;

    // Step 3-7: form the P x Wp residual (Q_chosen - L * L_AB[Π,:]^T) and
    // solve · U^{-T} to get L_new.
    //
    //   L_new = ( Q[:, perm[0..Wp-1]]
    //             - L[:, :M_old] * L[(A,B), :M_old]^T[perm[0..Wp-1], :] )
    //           * U_block^{-T}
    //
    // where U_block = U.topLeftCorner(Wp, Wp) is W' x W' lower triangular.

    Eigen::MatrixXd Q_chosen(P, Wp);
    for (Eigen::Index j = 0; j < Wp; ++j) Q_chosen.col(j) = Q.col(perm[j]);

    Eigen::MatrixXd L_new = std::move(Q_chosen);
    if (M_old > 0) {
      Eigen::MatrixXd L_AB_chosen(Wp, M_old);
      for (Eigen::Index j = 0; j < Wp; ++j) {
        L_AB_chosen.row(j) = L.block(pair_begin + perm[j], 0, 1, M_old);
      }
      // BLAS-3: L_new -= L[:, :M_old] * L_AB_chosen^T   (P x Wp)
      L_new.noalias() -= L.leftCols(M_old) * L_AB_chosen.transpose();
    }

    // Now L_new currently holds the residual P x Wp matrix. Solve
    // L_new * U_block^T = residual ⇔ U_block * L_new^T = residual^T.
    Eigen::MatrixXd U_block = U.topLeftCorner(Wp, Wp);
    Eigen::MatrixXd L_new_T = L_new.transpose();
    U_block.triangularView<Eigen::Lower>().solveInPlace(L_new_T);
    L_new = L_new_T.transpose();

    // Append L_new columns to L.
    reserve_columns(static_cast<std::size_t>(Wp));
    L.middleCols(M_current, Wp) = L_new;

    // Diagonal update: d -= rowwise sum of L_new^2.
    d.noalias() -= L_new.cwiseAbs2().rowwise().sum();
    d = d.cwiseMax(0.0);

    M_current += static_cast<std::size_t>(Wp);
    return static_cast<std::size_t>(Wp);
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
  ttg::Edge<int, void> e_terminate{"e_terminate"};

  std::unique_ptr<ttg::TTBase> t_diag_drive;
  std::unique_ptr<ttg::TTBase> t_diag_compute;
  std::unique_ptr<ttg::TTBase> t_diag_barrier;
  std::unique_ptr<ttg::TTBase> t_iter_dispatch;
  std::unique_ptr<ttg::TTBase> t_compute_slab;
  std::unique_ptr<ttg::TTBase> t_gather_pivot;
  std::unique_ptr<ttg::TTBase> t_terminate;

  CholeskyFlow(CholeskyDriver& d, double tol_) : drv(d), tol(tol_) { build(); }

  void build() {
    const std::size_t nsp = drv.D.nshell_pairs();
    const std::size_t P = drv.D.npairs();

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
                      << ", X_max(diag) = " << X_max << std::endl;
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

    auto gather_pivot = ttg::make_tt(
        [&, P, tol_local = tol](
            const int& iter, PivotSlabs&& slabs,
            std::tuple<ttg::Out<int, void>>& out) {
          prof.n_gather_calls.fetch_add(1, std::memory_order_relaxed);
          if (slabs.empty()) {
            ttg::sendk<0>(iter + 1, out);
            return;
          }
          const long t_assemble_0 = now_ns();
          const std::size_t W = slabs.front().W;
          const std::size_t pivot_sp = slabs.front().pivot_sp_idx;
          // Assemble Q as a P x W Eigen matrix (column-major).
          Eigen::MatrixXd Q(static_cast<Eigen::Index>(P),
                            static_cast<Eigen::Index>(W));
          Q.setZero();
          for (auto& s : slabs) {
            // s.rows is row-major npairs x W; copy into Q's rows
            // [pair_begin, pair_begin + npairs).
            for (std::size_t i = 0; i < s.npairs; ++i) {
              const std::size_t row = s.pair_begin + i;
              for (std::size_t j = 0; j < W; ++j) {
                Q(static_cast<Eigen::Index>(row),
                  static_cast<Eigen::Index>(j)) = s.rows[i * W + j];
              }
            }
          }
          prof.ns_gather_assemble.fetch_add(now_ns() - t_assemble_0,
                                            std::memory_order_relaxed);

          const long t_block_0 = now_ns();
          const std::size_t Wp =
              drv.apply_block_cholesky_update(pivot_sp, Q, tol_local);
          prof.ns_gather_block.fetch_add(now_ns() - t_block_0,
                                         std::memory_order_relaxed);
          prof.n_block_updates.fetch_add(1, std::memory_order_relaxed);
          prof.n_new_columns.fetch_add(static_cast<int>(Wp),
                                       std::memory_order_relaxed);
          if (drv.trace_each_vector) {
            std::cout << "  iter " << std::setw(5) << iter
                      << "  rank=" << std::setw(5) << drv.M_current
                      << "  +" << std::setw(2) << Wp << " from sp=" << pivot_sp
                      << std::endl;
          }
          ttg::sendk<0>(iter + 1, out);
        },
        ttg::edges(e_pivot_collect), ttg::edges(e_loop_iter),
        "gather_pivot", {"slabs"}, {"to_iter"});

    diag_barrier->template set_input_reducer<0>(
        [](int& acc, const int& in) { acc += in; });
    diag_barrier->template set_static_argstream_size<0>(nsp);

    gather_pivot->template set_input_reducer<0>(
        [](PivotSlabs& acc, const PivotSlabs& in) {
          for (const auto& s : in) acc.push_back(s);
        });
    gather_pivot->template set_static_argstream_size<0>(nsp);

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
    std::ofstream out("eri-pch_v3.dot");
    out << graph_str;
    std::cout << "[eri-pch] wrote TTG graph to eri-pch_v3.dot" << std::endl;
  }

  flow.run();

  if (drv.verbose) {
    const auto ms = [](long ns) { return ns / 1.0e6; };
    std::cout << "[eri-pch] profile (cumulative across worker threads):\n"
              << "  iter_dispatch   : " << std::fixed << std::setprecision(1)
              << ms(flow.prof.ns_iter_dispatch.load()) << " ms\n"
              << "  compute_slab    : "
              << ms(flow.prof.ns_compute_slab.load()) << " ms over "
              << flow.prof.n_compute_slab_calls.load() << " calls on "
              << flow.prof.tids_compute_slab.size() << " threads\n"
              << "  gather assemble : "
              << ms(flow.prof.ns_gather_assemble.load()) << " ms over "
              << flow.prof.n_gather_calls.load() << " calls\n"
              << "  gather block    : "
              << ms(flow.prof.ns_gather_block.load()) << " ms over "
              << flow.prof.n_block_updates.load() << " block updates ("
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
