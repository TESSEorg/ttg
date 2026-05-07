// SPDX-License-Identifier: BSD-3-Clause
//
// eri-pch v2 -- single TTG dataflow graph with feedback edge for outer
// iterations. See v2.md for the full description; the libint/
// AO-pair/CLI boilerplate lives in common.{h,cc}.

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
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace eri_pch {

// Per-phase wall-time and thread-fanout counters, accumulated atomically
// across worker threads. Reported once at the end of run_cholesky().
struct FlowProfile {
  std::atomic<long> ns_compute_diag{0};
  std::atomic<long> ns_compute_slab{0};
  std::atomic<long> ns_gather_assemble{0};
  std::atomic<long> ns_gather_subdecomp{0};
  std::atomic<long> ns_iter_dispatch{0};
  std::atomic<int> n_compute_slab_calls{0};
  std::atomic<int> n_gather_calls{0};
  std::atomic<int> n_subpivots{0};
  std::mutex tids_mu;
  std::set<std::thread::id> tids_compute_slab;
};
inline long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// One shell pair's contribution to Q for the iteration's pivot shell pair.
// This is the unit of dataflow between compute_slab and gather_pivot.
struct PivotSlab {
  std::size_t sp_idx;        // shell pair this slab corresponds to (rows)
  std::size_t pivot_sp_idx;  // the iteration's pivot shell pair (columns)
  std::size_t pair_begin;
  std::size_t npairs;
  std::size_t W;
  std::vector<double> rows;  // size = npairs * W, row-major

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

// Algorithm state. Only gather_pivot writes to d[]/L[]/recompute counter,
// and only iter_dispatch reads them in between -- both serialized by the
// streaming-reducer barrier and the feedback edge.
class CholeskyDriver {
 public:
  IntegralData D;
  EnginePool pool;
  std::vector<double> d;
  std::vector<std::vector<double>> L;
  std::size_t total_shellpair_recomputes = 0;

  bool verbose = false;
  bool trace_each_vector = false;

  CholeskyDriver(IntegralData data)
      : D(std::move(data)),
        pool(D.basis.max_nprim(), D.basis.max_l()),
        d(D.npairs(), 0.0) {}

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

  // Apply one Cholesky update with sub-pivot at pair index pivot_p (which
  // lies in pivot shell pair pivot_sp_idx, so J_local = pivot_p - pair_begin).
  void apply_cholesky_update(std::size_t pivot_p, std::size_t pivot_sp_idx,
                             const std::vector<double>& Q, std::size_t W) {
    const auto& pivot_sp = D.shell_pairs[pivot_sp_idx];
    const std::size_t J_local = pivot_p - pivot_sp.pair_begin;
    const double diag = d[pivot_p];
    if (diag <= 0.0) {
      d[pivot_p] = 0.0;
      return;
    }
    const double inv_sqrt_diag = 1.0 / std::sqrt(diag);
    const std::size_t P = D.npairs();
    std::vector<double> Lnew(P, 0.0);
    const std::size_t nL = L.size();
    std::vector<double> Lk_at_J(nL);
    for (std::size_t k = 0; k < nL; ++k) Lk_at_J[k] = L[k][pivot_p];
    for (std::size_t p = 0; p < P; ++p) {
      double m = Q[p * W + J_local];
      for (std::size_t k = 0; k < nL; ++k) m -= L[k][p] * Lk_at_J[k];
      const double lp = m * inv_sqrt_diag;
      Lnew[p] = lp;
      d[p] -= lp * lp;
      if (d[p] < 0.0) d[p] = 0.0;
    }
    L.push_back(std::move(Lnew));
  }
};

// Single connected dataflow graph. drive_diag -> compute_diag -> diag_barrier
// (streaming) seeds iter_dispatch; iter_dispatch fans out compute_slab tasks
// whose PivotSlab values stream into gather_pivot; gather_pivot runs the
// sub-decomp and emits iter+1 back into iter_dispatch via fuse(). When
// X_max < tol, iter_dispatch's second output terminal goes to terminate.
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
          double X_max = 0.0;
          for (std::size_t p = 0; p < P; ++p)
            X_max = std::max(X_max, drv.d[p]);
          const double prescreen_thr =
              (tol_local * tol_local) / std::max(X_max, 1e-300);
          for (std::size_t p = 0; p < P; ++p)
            if (drv.d[p] < prescreen_thr) drv.d[p] = 0.0;
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
        [&, nsp, P, tol_local = tol](
            const int& iter,
            std::tuple<ttg::Out<ttg::MultiIndex<2>, std::size_t>,
                       ttg::Out<int, void>>& out) {
          const long t0 = now_ns();
          std::size_t pivot_p = 0;
          double pivot_val = -1.0;
          for (std::size_t p = 0; p < P; ++p) {
            if (drv.d[p] > pivot_val) {
              pivot_val = drv.d[p];
              pivot_p = p;
            }
          }
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
          std::vector<double> Q(P * W, 0.0);
          for (auto& s : slabs) {
            const auto base = s.pair_begin * W;
            std::memcpy(Q.data() + base, s.rows.data(),
                        s.npairs * W * sizeof(double));
          }
          prof.ns_gather_assemble.fetch_add(now_ns() - t_assemble_0,
                                            std::memory_order_relaxed);
          const long t_subdecomp_0 = now_ns();
          const auto& pvsp = drv.D.shell_pairs[pivot_sp];
          while (true) {
            std::size_t sub = pvsp.pair_begin;
            double sv = drv.d[sub];
            for (std::size_t p = pvsp.pair_begin + 1;
                 p < pvsp.pair_begin + pvsp.npairs; ++p) {
              if (drv.d[p] > sv) { sv = drv.d[p]; sub = p; }
            }
            double X_max_now = 0.0;
            for (std::size_t p = 0; p < P; ++p)
              X_max_now = std::max(X_max_now, drv.d[p]);
            if (sv <= X_max_now * 1e-3 || sv < tol_local) break;
            drv.apply_cholesky_update(sub, pivot_sp, Q, W);
            prof.n_subpivots.fetch_add(1, std::memory_order_relaxed);
            if (drv.trace_each_vector) {
              std::cout << "  iter " << std::setw(5) << iter
                        << "  rank=" << std::setw(5) << drv.L.size()
                        << "  pivot p=" << std::setw(6) << sub
                        << "  d[pivot]=" << std::scientific
                        << std::setprecision(4) << sv
                        << "  X_max_now=" << X_max_now << std::defaultfloat
                        << std::endl;
            }
          }
          prof.ns_gather_subdecomp.fetch_add(now_ns() - t_subdecomp_0,
                                             std::memory_order_relaxed);
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
                      << ": rank = " << drv.L.size()
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
    std::ofstream out("eri-pch_v2.dot");
    out << graph_str;
    std::cout << "[eri-pch] wrote TTG graph to eri-pch_v2.dot" << std::endl;
  }

  flow.run();

  if (drv.verbose) {
    const auto ms = [](long ns) { return ns / 1.0e6; };
    std::cout << "[eri-pch] profile (cumulative across worker threads):\n"
              << "  iter_dispatch   : "
              << std::fixed << std::setprecision(1)
              << ms(flow.prof.ns_iter_dispatch.load()) << " ms\n"
              << "  compute_slab    : "
              << ms(flow.prof.ns_compute_slab.load()) << " ms over "
              << flow.prof.n_compute_slab_calls.load() << " calls on "
              << flow.prof.tids_compute_slab.size() << " threads\n"
              << "  gather assemble : "
              << ms(flow.prof.ns_gather_assemble.load()) << " ms over "
              << flow.prof.n_gather_calls.load() << " calls\n"
              << "  gather subdecomp: "
              << ms(flow.prof.ns_gather_subdecomp.load()) << " ms over "
              << flow.prof.n_subpivots.load() << " sub-pivots"
              << std::defaultfloat << std::endl;
  }

  const std::size_t P = drv.D.npairs();
  const std::size_t M = drv.L.size();
  Eigen::MatrixXd Lmat(P, M);
  for (std::size_t k = 0; k < M; ++k) {
    for (std::size_t p = 0; p < P; ++p) Lmat(p, k) = drv.L[k][p];
  }
  return Lmat;
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
