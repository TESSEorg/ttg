// SPDX-License-Identifier: BSD-3-Clause
//
// eri-pch v1 -- bulk-synchronous TTG dispatch with C++ outer loop.
// See v1.md for the full description; the libint/AO-pair/CLI
// boilerplate lives in common.{h,cc}.

#include "common.h"

#include <ttg.h>
#include <ttg/util/dot.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

namespace eri_pch {

// State for the v1 algorithm. The outer C++ loop in run_cholesky() is the
// only writer of cur_pivot_sp / Q / L. TTG tasks dispatched per phase
// write disjoint slices of d[] (during diag init) or disjoint row-strips
// of Q (during the per-iter pivot-block compute).
class CholeskyDriver {
 public:
  IntegralData D;
  EnginePool pool;
  std::vector<double> d;                 // diagonals, size = D.npairs()
  std::vector<std::vector<double>> L;    // finished Cholesky vectors
  std::vector<double> Q;                 // [P x cur_pivot_npairs] column block
  std::size_t cur_pivot_sp = IntegralData::npos;
  std::size_t cur_pivot_npairs = 0;

  bool verbose = false;
  bool trace_each_vector = false;

  CholeskyDriver(IntegralData data)
      : D(std::move(data)),
        pool(D.basis.max_nprim(), D.basis.max_l()),
        d(D.npairs(), 0.0) {}

  void compute_diagonal_for_shell_pair(std::size_t sp_idx) {
    compute_diag_block(D, pool, sp_idx, d.data());
  }

  void compute_pivot_block_for_shell_pair(std::size_t sp_idx) {
    const auto& sp = D.shell_pairs[sp_idx];
    const std::size_t W = cur_pivot_npairs;
    compute_pivot_block(D, pool, sp_idx, cur_pivot_sp,
                        Q.data() + sp.pair_begin * W);
  }

  // Apply one Cholesky update with sub-pivot at pair index pivot_p (which
  // must lie in cur_pivot_sp). Computes
  //   L^new[p] = (Q[p, J_local] - sum_K L^K[p]*L^K[J]) / sqrt(d[J])
  // for all p, then d[p] -= L^new[p]^2.
  void apply_cholesky_update(std::size_t pivot_p) {
    const auto& pivot_sp = D.shell_pairs[cur_pivot_sp];
    const std::size_t W = cur_pivot_npairs;
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

// Two disconnected fan-out sub-graphs, each invoked once per phase from the
// outer C++ driver loop in run_cholesky().
struct CholeskyFlow {
  CholeskyDriver& drv;

  ttg::Edge<std::size_t, void> e_diag{"e_diag"};
  ttg::Edge<std::size_t, void> e_pivot{"e_pivot"};

  std::unique_ptr<ttg::TTBase> t_diag_drive;
  std::unique_ptr<ttg::TTBase> t_diag_compute;
  std::unique_ptr<ttg::TTBase> t_pivot_drive;
  std::unique_ptr<ttg::TTBase> t_pivot_compute;

  CholeskyFlow(CholeskyDriver& d) : drv(d) { build(); }

  void build() {
    auto diag_drive = ttg::make_tt<void>(
        [&](std::tuple<ttg::Out<std::size_t, void>>& out) {
          for (std::size_t i = 0; i < drv.D.nshell_pairs(); ++i) {
            ttg::sendk<0>(i, out);
          }
        },
        ttg::edges(), ttg::edges(e_diag), "drive_diag", {}, {"to_diag"});

    auto diag_compute = ttg::make_tt(
        [&](const std::size_t& sp_idx) {
          drv.compute_diagonal_for_shell_pair(sp_idx);
        },
        ttg::edges(e_diag), ttg::edges(), "compute_diag", {"sp_idx"}, {});

    auto pivot_drive = ttg::make_tt<void>(
        [&](std::tuple<ttg::Out<std::size_t, void>>& out) {
          for (std::size_t i = 0; i < drv.D.nshell_pairs(); ++i) {
            ttg::sendk<0>(i, out);
          }
        },
        ttg::edges(), ttg::edges(e_pivot), "drive_pivot", {}, {"to_pivot"});

    auto pivot_compute = ttg::make_tt(
        [&](const std::size_t& sp_idx) {
          drv.compute_pivot_block_for_shell_pair(sp_idx);
        },
        ttg::edges(e_pivot), ttg::edges(), "compute_pivot_block", {"sp_idx"}, {});

    t_diag_drive = std::move(diag_drive);
    t_diag_compute = std::move(diag_compute);
    t_pivot_drive = std::move(pivot_drive);
    t_pivot_compute = std::move(pivot_compute);

    ttg::make_graph_executable(t_diag_drive.get());
    ttg::make_graph_executable(t_pivot_drive.get());
    ttg::execute();
  }

  void run_diagonals() {
    if (ttg::default_execution_context().rank() == 0) t_diag_drive->invoke();
    ttg::fence();
  }

  void run_pivot_block() {
    if (ttg::default_execution_context().rank() == 0) t_pivot_drive->invoke();
    ttg::fence();
  }
};

Eigen::MatrixXd run_cholesky(CholeskyDriver& drv, double tol,
                             bool dump_dot_on_root) {
  CholeskyFlow flow(drv);

  if (dump_dot_on_root && ttg::default_execution_context().rank() == 0) {
    ttg::Dot dot;
    auto graph_str = dot(flow.t_diag_drive.get(), flow.t_pivot_drive.get());
    std::ofstream out("eri-pch_v1.dot");
    out << graph_str;
    std::cout << "[eri-pch] wrote TTG graph to eri-pch_v1.dot" << std::endl;
  }

  const std::size_t P = drv.D.npairs();

  flow.run_diagonals();

  double X_max = 0.0;
  for (std::size_t p = 0; p < P; ++p) X_max = std::max(X_max, drv.d[p]);

  if (drv.verbose) {
    std::cout << "[eri-pch] N = " << drv.D.nbf
              << ", #shells = " << drv.D.nshell
              << ", #shell pairs = " << drv.D.nshell_pairs()
              << ", #AO pairs P = " << P
              << ", X_max(diag) = " << X_max << std::endl;
  }

  const double prescreen_thr = (tol * tol) / std::max(X_max, 1e-300);
  for (std::size_t p = 0; p < P; ++p) {
    if (drv.d[p] < prescreen_thr) drv.d[p] = 0.0;
  }

  std::size_t iter = 0;
  std::size_t total_shellpair_recomputes = 0;
  while (true) {
    std::size_t pivot_p = 0;
    double pivot_val = -1.0;
    for (std::size_t p = 0; p < P; ++p) {
      if (drv.d[p] > pivot_val) {
        pivot_val = drv.d[p];
        pivot_p = p;
      }
    }
    if (pivot_val < tol) break;

    const std::size_t sp_idx = drv.D.pairs[pivot_p].shell_pair_idx;
    drv.cur_pivot_sp = sp_idx;
    drv.cur_pivot_npairs = drv.D.shell_pairs[sp_idx].npairs;
    drv.Q.assign(P * drv.cur_pivot_npairs, 0.0);
    flow.run_pivot_block();
    ++total_shellpair_recomputes;

    while (true) {
      const auto& sp = drv.D.shell_pairs[sp_idx];
      std::size_t sub_pivot = sp.pair_begin;
      double sub_val = drv.d[sub_pivot];
      for (std::size_t p = sp.pair_begin + 1; p < sp.pair_begin + sp.npairs; ++p) {
        if (drv.d[p] > sub_val) {
          sub_val = drv.d[p];
          sub_pivot = p;
        }
      }
      double X_max_now = 0.0;
      for (std::size_t p = 0; p < P; ++p) X_max_now = std::max(X_max_now, drv.d[p]);
      const double local_thr = X_max_now * 1e-3;
      if (sub_val <= local_thr || sub_val < tol) break;

      drv.apply_cholesky_update(sub_pivot);
      ++iter;

      if (drv.trace_each_vector) {
        std::cout << "  iter " << std::setw(5) << iter
                  << "  rank=" << std::setw(5) << drv.L.size()
                  << "  pivot p=" << std::setw(6) << sub_pivot
                  << "  d[pivot]=" << std::scientific << std::setprecision(4)
                  << sub_val << "  X_max_now=" << X_max_now
                  << std::defaultfloat << std::endl;
      }
    }
  }

  if (drv.verbose) {
    std::cout << "[eri-pch] converged: rank = " << drv.L.size()
              << ", shell-pair recomputes = " << total_shellpair_recomputes
              << " (out of " << drv.D.nshell_pairs() << " possible)" << std::endl;
  }

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
