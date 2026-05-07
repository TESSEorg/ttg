// SPDX-License-Identifier: BSD-3-Clause

#include "common.h"

#include <libint2/pivoted_cholesky.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace eri_pch {

std::vector<std::pair<std::size_t, std::size_t>>
compute_significant_shellpairs(const libint2::BasisSet& basis,
                               double threshold) {
  const std::size_t nsh = basis.size();
  libint2::Engine engine(libint2::Operator::overlap, basis.max_nprim(),
                         basis.max_l(), 0);
  engine.set_precision(0.);

  std::vector<std::pair<std::size_t, std::size_t>> sps;
  sps.reserve(nsh * (nsh + 1) / 2);
  for (std::size_t s1 = 0; s1 < nsh; ++s1) {
    const std::size_t n1 = basis[s1].size();
    for (std::size_t s2 = s1; s2 < nsh; ++s2) {
      const bool on_same_center = (basis[s1].O == basis[s2].O);
      bool significant = on_same_center;
      if (!on_same_center) {
        const std::size_t n2 = basis[s2].size();
        const auto& result = engine.compute(basis[s1], basis[s2]);
        if (result[0] != nullptr) {
          Eigen::Map<const Eigen::MatrixXd> buf_mat(result[0],
                                                    static_cast<Eigen::Index>(n1),
                                                    static_cast<Eigen::Index>(n2));
          significant = (buf_mat.norm() >= threshold);
        }
      }
      if (significant) sps.emplace_back(s1, s2);
    }
  }
  return sps;
}

void IntegralData::build(std::vector<libint2::Atom> atoms_in,
                         const std::string& basis_name,
                         double screening_threshold) {
  atoms = std::move(atoms_in);
  basis = libint2::BasisSet(basis_name, atoms);
  basis.set_pure(true);
  nbf = static_cast<std::size_t>(basis.nbf());
  nshell = basis.size();
  shell2bf = basis.shell2bf();
  shell_size.resize(nshell);
  for (std::size_t s = 0; s < nshell; ++s) shell_size[s] = basis[s].size();

  pair_lookup.assign(nbf * nbf, npos);
  pairs.clear();
  shell_pairs.clear();

  std::vector<std::pair<std::size_t, std::size_t>> sps;
  if (screening_threshold > 0.0) {
    sps = compute_significant_shellpairs(basis, screening_threshold);
  } else {
    sps.reserve(nshell * (nshell + 1) / 2);
    for (std::size_t s1 = 0; s1 < nshell; ++s1)
      for (std::size_t s2 = s1; s2 < nshell; ++s2)
        sps.emplace_back(s1, s2);
  }

  for (auto [s1, s2] : sps) {
    ShellPair sp;
    sp.s1 = s1;
    sp.s2 = s2;
    sp.pair_begin = pairs.size();
    const std::size_t bf1 = shell2bf[s1], n1 = shell_size[s1];
    const std::size_t bf2 = shell2bf[s2], n2 = shell_size[s2];
    const std::size_t sp_idx = shell_pairs.size();
    std::size_t local = 0;
    for (std::size_t i = 0; i < n1; ++i) {
      const std::size_t lo_j = (s1 == s2) ? i : 0;
      for (std::size_t j = lo_j; j < n2; ++j) {
        PairInfo p;
        p.mu = bf1 + i;
        p.nu = bf2 + j;
        p.shell_pair_idx = sp_idx;
        p.local_idx = local++;
        const std::size_t pidx = pairs.size();
        pair_lookup[p.mu * nbf + p.nu] = pidx;
        pair_lookup[p.nu * nbf + p.mu] = pidx;
        pairs.push_back(p);
      }
    }
    sp.npairs = local;
    shell_pairs.push_back(sp);
  }
}

libint2::Engine& EnginePool::get() {
  const auto tid = std::this_thread::get_id();
  {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto it = per_thread_.find(tid);
    if (it != per_thread_.end()) return *it->second;
  }
  auto eng = std::make_unique<libint2::Engine>(libint2::Operator::coulomb,
                                               max_nprim_, max_l_, 0);
  eng->set_precision(0.0);
  std::unique_lock<std::shared_mutex> lock(mu_);
  auto [it, inserted] = per_thread_.emplace(tid, std::move(eng));
  return *it->second;
}

std::size_t EnginePool::size() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return per_thread_.size();
}

void compute_diag_block(const IntegralData& D, EnginePool& pool,
                        std::size_t sp_idx, double* d_out) {
  const auto& sp = D.shell_pairs[sp_idx];
  auto& engine = pool.get();
  const auto& result =
      engine.compute(D.basis[sp.s1], D.basis[sp.s2], D.basis[sp.s1], D.basis[sp.s2]);
  if (result[0] == nullptr) return;  // fully screened
  const double* buf = result[0];
  const std::size_t n1 = D.shell_size[sp.s1];
  const std::size_t n2 = D.shell_size[sp.s2];
  std::size_t local = 0;
  for (std::size_t i = 0; i < n1; ++i) {
    const std::size_t lo_j = (sp.s1 == sp.s2) ? i : 0;
    for (std::size_t j = lo_j; j < n2; ++j, ++local) {
      const std::size_t idx = ((i * n2 + j) * n1 + i) * n2 + j;
      d_out[sp.pair_begin + local] = buf[idx];
    }
  }
}

void compute_pivot_block(const IntegralData& D, EnginePool& pool,
                         std::size_t sp_idx, std::size_t pivot_sp_idx,
                         double* rows_out) {
  const auto& sp = D.shell_pairs[sp_idx];
  const auto& pivot = D.shell_pairs[pivot_sp_idx];
  auto& engine = pool.get();
  const auto& result = engine.compute(D.basis[sp.s1], D.basis[sp.s2],
                                      D.basis[pivot.s1], D.basis[pivot.s2]);
  if (result[0] == nullptr) return;
  const double* buf = result[0];
  const std::size_t n1 = D.shell_size[sp.s1];
  const std::size_t n2 = D.shell_size[sp.s2];
  const std::size_t nA = D.shell_size[pivot.s1];
  const std::size_t nB = D.shell_size[pivot.s2];
  const std::size_t W = pivot.npairs;
  std::size_t row_local = 0;
  for (std::size_t i = 0; i < n1; ++i) {
    const std::size_t lo_j = (sp.s1 == sp.s2) ? i : 0;
    for (std::size_t j = lo_j; j < n2; ++j, ++row_local) {
      std::size_t col_local = 0;
      for (std::size_t k = 0; k < nA; ++k) {
        const std::size_t lo_l = (pivot.s1 == pivot.s2) ? k : 0;
        for (std::size_t l = lo_l; l < nB; ++l, ++col_local) {
          const std::size_t idx = ((i * n2 + j) * nA + k) * nB + l;
          rows_out[row_local * W + col_local] = buf[idx];
        }
      }
    }
  }
}

Eigen::MatrixXd build_dense_M(const IntegralData& D, EnginePool& pool) {
  const std::size_t P = D.npairs();
  Eigen::MatrixXd M(P, P);
  M.setZero();
  auto& engine = pool.get();
  for (std::size_t spA = 0; spA < D.nshell_pairs(); ++spA) {
    const auto& A = D.shell_pairs[spA];
    for (std::size_t spB = 0; spB < D.nshell_pairs(); ++spB) {
      const auto& B = D.shell_pairs[spB];
      const auto& result = engine.compute(D.basis[A.s1], D.basis[A.s2],
                                          D.basis[B.s1], D.basis[B.s2]);
      if (result[0] == nullptr) continue;
      const double* buf = result[0];
      const std::size_t n1 = D.shell_size[A.s1];
      const std::size_t n2 = D.shell_size[A.s2];
      const std::size_t n3 = D.shell_size[B.s1];
      const std::size_t n4 = D.shell_size[B.s2];
      std::size_t row_local = 0;
      for (std::size_t i = 0; i < n1; ++i) {
        const std::size_t lo_j = (A.s1 == A.s2) ? i : 0;
        for (std::size_t j = lo_j; j < n2; ++j, ++row_local) {
          const std::size_t p = A.pair_begin + row_local;
          std::size_t col_local = 0;
          for (std::size_t k = 0; k < n3; ++k) {
            const std::size_t lo_l = (B.s1 == B.s2) ? k : 0;
            for (std::size_t l = lo_l; l < n4; ++l, ++col_local) {
              const std::size_t idx = ((i * n2 + j) * n3 + k) * n4 + l;
              M(p, B.pair_begin + col_local) = buf[idx];
            }
          }
        }
      }
    }
  }
  return M;
}

std::vector<libint2::Atom> default_water() {
  std::vector<libint2::Atom> atoms(3);
  // Coordinates in libint2::Atom are atomic units (bohr).
  // Experimental r(OH) = 0.9572 A, theta = 104.52 deg.
  constexpr double a2b = 1.8897259886;
  atoms[0].atomic_number = 8;
  atoms[0].x = 0.0; atoms[0].y = 0.0; atoms[0].z = 0.0;
  atoms[1].atomic_number = 1;
  atoms[1].x = 0.0; atoms[1].y = 0.7572 * a2b; atoms[1].z = 0.5860 * a2b;
  atoms[2].atomic_number = 1;
  atoms[2].x = 0.0; atoms[2].y = -0.7572 * a2b; atoms[2].z = 0.5860 * a2b;
  return atoms;
}

std::vector<libint2::Atom> read_xyz(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open xyz file: " + path);
  return libint2::read_dotxyz(in);
}

CliArgs parse_cli(int argc, char* argv[]) {
  CliArgs args;
  if (argc >= 2) args.xyz_path = argv[1];
  if (argc >= 3) args.basis_name = argv[2];
  if (argc >= 4) args.tol = std::stod(argv[3]);
  return args;
}

void print_run_header(const CliArgs& args) {
  std::cout << "[eri-pch] geometry: "
            << (args.xyz_path.empty() ? std::string("default water")
                                      : args.xyz_path)
            << ",  basis: " << args.basis_name
            << ",  tol: " << std::scientific << std::setprecision(3)
            << args.tol << std::defaultfloat << std::endl;
}

void validate_decomposition(const IntegralData& D, EnginePool& pool,
                            const Eigen::MatrixXd& Lmat, double tol) {
  const auto P = D.npairs();
  if (P > 600) {
    std::cout << "[validate] skipped (P=" << P
              << " > 600); reconstruction check is O(P^2) memory." << std::endl;
    return;
  }
  Eigen::MatrixXd Mdense = build_dense_M(D, pool);
  Eigen::MatrixXd Mrec = Lmat * Lmat.transpose();
  Eigen::MatrixXd Err = Mdense - Mrec;
  double max_err = Err.cwiseAbs().maxCoeff();
  double frob_err = Err.norm();
  std::cout << std::scientific << std::setprecision(6)
            << "[validate] max |M - L L^T| = " << max_err
            << ", ||M - L L^T||_F = " << frob_err
            << ", tol = " << tol << std::endl;

  std::vector<std::size_t> initial_pivot(P);
  std::iota(initial_pivot.begin(), initial_pivot.end(), 0);
  auto ref_piv = libint2::pivoted_cholesky(Mdense, tol, initial_pivot);
  std::cout << "[validate] libint2::pivoted_cholesky rank = "
            << ref_piv.size() << "; our rank = " << Lmat.cols() << std::endl;
}

}  // namespace eri_pch
