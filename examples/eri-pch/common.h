// SPDX-License-Identifier: BSD-3-Clause
//
// Shared building blocks for the eri-pch v* TTG examples: basis/AO-pair/
// shell-pair layout, libint engine pool, integral kernels, dense reference,
// CLI parsing, and validation. Anything that is *not* specific to a
// particular TTG flow lives here.

#pragma once

#include <Eigen/Dense>
#include <libint2.hpp>

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace eri_pch {

// One AO pair (mu, nu) with mu <= nu, plus the shell pair it belongs to.
struct PairInfo {
  std::size_t mu;
  std::size_t nu;
  std::size_t shell_pair_idx;
  std::size_t local_idx;  // position within the shell pair's pair list
};

// One shell pair (s1, s2) with s1 <= s2 and the contiguous range of AO
// pairs it owns in the flat pairs[] array.
struct ShellPair {
  std::size_t s1;
  std::size_t s2;
  std::size_t pair_begin;
  std::size_t npairs;
};

// All static data describing the basis, AO-pair, and shell-pair layout.
class IntegralData {
 public:
  std::vector<libint2::Atom> atoms;
  libint2::BasisSet basis;
  std::size_t nbf = 0;
  std::size_t nshell = 0;
  std::vector<std::size_t> shell2bf;
  std::vector<std::size_t> shell_size;
  std::vector<PairInfo> pairs;
  std::vector<ShellPair> shell_pairs;
  std::vector<std::size_t> pair_lookup;  // [mu * nbf + nu] -> pair index
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  std::size_t pair_index(std::size_t mu, std::size_t nu) const {
    return pair_lookup[mu * nbf + nu];
  }
  std::size_t npairs() const { return pairs.size(); }
  std::size_t nshell_pairs() const { return shell_pairs.size(); }

  // Build the AO-pair / shell-pair layout. If `screening_threshold > 0`, the
  // shell-pair list is restricted to *significant* pairs in the sense of
  // Almlöf — those whose overlap matrix block has Frobenius norm at least
  // the threshold (same-center pairs are always kept). With the default
  // value of 0 every (s1 <= s2) pair is included, matching the original
  // unscreened behavior.
  void build(std::vector<libint2::Atom> atoms_in, const std::string& basis_name,
             double screening_threshold = 0.0);
};

// List of (s1, s2) shell pairs with s1 <= s2 whose overlap-matrix block
// has Frobenius norm >= `threshold`. Pairs whose two shells sit on the same
// atomic center are always kept (their overlap is bounded below by the
// shell normalizations). Mirrors the screening done by libint2's
// hartree-fock++ test (`compute_shellpairs`).
std::vector<std::pair<std::size_t, std::size_t>>
compute_significant_shellpairs(const libint2::BasisSet& basis,
                               double threshold = 1e-12);

// Per-thread cache of libint2::Engine instances. libint engines are
// stateful and not thread-safe to share. Each calling thread gets its own
// engine, lazily constructed on first use.
class EnginePool {
 public:
  EnginePool(std::size_t max_nprim, int max_l)
      : max_nprim_(max_nprim), max_l_(max_l) {}

  libint2::Engine& get();
  std::size_t size() const;

 private:
  std::size_t max_nprim_;
  int max_l_;
  mutable std::shared_mutex mu_;
  std::unordered_map<std::thread::id, std::unique_ptr<libint2::Engine>> per_thread_;
};

// Compute the diagonals (mu nu | mu nu) for AO pairs in shell pair sp_idx
// (one shell quartet (s1 s2 | s1 s2)) and write them into d_out at offset
// sp.pair_begin. d_out must point at the start of a P-element array.
void compute_diag_block(const IntegralData& D, EnginePool& pool,
                        std::size_t sp_idx, double* d_out);

// Compute (mu nu | alpha beta) for (mu,nu) in shell pair sp_idx and
// (alpha,beta) in pivot_sp_idx (one shell quartet (s1 s2 | sA sB)). Writes
// the row strip into rows_out as an `npairs(sp) x npairs(pivot)` row-major
// block. Caller is responsible for zero-initialization if screening can
// drop values.
void compute_pivot_block(const IntegralData& D, EnginePool& pool,
                         std::size_t sp_idx, std::size_t pivot_sp_idx,
                         double* rows_out);

// Build the dense P x P AO-pair integral matrix once (for validation only).
Eigen::MatrixXd build_dense_M(const IntegralData& D, EnginePool& pool);

// Default geometry: water at the experimental geometry, in bohr.
std::vector<libint2::Atom> default_water();

// Read an .xyz file via libint2::read_dotxyz.
std::vector<libint2::Atom> read_xyz(const std::string& path);

// CLI argument parsing: `[<molecule.xyz> [<basis-name> [<tolerance>]]]`.
struct CliArgs {
  std::string xyz_path;            // empty → default_water()
  std::string basis_name = "def2-svp";
  double tol = 1e-8;
};
CliArgs parse_cli(int argc, char* argv[]);

// Print one-line summary of the run on rank 0.
void print_run_header(const CliArgs& args);

// Compare reconstructed M = L L^T against the dense reference matrix and
// against libint2::pivoted_cholesky on the dense matrix. Prints results.
// Skips when P > 600 (dense matrix would exceed ~3 MB).
void validate_decomposition(const IntegralData& D, EnginePool& pool,
                            const Eigen::MatrixXd& Lmat, double tol);

}  // namespace eri_pch
