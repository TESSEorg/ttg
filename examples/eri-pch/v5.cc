// SPDX-License-Identifier: BSD-3-Clause
//
// eri-pch v5 -- data-flow dataflow.
//
// In v1..v4 the application's per-row state (the running diagonals d[]
// and the Cholesky-vector matrix L[]) lived as metadata on the
// CholeskyDriver, accessed by reference from inside TTG task lambdas.
// That works in shared memory but rules out distributed execution: a TT
// running on rank r cannot follow a reference into rank 0's process
// memory. To make the application portable to distributed memory, the
// payload has to flow on edges instead.
//
// v5 makes the data part of the flow:
//
//   - Per-row-chunk tiles flow iter-to-iter:
//       e_L_tile_in(iter, chunk)  : Eigen::Tensor<double, 2>  (P_chunk × M_old)
//       e_d_tile_in(iter, chunk)  : Eigen::Tensor<double, 1>  (P_chunk)
//
//   - The pivot-block payload, the integral row slabs, and the per-chunk
//     setup data all flow as Eigen::Tensor<double, *>.
//
//   - Global argmax over d is a streaming reduction across chunks.
//
//   - L_AB rows for each iter's pivot shell pair are produced by N_chunks
//     L_AB_extract tasks (each chunk publishes its overlap with the AB
//     row range, possibly empty) and concatenated by a streaming reducer
//     into gather_pivot.
//
// To make Eigen::Tensor flow on edges we provide a small TensorTile<N>
// wrapper that exposes the boost-style and madness-style serialize()
// methods TTG requires. The actual numerical kernels still operate via
// Eigen::Map<MatrixXd> over the tensor's storage so that BLAS-3 paths
// remain intact.
//
// Algorithm reference: Koch, Sanchez de Meras, Pedersen,
//   J. Chem. Phys. 118, 9481 (2003).
//
// Usage: eri-pch_v5-{mad,parsec} [<molecule.xyz> [<basis-name> [<tolerance>]]]
//
// Side effect: writes the TTG flow graph to `eri-pch_v5.dot`.

#include "common.h"

#include <ttg.h>
#include <ttg/util/dot.h>
#include <ttg/util/multiindex.h>
#include <ttg/serialization/traits.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <array>
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

// --------------------------------------------------------------------------
// TensorTile<Rank, StorageOrder>: thin Eigen::Tensor wrapper that exposes
// the serialize methods TTG looks for (both boost-style and madness-style).
// Rank is fixed at compile time; dimensions are dynamic and serialized
// along with the data. StorageOrder selects column-major (default) or
// row-major in-memory layout, and as_matrix() returns a matching
// Eigen::Map<...> view so no copy/permute is needed when interfacing with
// row-major external buffers (e.g. integrals from compute_pivot_block,
// which writes buf[i*W+j]).
// --------------------------------------------------------------------------
template <int Rank, int StorageOrder = Eigen::ColMajor>
struct TensorTile {
  using TensorT = Eigen::Tensor<double, Rank, StorageOrder>;
  using MatrixT = std::conditional_t<
      StorageOrder == Eigen::RowMajor,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;
  TensorT data;

  TensorTile() = default;
  TensorTile(const TensorTile&) = default;
  TensorTile(TensorTile&&) noexcept = default;
  TensorTile& operator=(const TensorTile&) = default;
  TensorTile& operator=(TensorTile&&) noexcept = default;

  explicit TensorTile(TensorT&& t) : data(std::move(t)) {}
  explicit TensorTile(const TensorT& t) : data(t) {}

  // Construct an empty tile of given dimensions.
  template <typename... Dims>
  static TensorTile zeros(Dims... dims) {
    static_assert(sizeof...(Dims) == Rank, "wrong number of dims");
    TensorTile t;
    t.data.resize(static_cast<Eigen::Index>(dims)...);
    t.data.setZero();
    return t;
  }

  Eigen::Index dimension(int i) const { return data.dimension(i); }
  Eigen::Index size() const { return data.size(); }
  double* tensor_data() { return data.data(); }
  const double* tensor_data() const { return data.data(); }

  // Matrix view with the same storage order as the tensor; no permute.
  Eigen::Map<MatrixT> as_matrix() {
    static_assert(Rank == 2, "as_matrix requires Rank == 2");
    return Eigen::Map<MatrixT>(data.data(), dimension(0), dimension(1));
  }
  Eigen::Map<const MatrixT> as_matrix() const {
    static_assert(Rank == 2, "as_matrix requires Rank == 2");
    return Eigen::Map<const MatrixT>(data.data(), dimension(0), dimension(1));
  }
  Eigen::Map<Eigen::VectorXd> as_vector() {
    static_assert(Rank == 1, "as_vector requires Rank == 1");
    return Eigen::Map<Eigen::VectorXd>(data.data(), dimension(0));
  }
  Eigen::Map<const Eigen::VectorXd> as_vector() const {
    static_assert(Rank == 1, "as_vector requires Rank == 1");
    return Eigen::Map<const Eigen::VectorXd>(data.data(), dimension(0));
  }

  // Boost-style serialization (two-arg).
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { archive_io(ar); }
  // madness-style serialization (one-arg).
  template <typename Archive>
  void serialize(Archive& ar) { archive_io(ar); }

 private:
  template <typename Archive>
  void archive_io(Archive& ar) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>) {
      for (int i = 0; i < Rank; ++i) {
        Eigen::Index d = data.dimension(i);
        ar & d;
      }
      const Eigen::Index n = data.size();
      for (Eigen::Index i = 0; i < n; ++i) ar & data.data()[i];
    } else {
      std::array<Eigen::Index, Rank> dims;
      for (int i = 0; i < Rank; ++i) {
        Eigen::Index d;
        ar & d;
        dims[i] = d;
      }
      // Eigen::Tensor::resize takes Eigen::array (== std::array on most builds).
      Eigen::array<Eigen::Index, Rank> earr;
      for (int i = 0; i < Rank; ++i) earr[i] = dims[i];
      data.resize(earr);
      const Eigen::Index n = data.size();
      for (Eigen::Index i = 0; i < n; ++i) ar & data.data()[i];
    }
  }
};

// One shell pair's contribution to Q for the iteration's pivot shell pair.
// Same role as in v2..v4 but the row data is now an Eigen::Tensor<double, 2>.
// X_max_now (global running max diagonal at the start of the iter) is
// carried along so gather_pivot can apply Koch's X_max/1000 sub-pivot
// threshold without needing a separate side channel.
struct PivotSlab {
  std::size_t sp_idx = 0;
  std::size_t pivot_sp_idx = 0;
  std::size_t pair_begin = 0;
  std::size_t npairs = 0;
  std::size_t W = 0;
  double X_max_now = 0.0;
  bool terminate = false;  // drain marker (no integrals computed)
  // Row-major storage matches compute_pivot_block's buf[i*W+j] layout, so
  // the integral kernel writes directly into the tile with no permute.
  TensorTile<2, Eigen::RowMajor> rows;  // shape (npairs, W)

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & sp_idx & pivot_sp_idx & pair_begin & npairs & W & X_max_now
       & terminate & rows;
  }
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & sp_idx & pivot_sp_idx & pair_begin & npairs & W & X_max_now
       & terminate & rows;
  }
};
using PivotSlabs = std::vector<PivotSlab>;

// What iter_dispatch hands to compute_slab and L_AB_extract: pivot shell
// pair plus the global X_max_now for thresholding. When `terminate` is set
// it instead acts as a drain marker that propagates through compute_slab,
// L_AB_extract, gather_pivot, and compute_l_chunk so the orphan L_tile /
// d_tile values for the converged iter get consumed (otherwise ttg::fence
// would never return).
struct DispatchInfo {
  std::size_t pivot_sp_idx = 0;
  double X_max_now = 0.0;
  bool terminate = false;

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & pivot_sp_idx & X_max_now & terminate;
  }
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & pivot_sp_idx & X_max_now & terminate;
  }
};

// Per-chunk "bag of L tiles". Each entry is one iter's L_new for this
// chunk, shape (P_chunk, Wp_iter). The bag length grows by one per
// successful iter (Wp > 0). Avoids the per-iter (P_chunk × M_new) realloc
// + memcpy that drives v5's macOS sys time.
//
// The bag flows on the loopback edge `e_L_tile_in` and stays on chunk
// c's home rank for the life of the run (chunks are pinned by keymap
// to `c % nranks`), so no cross-rank ship of `vector<TensorTile<2>>`
// happens during the iters. Only `collect_final_L` ships one bag per
// chunk, once, at termination.
using LStack = std::vector<TensorTile<2>>;

// One chunk's contribution of L_AB rows for the current iter's pivot
// shell pair, packed as a SINGLE (n_rows × M_old) column-major matrix.
// The columns correspond to the column-block layout of the L bag
// (concatenated left-to-right); compute_l_chunk's multi-GEMM walks
// through it with column offsets matching the per-tile Wp_k counts in
// `L_in`. Single matrix == single TTG payload, single MPI send.
struct LABContribution {
  std::size_t chunk = 0;
  std::size_t row_offset_in_AB = 0;  // global row index relative to pair_begin
  std::size_t n_rows = 0;            // 0 means "no overlap"
  std::size_t M_old = 0;             // number of columns in `rows`
  TensorTile<2> rows;                // shape (n_rows, M_old)

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & chunk & row_offset_in_AB & n_rows & M_old & rows;
  }
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & chunk & row_offset_in_AB & n_rows & M_old & rows;
  }
};
using LABContributions = std::vector<LABContribution>;

// Output of the small W×W pivoted Cholesky on R_AB. Broadcast to all
// chunks for the iter. L_AB_chosen is one (Wp × M_old) matrix; chunks
// do their multi-GEMM by walking through it with column offsets that
// match the per-tile Wp_k of the L bag:
//
//   L_new = Q_chunk - sum_k L_in[k] · L_AB_chosen.middleCols(off_k, Wp_k)^T
//
// Same FLOP count as one big dgemm, but no monolithic L_in needed
// on the chunk side and no list-of-blocks payload on the wire.
struct IterSetup {
  std::size_t pivot_sp_idx = 0;
  std::size_t M_old = 0;
  std::size_t Wp = 0;          // 0 means "no progress this iter; pass through"
  std::size_t W = 0;           // total AB pairs (so chunks know Q's column count)
  bool terminate = false;      // drain marker (compute_l_chunk consumes and exits)
  TensorTile<2> U_block;       // Wp × Wp lower triangular
  TensorTile<2> L_AB_chosen;   // Wp × M_old, packed left-to-right by L bag tile
  std::vector<std::size_t> perm;  // length W; perm[0..Wp-1] are the chosen sub-pivots

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & pivot_sp_idx & M_old & Wp & W & terminate & U_block
       & L_AB_chosen & perm;
  }
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & pivot_sp_idx & M_old & Wp & W & terminate & U_block
       & L_AB_chosen & perm;
  }
};

// One chunk's (val, p) summary for global argmax reduction across chunks.
struct ArgmaxItem {
  double val = -1.0;
  std::size_t p = 0;  // global pair index
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & val & p; }
  template <typename Archive>
  void serialize(Archive& ar) { ar & val & p; }
};

// Result fanned out from argmax_reducer to iter_dispatch.
struct ArgmaxResult {
  double val = -1.0;
  std::size_t p = 0;
  std::size_t pivot_sp_idx = 0;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & val & p & pivot_sp_idx; }
  template <typename Archive>
  void serialize(Archive& ar) { ar & val & p & pivot_sp_idx; }
};

// --------------------------------------------------------------------------
// Row-chunk layout. Each chunk owns rows [p_lo, p_hi) of the global
// (P-row) state. We deliberately use simple uniform chunking; routing
// between chunks and shell-pair-local AO ranges happens at the consumers
// (gather_pivot, L_AB_extract).
// --------------------------------------------------------------------------
struct ChunkLayout {
  std::size_t n_chunks = 0;
  std::vector<std::size_t> p_lo;  // length n_chunks + 1, with p_lo[n_chunks] = P
  std::size_t P = 0;

  std::size_t lo(std::size_t c) const { return p_lo[c]; }
  std::size_t hi(std::size_t c) const { return p_lo[c + 1]; }
  std::size_t size_of(std::size_t c) const { return hi(c) - lo(c); }

  void build(std::size_t P_total, std::size_t Nc) {
    P = P_total;
    n_chunks = Nc;
    p_lo.assign(n_chunks + 1, 0);
    for (std::size_t c = 0; c <= n_chunks; ++c) {
      p_lo[c] = (c * P) / n_chunks;
    }
  }

  // Compute [overlap_lo, overlap_hi) of [a, b) with chunk c's row range.
  std::pair<std::size_t, std::size_t> overlap(std::size_t c,
                                              std::size_t a,
                                              std::size_t b) const {
    return {std::max(lo(c), a), std::min(hi(c), b)};
  }
};

inline std::size_t default_chunk_count() {
  if (const char* env = std::getenv("TTG_ERI_PCH_CHUNKS")) {
    if (auto v = std::atoi(env); v > 0) return static_cast<std::size_t>(v);
  }
  unsigned int hc = std::thread::hardware_concurrency();
  if (hc == 0) hc = 8;
  return static_cast<std::size_t>(std::min<unsigned int>(hc, 24));
}

struct FlowProfile {
  std::atomic<long> ns_compute_diag{0};
  std::atomic<long> ns_compute_slab{0};
  std::atomic<long> ns_l_ab_extract{0};
  std::atomic<long> ns_gather_setup{0};
  std::atomic<long> ns_chunk_compute{0};
  std::atomic<long> ns_iter_dispatch{0};
  std::atomic<long> ns_argmax{0};
  std::atomic<int> n_chunk_calls{0};
  std::atomic<int> n_new_columns{0};
  std::mutex tids_mu;
  std::set<std::thread::id> tids_chunk;
};
inline long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// --------------------------------------------------------------------------
// CholeskyDriver: just the static problem definition + the engine pool.
// Per-iteration state (d, L) is no longer here -- it flows on edges.
// d_init and L are kept as scratch only during the diagonal-init phase
// before the iteration loop starts.
// --------------------------------------------------------------------------
class CholeskyDriver {
 public:
  IntegralData D;
  EnginePool pool;
  ChunkLayout layout;
  std::vector<double> d_init;  // scratch for diagonal init only
  // Note: the cumulative L matrix (concatenation of all per-chunk
  // L_tiles) is reconstructed at the end from the final L_tile values
  // arriving at the terminate task; no global L is held here.
  std::size_t total_iters = 0;
  std::size_t total_new_columns = 0;
  bool verbose = false;
  bool trace_each_vector = false;

  // Final result: drain pass snapshots each chunk's L bag here, then
  // run_cholesky repacks them into the global L matrix.
  std::vector<LStack> final_L_bags;
  std::mutex final_mu;

  CholeskyDriver(IntegralData data, std::size_t n_chunks)
      : D(std::move(data)),
        pool(D.basis.max_nprim(), D.basis.max_l()) {
    layout.build(D.npairs(), n_chunks);
    d_init.assign(D.npairs(), 0.0);
    final_L_bags.resize(n_chunks);
  }

  void compute_diagonal_for_shell_pair(std::size_t sp_idx) {
    compute_diag_block(D, pool, sp_idx, d_init.data());
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
    // Row-major TensorTile matches compute_pivot_block's buf[i*W+j] layout
    // exactly, so the integral kernel writes straight into the tile.
    slab.rows = decltype(slab.rows)::zeros(slab.npairs, slab.W);
    compute_pivot_block(D, pool, sp_idx, pivot_sp_idx, slab.rows.tensor_data());
    return slab;
  }
};

// --------------------------------------------------------------------------
// CholeskyFlow: the dataflow graph.
// --------------------------------------------------------------------------
struct CholeskyFlow {
  CholeskyDriver& drv;
  const double tol;
  FlowProfile prof;

  using IterChunk = ttg::MultiIndex<2>;     // (iter, chunk)
  using IterSp    = ttg::MultiIndex<2>;     // (iter, sp_idx)

  // diag init
  ttg::Edge<std::size_t, void>           e_diag_in{"e_diag_in"};
  ttg::Edge<int, int>                    e_diag_done{"e_diag_done"};
  // initial broadcast
  ttg::Edge<IterChunk, LStack>           e_L_tile_in{"e_L_tile_in"};
  ttg::Edge<IterChunk, TensorTile<1>>    e_d_tile_in{"e_d_tile_in"};
  ttg::Edge<int, ArgmaxItem>             e_argmax_in{"e_argmax_in"};
  // argmax → iter_dispatch
  ttg::Edge<int, ArgmaxResult>           e_pivot_decided{"e_pivot_decided"};
  // iter_dispatch fan-outs (carry DispatchInfo so X_max_now propagates)
  ttg::Edge<IterSp, DispatchInfo>        e_compute_slab_in{"e_compute_slab_in"};
  ttg::Edge<IterChunk, DispatchInfo>     e_l_ab_request{"e_l_ab_request"};
  ttg::Edge<int, void>                   e_terminate{"e_terminate"};
  // compute_slab → gather (streaming)
  ttg::Edge<int, PivotSlabs>             e_pivot_collect{"e_pivot_collect"};
  // L_AB_extract → gather (streaming reducer on input 1)
  ttg::Edge<int, LABContributions>       e_l_ab_collect{"e_l_ab_collect"};
  // gather → compute_l_chunk
  ttg::Edge<IterChunk, IterSetup>        e_setup{"e_setup"};
  ttg::Edge<IterChunk, TensorTile<2>>    e_q_chunk{"e_q_chunk"};
  // compute_l_chunk drain → collect_final_L (gathers per-chunk L bags
  // onto rank 0 so the global L can be repacked there for validation).
  ttg::Edge<std::size_t, LStack>         e_final_L{"e_final_L"};

  std::unique_ptr<ttg::TTBase> t_diag_drive;
  std::unique_ptr<ttg::TTBase> t_diag_compute;
  std::unique_ptr<ttg::TTBase> t_init_phase;
  std::unique_ptr<ttg::TTBase> t_argmax;
  std::unique_ptr<ttg::TTBase> t_iter_dispatch;
  std::unique_ptr<ttg::TTBase> t_compute_slab;
  std::unique_ptr<ttg::TTBase> t_l_ab_extract;
  std::unique_ptr<ttg::TTBase> t_gather_pivot;
  std::unique_ptr<ttg::TTBase> t_compute_l_chunk;
  std::unique_ptr<ttg::TTBase> t_terminate;
  std::unique_ptr<ttg::TTBase> t_collect_final_L;

  CholeskyFlow(CholeskyDriver& d, double tol_) : drv(d), tol(tol_) { build(); }

  // Helpers

  static IterChunk ic(int iter, std::size_t chunk) {
    return IterChunk{static_cast<unsigned long>(iter),
                     static_cast<unsigned long>(chunk)};
  }

  void build() {
    const std::size_t nsp = drv.D.nshell_pairs();
    const std::size_t Nc = drv.layout.n_chunks;
    const std::size_t P = drv.D.npairs();

    // ---- Diagonal init phase ----
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

    // init_phase: streaming x nsp at key=0. After all diag tasks done,
    // emits initial tiles for each chunk.
    auto init_phase = ttg::make_tt(
        [&, P, Nc, tol_local = tol](
            const int& iter_key, const int& /*count*/,
            std::tuple<ttg::Out<IterChunk, LStack>,
                       ttg::Out<IterChunk, TensorTile<1>>,
                       ttg::Out<int, ArgmaxItem>>& out) {
          // Prescreen drv.d_init using the global X_max.
          double X_max = 0.0;
          for (std::size_t p = 0; p < P; ++p)
            X_max = std::max(X_max, drv.d_init[p]);
          const double prescreen_thr =
              (tol_local * tol_local) / std::max(X_max, 1e-300);
          for (std::size_t p = 0; p < P; ++p)
            if (drv.d_init[p] < prescreen_thr) drv.d_init[p] = 0.0;

          if (drv.verbose) {
            std::cout << "[eri-pch] N = " << drv.D.nbf
                      << ", #shells = " << drv.D.nshell
                      << ", #shell pairs = " << drv.D.nshell_pairs()
                      << ", #AO pairs P = " << P
                      << ", X_max(diag) = " << X_max
                      << ", chunks = " << Nc << std::endl;
          }
          // Emit initial tiles per chunk.
          for (std::size_t c = 0; c < Nc; ++c) {
            const std::size_t lo = drv.layout.lo(c);
            const std::size_t hi = drv.layout.hi(c);
            const std::size_t nrow = hi - lo;
            // L bag starts empty -- no iters have produced anything yet.
            LStack bag;
            // d_tile(0, c) is the slice of d_init.
            TensorTile<1> dtile = TensorTile<1>::zeros(nrow);
            std::memcpy(dtile.tensor_data(), drv.d_init.data() + lo,
                        nrow * sizeof(double));
            // argmax contribution
            double v = -1.0;
            std::size_t pmax = lo;
            for (std::size_t p = lo; p < hi; ++p) {
              if (drv.d_init[p] > v) { v = drv.d_init[p]; pmax = p; }
            }
            ArgmaxItem item{v, pmax};
            ttg::send<0>(ic(iter_key, c), std::move(bag), out);
            ttg::send<1>(ic(iter_key, c), std::move(dtile), out);
            ttg::send<2>(iter_key, item, out);
          }
        },
        ttg::edges(e_diag_done),
        ttg::edges(e_L_tile_in, e_d_tile_in, e_argmax_in),
        "init_phase", {"tokens"},
        {"to_L_tile", "to_d_tile", "to_argmax"});

    // ---- Argmax reducer ----
    auto argmax = ttg::make_tt(
        [&](const int& iter, ArgmaxItem&& acc,
            std::tuple<ttg::Out<int, ArgmaxResult>>& out) {
          const long t0 = now_ns();
          ArgmaxResult r;
          r.val = acc.val;
          r.p = acc.p;
          r.pivot_sp_idx =
              drv.D.pairs[acc.p].shell_pair_idx;
          ttg::send<0>(iter, r, out);
          prof.ns_argmax.fetch_add(now_ns() - t0,
                                   std::memory_order_relaxed);
        },
        ttg::edges(e_argmax_in), ttg::edges(e_pivot_decided),
        "argmax_reducer", {"per-chunk argmax"}, {"global argmax"});

    // ---- iter_dispatch ----
    auto iter_dispatch = ttg::make_tt(
        [&, nsp, Nc, tol_local = tol](
            const int& iter, const ArgmaxResult& r,
            std::tuple<ttg::Out<IterSp, DispatchInfo>,
                       ttg::Out<IterChunk, DispatchInfo>,
                       ttg::Out<int, void>>& out) {
          const long t0 = now_ns();
          if (r.val < tol_local) {
            // Converged. Fire terminate AND fan out a drain DispatchInfo so
            // the (already-emitted) L_tile/d_tile values for this iter get
            // consumed by L_AB_extract and compute_l_chunk; otherwise those
            // pending tasks would block ttg::fence() forever.
            ttg::sendk<2>(iter, out);
            DispatchInfo drain{0, 0.0, /*terminate=*/true};
            for (std::size_t sp = 0; sp < nsp; ++sp) {
              ttg::send<0>(IterSp{static_cast<unsigned long>(iter), sp},
                           drain, out);
            }
            for (std::size_t c = 0; c < Nc; ++c) {
              ttg::send<1>(ic(iter, c), drain, out);
            }
            prof.ns_iter_dispatch.fetch_add(
                now_ns() - t0, std::memory_order_relaxed);
            return;
          }
          ++drv.total_iters;
          DispatchInfo di{r.pivot_sp_idx, r.val, /*terminate=*/false};
          for (std::size_t sp = 0; sp < nsp; ++sp) {
            ttg::send<0>(IterSp{static_cast<unsigned long>(iter), sp},
                         di, out);
          }
          for (std::size_t c = 0; c < Nc; ++c) {
            ttg::send<1>(ic(iter, c), di, out);
          }
          prof.ns_iter_dispatch.fetch_add(
              now_ns() - t0, std::memory_order_relaxed);
        },
        ttg::edges(e_pivot_decided),
        ttg::edges(e_compute_slab_in, e_l_ab_request, e_terminate),
        "iter_dispatch", {"argmax"},
        {"to_compute_slab", "to_l_ab_extract", "to_terminate"});

    // ---- compute_slab ----
    auto compute_slab = ttg::make_tt(
        [&](const IterSp& key, const DispatchInfo& di,
            std::tuple<ttg::Out<int, PivotSlabs>>& out) {
          const long t0 = now_ns();
          const auto iter = static_cast<int>(key[0]);
          const auto sp_idx = static_cast<std::size_t>(key[1]);
          PivotSlabs one;
          if (di.terminate) {
            PivotSlab marker;
            marker.terminate = true;
            one.push_back(std::move(marker));
          } else {
            auto slab = drv.compute_pivot_slab(sp_idx, di.pivot_sp_idx);
            slab.X_max_now = di.X_max_now;
            one.push_back(std::move(slab));
          }
          ttg::send<0>(iter, std::move(one), out);
          prof.ns_compute_slab.fetch_add(
              now_ns() - t0, std::memory_order_relaxed);
        },
        ttg::edges(e_compute_slab_in), ttg::edges(e_pivot_collect),
        "compute_slab", {"sp/pivot_sp"}, {"slab"});

    // ---- L_AB_extract ----
    // For each chunk, take its L_tile at the current iter (input) and
    // emit the slice of rows that overlaps the pivot AB row range. This
    // is read-only on L_tile_in.
    auto l_ab_extract = ttg::make_tt(
        [&](const IterChunk& key,
            const LStack& Lbag,
            const DispatchInfo& di,
            std::tuple<ttg::Out<int, LABContributions>>& out) {
          const long t0 = now_ns();
          const auto iter = static_cast<int>(key[0]);
          const auto c = static_cast<std::size_t>(key[1]);
          // Total cols = sum of Wp_k across the bag (= M_old at this iter).
          std::size_t M_old = 0;
          for (const auto& tile : Lbag) {
            M_old += static_cast<std::size_t>(tile.dimension(1));
          }
          if (di.terminate) {
            LABContributions out_vec;
            LABContribution contrib;
            contrib.chunk = c;
            contrib.n_rows = 0;
            contrib.M_old = M_old;
            contrib.rows = TensorTile<2>::zeros(std::size_t{0}, M_old);
            out_vec.push_back(std::move(contrib));
            ttg::send<0>(iter, std::move(out_vec), out);
            prof.ns_l_ab_extract.fetch_add(
                now_ns() - t0, std::memory_order_relaxed);
            return;
          }
          const auto& pvsp = drv.D.shell_pairs[di.pivot_sp_idx];
          const std::size_t ab_lo = pvsp.pair_begin;
          const std::size_t ab_hi = pvsp.pair_begin + pvsp.npairs;

          auto [olo, ohi] = drv.layout.overlap(c, ab_lo, ab_hi);
          LABContributions out_vec;
          LABContribution contrib;
          contrib.chunk = c;
          contrib.M_old = M_old;
          if (olo < ohi) {
            const auto nrow = static_cast<Eigen::Index>(ohi - olo);
            contrib.n_rows = static_cast<std::size_t>(nrow);
            contrib.row_offset_in_AB = olo - ab_lo;
            const auto local_lo =
                static_cast<Eigen::Index>(olo - drv.layout.lo(c));
            contrib.rows = TensorTile<2>::zeros(
                static_cast<std::size_t>(nrow), M_old);
            auto rows_m = contrib.rows.as_matrix();
            // Pack each L bag tile's AB-row slice as a column block of
            // `rows`, in the same left-to-right order as the bag itself.
            Eigen::Index col_off = 0;
            for (const auto& tile : Lbag) {
              const auto Wp_k = tile.dimension(1);
              if (Wp_k == 0) continue;
              rows_m.middleCols(col_off, Wp_k) =
                  tile.as_matrix().middleRows(local_lo, nrow);
              col_off += Wp_k;
            }
          } else {
            contrib.n_rows = 0;
            contrib.row_offset_in_AB = 0;
            contrib.rows = TensorTile<2>::zeros(std::size_t{0}, M_old);
          }
          out_vec.push_back(std::move(contrib));
          ttg::send<0>(iter, std::move(out_vec), out);
          prof.ns_l_ab_extract.fetch_add(
              now_ns() - t0, std::memory_order_relaxed);
        },
        ttg::edges(e_L_tile_in, e_l_ab_request),
        ttg::edges(e_l_ab_collect),
        "L_AB_extract", {"L_tile_in", "pivot_sp_idx"}, {"l_ab"});

    // ---- gather_pivot ----
    // Two streaming inputs: PivotSlabs (nsp arrivals) and LABContributions
    // (Nc arrivals). Once both complete for a given iter, build Q + R_AB,
    // run pivoted Cholesky, fan out per-chunk Q tiles + IterSetup.
    auto gather_pivot = ttg::make_tt(
        [&, P, Nc, tol_local = tol](
            const int& iter,
            PivotSlabs&& slabs,
            LABContributions&& labs,
            std::tuple<ttg::Out<IterChunk, IterSetup>,
                       ttg::Out<IterChunk, TensorTile<2>>>& out) {
          const long t0 = now_ns();
          // Drain pass: any slab with terminate=true means iter_dispatch
          // converged; fan out a terminate IterSetup to each chunk so they
          // can consume their stale L_tile/d_tile.
          const bool drain =
              !slabs.empty() && slabs.front().terminate;
          if (drain) {
            IterSetup setup;
            setup.terminate = true;
            for (std::size_t c = 0; c < Nc; ++c) {
              const std::size_t nrow = drv.layout.size_of(c);
              TensorTile<2> Qchunk = TensorTile<2>::zeros(nrow, std::size_t{0});
              ttg::send<0>(ic(iter, c), setup, out);
              ttg::send<1>(ic(iter, c), std::move(Qchunk), out);
            }
            prof.ns_gather_setup.fetch_add(
                now_ns() - t0, std::memory_order_relaxed);
            return;
          }
          const std::size_t W = slabs.empty() ? 0 : slabs.front().W;
          const std::size_t pivot_sp =
              slabs.empty() ? 0 : slabs.front().pivot_sp_idx;
          // All chunks ship the same M_old (sum of L bag column counts).
          std::size_t M_old = 0;
          for (auto& lab : labs) M_old = std::max(M_old, lab.M_old);
          // Assemble Q (P x W).
          Eigen::MatrixXd Q(static_cast<Eigen::Index>(P),
                            static_cast<Eigen::Index>(W));
          Q.setZero();
          for (auto& s : slabs) {
            auto Sm = s.rows.as_matrix();  // (npairs x W)
            for (std::size_t i = 0; i < s.npairs; ++i) {
              for (std::size_t j = 0; j < W; ++j) {
                Q(static_cast<Eigen::Index>(s.pair_begin + i),
                  static_cast<Eigen::Index>(j)) = Sm(i, j);
              }
            }
          }
          // Stitch L_AB into one (W × M_old) matrix.
          Eigen::MatrixXd L_AB =
              Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(W),
                                    static_cast<Eigen::Index>(M_old));
          for (auto& lab : labs) {
            if (lab.n_rows == 0 || lab.M_old == 0) continue;
            const auto off = static_cast<Eigen::Index>(lab.row_offset_in_AB);
            const auto nrow = static_cast<Eigen::Index>(lab.n_rows);
            L_AB.middleRows(off, nrow) = lab.rows.as_matrix();
          }
          // R_AB = Q[(A,B), (A,B)] - L_AB · L_AB^T
          const auto& pvsp = drv.D.shell_pairs[pivot_sp];
          const auto pair_begin = static_cast<Eigen::Index>(pvsp.pair_begin);
          Eigen::MatrixXd R_AB = Q.block(pair_begin, 0, W, W);
          if (M_old > 0) R_AB.noalias() -= L_AB * L_AB.transpose();
          R_AB = 0.5 * (R_AB + R_AB.transpose());
          // Koch's sub-pivot threshold: drop any candidate diagonal below
          // X_max(global) / 1000. The global running max diagonal is
          // carried in via DispatchInfo on every slab.
          const double X_max_now =
              slabs.empty() ? 0.0 : slabs.front().X_max_now;
          const double local_thr = X_max_now * 1e-3;

          // Pivoted Cholesky on R_AB.
          std::vector<Eigen::Index> perm(W);
          std::iota(perm.begin(), perm.end(), 0);
          Eigen::VectorXd diag = R_AB.diagonal();
          Eigen::MatrixXd U = Eigen::MatrixXd::Zero(W, W);
          Eigen::Index Wp = 0;
          for (Eigen::Index k = 0;
               k < static_cast<Eigen::Index>(W); ++k) {
            Eigen::Index best = k;
            double best_val = diag(perm[k]);
            for (Eigen::Index i = k + 1;
                 i < static_cast<Eigen::Index>(W); ++i) {
              if (diag(perm[i]) > best_val) {
                best_val = diag(perm[i]);
                best = i;
              }
            }
            if (best != k) {
              std::swap(perm[k], perm[best]);
              if (k > 0) U.row(k).head(k).swap(U.row(best).head(k));
            }
            if (best_val <= local_thr || best_val < tol_local) break;
            const double sqd = std::sqrt(best_val);
            U(k, k) = sqd;
            const Eigen::Index pk = perm[k];
            for (Eigen::Index i = k + 1;
                 i < static_cast<Eigen::Index>(W); ++i) {
              const Eigen::Index pi = perm[i];
              double r = R_AB(pi, pk);
              for (Eigen::Index m = 0; m < k; ++m) r -= U(i, m) * U(k, m);
              U(i, k) = r / sqd;
              diag(pi) -= U(i, k) * U(i, k);
            }
            ++Wp;
          }

          // Build IterSetup.
          IterSetup setup;
          setup.pivot_sp_idx = pivot_sp;
          setup.M_old = M_old;
          setup.Wp = static_cast<std::size_t>(Wp);
          setup.W = W;
          setup.perm.resize(W);
          for (std::size_t i = 0; i < W; ++i) setup.perm[i] = perm[i];
          if (Wp > 0) {
            setup.U_block = TensorTile<2>::zeros(
                static_cast<std::size_t>(Wp),
                static_cast<std::size_t>(Wp));
            auto Um = setup.U_block.as_matrix();
            Um = U.topLeftCorner(Wp, Wp);
            setup.L_AB_chosen = TensorTile<2>::zeros(
                static_cast<std::size_t>(Wp), M_old);
            if (M_old > 0) {
              auto Lm = setup.L_AB_chosen.as_matrix();
              for (Eigen::Index j = 0; j < Wp; ++j) {
                Lm.row(j) = L_AB.row(perm[j]);
              }
            }
          } else {
            setup.U_block = TensorTile<2>::zeros(std::size_t{0},
                                                 std::size_t{0});
            setup.L_AB_chosen = TensorTile<2>::zeros(std::size_t{0}, M_old);
          }

          drv.total_new_columns += setup.Wp;

          if (drv.trace_each_vector) {
            std::cout << "  iter " << std::setw(5) << iter
                      << "  +" << std::setw(2) << setup.Wp
                      << " from sp=" << pivot_sp
                      << " (M_old=" << M_old << ")" << std::endl;
          }

          // Per-chunk fan-out: Q_chunk for chunk c = Q_chosen[p_lo(c)..p_hi(c)).
          // Q_chosen has W' columns indexed by perm[0..Wp-1] of original W.
          for (std::size_t c = 0; c < Nc; ++c) {
            const std::size_t cl = drv.layout.lo(c);
            const std::size_t ch = drv.layout.hi(c);
            const std::size_t nrow = ch - cl;
            TensorTile<2> Qchunk = TensorTile<2>::zeros(nrow, setup.Wp);
            if (setup.Wp > 0) {
              auto Qm = Qchunk.as_matrix();
              for (Eigen::Index j = 0; j < Wp; ++j) {
                Qm.col(j) = Q.col(perm[j]).segment(
                    static_cast<Eigen::Index>(cl),
                    static_cast<Eigen::Index>(nrow));
              }
            }
            ttg::send<0>(ic(iter, c), setup, out);
            ttg::send<1>(ic(iter, c), std::move(Qchunk), out);
          }
          prof.ns_gather_setup.fetch_add(
              now_ns() - t0, std::memory_order_relaxed);
        },
        ttg::edges(e_pivot_collect, e_l_ab_collect),
        ttg::edges(e_setup, e_q_chunk),
        "gather_pivot", {"slabs", "L_AB"}, {"setup", "Q_chunk"});

    // ---- compute_l_chunk ----
    // 4 inputs at the same key (iter, chunk). All persistent values flow
    // through edges; on output we relabel iter->iter+1 for the loopback.
    auto compute_l_chunk = ttg::make_tt(
        [&](const IterChunk& key,
            LStack&& L_in,
            const TensorTile<1>& d_in,
            const TensorTile<2>& Q_chunk,
            const IterSetup& setup,
            std::tuple<ttg::Out<IterChunk, LStack>,
                       ttg::Out<IterChunk, TensorTile<1>>,
                       ttg::Out<int, ArgmaxItem>,
                       ttg::Out<std::size_t, LStack>>& out) {
          const long t0 = now_ns();
          const auto iter = static_cast<int>(key[0]);
          const auto c = static_cast<std::size_t>(key[1]);
          // Drain pass: consume L_in/d_in/Q_chunk/setup for the converged
          // iter and ship L_in (the bag) to collect_final_L on rank 0,
          // which repacks the global L for validation. Without this,
          // chunks owned by other ranks would never reach rank 0's
          // CholeskyDriver in distributed runs.
          if (setup.terminate) {
            ttg::send<3>(c, std::move(L_in), out);
            (void)d_in; (void)Q_chunk;
            return;
          }
          const std::size_t cl = drv.layout.lo(c);
          const std::size_t nrow = drv.layout.size_of(c);
          const auto Wp = static_cast<Eigen::Index>(setup.Wp);

          TensorTile<1> d_out = TensorTile<1>::zeros(nrow);
          std::memcpy(d_out.tensor_data(), d_in.tensor_data(),
                      nrow * sizeof(double));

          if (setup.Wp == 0) {
            // No new columns; pass the bag through unchanged.
          } else {
            // L_new = Q_chunk - sum_k L_in[k] · L_AB_chosen[col_off_k, Wp_k]^T,
            // then divided by U_block^{-T}.
            Eigen::MatrixXd L_new(static_cast<Eigen::Index>(nrow), Wp);
            L_new = Q_chunk.as_matrix();
            auto LAB_chosen_m = setup.L_AB_chosen.as_matrix();  // (Wp, M_old)
            Eigen::Index col_off = 0;
            for (std::size_t k = 0; k < L_in.size(); ++k) {
              const auto& tile = L_in[k];
              const auto Wp_k = tile.dimension(1);
              if (Wp_k == 0) continue;
              L_new.noalias() -=
                  tile.as_matrix() *
                  LAB_chosen_m.middleCols(col_off, Wp_k).transpose();
              col_off += Wp_k;
            }
            // Solve U * L_new^T = residual^T → L_new = residual · U^{-T}.
            auto Um = setup.U_block.as_matrix();  // Wp × Wp lower triangular
            Eigen::MatrixXd L_new_T = L_new.transpose();
            Um.triangularView<Eigen::Lower>().solveInPlace(L_new_T);
            L_new = L_new_T.transpose();

            // d_out -= rowwise sum-of-squares of L_new
            auto d_out_v = d_out.as_vector();
            d_out_v.noalias() -= L_new.cwiseAbs2().rowwise().sum();
            d_out_v = d_out_v.cwiseMax(0.0);

            // Append L_new as the next bag tile. This is a single small
            // (nrow × Wp) allocation, NOT a (nrow × M_new) allocation
            // followed by a memcpy of the previous tile.
            TensorTile<2> new_tile = TensorTile<2>::zeros(
                nrow, static_cast<std::size_t>(Wp));
            new_tile.as_matrix() = L_new;
            L_in.push_back(std::move(new_tile));
          }

          // Per-chunk argmax over d_out for argmax_reducer.
          ArgmaxItem item;
          item.val = -1.0;
          item.p = cl;
          for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nrow); ++i) {
            const double v = d_out.tensor_data()[i];
            if (v > item.val) {
              item.val = v;
              item.p = cl + static_cast<std::size_t>(i);
            }
          }
          ttg::send<0>(ic(iter + 1, c), std::move(L_in), out);
          ttg::send<1>(ic(iter + 1, c), std::move(d_out), out);
          ttg::send<2>(iter + 1, item, out);

          prof.ns_chunk_compute.fetch_add(
              now_ns() - t0, std::memory_order_relaxed);
          prof.n_chunk_calls.fetch_add(1, std::memory_order_relaxed);
          prof.n_new_columns.fetch_add(static_cast<int>(setup.Wp),
                                       std::memory_order_relaxed);
          {
            std::lock_guard<std::mutex> lk(prof.tids_mu);
            prof.tids_chunk.insert(std::this_thread::get_id());
          }
        },
        ttg::edges(e_L_tile_in, e_d_tile_in, e_q_chunk, e_setup),
        ttg::edges(e_L_tile_in, e_d_tile_in, e_argmax_in, e_final_L),
        "compute_l_chunk",
        {"L_in", "d_in", "Q_chunk", "setup"},
        {"L_out", "d_out", "argmax_out", "final_L"});

    // ---- terminate ----
    auto terminate = ttg::make_tt(
        [&](const int& iter) {
          if (drv.verbose) {
            std::cout << "[eri-pch] converged at iter " << iter
                      << ": rank = " << drv.total_new_columns
                      << ", iters = " << drv.total_iters
                      << " (out of " << drv.D.nshell_pairs() << " possible)"
                      << std::endl;
          }
        },
        ttg::edges(e_terminate), ttg::edges(),
        "terminate", {"iter"}, {});

    // ---- collect_final_L ----
    // Per-chunk drain emits its L bag at key=c. This task pins all bags
    // to rank 0 so they land in rank-0's CholeskyDriver for the repack.
    auto collect_final_L = ttg::make_tt(
        [&](const std::size_t& c, LStack&& bag) {
          std::lock_guard<std::mutex> lk(drv.final_mu);
          if (c < drv.final_L_bags.size()) {
            drv.final_L_bags[c] = std::move(bag);
          }
        },
        ttg::edges(e_final_L), ttg::edges(),
        "collect_final_L", {"L_bag"}, {});

    // ---- streaming reducers ----
    init_phase->template set_input_reducer<0>(
        [](int& acc, const int& in) { acc += in; });
    init_phase->template set_static_argstream_size<0>(nsp);

    argmax->template set_input_reducer<0>(
        [](ArgmaxItem& acc, const ArgmaxItem& in) {
          if (in.val > acc.val) acc = in;
        });
    argmax->template set_static_argstream_size<0>(Nc);

    gather_pivot->template set_input_reducer<0>(
        [](PivotSlabs& acc, const PivotSlabs& in) {
          for (const auto& s : in) acc.push_back(s);
        });
    gather_pivot->template set_static_argstream_size<0>(nsp);
    gather_pivot->template set_input_reducer<1>(
        [](LABContributions& acc, const LABContributions& in) {
          for (const auto& s : in) acc.push_back(s);
        });
    gather_pivot->template set_static_argstream_size<1>(Nc);

    // ---- distribution of work across MPI ranks ----
    //
    // Control plane (single-key tasks) is pinned to rank 0:
    //   - drv.d_init is built and prescreened on rank 0
    //   - iter_dispatch / argmax_reducer / gather_pivot all touch
    //     drv.{total_iters, total_new_columns}, which are local state
    //   - terminate's verbose print reads those counters
    //
    // Data plane (per-chunk and per-shell-pair tasks) distributes by
    // key. Chunks are pinned by chunk index so the L bag never travels
    // between ranks across iters; shell pairs round-robin so integral
    // compute parallelizes across nodes.
    const int nranks = static_cast<int>(ttg::default_execution_context().size());
    auto chunk_keymap = [nranks](const IterChunk& key) {
      return static_cast<int>(static_cast<std::size_t>(key[1]) % nranks);
    };
    auto sp_keymap = [nranks](const IterSp& key) {
      return static_cast<int>(static_cast<std::size_t>(key[1]) % nranks);
    };

    // diag_compute writes drv.d_init via shared-memory side-effects; pin
    // to rank 0 so init_phase (also on rank 0) sees the complete diag.
    // A v6 follow-up would have compute_diag emit its slice on an edge
    // and init_phase reduce them into a local d_init, freeing diag
    // compute to distribute.
    diag_drive->set_keymap([]() { return 0; });
    diag_compute->set_keymap([](const std::size_t&) { return 0; });
    init_phase->set_keymap([](const int&) { return 0; });
    argmax->set_keymap([](const int&) { return 0; });
    iter_dispatch->set_keymap([](const int&) { return 0; });
    compute_slab->set_keymap(sp_keymap);
    l_ab_extract->set_keymap(chunk_keymap);
    gather_pivot->set_keymap([](const int&) { return 0; });
    compute_l_chunk->set_keymap(chunk_keymap);
    terminate->set_keymap([](const int&) { return 0; });
    collect_final_L->set_keymap([](const std::size_t&) { return 0; });

    t_diag_drive       = std::move(diag_drive);
    t_diag_compute     = std::move(diag_compute);
    t_init_phase       = std::move(init_phase);
    t_argmax           = std::move(argmax);
    t_iter_dispatch    = std::move(iter_dispatch);
    t_compute_slab     = std::move(compute_slab);
    t_l_ab_extract     = std::move(l_ab_extract);
    t_gather_pivot     = std::move(gather_pivot);
    t_compute_l_chunk  = std::move(compute_l_chunk);
    t_terminate        = std::move(terminate);
    t_collect_final_L  = std::move(collect_final_L);

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
    std::ofstream out("eri-pch_v5.dot");
    out << graph_str;
    std::cout << "[eri-pch] wrote TTG graph to eri-pch_v5.dot" << std::endl;
  }

  flow.run();

  if (drv.verbose) {
    const auto ms = [](long ns) { return ns / 1.0e6; };
    std::cout << "[eri-pch] profile (cumulative across worker threads):\n"
              << "  iter_dispatch    : " << std::fixed << std::setprecision(1)
              << ms(flow.prof.ns_iter_dispatch.load()) << " ms\n"
              << "  argmax_reducer   : "
              << ms(flow.prof.ns_argmax.load()) << " ms\n"
              << "  compute_slab     : "
              << ms(flow.prof.ns_compute_slab.load()) << " ms\n"
              << "  L_AB_extract     : "
              << ms(flow.prof.ns_l_ab_extract.load()) << " ms\n"
              << "  gather setup     : "
              << ms(flow.prof.ns_gather_setup.load()) << " ms\n"
              << "  chunk compute    : "
              << ms(flow.prof.ns_chunk_compute.load()) << " ms over "
              << flow.prof.n_chunk_calls.load() << " calls on "
              << flow.prof.tids_chunk.size() << " threads ("
              << flow.prof.n_new_columns.load() << " columns)"
              << std::defaultfloat << std::endl;
  }

  // Reconstruct global L from the per-chunk final L bags snapshotted by
  // compute_l_chunk's drain pass. Each bag is a list of per-iter (P_chunk,
  // Wp_k) tiles; M = sum of Wp_k. All chunks have the same Wp_k sequence.
  const std::size_t P = drv.D.npairs();
  std::size_t M = 0;
  for (auto& bag : drv.final_L_bags) {
    if (!bag.empty()) {
      M = 0;
      for (auto& tile : bag) M += static_cast<std::size_t>(tile.dimension(1));
      break;
    }
  }
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(P, M);
  for (std::size_t c = 0; c < drv.final_L_bags.size(); ++c) {
    const auto& bag = drv.final_L_bags[c];
    if (bag.empty()) continue;
    const auto lo = static_cast<Eigen::Index>(drv.layout.lo(c));
    const auto nrow = static_cast<Eigen::Index>(drv.layout.size_of(c));
    Eigen::Index col_off = 0;
    for (const auto& tile : bag) {
      const auto Wp_k = tile.dimension(1);
      if (Wp_k == 0) continue;
      L.block(lo, col_off, nrow, Wp_k) = tile.as_matrix();
      col_off += Wp_k;
    }
  }
  return L;
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

  eri_pch::CholeskyDriver drv(std::move(D), eri_pch::default_chunk_count());
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
    if (Lmat.cols() > 0) {
      eri_pch::EnginePool refpool(drv.D.basis.max_nprim(), drv.D.basis.max_l());
      eri_pch::validate_decomposition(drv.D, refpool, Lmat, args.tol);
    } else {
      std::cout << "[validate] skipped (no L matrix produced)" << std::endl;
    }
  }

  libint2::finalize();
  ttg::finalize();
  return 0;
}
