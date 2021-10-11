#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>
#ifdef BLOCK_SPARSE_GEMM
#include <btas/features.h>
#ifdef BTAS_IS_USABLE
#include <btas/btas.h>
#include <btas/optimize/contract.h>
#include <btas/util/mohndle.h>
#else  // defined(BTAS_IS_USABLE)
#error "btas/features.h does not define BTAS_IS_USABLE ... broken BTAS?"
#endif  // defined(BTAS_IS_USABLE)
#endif  // defined(BLOCK_SPARSE_GEMM)

#include <sys/time.h>
#include <boost/graph/rmat_graph_generator.hpp>
#if !defined(BLOCK_SPARSE_GEMM)
#include <boost/graph/directed_graph.hpp>
#include <boost/random/linear_congruential.hpp>
#include <unsupported/Eigen/SparseExtra>
#endif

#ifdef BSPMM_HAS_LIBINT
#include <libint2.hpp>
#include <thread>
#endif

// TA is only usable if MADNESS backend is used
#if defined(BSPMM_HAS_TILEDARRAY) && defined(TTG_USE_MADNESS)
# define BSPMM_BUILD_TA_TEST
# include <tiledarray.h>
# include <TiledArray/pmap/user_pmap.h>
#endif

#include "ttg.h"

using namespace ttg;

#include "ttg/util/future.h"

#include "ttg/util/bug.h"

#include "active-set-strategy.h"

#if defined(BLOCK_SPARSE_GEMM)
// shallow-copy storage
using storage_type = btas::mohndle<btas::varray<double>, btas::Handle::shared_ptr>;
// deep-copy storage
//using storage_type = btas::varray<double>;
# ifndef BSPMM_BUILD_TA_TEST  // TA overloads btas's impl of btas::dot with its own, but must use TA::Range
using blk_t = btas::Tensor<double, btas::DEFAULT::range, storage_type>;
# else
using blk_t = btas::Tensor<double, TA::Range, storage_type>;
# endif

#if defined(TTG_USE_PARSEC)
namespace ttg {
  template <>
  struct SplitMetadataDescriptor<blk_t> {
    // TODO: this is a quick and dirty approach.
    //   - blk_t could have any number of dimensions, this code only works for 2 dim blocks
    //   - we use Blk{} to send a control flow in some tasks below, these blocks have only
    //     1 dimension (of size 0), to code this, we set the second dimension to 0 in our
    //     quick and dirty linearization, then have a case when we create the object
    //   - when we create the object with the metadata, we use a constructor that initializes
    //     the data to 0, which is useless: the data could be left uninitialized
    static auto get_metadata(const blk_t &b) {
      std::pair<int, int> dim{0, 0};
      if (!b.empty()) {
        assert(b.range().extent().size() == 2);
        std::get<0>(dim) = (int)b.range().extent(0);
        std::get<1>(dim) = (int)b.range().extent(1);
      }
      return dim;
    }
    static auto get_data(blk_t &b) {
      if (!b.empty())
        return boost::container::small_vector<iovec, 1>(1, iovec{b.size() * sizeof(double), b.data()});
      else
        return boost::container::small_vector<iovec, 1>{};
    }
    static auto create_from_metadata(const std::pair<int, int> &meta) {
      if (meta != std::pair{0, 0})
        return blk_t(btas::Range(std::get<0>(meta), std::get<1>(meta)), 0.0);
      else
        return blk_t{};
    }
  };
}  // namespace ttg
#endif /* TTG_USE_PARSEC */

// declare btas::Tensor serializable by Boost
#include "ttg/serialization/backends/boost.h"
namespace ttg::detail {
  // BTAS defines all of its Boost serializers in boost::serialization namespace ... as explained in
  // ttg/serialization/boost.h such functions are not detectable via SFINAE, so must explicitly define serialization
  // traits here
  template <typename Archive>
  inline static constexpr bool is_boost_serializable_v<Archive, blk_t> = is_boost_archive_v<Archive>;
  template <typename Archive>
  inline static constexpr bool is_boost_serializable_v<Archive, const blk_t> = is_boost_archive_v<Archive>;
}  // namespace ttg::detail

#else
using blk_t = double;
#endif
template <typename T = blk_t>
using SpMatrix = Eigen::SparseMatrix<T>;
template <typename T = blk_t>
using SpMatrixTriplet = Eigen::Triplet<T>;  // {row,col,value}

#ifdef BLOCK_SPARSE_GEMM

#if __has_include(<madness/world/archive.h>)

#include <madness/world/archive.h>

#endif  // __has_include(<madness/world/archive.h>)

namespace btas {
  template <typename T_, class Range_, class Store_>
  inline btas::Tensor<T_, Range_, Store_> operator*(const btas::Tensor<T_, Range_, Store_> &A,
                                                    const btas::Tensor<T_, Range_, Store_> &B) {
    btas::Tensor<T_, Range_, Store_> C;
    btas::contract(1.0, A, {1, 2}, B, {2, 3}, 0.0, C, {1, 3});
    return C;
  }

  template <typename T_, class Range_, class Store_>
  btas::Tensor<T_, Range_, Store_> gemm(btas::Tensor<T_, Range_, Store_> &&C, const btas::Tensor<T_, Range_, Store_> &A,
                                        const btas::Tensor<T_, Range_, Store_> &B) {
    using array = btas::DEFAULT::index<int>;
    if (C.empty()) {
      C = btas::Tensor<T_, Range_, Store_>(btas::Range(A.range().extent(0), B.range().extent(1)), 0.0);
    }
    btas::contract(1.0, A, {1, 2}, B, {2, 3}, 1.0, C, {1, 3});
    return std::move(C);
  }
}  // namespace btas
#endif  // defined(BLOCK_SPARSE_GEMM)
double gemm(double C, double A, double B) { return C + A * B; }
/////////////////////////////////////////////

// template <typename _Scalar, int _Options, typename _StorageIndex>
// struct colmajor_layout;
// template <typename _Scalar, typename _StorageIndex>
// struct colmajor_layout<_Scalar, Eigen::ColMajor, _StorageIndex> : public std::true_type {};
// template <typename _Scalar, typename _StorageIndex>
// struct colmajor_layout<_Scalar, Eigen::RowMajor, _StorageIndex> : public std::false_type {};

template <std::size_t Rank>
struct Key : public std::array<long, Rank> {
  static constexpr const long max_index = 1 << 21;
  Key() = default;
  template <typename Integer>
  Key(std::initializer_list<Integer> ilist) {
    std::copy(ilist.begin(), ilist.end(), this->begin());
  }
  std::size_t hash() const {
    static_assert(Rank == 2 || Rank == 3 || Rank == 4, "Key<Rank>::hash only implemented for Rank={2,3,4}");
    if (Rank == 4) {
      return (((*this)[0] * (1 << 15) + (*this)[1]) * (1 << 15) + (*this)[2]) * (1 << 15) + (*this)[3];
    }
    return Rank == 2 ? (*this)[0] * max_index + (*this)[1]
                     : ((*this)[0] * max_index + (*this)[1]) * max_index + (*this)[2];
  }
};

template <std::size_t Rank>
std::ostream &operator<<(std::ostream &os, const Key<Rank> &key) {
  os << "{";
  for (size_t i = 0; i != Rank; ++i) os << key[i] << (i + 1 != Rank ? "," : "");
  os << "}";
  return os;
}

// block-cyclic map of tile index {i,j} onto the (2d) PxQ grid
// return process rank, obtained as col-major map of the process grid coordinate
inline int tile2rank(int i, int j, int P, int Q) {
  int p = (i % P);
  int q = (j % Q);
  int pq = (q * P) + p;
  return pq;
}

// flow (move?) data into an existing SpMatrix on rank 0
template <typename Blk = blk_t>
class Write_SpMatrix : public Op<Key<2>, std::tuple<>, Write_SpMatrix<Blk>, const Blk> {
 public:
  using baseT = Op<Key<2>, std::tuple<>, Write_SpMatrix<Blk>, const Blk>;

  template <typename Keymap>
  Write_SpMatrix(SpMatrix<Blk> &matrix, Edge<Key<2>, Blk> &in, Keymap &&keymap)
      : baseT(edges(in), edges(), "write_spmatrix", {"Cij"}, {}, keymap), matrix_(matrix) {}

  void op(const Key<2> &key, typename baseT::input_values_tuple_type &&elem, std::tuple<> &) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (ttg::tracing()) {
      auto &w = get_default_world();
      ttg::print("rank =", w.rank(), "/ thread_id =", reinterpret_cast<std::uintptr_t>(pthread_self()),
                 "spmm.cc Write_SpMatrix wrote {", key[0], ",", key[1], "} = ", baseT::template get<0>(elem), " in ",
                 static_cast<void *>(&matrix_), " with mutex @", static_cast<void *>(&mtx_), " for object @",
                 static_cast<void *>(this));
    }
    values_.emplace_back(key[0], key[1], baseT::template get<0>(elem));
  }

  /// grab completion status as a future<void>
  /// \note cannot be called once this is executable
  const std::shared_future<void> &status() const {
    assert(!this->is_executable());
    if (!completion_status_) {  // if not done yet, register completion work with the world
      auto promise = std::make_shared<std::promise<void>>();
      completion_status_ = std::make_shared<std::shared_future<void>>(promise->get_future());
      ttg_register_status(this->get_world(), std::move(promise));
      ttg_register_callback(this->get_world(),
                            [this]() { this->matrix_.setFromTriplets(this->values_.begin(), this->values_.end()); });
    } else {  // if done already, commit the result
      this->matrix_.setFromTriplets(this->values_.begin(), this->values_.end());
    }
    return *completion_status_;
  }

 private:
  std::mutex mtx_;
  SpMatrix<Blk> &matrix_;
  std::vector<SpMatrixTriplet<Blk>> values_;
  mutable std::shared_ptr<std::shared_future<void>> completion_status_;
};

struct Control {
  template <typename Archive>
  void serialize(Archive &ar) {}
};

// sparse mm
template <typename Keymap = std::function<int(const Key<2> &)>, typename Blk = blk_t>
class SpMM {
 public:
  SpMM(Edge<Key<2>, Control> &progress_ctl, Edge<Key<2>, Blk> &c_flow, const SpMatrix<Blk> &a_mat,
       const SpMatrix<Blk> &b_mat, const std::vector<std::vector<long>> &a_rowidx_to_colidx,
       const std::vector<std::vector<long>> &a_colidx_to_rowidx,
       const std::vector<std::vector<long>> &b_rowidx_to_colidx,
       const std::vector<std::vector<long>> &b_colidx_to_rowidx, const std::vector<long> &mTiles,
       const std::vector<long> &nTiles, const std::vector<long> &kTiles, const Keymap &keymap, const long P,
       const long Q, size_t memory, const long forced_split, const long lookahead, const long comm_threshold)
      : a_ik_()
      , b_kj_()
      , a_rik_()
      , a_riks_()
      , b_rkj_()
      , b_rkjs_()
      , ctl_riks_()
      , ctl_rkjs_()
      , c2c_ctl_()
      , a_ijk_()
      , b_ijk_()
      , c_ijk_()
      , a_comm_ctl_()
      , b_comm_ctl_()
      , plan_(nullptr) {
    plan_ =
        std::make_shared<Plan>(a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, mTiles,
                               nTiles, kTiles, keymap, P, Q, memory, forced_split, lookahead, comm_threshold);

    lbcast_a_ = std::make_unique<LBcastA>(a_riks_, ctl_riks_, a_ijk_, plan_, keymap);
    lstore_a_ = std::make_unique<LStoreA>(a_rik_, a_riks_, a_comm_ctl_, plan_, keymap);
    bcast_a_ = std::make_unique<BcastA>(a_ik_, a_rik_, plan_, keymap);
    read_a_ = std::make_unique<Read_SpMatrix>("A", a_mat, a_comm_ctl_, a_ik_, plan_, keymap);

    lbcast_b_ = std::make_unique<LBcastB>(b_rkjs_, ctl_rkjs_, b_ijk_, plan_, keymap);
    lstore_b_ = std::make_unique<LStoreB>(b_rkj_, b_rkjs_, b_comm_ctl_, plan_, keymap);
    bcast_b_ = std::make_unique<BcastB>(b_kj_, b_rkj_, plan_, keymap);
    read_b_ = std::make_unique<Read_SpMatrix>("B", b_mat, b_comm_ctl_, b_kj_, plan_, keymap);

    coordinator_ = std::make_unique<Coordinator>(progress_ctl, ctl_riks_, ctl_rkjs_, c2c_ctl_, plan_, keymap);
    multiplyadd_ = std::make_unique<MultiplyAdd>(a_ijk_, b_ijk_, c_ijk_, c_flow, progress_ctl, plan_, keymap);

    TTGUNUSED(bcast_a_);
    TTGUNUSED(bcast_b_);
    TTGUNUSED(multiplyadd_);
  }

  long initbound() const {
    if (plan_->nb_steps() < plan_->lookahead_) return plan_->nb_steps();
    return plan_->lookahead_;
  }

  long nbphases() const { return plan_->nb_steps(); }

  std::pair<double, double> gemmsperrankperphase() const { return plan_->gemmsperrankperphase(); }

  /// Plan: group all GEMMs in blocks of efficient size
  class Plan {
   public:
    const std::vector<std::vector<long>> &a_rowidx_to_colidx_;
    const std::vector<std::vector<long>> &a_colidx_to_rowidx_;
    const std::vector<std::vector<long>> &b_rowidx_to_colidx_;
    const std::vector<std::vector<long>> &b_colidx_to_rowidx_;
    const std::vector<long> &mTiles_;
    const std::vector<long> &nTiles_;
    const std::vector<long> &kTiles_;
    const Keymap keymap_;
    const long P_;
    const long Q_;
    const long lookahead_;
    const long comm_threshold_;

   private:
    struct long_tuple_hash : public std::unary_function<std::tuple<long, long, long>, std::size_t> {
      std::size_t operator()(const std::tuple<long, long, long> &k) const {
        return static_cast<size_t>(std::get<0>(k)) | (static_cast<size_t>(std::get<1>(k)) << 21) |
               (static_cast<size_t>(std::get<2>(k)) << 21);
      }
    };

    using gemmset_t = std::set<std::tuple<long, long, long>>;
    using step_vector_t = std::vector<std::tuple<gemmset_t, long, gemmset_t>>;
    using step_per_tile_t = std::unordered_map<std::tuple<long, long, long>, std::set<long>, long_tuple_hash>;
    using bcastset_t = std::set<std::tuple<long, long>>;
    using comm_plan_t = std::vector<std::vector<bcastset_t>>;
    using step_t = std::tuple<step_vector_t, step_per_tile_t, step_per_tile_t, comm_plan_t, comm_plan_t>;

    const step_t steps_;

   public:
    Plan(const std::vector<std::vector<long>> &a_rowidx_to_colidx,
         const std::vector<std::vector<long>> &a_colidx_to_rowidx,
         const std::vector<std::vector<long>> &b_rowidx_to_colidx,
         const std::vector<std::vector<long>> &b_colidx_to_rowidx, const std::vector<long> &mTiles,
         const std::vector<long> &nTiles, const std::vector<long> &kTiles, const Keymap &keymap, const long P,
         const long Q, const size_t memory, const long forced_split, const long lookahead, const long threshold)
        : a_rowidx_to_colidx_(a_rowidx_to_colidx)
        , a_colidx_to_rowidx_(a_colidx_to_rowidx)
        , b_rowidx_to_colidx_(b_rowidx_to_colidx)
        , b_colidx_to_rowidx_(b_colidx_to_rowidx)
        , mTiles_(mTiles)
        , nTiles_(nTiles)
        , kTiles_(kTiles)
        , keymap_(keymap)
        , P_(P)
        , Q_(Q)
        , lookahead_(lookahead + 1)  // users generally understand that a lookahead of 0 still progresses
        , steps_(strategy_selector(memory, forced_split))
        , comm_threshold_(3) {
      if (tracing()) display_plan();
    }

    step_t strategy_selector(size_t memory, long forced_split) const {
      if (0 == forced_split) return active_set_strategy(memory);
      return regular_cube_strategy(forced_split);
    }

    step_t active_set_strategy(size_t memory) const {
      ActiveSetStrategy st(a_rowidx_to_colidx_, a_colidx_to_rowidx_, b_rowidx_to_colidx_, b_colidx_to_rowidx_, mTiles_,
                           nTiles_, kTiles_, memory);

      /* First, we sample a couple of places to find what 'cube' dimension is ok for this memory size */
      // Sometimes, it's worth not splitting at all, so check the corner 0,0 first
      long cube_dim, delta, cube_max, sample_min, sample_max, sample_avg, nb_sample;
      size_t cube_size, size_min, size_max, size_avg;
      std::tie(cube_dim, cube_size) = st.best_cube_dim_and_size(0, 0, 0);
      nb_sample = 1;
      if (cube_dim != st.mt && cube_dim != st.nt && cube_dim != st.kt) {
        // OK... the best cube dim to fit in (0, 0, 0) is not to fill up an entire dimension...
        // Check a couple of times: in the middle of the search space, and in corners.
        sample_min = cube_dim;
        sample_max = cube_dim;
        sample_avg = cube_dim;
        size_min = cube_size;
        size_max = cube_size;
        size_avg = cube_size;
        for (int m = 0; m < 2; m++) {
          for (int n = 0; n < 2; n++) {
            for (int k = 0; k < 2; k++) {
              if (m != 0 || n != 0 || k != 0) {
                long sample;
                size_t size;
                if (m * (st.mt - cube_dim) / 3 < 0 || n * (st.nt - cube_dim) / 3 < 0 || k * (st.kt - cube_dim) / 3 < 0)
                  continue;
                std::tie(sample, size) = st.best_cube_dim_and_size(
                    m * (st.mt - cube_dim) / 3, n * (st.nt - cube_dim) / 3, k * (st.kt - cube_dim) / 3);
                if (sample < sample_min) {
                  sample_min = sample;
                  size_min = size;
                }
                if (sample > sample_max) {
                  sample_max = sample;
                  size_max = size;
                }
                sample_avg += sample;
                size_avg += size;
                nb_sample++;
              }
            }
          }
        }
      }
      if (nb_sample > 1) {
        if (tracing() && ttg_default_execution_context().rank() == 0) {
          ttg::print("SpMM::Plan -- Sample (min/avg/max/initial): ", sample_min, "(", size_min, ") / ",
                     ((double)sample_avg / (double)nb_sample), "(", ((double)size_avg / (double)nb_sample), ") / ",
                     sample_max, "(", size_max, ") / ", cube_dim, "(", cube_size, ")");
        }
        cube_dim = (long)((double)sample_avg / (double)nb_sample);

        delta = sample_max / 2;
        if (delta < 10) delta = 10;
        cube_max = sample_max;
      } else {
        if (tracing() && ttg_default_execution_context().rank() == 0) {
          ttg::print("SpMM::Plan -- Cube is filling up the domain: ", cube_dim, "(", cube_size, ")");
        }
        delta = cube_dim / 2;
        if (delta < 10) delta = 10;
        cube_max = cube_dim;
      }

      auto excess_fct = [st](const long d) {
        return (st.mt % d) * (st.nt % d) + (st.mt % d) * (st.kt % d) + (st.kt % d) * (st.nt % d);
      };
      long excess = excess_fct(cube_dim);
      for (long d = 1; d < delta; d++) {
        long candidate = cube_dim + d;
        if (candidate <= 1) continue;
        if (candidate > cube_max) continue;
        long tmp = excess_fct(candidate);
        if (0 == tmp) {
          excess = tmp;
          cube_dim = candidate;
          break;
        }
        if (tmp > excess) {
          excess = tmp;
          cube_dim = candidate;
        }
        candidate = cube_dim - d;
        if (candidate <= 1) continue;
        tmp = excess_fct(candidate);
        if (0 == tmp) {
          excess = tmp;
          cube_dim = candidate;
          break;
        }
        if (tmp > excess) {
          excess = tmp;
          cube_dim = candidate;
        }
        if (tracing() && ttg_default_execution_context().rank() == 0) {
          ttg::print("SpMM::Plan -- candidate cube_dim is ", cube_dim, " (mt/nt/kt=", st.mt, "/", st.nt, "/", st.kt,
                     ") producing an excess of ", excess, "/", 3 * cube_dim * cube_dim, " = ",
                     (double)excess / (double)(3 * cube_dim * cube_dim));
        }
      }
      if (tracing() && ttg_default_execution_context().rank() == 0) {
        ttg::print("SpMM::Plan -- selected cube_dim is ", cube_dim, " (mt/nt/kt=", st.mt, "/", st.nt, "/", st.kt,
                   ") producing an excess of ", excess, "/", 3 * cube_dim * cube_dim, " = ",
                   (double)excess / (double)(3 * cube_dim * cube_dim));
      }

      return regular_cube_strategy(cube_dim);
    }

    step_t regular_cube_strategy(long cube_dim) const {
      step_vector_t steps;
      step_per_tile_t steps_per_tile_A;
      step_per_tile_t steps_per_tile_B;
      comm_plan_t comm_plan_A;
      comm_plan_t comm_plan_B;
      auto rank = ttg_default_execution_context().rank();
      long mt;
      long nt;
      long kt;
      mt = a_rowidx_to_colidx_.size();
      nt = b_colidx_to_rowidx_.size();
      kt = a_colidx_to_rowidx_.size();
      long tmp = b_rowidx_to_colidx_.size();
      if (tmp > kt) kt = tmp;

      long mns = (mt + cube_dim - 1) / cube_dim;
      long nns = (nt + cube_dim - 1) / cube_dim;
      long kns = (kt + cube_dim - 1) / cube_dim;
      if (tracing())
        ttg::print("On rank ", ttg_default_execution_context().rank(), " Planning with a cube_dim of ", cube_dim,
                   " over a problem of ", mt, "x", nt, "x", kt, " gives a plan of ", mns, "x", nns, "x", kns);

      std::vector<bcastset_t> a_sent;
      std::vector<bcastset_t> b_sent;
      std::vector<bcastset_t> a_in_comm_step;
      std::vector<bcastset_t> b_in_comm_step;
      a_sent.resize(P_ * Q_);
      b_sent.resize(P_ * Q_);
      a_in_comm_step.resize(P_ * Q_);
      b_in_comm_step.resize(P_ * Q_);
      comm_plan_A.resize(P_ * Q_);
      comm_plan_B.resize(P_ * Q_);
      for (long mm = 0; mm < mns; mm++) {
        for (long nn = 0; nn < nns; nn++) {
          for (long kk = 0; kk < kns; kk++) {
            gemmset_t gemms;
            gemmset_t local_gemms;
            long nb_local_gemms = 0;
            for (long m = mm * cube_dim; m < (mm + 1) * cube_dim && m < mt; m++) {
              if (m >= a_rowidx_to_colidx_.size() || a_rowidx_to_colidx_[m].empty()) continue;
              for (long k = kk * cube_dim; k < (kk + 1) * cube_dim && k < kt; k++) {
                if (k >= b_rowidx_to_colidx_.size() || b_rowidx_to_colidx_[k].empty()) continue;
                if (std::find(a_rowidx_to_colidx_[m].begin(), a_rowidx_to_colidx_[m].end(), k) ==
                    a_rowidx_to_colidx_[m].end())
                  continue;
                for (long n = nn * cube_dim; n < (nn + 1) * cube_dim && n < nt; n++) {
                  if (n >= b_colidx_to_rowidx_.size() || b_colidx_to_rowidx_[n].empty()) continue;
                  if (std::find(b_colidx_to_rowidx_[n].begin(), b_colidx_to_rowidx_[n].end(), k) ==
                      b_colidx_to_rowidx_[n].end())
                    continue;
                  auto r = keymap_(Key<2>({m, n}));
                  if (r == rank) {
                    local_gemms.insert({m, n, k});
                    nb_local_gemms++;
                  }
                  gemms.insert({m, n, k});
                  auto it = steps_per_tile_A.find(std::make_tuple(r, m, k));
                  if (it == steps_per_tile_A.end()) {
                    std::set<long> f;
                    f.insert(steps.size());
                    steps_per_tile_A.insert({std::make_tuple(r, m, k), f});
                  } else {
                    it->second.insert(steps.size());
                  }

                  it = steps_per_tile_B.find(std::make_tuple(r, k, n));
                  if (it == steps_per_tile_B.end()) {
                    std::set<long> f;
                    f.insert(steps.size());
                    steps_per_tile_B.insert({std::make_tuple(r, k, n), f});
                  } else {
                    it->second.insert(steps.size());
                  }
                  auto a_rank = keymap_(Key<2>{m, k});
                  if (a_sent[a_rank].find({m, k}) == a_sent[a_rank].end()) {
                    a_sent[a_rank].insert({m, k});
                    a_in_comm_step[a_rank].insert(std::make_pair(m, k));
                    if (a_in_comm_step[a_rank].size() >= comm_threshold_) {
                      comm_plan_A[a_rank].push_back(a_in_comm_step[a_rank]);
                      a_in_comm_step[a_rank].clear();
                    }
                  }
                  auto b_rank = keymap_(Key<2>{k, n});
                  if (b_sent[b_rank].find({k, n}) == b_sent[b_rank].end()) {
                    b_sent[b_rank].insert({k, n});
                    b_in_comm_step[b_rank].insert(std::make_pair(k, n));
                    if (b_in_comm_step[b_rank].size() >= comm_threshold_) {
                      comm_plan_B[b_rank].push_back(b_in_comm_step[b_rank]);
                      b_in_comm_step[b_rank].clear();
                    }
                  }
                }
              }
            }
            steps.emplace_back(std::make_tuple(gemms, nb_local_gemms, local_gemms));
          }
        }
      }
      for (long r = 0; r < b_in_comm_step.size(); r++) {
        if (!b_in_comm_step[r].empty()) {
          comm_plan_B[r].push_back(b_in_comm_step[r]);
          b_in_comm_step[r].clear();
        }
      }
      for (long r = 0; r < a_in_comm_step.size(); r++) {
        if (!a_in_comm_step[r].empty()) {
          comm_plan_A[r].push_back(a_in_comm_step[r]);
          a_in_comm_step[r].clear();
        }
      }
      return std::make_tuple(steps, steps_per_tile_A, steps_per_tile_B, comm_plan_A, comm_plan_B);
    }

    void display_plan() const {
      if (!tracing()) return;
      auto rank = ttg_default_execution_context().rank();
      for (long i = 0; 0 == rank && i < std::get<0>(steps_).size(); i++) {
        auto step = std::get<0>(steps_)[i];
        ttg::print("On rank", rank, "step", i, "has", std::get<1>(step), "local GEMMS and", std::get<0>(step).size(),
                   "GEMMs in total");
        if (rank == 0 && std::get<0>(step).size() < 30) {
          std::ostringstream dbg;
          dbg << "On rank " << rank << ", Step " << i << " is ";
          for (auto it : std::get<0>(step)) {
            dbg << "(" << std::get<0>(it) << "," << std::get<1>(it) << "," << std::get<2>(it) << ") ";
          }
          ttg::print(dbg.str());
        } else {
          ttg::print("On rank", rank,
                     "full plan is not displayed because it is too large or displayed by another process");
        }
      }

      const auto &steps_per_tile_A = std::get<1>(steps_);
      const auto &steps_per_tile_B = std::get<2>(steps_);
      if (0 == rank && steps_per_tile_A.size() <= 32 && steps_per_tile_B.size() <= 32) {
        ttg::print("Displaying step list per tile of A on rank", rank);
        for (auto const &it : steps_per_tile_A) {
          std::stringstream steplist;
          for (auto const s : it.second) {
            steplist << s << ",";
          }
          ttg::print("On rank", rank, "rank", std::get<0>(it.first), "runs the following steps for A(",
                     std::get<1>(it.first), ",", std::get<2>(it.first), "):", steplist.str());
        }
        ttg::print("Displaying step list per tile of B on rank", rank);
        for (auto const &it : steps_per_tile_B) {
          std::stringstream steplist;
          for (auto const s : it.second) {
            steplist << s << ",";
          }
          ttg::print("On rank", rank, "rank", std::get<0>(it.first), "runs the following steps for B(",
                     std::get<1>(it.first), ",", std::get<2>(it.first), "):", steplist.str());
        }
      } else {
        ttg::print("On rank", rank, "steps per tile of A is", steps_per_tile_A.size(), "too big to display");
        ttg::print("On rank", rank, "steps per tile of B is", steps_per_tile_B.size(), "too big to display");
      }

      const auto &comm_plan_A = std::get<3>(steps_);
      const auto &comm_plan_B = std::get<4>(steps_);
      bool display = (rank == 0);
      for (auto r = 0; display && r < comm_plan_A.size(); r++) {
        if (comm_plan_A[r].size() > 900) {
          display = false;
        }
      }
      for (auto r = 0; display && r < comm_plan_B.size(); r++) {
        if (comm_plan_B[r].size() > 900) {
          display = false;
        }
      }
      if (display) {
        for (auto r = 0; r < comm_plan_A.size(); r++) {
          ttg::print("On rank", rank, "this is the communication plan for matrix A on rank", r);
          for (auto cs = 0; cs < comm_plan_A[r].size(); cs++) {
            const auto &bs = comm_plan_A[r][cs];
            std::stringstream ss;
            for (auto x : bs) {
              ss << "A(" << std::get<0>(x) << ", " << std::get<1>(x) << ") ";
            }
            ttg::print("On rank", rank, "rank", r, "broadcasts", ss.str(), "at step", cs);
          }
        }
        for (auto r = 0; r < comm_plan_B.size(); r++) {
          ttg::print("On rank", rank, "this is the communication plan for matrix B on rank", r);
          for (auto cs = 0; cs < comm_plan_B[r].size(); cs++) {
            const auto &bs = comm_plan_B[r][cs];
            std::stringstream ss;
            for (auto x : bs) {
              ss << "B(" << std::get<0>(x) << ", " << std::get<1>(x) << ") ";
            }
            ttg::print("On rank", rank, "rank", r, "broadcasts", ss.str(), "at step", cs);
          }
        }
      } else {
        ttg::print("On rank", rank, "comm plan for A or B is too big to display (or is displayed by another rank)");
      }
    }

    /* Compute the length of the remaining sequence on that tile */
    int32_t prio(const Key<3> &key) const {
      const auto i = key[0];
      const auto j = key[1];
      const auto k = key[2];
      int32_t len = -1;  // will be incremented at least once
      long next_k;
      bool have_next_k;
#ifndef _NDEBUG
      std::tie(next_k, have_next_k) = compute_first_k(i, j);
      assert(have_next_k);
      assert(k >= next_k);
#endif
      next_k = k;
      do {
        std::tie(next_k, have_next_k) = compute_next_k(i, j, next_k);
        ++len;
      } while (have_next_k);
      return len;
    }

    // given {i,j} return first k such that A[i][k] and B[k][j] exist
    std::tuple<long, bool> compute_first_k(long i, long j) const {
      const auto &a_k_range = a_rowidx_to_colidx_.at(i);
      auto a_iter = a_k_range.begin();
      auto a_iter_fence = a_k_range.end();
      if (a_iter == a_iter_fence) return std::make_tuple(-1, false);
      const auto &b_k_range = b_colidx_to_rowidx_.at(j);
      auto b_iter = b_k_range.begin();
      auto b_iter_fence = b_k_range.end();
      if (b_iter == b_iter_fence) return std::make_tuple(-1, false);

      {
        auto a_colidx = *a_iter;
        auto b_rowidx = *b_iter;
        while (a_colidx != b_rowidx) {
          if (a_colidx < b_rowidx) {
            ++a_iter;
            if (a_iter == a_iter_fence) return std::make_tuple(-1, false);
            a_colidx = *a_iter;
          } else {
            ++b_iter;
            if (b_iter == b_iter_fence) return std::make_tuple(-1, false);
            b_rowidx = *b_iter;
          }
        }
        return std::make_tuple(a_colidx, true);
      }
      assert(false);
    }

    // given {i,j,k} such that A[i][k] and B[k][j] exist
    // return next k such that this condition holds
    std::tuple<long, bool> compute_next_k(long i, long j, long k) const {
      const auto &a_k_range = a_rowidx_to_colidx_.at(i);
      auto a_iter_fence = a_k_range.end();
      auto a_iter = std::lower_bound(a_k_range.begin(), a_iter_fence, k);
      assert(a_iter != a_iter_fence);
      const auto &b_k_range = b_colidx_to_rowidx_.at(j);
      auto b_iter_fence = b_k_range.end();
      auto b_iter = std::lower_bound(b_k_range.begin(), b_iter_fence, k);
      assert(b_iter != b_iter_fence);
      while (a_iter != a_iter_fence && b_iter != b_iter_fence) {
        ++a_iter;
        ++b_iter;
        if (a_iter == a_iter_fence || b_iter == b_iter_fence) return std::make_tuple(-1, false);
        auto a_colidx = *a_iter;
        auto b_rowidx = *b_iter;
        while (a_colidx != b_rowidx) {
          if (a_colidx < b_rowidx) {
            ++a_iter;
            if (a_iter == a_iter_fence) return std::make_tuple(-1, false);
            a_colidx = *a_iter;
          } else {
            ++b_iter;
            if (b_iter == b_iter_fence) return std::make_tuple(-1, false);
            b_rowidx = *b_iter;
          }
        }
        return std::make_tuple(a_colidx, true);
      }
      abort();  // unreachable
    }

    long nb_steps() const { return std::get<0>(steps_).size(); }

    std::tuple<long, long> gemm_coordinates(long i, long j, long k) const {
      long p = i % this->p();
      long q = j % this->q();
      long r = q * this->p() + p;
      for (long s = 0l; s < std::get<0>(steps_).size(); s++) {
        const gemmset_t *gs = &std::get<0>(std::get<0>(steps_)[s]);
        if (gs->find({i, j, k}) != gs->end()) {
          return std::make_tuple(r, s);
        }
      }
      abort();
      return std::make_tuple(r, -1);
    }

    struct GemmCoordinate {
      long r_;
      long c_;
      const Blk v_;

      long row() { return r_; }
      long col() { return c_; }
      const Blk &value() { return v_; }
    };

    const gemmset_t &gemms(long s) const { return std::get<0>(std::get<0>(steps_)[s]); }
    const gemmset_t &local_gemms(long s) const { return std::get<2>(std::get<0>(steps_)[s]); }

    long nb_local_gemms(long s) const { return std::get<1>(std::get<0>(steps_)[s]); }

    /// Accessors to the local broadcast steps

    long first_step_A(long r, long i, long k) const {
      const std::set<long> &sv = std::get<1>(steps_).at(std::make_tuple(r, i, k));
      return *sv.begin();
    }

    long first_step_B(long r, long k, long j) const {
      const std::set<long> &sv = std::get<2>(steps_).at(std::make_tuple(r, k, j));
      return *sv.begin();
    }

    long next_step_A(long r, long i, long k, long s) const {
      const std::set<long> &sv = std::get<1>(steps_).at(std::make_tuple(r, i, k));
      auto it = sv.find(s);
      assert(it != sv.end());
      it++;
      if (it == sv.end()) return -1;
      return *it;
    }

    long next_step_B(long r, long k, long j, long s) const {
      const std::set<long> &sv = std::get<2>(steps_).at(std::make_tuple(r, k, j));
      auto it = sv.find(s);
      assert(it != sv.end());
      it++;
      if (it == sv.end()) return -1;
      return *it;
    }

    /// Accessors to the communication plan

    long nb_comm_steps(long rank, bool is_a) const {
      const std::vector<bcastset_t> *cp;
      if (is_a) {
        cp = &std::get<3>(steps_)[rank];
      } else {
        cp = &std::get<4>(steps_)[rank];
      }
      return cp->size();
    }

    std::vector<GemmCoordinate> comms_in_comm_step(long rank, long comm_step, bool is_a,
                                                   const SpMatrix<Blk> &matrix) const {
      std::vector<GemmCoordinate> res;
      const bcastset_t *bset;
      if (is_a) {
        const auto &comm_plan = std::get<3>(steps_)[rank];
        if (comm_step >= comm_plan.size()) return res;
        bset = &(comm_plan[comm_step]);
      } else {
        const auto &comm_plan = std::get<4>(steps_)[rank];
        if (comm_step >= comm_plan.size()) return res;
        bset = &(comm_plan[comm_step]);
      }

      for (auto it : *bset) {
        long r;
        long c;
        std::tie(r, c) = it;
        const Blk v = matrix.coeff(r, c);
        res.emplace_back(GemmCoordinate{r, c, v});
      }

      return res;
    }

    long comm_next_step(long i, long j, bool is_a) const {
      const std::vector<bcastset_t> *cp;
      auto r = keymap_(Key<2>({i, j}));
      if (is_a) {
        cp = &std::get<3>(steps_)[r];
      } else {
        cp = &std::get<4>(steps_)[r];
      }
      long s;
      for (s = 0; s < cp->size() - 1; s++) {
        const bcastset_t &bs = (*cp)[s];
        if (bs.find(std::make_pair(i, j)) != bs.end()) return s + 1;
      }
      return -1;
    }

    /// Globals and statistics

    long p() const { return P_; }

    long q() const { return Q_; }

    std::pair<double, double> gemmsperrankperphase() const {
      double mean = 0.0, M2 = 0.0, delta, delta2;
      long count = 0;
      for (long phase = 0; phase < nb_steps(); phase++) {
        const gemmset_t &gemms_in_phase = gemms(phase);
        for (long rank = 0; rank < p() * q(); rank++) {
          long nbgemm_in_phase_for_rank = 0;
          for (auto g : gemms_in_phase) {
            if (keymap_(Key<2>({std::get<0>(g), std::get<1>(g)})) == rank) nbgemm_in_phase_for_rank++;
          }
          double x = (double)nbgemm_in_phase_for_rank;
          count++;
          delta = x - mean;
          mean += delta / count;
          delta2 = x - mean;
          M2 += delta * delta2;
        }
      }
      if (count > 0) {
        return std::make_pair(mean, sqrt(M2 / count));
      } else {
        return std::make_pair(mean, nan("undefined"));
      }
    }
  };

  /// Central coordinator: ensures that all progress according to the plan
  class Coordinator : public Op<Key<2>, std::tuple<Out<Key<4>, Control>, Out<Key<4>, Control>, Out<Key<2>, Control>>,
                                Coordinator, const Control> {
   public:
    using baseT =
        Op<Key<2>, std::tuple<Out<Key<4>, Control>, Out<Key<4>, Control>, Out<Key<2>, Control>>, Coordinator, const Control>;

    Coordinator(Edge<Key<2>, Control> progress_ctl, Edge<Key<4>, Control> &a_ctl, Edge<Key<4>, Control> &b_ctl,
                Edge<Key<2>, Control> &c2c_ctl, std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(fuse(progress_ctl, c2c_ctl)), edges(a_ctl, b_ctl, c2c_ctl), std::string("SpMM::Coordinator"),
                {"ctl_rs"}, {"a_ctl_riks", "b_ctl_rkjs", "ctl_rs"}, [](const Key<2> &key) { return (int)key[0]; })
        , plan_(plan)
        , keymap_(keymap) {
      baseT::template set_input_reducer<0>([](Control &&a, Control &&b) { return a; });
      auto r = ttg_default_execution_context().rank();
      for (long l = 0; l < plan_->lookahead_ && l < plan_->nb_steps(); l++) {
        if (tracing())
          ttg::print("On rank ", r, ": at bootstrap, setting the number of local GEMMS to trigger step ", l, " to 1");
        baseT::template set_argstream_size<0>(Key<2>{(long)r, l}, 1);
      }
    }

    void op(const Key<2> &key, typename baseT::input_values_tuple_type &&input,
            std::tuple<Out<Key<4>, Control>, Out<Key<4>, Control>, Out<Key<2>, Control>> &out) {
      auto r = key[0];
      auto s = key[1];
      if (s + plan_->lookahead_ < plan_->nb_steps()) {
        auto nb = plan_->nb_local_gemms(s);
        if (nb > 0) {  // We set the number of reductions before triggering the first GEMM (through trigger of bcasts)
          if (tracing())
            ttg::print("Coordinator(", r, ",", s, "): setting the number of local GEMMS to trigger step",
                       s + plan_->lookahead_, "to", nb);
          baseT::template set_argstream_size<0>(Key<2>({r, s + plan_->lookahead_}), nb);
        } else {
          if (tracing())
            ttg::print("Coordinator(", r, ",", s, "): there are 0 local GEMMS in step", s,
                       "; triggering next coordinator step", s + plan_->lookahead_);
          baseT::template set_argstream_size<0>(Key<2>({r, s + plan_->lookahead_}), 1);
          ::send<2>(Key<2>({r, s + plan_->lookahead_}), std::get<0>(input), out);
        }
      }

      struct tuple_hash : public std::unary_function<std::tuple<int, int>, std::size_t> {
        std::size_t operator()(const std::tuple<int, int> &k) const {
          return static_cast<size_t>(std::get<0>(k)) | (static_cast<size_t>(std::get<1>(k)) << 32);
        }
      };

      std::unordered_set<std::tuple<int, int>, tuple_hash> seen_a;
      std::unordered_set<std::tuple<int, int>, tuple_hash> seen_b;
      for (auto x : plan_->local_gemms(s)) {
        long gi, gj, gk;
        std::tie(gi, gj, gk) = x;
        if (seen_a.find(std::make_tuple(gi, gk)) == seen_a.end()) {
          if (tracing())
            ttg::print("On rank", r, "Coordinator(", r, ", ", s, "): Sending control to LBCastA(", r, ",", gi, ",", gk,
                       ",", s, ")");
          ::send<0>(Key<4>({r, gi, gk, s}), std::get<0>(input), out);
          seen_a.insert(std::make_tuple(gi, gk));
        }
        if (seen_b.find(std::make_tuple(gk, gj)) == seen_b.end()) {
          if (tracing())
            ttg::print("On rank", r, "Coordinator(", r, ", ", s, "): Sending control to LBCastB(", r, ",", gk, ",", gj,
                       ",", s, ")");
          ::send<1>(Key<4>({r, gk, gj, s}), std::get<0>(input), out);
          seen_b.insert(std::make_tuple(gk, gj));
        }
      }
    }

   private:
    std::shared_ptr<const Plan> plan_;
    const Keymap &keymap_;
  };

  // flow data from an existing SpMatrix
  class Read_SpMatrix : public Op<Key<2>, std::tuple<Out<Key<2>, Blk>>, Read_SpMatrix, Control> {
   public:
    using baseT = Op<Key<2>, std::tuple<Out<Key<2>, Blk>>, Read_SpMatrix, Control>;

    Read_SpMatrix(const char *label, const SpMatrix<Blk> &matrix, Edge<Key<2>, Control> &progress_ctl,
                  Edge<Key<2>, Blk> &out, std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(progress_ctl), edges(out), std::string("read_spmatrix(") + label + ")", {"progress_ctl"},
                {std::string(label) + "_mn"}, [](const Key<2> &key) { return (int)key[0]; })
        , matrix_(matrix)
        , plan_(plan)
        , is_a_(is_label_a(label))
        , keymap_(keymap) {
      baseT::template set_input_reducer<0>([](Control &&a, Control &&b) { return a; });
      auto r = ttg_default_execution_context().rank();
      if (tracing())
        ttg::print("On rank ", r, ": at bootstrap, setting the number of local bcast on ", is_a_ ? "A" : "B",
                   "to trigger comm step 0 to 1");
      baseT::template set_argstream_size<0>(Key<2>{(long)r, 0l}, 1);
      this->template in<0>()->send(Key<2>({(long)r, 0l}), Control{});
    }

    void op(const Key<2> &key, typename baseT::input_values_tuple_type &&inputs, std::tuple<Out<Key<2>, Blk>> &out) {
      auto rank = key[0];
      auto comm_step = key[1];
      auto world = ttg_default_execution_context();
      if (tracing())
        ttg::print("On Rank", world.rank(), ": Read_SpMatrix", is_a_ ? "A(" : "B(", rank, ",", comm_step,
                   ") is executing");
      auto comms = plan_->comms_in_comm_step(rank, comm_step, is_a_, matrix_);
      if (comm_step + 1 < plan_->nb_comm_steps(rank, is_a_)) {
        long nb_seen = 0;
        for (auto &it : comms) {
          if (is_a_) {
            auto i = it.row();
            auto k = it.col();
            long nb_seen_ik = 0;
            std::vector<bool> seen_rank(world.size(), false);
            if (k < plan_->b_rowidx_to_colidx_.size()) {
              for (auto jj = 0; nb_seen_ik < world.size() && jj < plan_->b_rowidx_to_colidx_[k].size(); jj++) {
                auto j = plan_->b_rowidx_to_colidx_[k][jj];
                auto r = keymap_(Key<2>({i, j}));
                if (seen_rank[r]) continue;
                seen_rank[r] = true;
                nb_seen++;
                nb_seen_ik++;
              }
            }
          } else {
            auto k = it.row();
            auto j = it.col();
            std::vector<bool> seen_rank(world.size(), false);
            long nb_seen_kj = 0;
            if (k < plan_->a_colidx_to_rowidx_.size()) {
              for (auto ii = 0; nb_seen_kj < world.size() && ii < plan_->a_colidx_to_rowidx_[k].size(); ii++) {
                auto i = plan_->a_colidx_to_rowidx_[k][ii];
                auto r = keymap_(Key<2>({i, j}));
                if (seen_rank[r]) continue;
                seen_rank[r] = true;
                nb_seen++;
                nb_seen_kj++;
              }
            }
          }
        }
        if (tracing()) {
          ttg::print("On Rank", ttg_default_execution_context().rank(), ": Read_SpMatrix", is_a_ ? "A(" : "B(", rank,
                     ",", comm_step, ") sets the number of LBCast for step", comm_step + 1, "to", nb_seen);
        }
        if (0 == nb_seen) {
          ttg::print("On Rank", ttg_default_execution_context().rank(), ": Read_SpMatrix", is_a_ ? "A(" : "B(", rank,
                     ",", comm_step, ") -- there are no local communications for this step");
          baseT::template set_argstream_size<0>(Key<2>{(long)rank, comm_step + 1}, 1);
          this->template in<0>()->send(Key<2>({(long)rank, comm_step + 1}), Control{});
        } else {
          baseT::template set_argstream_size<0>(Key<2>{(long)rank, comm_step + 1}, nb_seen);
        }
      } else if (tracing()) {
        ttg::print("On Rank", ttg_default_execution_context().rank(), ": Read_SpMatrix", is_a_ ? "A(" : "B(", rank, ",",
                   comm_step, ") this is the last comm step, not setting any reduce values for the next");
      }
      for (auto &it : comms) {
        assert(rank == keymap_(Key<2>({it.row(), it.col()})));
        if (tracing()) {
          ttg::print("On Rank", ttg_default_execution_context().rank(), ": Read_SpMatrix", is_a_ ? "A(" : "B(", rank,
                     ",", comm_step, ") provides ", is_a_ ? "A[" : "B[", it.row(), ",", it.col(), "]");
        }
        ::send<0>(Key<2>({it.row(), it.col()}), it.value(), out);
      }
    }

   private:
    const SpMatrix<Blk> &matrix_;
    std::shared_ptr<const Plan> plan_;
    const bool is_a_;
    const Keymap &keymap_;

    bool is_label_a(const char *label) const {
      if (strcmp(label, "A") == 0) {
        return true;
      } else {
        assert(strcmp(label, "B") == 0);
        return false;
      }
    }
  };

  /// broadcast A[i][k] to all procs where B[j][k]
  class BcastA : public Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastA, const Blk> {
   public:
    using baseT = Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastA, const Blk>;

    BcastA(Edge<Key<2>, Blk> &a_mn, Edge<Key<3>, Blk> &a_rik, std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(a_mn), edges(a_rik), "SpMM::BcastA", {"a_mn"}, {"a_rik"}, keymap), plan_(plan), keymap_(keymap) {}

    void op(const Key<2> &key, typename baseT::input_values_tuple_type &&a_ik, std::tuple<Out<Key<3>, Blk>> &a_rik) {
      auto world = get_default_world();
      const auto i = key[0];
      const auto k = key[1];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank", rank, "BcastA(", i, ", ", k, ")");
      // broadcast a_ik to nodes that use it (any time). Broadcast it to
      // the first LBcast_A that will use it, though.
      std::vector<Key<3>> rik_keys;
      std::vector<bool> seen_rank(world.size(), false);
      long nb_seen = 0;
      if (k < plan_->b_rowidx_to_colidx_.size()) {
        for (auto jj = 0; nb_seen < world.size() && jj < plan_->b_rowidx_to_colidx_[k].size(); jj++) {
          auto j = plan_->b_rowidx_to_colidx_[k][jj];
          auto r = keymap_(Key<2>({i, j}));
          if (seen_rank[r]) continue;
          seen_rank[r] = true;
          nb_seen++;
          if (tracing()) ttg::print("On rank", rank, "Broadcasting A[", i, ",", k, "] to rank", r);
          rik_keys.emplace_back(Key<3>({(long)r, i, k}));
        }
      }
      ::broadcast<0>(rik_keys, baseT::template get<0>(a_ik), a_rik);
    }

   private:
    std::shared_ptr<const Plan> plan_;
    const Keymap &keymap_;
  };  // class BcastA

  /// Provide a local copy of A[i][k] to all local tasks
  class LStoreA : public Op<Key<3>, std::tuple<Out<Key<4>, Blk>, Out<Key<2>, Control>>, LStoreA, Blk> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<4>, Blk>, Out<Key<2>, Control>>, LStoreA, Blk>;

    LStoreA(Edge<Key<3>, Blk> &a_rik, Edge<Key<4>, Blk> &a_riks, Edge<Key<2>, Control> &comm_ctl_a,
            std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(a_rik), edges(a_riks, comm_ctl_a), "SpMM::LStoreA", {"a_rik"}, {"a_riks", "comm_ctl_a"},
                [](const Key<3> &keys) { return keys[0]; })
        , plan_(plan)
        , keymap_(keymap) {}

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&a_rik,
            std::tuple<Out<Key<4>, Blk>, Out<Key<2>, Control>> &out) {
      auto world = get_default_world();
      const auto r = key[0];
      const auto i = key[1];
      const auto k = key[2];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank", rank, "LStoreA(", r, ",", i, ",", k, ")");

      auto s = plan_->first_step_A(r, i, k);
      if (tracing()) ttg::print("On rank", rank, "Starting with A[", i, ",", k, "] at starting step", s);
      ::send<0>(Key<4>({r, i, k, s}), baseT::template get<0>(a_rik), out);

      auto comm_s = plan_->comm_next_step(i, k, true);
      if (comm_s > -1) {
        long src = keymap_(Key<2>{i, k});
        if (tracing())
          ttg::print("On rank", rank, "Notifying Read_SpMatrix_A(", src, ",", comm_s, ") that A[", i, ",", k,
                     "] is received");
        ::send<1>(Key<2>{src, comm_s}, Control{}, out);
      }
    }

   private:
    std::shared_ptr<const Plan> plan_;
    const Keymap &keymap_;
  };  // class LStoreA

  /// broadcast A[i][k] to all local tasks that belong to this step
  class LBcastA : public Op<Key<4>, std::tuple<Out<Key<3>, Blk>, Out<Key<4>, Blk>>, LBcastA, const Blk, const Control> {
   public:
    using baseT = Op<Key<4>, std::tuple<Out<Key<3>, Blk>, Out<Key<4>, Blk>>, LBcastA, const Blk, const Control>;

    LBcastA(Edge<Key<4>, Blk> &a_riks, Edge<Key<4>, Control> &ctl_riks, Edge<Key<3>, Blk> &a_ijk,
            std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(a_riks, ctl_riks), edges(a_ijk, a_riks), "SpMM::LBcastA", {"a_riks", "ctl_riks"},
                {"a_ijk", "a_riks"}, [keymap](const Key<4> &keys) { return keys[0]; })
        , plan_(plan)
        , keymap_(keymap) {}

    void op(const Key<4> &key, typename baseT::input_values_tuple_type &&a_riks,
            std::tuple<Out<Key<3>, Blk>, Out<Key<4>, Blk>> &out) {
      auto world = get_default_world();
      const auto r = key[0];
      const auto i = key[1];
      const auto k = key[2];
      const auto s = key[3];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank", rank, "LBcastA(", r, ",", i, ",", k, ",", s, ")");
      // broadcast A[i][k] to all local GEMMs in step s, then pass the data to the next step
      std::vector<Key<3>> ijk_keys;
      for (const auto& x : plan_->local_gemms(s)) {
        long gi, gj, gk;
        std::tie(gi, gj, gk) = x;
        if (gi != i || gk != k) continue;
        ijk_keys.emplace_back(Key<3>{gi, gj, gk});
        if (tracing())
          ttg::print("On rank", rank, "Giving A[", gi, ",", gk, "]", "to GEMM(", gi, ",", gj, ",", gk, ") during step",
                     s);
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(a_riks), out);
      auto ns = plan_->next_step_A(r, i, k, s);
      if (ns > -1) {
        Key<4> next_key{r, i, k, ns};
        if (tracing())
          ttg::print("On rank", rank, "Sending A[", i, ",", k, "] to LBcastA(", r, ",", i, ",", k, ",", ns, ")");
        ::send<1>(next_key, baseT::template get<0>(a_riks), out);
      } else {
        if (tracing())
          ttg::print("On rank", rank, "not Sending A[", i, ",", k,
                     "] to LBcastA, because there is no more step using it");
      }
    }

   private:
    std::shared_ptr<const Plan> plan_;
    const Keymap &keymap_;
  };  // class LBcastA

  /// broadcast B[k][j] to all procs where A[i][k]
  class BcastB : public Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastB, const Blk> {
   public:
    using baseT = Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastB, const Blk>;

    BcastB(Edge<Key<2>, Blk> &b_mn, Edge<Key<3>, Blk> &b_rkj, std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(b_mn), edges(b_rkj), "SpMM::BcastB", {"b_mn"}, {"b_rkj"}, keymap), plan_(plan), keymap_(keymap) {}

    void op(const Key<2> &key, typename baseT::input_values_tuple_type &&b_kj, std::tuple<Out<Key<3>, Blk>> &b_rkj) {
      auto world = get_default_world();
      const auto k = key[0];
      const auto j = key[1];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank", rank, "BcastB(", k, ", ", j, ")");
      // broadcast b_kj to nodes that use it (any time). Broadcast it to
      // the first LBcast_B that will use it, though.
      std::vector<Key<3>> rkj_keys;
      std::vector<bool> seen_rank(world.size(), false);
      long nb_seen = 0;
      if (k < plan_->a_colidx_to_rowidx_.size()) {
        for (auto ii = 0; nb_seen < world.size() && ii < plan_->a_colidx_to_rowidx_[k].size(); ii++) {
          auto i = plan_->a_colidx_to_rowidx_[k][ii];
          auto r = keymap_(Key<2>({i, j}));
          if (seen_rank[r]) continue;
          seen_rank[r] = true;
          nb_seen++;
          if (tracing()) ttg::print("On rank", rank, "Broadcasting B[", k, ",", j, "] to rank", r);
          rkj_keys.emplace_back(Key<3>({(long)r, k, j}));
        }
      }
      ::broadcast<0>(rkj_keys, baseT::template get<0>(b_kj), b_rkj);
    }

   private:
    std::shared_ptr<const Plan> plan_;
    const Keymap &keymap_;
  };  // class BcastB

  /// Provide a local copy of B[k][j] to all local tasks
  class LStoreB : public Op<Key<3>, std::tuple<Out<Key<4>, Blk>, Out<Key<2>, Control>>, LStoreB, Blk> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<4>, Blk>, Out<Key<2>, Control>>, LStoreB, Blk>;

    LStoreB(Edge<Key<3>, Blk> &b_rkj, Edge<Key<4>, Blk> &b_rkjs, Edge<Key<2>, Control> &comm_ctl_b,
            std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(b_rkj), edges(b_rkjs, comm_ctl_b), "SpMM::LStoreB", {"b_rkj"}, {"b_rkjs", "comm_ctl_b"},
                [](const Key<3> &keys) { return keys[0]; })
        , plan_(plan)
        , keymap_(keymap) {}

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&b_rkj,
            std::tuple<Out<Key<4>, Blk>, Out<Key<2>, Control>> &out) {
      auto world = get_default_world();
      const auto r = key[0];
      const auto k = key[1];
      const auto j = key[2];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank", rank, "LStoreB(", r, ",", k, ",", j, ")");

      auto s = plan_->first_step_B(r, k, j);
      if (tracing()) ttg::print("On rank", rank, "Starting with B[", k, ",", j, "] at starting step", s);
      ::send<0>(Key<4>({r, k, j, s}), baseT::template get<0>(b_rkj), out);

      auto comm_s = plan_->comm_next_step(k, j, false);
      if (comm_s > -1) {
        long src = keymap_(Key<2>{k, j});
        if (tracing())
          ttg::print("On rank", rank, "Notifying Read_SpMatrix_B(", src, ",", comm_s, ") that B[", k, ",", j,
                     "] is received");
        ::send<1>(Key<2>{src, comm_s}, Control{}, out);
      }
    }

   private:
    std::shared_ptr<const Plan> plan_;
    const Keymap &keymap_;
  };  // class LStoreB

  /// broadcast B[k][j] to all local tasks that belong to this step
  class LBcastB : public Op<Key<4>, std::tuple<Out<Key<3>, Blk>, Out<Key<4>, Blk>>, LBcastB, const Blk, const Control> {
   public:
    using baseT = Op<Key<4>, std::tuple<Out<Key<3>, Blk>, Out<Key<4>, Blk>>, LBcastB, const Blk, const Control>;

    LBcastB(Edge<Key<4>, Blk> &b_rkjs, Edge<Key<4>, Control> &ctl_rkjs, Edge<Key<3>, Blk> &b_ijk,
            std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(b_rkjs, ctl_rkjs), edges(b_ijk, b_rkjs), "SpMM::LBcastB", {"b_rkjs", "ctl_rkjs"},
                {"b_ijk", "b_rkjs"}, [keymap](const Key<4> &keys) { return keys[0]; })
        , plan_(plan)
        , keymap_(keymap) {}

    void op(const Key<4> &key, typename baseT::input_values_tuple_type &&b_rkjs,
            std::tuple<Out<Key<3>, Blk>, Out<Key<4>, Blk>> &out) {
      auto world = get_default_world();
      const auto r = key[0];
      const auto k = key[1];
      const auto j = key[2];
      const auto s = key[3];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank", r, "LBcastB(", r, ",", k, ",", j, ",", s, ")");
      // broadcast B[k][j] to all local GEMMs in step s, then pass the data to the next step
      std::vector<Key<3>> ijk_keys;
      for (const auto& x : plan_->local_gemms(s)) {
        long gi, gj, gk;
        std::tie(gi, gj, gk) = x;
        if (gj != j || gk != k) continue;
        ijk_keys.emplace_back(Key<3>{gi, gj, gk});
        if (tracing())
          ttg::print("On rank", rank, "Giving B[", gk, ",", gj, "]", "to GEMM(", gi, ",", gj, ",", gk, ") during step",
                     s);
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(b_rkjs), out);
      auto ns = plan_->next_step_B(r, k, j, s);
      if (ns > -1) {
        Key<4> next_key{r, k, j, ns};
        if (tracing())
          ttg::print("On rank", rank, "Sending B[", k, ",", j, "] to LBcastB(", r, ",", k, ",", j, ",", ns, ")");
        ::send<1>(next_key, baseT::template get<0>(b_rkjs), out);
      } else {
        if (tracing())
          ttg::print("On rank", rank, "not Sending B[", k, ",", j,
                     "] to LBcastB, because there is no more step using it");
      }
    }

   private:
    std::shared_ptr<const Plan> plan_;
    const Keymap &keymap_;
  };  // class LBcastB

  /// multiply task has 3 input flows: a_ijk, b_ijk, and c_ijk, c_ijk contains the running total
  class MultiplyAdd : public Op<Key<3>, std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>, Out<Key<2>, Control>>,
                                MultiplyAdd, const Blk, const Blk, Blk> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>, Out<Key<2>, Control>>, MultiplyAdd,
                     const Blk, const Blk, Blk>;

    MultiplyAdd(Edge<Key<3>, Blk> &a_ijk, Edge<Key<3>, Blk> &b_ijk, Edge<Key<3>, Blk> &c_ijk, Edge<Key<2>, Blk> &c,
                Edge<Key<2>, Control> &progress_ctl, std::shared_ptr<const Plan> plan, Keymap keymap)
        : baseT(edges(a_ijk, b_ijk, c_ijk), edges(c, c_ijk, progress_ctl), "SpMM::MultiplyAdd",
                {"a_ijk", "b_ijk", "c_ijk"}, {"c_ij", "c_ijk", "ctl_pqs"},
                [keymap](const Key<3> &key) {
                  auto key2 = Key<2>({key[0], key[1]});
                  return keymap(key2);
                })
        , plan_(std::move(plan)) {
      this->set_priomap([=](const Key<3> &key) { return plan_->prio(key); });

      // for each i and j that belongs to this node
      // determine first k that contributes, initialize input {i,j,first_k} flow to 0
      for (auto i = 0ul; i != plan_->a_rowidx_to_colidx_.size(); ++i) {
        if (plan_->a_rowidx_to_colidx_[i].empty()) continue;
        for (auto j = 0ul; j != plan_->b_colidx_to_rowidx_.size(); ++j) {
          if (plan_->b_colidx_to_rowidx_[j].empty()) continue;

          // assuming here {i,j,k} for all k map to same node
          auto owner = keymap(Key<2>({i, j}));
          if (owner == ttg_default_execution_context().rank()) {
            if (true) {
              decltype(i) k;
              bool have_k;
              std::tie(k, have_k) = plan_->compute_first_k(i, j);
              if (have_k) {
                if (tracing())
                  ttg::print("On rank", owner, "Initializing C[", i, "][", j, "] to zero (first k ", k, ")");
#if BLOCK_SPARSE_GEMM
                Blk zero(btas::Range(plan_->mTiles_[i], plan_->nTiles_[j]), 0.0);
#else
                Blk zero{0.0};
#endif
                this->template in<2>()->send(Key<3>({i, j, k}), zero);
              } else {
                if (tracing() && plan_->a_rowidx_to_colidx_.size() * plan_->b_colidx_to_rowidx_.size() < 400)
                  ttg::print("C[", i, "][", j, "] is empty");
              }
            }
          }
        }
      }
    }

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&_ijk,
            std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>, Out<Key<2>, Control>> &result) {
      const auto i = key[0];
      const auto j = key[1];
      const auto k = key[2];
      long r, s;
      std::tie(r, s) = plan_->gemm_coordinates(i, j, k);
      long next_k;
      bool have_next_k;
      std::tie(next_k, have_next_k) = plan_->compute_next_k(i, j, k);
      if (tracing()) {
        ttg::print("On rank", ttg_default_execution_context().rank(), "step", s, "GEMM C[", i, "][", j, "]  += A[", i,
                   "][", k, "] by B[", k, "][", j, "],  next_k? ",
                   (have_next_k ? std::to_string(next_k) : "does not exist"));
      }
      // compute the contrib, pass the running total to the next flow, if needed
      // otherwise write to the result flow
      if (have_next_k) {
        ::send<1>(
            Key<3>({i, j, next_k}),
            gemm(std::move(baseT::template get<2>(_ijk)), baseT::template get<0>(_ijk), baseT::template get<1>(_ijk)),
            result);
      } else {
        ::send<0>(
            Key<2>({i, j}),
            gemm(std::move(baseT::template get<2>(_ijk)), baseT::template get<0>(_ijk), baseT::template get<1>(_ijk)),
            result);
      }
      if (s + plan_->lookahead_ < plan_->nb_steps()) {
        if (tracing()) {
          ttg::print("On rank", ttg_default_execution_context().rank(), "step", s, "Notifying coordinator of step",
                     s + plan_->lookahead_, "that GEMM(", i, ", ", j, ", ", k, ") completed");
        }
        ::send<2>(Key<2>({r, s + plan_->lookahead_}), Control{}, result);
      }
    }

   private:
    std::shared_ptr<const Plan> plan_;
  };

  Read_SpMatrix *get_reada() { return read_a_.get(); }
  Read_SpMatrix *get_readb() { return read_b_.get(); }

 private:
  Edge<Key<3>, Blk> a_ijk_;
  Edge<Key<3>, Blk> b_ijk_;
  Edge<Key<3>, Blk> c_ijk_;
  Edge<Key<2>, Blk> a_ik_;
  Edge<Key<3>, Blk> a_rik_;
  Edge<Key<4>, Blk> a_riks_;
  Edge<Key<2>, Blk> b_kj_;
  Edge<Key<3>, Blk> b_rkj_;
  Edge<Key<4>, Blk> b_rkjs_;
  Edge<Key<4>, Control> ctl_riks_;
  Edge<Key<4>, Control> ctl_rkjs_;
  Edge<Key<2>, Control> c2c_ctl_;
  Edge<Key<2>, Control> a_comm_ctl_;
  Edge<Key<2>, Control> b_comm_ctl_;
  std::unique_ptr<Coordinator> coordinator_;
  std::unique_ptr<BcastA> bcast_a_;
  std::unique_ptr<LStoreA> lstore_a_;
  std::unique_ptr<LBcastA> lbcast_a_;
  std::unique_ptr<BcastB> bcast_b_;
  std::unique_ptr<LStoreB> lstore_b_;
  std::unique_ptr<LBcastB> lbcast_b_;
  std::unique_ptr<MultiplyAdd> multiplyadd_;
  std::unique_ptr<Read_SpMatrix> read_a_;
  std::unique_ptr<Read_SpMatrix> read_b_;
  std::shared_ptr<Plan> plan_;
};

class StartupControl : public Op<void, std::tuple<Out<Key<2>, Control>>, StartupControl> {
  using baseT = Op<void, std::tuple<Out<Key<2>, Control>>, StartupControl>;
  long initbound;

 public:
  explicit StartupControl(Edge<Key<2>, Control> &ctl)
      : baseT(edges(), edges(ctl), "StartupControl", {}, {"ctl_rs"}), initbound(0) {}

  void op(std::tuple<Out<Key<2>, Control>> &out) const {
    auto world = ttg_default_execution_context();
    for (long r = 0; r < world.size(); r++) {
      for (long l = 0; l < initbound; l++) {
        Key<2> k{r, l};
        if (ttg::tracing()) ttg::print("On rank ", world.rank(), " StartupControl: enable {", r, ",", l, "}");
        ::send<0>(k, Control{}, out);
      }
    }
  }

  void start(const long _b) {
    initbound = _b;
    invoke();
  }
};

#ifdef BTAS_IS_USABLE
template <typename T_, class Range_, class Store_>
std::tuple<T_, T_> norms(const btas::Tensor<T_, Range_, Store_> &t) {
  T_ norm_2_square = 0.0;
  T_ norm_inf = 0.0;
  for (auto k : t) {
    norm_2_square += k * k;
    norm_inf = std::max(norm_inf, std::abs(k));
  }
  return std::make_tuple(norm_2_square, norm_inf);
}
#endif  // defined(BLOCK_SPARSE_GEMM)

std::tuple<double, double> norms(double t) { return std::make_tuple(t * t, std::abs(t)); }

template <typename Blk = blk_t>
std::tuple<double, double> norms(const SpMatrix<Blk> &A) {
  double norm_2_square = 0.0;
  double norm_inf = 0.0;
  for (int i = 0; i < A.outerSize(); ++i) {
    for (typename SpMatrix<Blk>::InnerIterator it(A, i); it; ++it) {
      //  cout << 1+it.row() << "\t"; // row index
      //  cout << 1+it.col() << "\t"; // col index (here it is equal to k)
      //  cout << it.value() << endl;
      auto elem = it.value();
      double elem_norm_2_square, elem_norm_inf;
      std::tie(elem_norm_2_square, elem_norm_inf) = norms(elem);
      norm_2_square += elem_norm_2_square;
      norm_inf = std::max(norm_inf, elem_norm_inf);
    }
  }
  return std::make_tuple(norm_2_square, norm_inf);
}

char *getCmdOption(char **begin, char **end, const std::string &option) {
  static char *empty = (char *)"";
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) return *itr;
  return empty;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

int cmdOptionIndex(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end) return (int)(itr - begin);
  return -1;
}

static int parseOption(std::string &option, int default_value) {
  size_t pos;
  std::string token;
  int N = default_value;
  if (option.length() == 0) return N;
  pos = option.find(':');
  if (pos == std::string::npos) {
    pos = option.length();
  }
  token = option.substr(0, pos);
  N = std::stoi(token);
  option.erase(0, pos + 1);
  return N;
}

static long parseOption(std::string &option, long default_value) {
  size_t pos;
  std::string token;
  long N = default_value;
  if (option.length() == 0) return N;
  pos = option.find(':');
  if (pos == std::string::npos) {
    pos = option.length();
  }
  token = option.substr(0, pos);
  N = std::stol(token);
  option.erase(0, pos + 1);
  return N;
}

static size_t parseOption(std::string &option, size_t default_value) {
  size_t pos;
  std::string token;
  size_t N = default_value;
  if (option.length() == 0) return N;
  pos = option.find(':');
  if (pos == std::string::npos) {
    pos = option.length();
  }
  token = option.substr(0, pos);
  N = std::stoul(token);
  option.erase(0, pos + 1);
  return N;
}

static double parseOption(std::string &option, double default_value = 0.25) {
  size_t pos;
  std::string token;
  double N = default_value;
  if (option.length() == 0) return N;
  pos = option.find(':');
  if (pos == std::string::npos) {
    pos = option.length();
  }
  token = option.substr(0, pos);
  N = std::stod(token);
  option.erase(0, pos + 1);
  return N;
}

#if !defined(BLOCK_SPARSE_GEMM)
static void initSpMatrixMarket(const std::function<int(const Key<2> &)> &keymap, const char *filename, SpMatrix<> &A,
                               SpMatrix<> &B, SpMatrix<> &C, int &M, int &N, int &K) {
  std::vector<long> sizes;
  // We load the entire matrix on each rank, but we only use the local part for the GEMM
  if (!loadMarket(A, filename)) {
    std::cerr << "Failed to load " << filename << ", bailing out..." << std::endl;
    ttg::ttg_abort();
  }
  if (0 == ttg_default_execution_context().rank()) {
    std::cout << "##MatrixMarket file " << filename << " -- " << A.rows() << " x " << A.cols() << " -- " << A.nonZeros()
              << " nnz (density: " << (float)A.nonZeros() / (float)A.rows() / (float)A.cols() << ")" << std::endl;
  }
  if (A.rows() != A.cols()) {
    B = A.transpose();
  } else {
    B = A;
  }
  C.resize(A.rows(), B.cols());
  M = (int)A.rows();
  N = (int)C.cols();
  K = (int)A.cols();
}

static void initSpRmat(const std::function<int(const Key<2> &)> &keymap, const char *opt, SpMatrix<> &A, SpMatrix<> &B,
                       SpMatrix<> &C, int &M, int &N, int &K, unsigned long seed) {
  int E;
  double a = 0.25, b = 0.25, c = 0.25, d = 0.25;
  size_t nnz = 0;

  if (nullptr == opt) {
    std::cerr << "Usage: -rmat <#nodes>[:<#edges>[:<a>[:<b>:[<c>[:<d>]]]]]" << std::endl;
    exit(1);
  }
  std::string token;
  std::string option = std::string(opt);
  N = parseOption(option, -1);
  K = N;
  M = N;

  // We build the entire sparse matrix on each rank, but use only the local part
  // on a given rank, according to keymap
  A.resize(N, N);

  E = parseOption(option, (int)(0.01 * N * N));
  a = parseOption(option, a);
  b = parseOption(option, b);
  c = parseOption(option, c);
  d = parseOption(option, d);

  if (ttg_default_execution_context().rank() == 0) {
    std::cout << "#R-MAT: " << N << " nodes, " << E << " edges, a/b/c/d = " << a << "/" << b << "/" << c << "/" << d
              << std::endl;
  }

  boost::minstd_rand gen(seed);
  boost::rmat_iterator<boost::minstd_rand, boost::directed_graph<>> rmat_it(gen, N, E, a, b, c, d);

  using triplet_t = Eigen::Triplet<blk_t>;
  std::vector<triplet_t> A_elements;
  for (int i = 0; i < N; i++) {
    nnz++;
    A_elements.emplace_back(i, i, 1.0);
  }
  for (int i = 0; i < E; i++) {
    auto x = *rmat_it++;
    if (x.first != x.second) {
      A_elements.emplace_back(x.first, x.second, 1.0);
      nnz++;
    }
  }
  A.setFromTriplets(A_elements.begin(), A_elements.end());

  B = A;
  C.resize(N, N);

  if (ttg_default_execution_context().rank() == 0) {
    std::cout << "#R-MAT: " << E << " nonzero elements, density: " << (double)nnz / (double)N / (double)N << std::endl;
  }
}

static void initSpHardCoded(const std::function<int(const Key<2> &)> &keymap, SpMatrix<> &A, SpMatrix<> &B,
                            SpMatrix<> &C, int &m, int &n, int &k) {
  m = 2;
  n = 3;
  k = 4;

  std::cout << "#HardCoded A, B, C" << std::endl;
  A.resize(m, k);
  B.resize(k, n);
  C.resize(m, n);
  // We initialize the same matrices on all the ranks, but we will use only the local part
  // following the keymap
  using triplet_t = Eigen::Triplet<blk_t>;
  std::vector<triplet_t> A_elements;
  A_elements.emplace_back(0, 1, 12.3);
  A_elements.emplace_back(0, 2, 10.7);
  A_elements.emplace_back(0, 3, -2.3);
  A_elements.emplace_back(1, 0, -0.3);
  A_elements.emplace_back(1, 2, 1.2);
  A.setFromTriplets(A_elements.begin(), A_elements.end());

  std::vector<triplet_t> B_elements;
  B_elements.emplace_back(0, 0, 12.3);
  B_elements.emplace_back(1, 0, 10.7);
  B_elements.emplace_back(3, 0, -2.3);
  B_elements.emplace_back(1, 1, -0.3);
  B_elements.emplace_back(1, 2, 1.2);
  B_elements.emplace_back(2, 2, 7.2);
  B_elements.emplace_back(3, 2, 0.2);
  B.setFromTriplets(B_elements.begin(), B_elements.end());
}
#endif

#if defined(BLOCK_SPARSE_GEMM)
static void initBlSpHardCoded(const std::function<int(const Key<2> &)> &keymap, SpMatrix<> &A, SpMatrix<> &B,
                              SpMatrix<> &C, SpMatrix<> &Aref, SpMatrix<> &Bref, bool buildRefs,
                              std::vector<long> &mTiles, std::vector<long> &nTiles, std::vector<long> &kTiles,
                              std::vector<std::vector<long>> &a_rowidx_to_colidx,
                              std::vector<std::vector<long>> &a_colidx_to_rowidx,
                              std::vector<std::vector<long>> &b_rowidx_to_colidx,
                              std::vector<std::vector<long>> &b_colidx_to_rowidx, int &m, int &n, int &k) {
  m = 2;
  n = 3;
  k = 4;

  std::cout << "#HardCoded A, B, C" << std::endl;
  A.resize(m, k);
  B.resize(k, n);
  C.resize(m, n);
  if (buildRefs) {
    Aref.resize(m, k);
    Bref.resize(k, n);
  }

  for (int mt = 0; mt < m; mt++) mTiles.push_back(128);
  for (int nt = 0; nt < n; nt++) nTiles.push_back(196);
  for (int kt = 0; kt < k; kt++) kTiles.push_back(256);

  int rank = ttg_default_execution_context().rank();

  using triplet_t = Eigen::Triplet<blk_t>;
  std::vector<triplet_t> A_elements;
  std::vector<triplet_t> Aref_elements;
#if defined(BTAS_IS_USABLE)
  if (keymap({0, 1}) == rank) {
    A_elements.emplace_back(0, 1, blk_t(btas::Range(128, 256), 12.3));
  }
  if (keymap({0, 2}) == rank) {
    A_elements.emplace_back(0, 2, blk_t(btas::Range(128, 256), 10.7));
  }
  if (keymap({0, 3}) == rank) {
    A_elements.emplace_back(0, 3, blk_t(btas::Range(128, 256), -2.3));
  }
  if (keymap({1, 0}) == rank) {
    A_elements.emplace_back(1, 0, blk_t(btas::Range(128, 256), -0.3));
  }
  if (keymap({1, 2}) == rank) {
    A_elements.emplace_back(1, 2, blk_t(btas::Range(128, 256), 1.2));
  }
  if (buildRefs && rank == 0) {
    Aref_elements.emplace_back(0, 1, blk_t(btas::Range(128, 256), 12.3));
    Aref_elements.emplace_back(0, 2, blk_t(btas::Range(128, 256), 10.7));
    Aref_elements.emplace_back(0, 3, blk_t(btas::Range(128, 256), -2.3));
    Aref_elements.emplace_back(1, 0, blk_t(btas::Range(128, 256), -0.3));
    Aref_elements.emplace_back(1, 2, blk_t(btas::Range(128, 256), 1.2));
  }
#else
  if ((buildRefs && rank == 0) || keymap({0, 1}) == rank) {
    A_elements.emplace_back(0, 1, 12.3);
  }
  if ((buildRefs && rank == 0) || keymap({0, 2}) == rank) {
    A_elements.emplace_back(0, 2, 10.7);
  }
  if ((buildRefs && rank == 0) || keymap({0, 3}) == rank) {
    A_elements.emplace_back(0, 3, -2.3);
  }
  if ((buildRefs && rank == 0) || keymap({1, 0}) == rank) {
    A_elements.emplace_back(1, 0, -0.3);
  }
  if ((buildRefs && rank == 0) || keymap({1, 2}) == rank) {
    A_elements.emplace_back(1, 2, .2);
  }
  if (buildRefs && rank == 0) {
    Aref_elements.emplace_back(0, 1, 12.3);
    Aref_elements.emplace_back(0, 2, 10.7);
    Aref_elements.emplace_back(0, 3, -2.3);
    Aref_elements.emplace_back(1, 0, -0.3);
    Aref_elements.emplace_back(1, 2, .2);
  }
#endif /* BTAS_IS_USABLE */
  a_rowidx_to_colidx.resize(2);
  a_rowidx_to_colidx[0].emplace_back(1);  // A[0][1]
  a_rowidx_to_colidx[0].emplace_back(2);  // A[0][2]
  a_rowidx_to_colidx[0].emplace_back(3);  // A[0][3]
  a_rowidx_to_colidx[1].emplace_back(0);  // A[1][0]
  a_rowidx_to_colidx[1].emplace_back(2);  // A[1][2]

  a_colidx_to_rowidx.resize(4);
  a_colidx_to_rowidx[0].emplace_back(1);  // A[1][0]
  a_colidx_to_rowidx[1].emplace_back(0);  // A[0][1]
  a_colidx_to_rowidx[2].emplace_back(0);  // A[0][2]
  a_colidx_to_rowidx[2].emplace_back(1);  // A[1][2]
  a_colidx_to_rowidx[3].emplace_back(0);  // A[0][3]

  A.setFromTriplets(A_elements.begin(), A_elements.end());
  if (buildRefs && 0 == rank) {
    Aref.setFromTriplets(Aref_elements.begin(), Aref_elements.end());
  }

  std::vector<triplet_t> B_elements;
  std::vector<triplet_t> Bref_elements;
#if defined(BTAS_IS_USABLE)
  if (keymap({0, 0}) == rank) {
    B_elements.emplace_back(0, 0, blk_t(btas::Range(256, 196), 12.3));
  }
  if (keymap({1, 0}) == rank) {
    B_elements.emplace_back(1, 0, blk_t(btas::Range(256, 196), 10.7));
  }
  if (keymap({3, 0}) == rank) {
    B_elements.emplace_back(3, 0, blk_t(btas::Range(256, 196), -2.3));
  }
  if (keymap({1, 1}) == rank) {
    B_elements.emplace_back(1, 1, blk_t(btas::Range(256, 196), -0.3));
  }
  if (keymap({1, 2}) == rank) {
    B_elements.emplace_back(1, 2, blk_t(btas::Range(256, 196), 1.2));
  }
  if (keymap({2, 2}) == rank) {
    B_elements.emplace_back(2, 2, blk_t(btas::Range(256, 196), 7.2));
  }
  if (keymap({3, 2}) == rank) {
    B_elements.emplace_back(3, 2, blk_t(btas::Range(256, 196), 0.2));
  }
  if (buildRefs && rank == 0) {
    Bref_elements.emplace_back(0, 0, blk_t(btas::Range(256, 196), 12.3));
    Bref_elements.emplace_back(1, 0, blk_t(btas::Range(256, 196), 10.7));
    Bref_elements.emplace_back(3, 0, blk_t(btas::Range(256, 196), -2.3));
    Bref_elements.emplace_back(1, 1, blk_t(btas::Range(256, 196), -0.3));
    Bref_elements.emplace_back(1, 2, blk_t(btas::Range(256, 196), 1.2));
    Bref_elements.emplace_back(2, 2, blk_t(btas::Range(256, 196), 7.2));
    Bref_elements.emplace_back(3, 2, blk_t(btas::Range(256, 196), 0.2));
  }
#else  /* BTAS_IS_USABLE */
  if (keymap({0, 0}) == rank) {
    B_elements.emplace_back(0, 0, 12.3);
  }
  if (keymap({1, 0}) == rank) {
    B_elements.emplace_back(1, 0, 10.7);
  }
  if (keymap({3, 0}) == rank) {
    B_elements.emplace_back(3, 0, -2.3);
  }
  if (keymap({1, 1}) == rank) {
    B_elements.emplace_back(1, 1, -0.3);
  }
  if (keymap({1, 2}) == rank) {
    B_elements.emplace_back(1, 2, 1.2);
  }
  if (keymap({2, 2}) == rank) {
    B_elements.emplace_back(2, 2, 7.2);
  }
  if (keymap({3, 2}) == rank) {
    B_elements.emplace_back(3, 2, 0.2);
  }
#endif /* BTAS_IS_USABLE */
  b_rowidx_to_colidx.resize(4);
  b_rowidx_to_colidx[0].emplace_back(0);  // B[0][0]
  b_rowidx_to_colidx[1].emplace_back(0);  // B[1][0]
  b_rowidx_to_colidx[1].emplace_back(1);  // B[1][1]
  b_rowidx_to_colidx[1].emplace_back(2);  // B[1][2]
  b_rowidx_to_colidx[2].emplace_back(2);  // B[2][2]
  b_rowidx_to_colidx[3].emplace_back(0);  // B[3][0]
  b_rowidx_to_colidx[3].emplace_back(2);  // B[3][2]

  b_colidx_to_rowidx.resize(3);
  b_colidx_to_rowidx[0].emplace_back(0);  // B[0][0]
  b_colidx_to_rowidx[0].emplace_back(1);  // B[1][0]
  b_colidx_to_rowidx[0].emplace_back(3);  // B[3][0]
  b_colidx_to_rowidx[1].emplace_back(1);  // B[1][1]
  b_colidx_to_rowidx[2].emplace_back(1);  // B[1][2]
  b_colidx_to_rowidx[2].emplace_back(2);  // B[2][2]
  b_colidx_to_rowidx[2].emplace_back(3);  // A[3][2]

  B.setFromTriplets(B_elements.begin(), B_elements.end());
  if (buildRefs && 0 == rank) {
    Bref.setFromTriplets(Bref_elements.begin(), Bref_elements.end());
  }
}

#if defined(BTAS_IS_USABLE)
static void initBlSpRandom(const std::function<int(const Key<2> &)> &keymap, long M, long N, long K, int minTs,
                           int maxTs, double avgDensity, SpMatrix<> &A, SpMatrix<> &B, SpMatrix<> &Aref,
                           SpMatrix<> &Bref, bool buildRefs, std::vector<long> &mTiles, std::vector<long> &nTiles,
                           std::vector<long> &kTiles, std::vector<std::vector<long>> &a_rowidx_to_colidx,
                           std::vector<std::vector<long>> &a_colidx_to_rowidx,
                           std::vector<std::vector<long>> &b_rowidx_to_colidx,
                           std::vector<std::vector<long>> &b_colidx_to_rowidx, double &average_tile_size,
                           double &Adensity, double &Bdensity, unsigned int seed, int P, int Q) {
  int rank = ttg_default_execution_context().rank();

  long ts;
  std::mt19937 gen(seed);
  std::mt19937 genv(seed + 1);

  std::uniform_int_distribution<> dist(minTs, maxTs);
  using triplet_t = Eigen::Triplet<blk_t>;
  std::vector<triplet_t> A_elements;
  std::vector<triplet_t> B_elements;
  std::vector<triplet_t> Aref_elements;
  std::vector<triplet_t> Bref_elements;

  for (long m = 0; m < M; m += ts) {
    ts = dist(gen);
    if (ts > M - m) ts = M - m;
    mTiles.push_back(ts);
  }
  for (long n = 0; n < N; n += ts) {
    ts = dist(gen);
    if (ts > N - n) ts = N - n;
    nTiles.push_back(ts);
  }
  for (long k = 0; k < K; k += ts) {
    ts = dist(gen);
    if (ts > K - k) ts = K - k;
    kTiles.push_back(ts);
  }

  A.resize((long)mTiles.size(), (long)kTiles.size());
  B.resize((long)kTiles.size(), (long)nTiles.size());
  if (buildRefs) {
    Aref.resize((long)mTiles.size(), (long)kTiles.size());
    Bref.resize((long)kTiles.size(), (long)nTiles.size());
  }

  std::uniform_int_distribution<> mDist(0, (int)mTiles.size() - 1);
  std::uniform_int_distribution<> nDist(0, (int)nTiles.size() - 1);
  std::uniform_int_distribution<> kDist(0, (int)kTiles.size() - 1);
  std::uniform_real_distribution<> vDist(-1.0, 1.0);

  size_t filling = 0;
  size_t avg_nb = 0;
  int avg_nb_nb = 0;

  struct tuple_hash : public std::unary_function<std::tuple<int, int>, std::size_t> {
    std::size_t operator()(const std::tuple<int, int> &k) const {
      return static_cast<size_t>(std::get<0>(k)) | (static_cast<size_t>(std::get<1>(k)) << 32);
    }
  };

  std::unordered_set<std::tuple<int, int>, tuple_hash> fills;

  fills.clear();
  while ((double)filling / (double)(M * K) < avgDensity) {
    int mt = mDist(gen);
    int kt = kDist(gen);

    if (fills.find({mt, kt}) != fills.end()) continue;
    fills.insert({mt, kt});

    if (mt >= a_rowidx_to_colidx.size()) a_rowidx_to_colidx.resize(mt + 1);
    a_rowidx_to_colidx[mt].emplace_back(kt);
    if (kt >= a_colidx_to_rowidx.size()) a_colidx_to_rowidx.resize(kt + 1);
    a_colidx_to_rowidx[kt].emplace_back(mt);

    filling += mTiles[mt] * kTiles[kt];
    avg_nb += mTiles[mt] * kTiles[kt];
    avg_nb_nb++;
    double value = vDist(genv);
    if (0 == rank && buildRefs) Aref_elements.emplace_back(mt, kt, blk_t(btas::Range(mTiles[mt], kTiles[kt]), value));
    if (rank != keymap({mt, kt})) continue;
    A_elements.emplace_back(mt, kt, blk_t(btas::Range(mTiles[mt], kTiles[kt]), value));
  }
  for (auto &row : a_rowidx_to_colidx) {
    std::sort(row.begin(), row.end());
  }
  for (auto &col : a_colidx_to_rowidx) {
    std::sort(col.begin(), col.end());
  }
  A.setFromTriplets(A_elements.begin(), A_elements.end());
  Adensity = (double)filling / (double)(M * K);
  if (0 == rank && buildRefs) Aref.setFromTriplets(Aref_elements.begin(), Aref_elements.end());

  filling = 0;
  fills.clear();
  while ((double)filling / (double)(K * N) < avgDensity) {
    int nt = nDist(gen);
    int kt = kDist(gen);

    if (fills.find({kt, nt}) != fills.end()) continue;
    fills.insert({kt, nt});

    if (kt >= b_rowidx_to_colidx.size()) b_rowidx_to_colidx.resize(kt + 1);
    b_rowidx_to_colidx[kt].emplace_back(nt);
    if (nt >= b_colidx_to_rowidx.size()) b_colidx_to_rowidx.resize(nt + 1);
    b_colidx_to_rowidx[nt].emplace_back(kt);

    filling += kTiles[kt] * nTiles[nt];
    avg_nb += kTiles[kt] * nTiles[nt];
    avg_nb_nb++;
    double value = vDist(genv);
    if (0 == rank && buildRefs) Bref_elements.emplace_back(kt, nt, blk_t(btas::Range(kTiles[kt], nTiles[nt]), value));
    if (rank != keymap({kt, nt})) continue;
    B_elements.emplace_back(kt, nt, blk_t(btas::Range(kTiles[kt], nTiles[nt]), value));
  }
  for (auto &row : b_rowidx_to_colidx) {
    std::sort(row.begin(), row.end());
  }
  for (auto &col : b_colidx_to_rowidx) {
    std::sort(col.begin(), col.end());
  }
  B.setFromTriplets(B_elements.begin(), B_elements.end());
  Bdensity = (double)filling / (double)(K * N);
  if (0 == rank && buildRefs) Bref.setFromTriplets(Bref_elements.begin(), Bref_elements.end());
  fills.clear();
  if (0 == rank && buildRefs) Bref.setFromTriplets(Bref_elements.begin(), Bref_elements.end());
  fills.clear();

  average_tile_size = (double)avg_nb / avg_nb_nb;
}

#ifdef BSPMM_HAS_LIBINT
static void initBlSpLibint2(libint2::Operator libint2_op, libint2::any libint2_op_params,
                            const std::vector<libint2::Atom> atoms, const std::string &basis_set_name,
                            double tile_perelem_2norm_threshold, const std::function<int(const Key<2> &)> &keymap,
                            int maxTs, int nthreads, SpMatrix<> &A, SpMatrix<> &B, SpMatrix<> &Aref, SpMatrix<> &Bref,
                            bool buildRefs, std::vector<long> &mTiles, std::vector<long> &nTiles,
                            std::vector<long> &kTiles, std::vector<std::vector<long>> &a_rowidx_to_colidx,
                            std::vector<std::vector<long>> &a_colidx_to_rowidx,
                            std::vector<std::vector<long>> &b_rowidx_to_colidx,
                            std::vector<std::vector<long>> &b_colidx_to_rowidx, double &average_tile_volume,
                            double &Adensity, double &Bdensity) {
  libint2::initialize();
  int rank = ttg_default_execution_context().rank();

  std::mutex mtx;  // will serialize access to non-concurrent data

  /// fires off nthreads instances of lambda in parallel, using use C++ threads
  auto parallel_do = [&nthreads, &mtx](auto &lambda) {
    std::vector<std::thread> threads;
    for (int thread_id = 0; thread_id != nthreads; ++thread_id) {
      if (thread_id != nthreads - 1)
        threads.push_back(std::thread(lambda, thread_id, &mtx));
      else
        lambda(thread_id, &mtx);
    }  // threads_id
    for (int thread_id = 0; thread_id < nthreads - 1; ++thread_id) threads[thread_id].join();
  };

  auto invert = [](const long dim2, const std::vector<size_t> &map12) {
    std::vector<long> map21(dim2, -1);
    for (size_t i1 = 0; i1 != map12.size(); ++i1) {
      const auto i2 = map12[i1];
      map21.at(i2) = i1;
    }
    return map21;
  };

  libint2::BasisSet bs(basis_set_name, atoms, /* throw_if_no_match = */ true);
  auto atom2shell = bs.atom2shell(atoms);
  auto shell2bf = bs.shell2bf();
  auto bf2shell = invert(bs.nbf(), shell2bf);
  std::cout << "basis set size = " << bs.nbf() << std::endl;

  // compute basis tilings by chopping into groups of atoms that are small enough
  std::vector<long> bsTiles;
  {
    const int natoms = atoms.size();
    int tile_size = 0;
    for (int a = 0; a != natoms; ++a) {
      auto &a_shells = atom2shell.at(a);
      const auto nbf_a = std::accumulate(a_shells.begin(), a_shells.end(), 0,
                                         [&bs](auto nbf, const auto &sh_idx) { return nbf + bs.at(sh_idx).size(); });
      if (tile_size + nbf_a <= maxTs) {
        tile_size += nbf_a;
      } else {
        if (tile_size == 0)  // 1 atom exceed max tile size, make the 1-atom tile
          bsTiles.emplace_back(nbf_a);
        else {
          bsTiles.emplace_back(tile_size);
          tile_size = nbf_a;
        }
        if (tile_size > maxTs) {
          bsTiles.emplace_back(tile_size);
          tile_size = 0;
        }
      }
    }
    if (tile_size > 0)  // last time
      bsTiles.emplace_back(tile_size);
  }
  mTiles = bsTiles;
  nTiles = bsTiles;
  kTiles = bsTiles;

  // fill the matrix, only insert tiles with norm greater than the threshold
  auto fill_matrix = [&](const auto &tiles) {
    SpMatrix<> M, Mref;

    const auto ntiles = tiles.size();

    M.resize(ntiles, ntiles);
    if (buildRefs && rank == 0) {
      Mref.resize(ntiles, ntiles);
    }

    // this data will be computed concurrently
    using triplet_t = Eigen::Triplet<blk_t>;
    std::vector<triplet_t> elements;
    std::vector<triplet_t> ref_elements;
    double total_tile_volume = 0.;
    std::vector<std::vector<long>> rowidx_to_colidx(ntiles), colidx_to_rowidx(ntiles);

    auto fill_matrix_impl = [&](int thread_id, std::mutex *mtx) {
      libint2::Engine engine(libint2_op, bs.max_nprim(), bs.max_l(), 0, std::numeric_limits<double>::epsilon(),
                             libint2_op_params, libint2::BraKet::xs_xs);

      const auto ntiles = tiles.size();
      const auto nshell = bs.size();
      const auto nbf = bs.nbf();
      long row_bf_offset = 0;
      for (auto row_tile_idx = 0; row_tile_idx != tiles.size(); ++row_tile_idx) {
        const auto row_bf_fence = row_bf_offset + tiles[row_tile_idx];
        const auto row_sh_offset = bf2shell.at(row_bf_offset);
        assert(row_sh_offset != -1);
        const auto row_sh_fence = (row_bf_fence != nbf) ? bf2shell.at(row_bf_fence) : nshell;
        assert(row_sh_fence != -1);

        long col_bf_offset = 0;
        for (auto col_tile_idx = 0; col_tile_idx != tiles.size(); ++col_tile_idx) {
          const auto col_bf_fence = col_bf_offset + tiles[col_tile_idx];

          // skip this tile if it does not belong to this rank
          const auto my_tile = (rank == keymap({row_tile_idx, col_tile_idx}) || (buildRefs && rank == 0)) &&
                               ((row_tile_idx * ntiles + col_tile_idx) % nthreads == thread_id);
          const auto really_my_tile = (rank == keymap({row_tile_idx, col_tile_idx})) &&
                                      ((row_tile_idx * ntiles + col_tile_idx) % nthreads == thread_id);
          if (my_tile) {
            const auto col_sh_offset = bf2shell.at(col_bf_offset);
            assert(col_sh_offset != -1);
            const auto col_sh_fence = (col_bf_fence != nbf) ? bf2shell.at(col_bf_fence) : nshell;
            assert(col_sh_fence != -1);

            blk_t tile(btas::Range({row_bf_offset, col_bf_offset}, {row_bf_fence, col_bf_fence}), 0.);

            for (auto row_sh_idx = row_sh_offset; row_sh_idx != row_sh_fence; ++row_sh_idx) {
              const auto &row_sh = bs.at(row_sh_idx);
              const auto row_sh_bf_offset = shell2bf.at(row_sh_idx);
              for (auto col_sh_idx = col_sh_offset; col_sh_idx != col_sh_fence; ++col_sh_idx) {
                const auto &col_sh = bs.at(col_sh_idx);
                const auto col_sh_bf_offset = shell2bf.at(col_sh_idx);

                engine.compute(row_sh, col_sh);

                // copy to the tile
                {
                  const auto *shellset = engine.results()[0];
                  for (auto bf0 = 0, bf01 = 0; bf0 != row_sh.size(); ++bf0)
                    for (auto bf1 = 0; bf1 != col_sh.size(); ++bf1, ++bf01)
                      tile(row_sh_bf_offset + bf0, col_sh_bf_offset + bf1) = shellset[bf01];
                }
              }
            }

            const auto tile_volume = tile.range().volume();
            const auto tile_perelem_2norm = std::sqrt(btas::dot(tile, tile)) / static_cast<double>(tile_volume);

            if (tile_perelem_2norm >= tile_perelem_2norm_threshold) {
              {
                std::scoped_lock<std::mutex> lock(*mtx);
                if (buildRefs && rank == 0) {
                  ref_elements.emplace_back(row_tile_idx, col_tile_idx, tile);
                }
                if (really_my_tile) {
                  elements.emplace_back(row_tile_idx, col_tile_idx, tile);
                  rowidx_to_colidx.at(row_tile_idx).emplace_back(col_tile_idx);
                  colidx_to_rowidx.at(col_tile_idx).emplace_back(row_tile_idx);
                  total_tile_volume += tile.range().volume();
                }
              }
            }
          }  // !my_tile

          col_bf_offset = col_bf_fence;
        }
        row_bf_offset = row_bf_fence;
      }
    };

    parallel_do(fill_matrix_impl);

    long nnz_tiles = elements.size();  // # of nonzero tiles, currently on this rank only

    // allreduce metadata: rowidx_to_colidx, colidx_to_rowidx, total_tile_volume, nnz_tiles
    ttg_sum(ttg_default_execution_context(), nnz_tiles);
    ttg_sum(ttg_default_execution_context(), total_tile_volume);
    auto allreduce_vevveclong = [&](std::vector<std::vector<long>> &vvl) {
      std::vector<std::vector<long>> vvl_result(vvl.size());
      for (long source_rank = 0; source_rank != ttg_default_execution_context().size(); ++source_rank) {
        for (auto rowidx = 0; rowidx != ntiles; ++rowidx) {
          long sz = static_cast<long>(vvl.at(rowidx).size());
          MPI_Bcast(&sz, 1, MPI_LONG, source_rank, ttg_default_execution_context().impl().comm());
          if (rank == source_rank) {
            MPI_Bcast(vvl[rowidx].data(), sz, MPI_LONG, source_rank, ttg_default_execution_context().impl().comm());
            vvl_result.at(rowidx).insert(vvl_result[rowidx].end(), vvl[rowidx].begin(), vvl[rowidx].end());
          } else {
            std::vector<long> colidxs(sz);
            MPI_Bcast(colidxs.data(), sz, MPI_LONG, source_rank, ttg_default_execution_context().impl().comm());
            vvl_result.at(rowidx).insert(vvl_result[rowidx].end(), colidxs.begin(), colidxs.end());
          }
        }
      }
      vvl = std::move(vvl_result);
    };
    allreduce_vevveclong(rowidx_to_colidx);
    allreduce_vevveclong(colidx_to_rowidx);

    for (auto &row : rowidx_to_colidx) {
      std::sort(row.begin(), row.end());
    }
    for (auto &col : colidx_to_rowidx) {
      std::sort(col.begin(), col.end());
    }

    const auto nbf = bs.nbf();
    const double density = total_tile_volume / (nbf * nbf);
    const auto avg_tile_volume = total_tile_volume / elements.size();
    M.setFromTriplets(elements.begin(), elements.end());
    if (buildRefs && rank == 0) Mref.setFromTriplets(ref_elements.begin(), ref_elements.end());

    return std::make_tuple(M, Mref, rowidx_to_colidx, colidx_to_rowidx, avg_tile_volume, density);
  };

  std::tie(A, Aref, a_rowidx_to_colidx, a_colidx_to_rowidx, average_tile_volume, Adensity) = fill_matrix(bsTiles);
  B = A;
  Bref = Aref;
  b_rowidx_to_colidx = a_rowidx_to_colidx;
  b_colidx_to_rowidx = a_colidx_to_rowidx;
  Bdensity = Adensity;

  libint2::finalize();
}
#endif  // defined(BSPMM_HAS_LIBINT)

#endif  // defined(BTAS_IS_USABLE)

#endif  // defined(BLOCK_SPARSE_GEMM)

static SpMatrix<> timed_measurement(SpMatrix<> &A, SpMatrix<> &B, const std::function<int(const Key<2> &)> &keymap,
                              const std::string &tiling_type, double gflops, double avg_nb, double Adensity,
                              double Bdensity, const std::vector<std::vector<long>> &a_rowidx_to_colidx,
                              const std::vector<std::vector<long>> &a_colidx_to_rowidx,
                              const std::vector<std::vector<long>> &b_rowidx_to_colidx,
                              const std::vector<std::vector<long>> &b_colidx_to_rowidx, std::vector<long> &mTiles,
                              std::vector<long> &nTiles, std::vector<long> &kTiles, int M, int N, int K, int P, int Q,
                              size_t memory, const long forced_split, long lookahead, long comm_threshold) {
  int MT = (int)A.rows();
  int NT = (int)B.cols();
  int KT = (int)A.cols();
  assert(KT == B.rows());

  SpMatrix<> C;
  C.resize(MT, NT);

  // flow graph needs to exist on every node
  Edge<Key<2>, Control> ctl("StartupControl");
  StartupControl control(ctl);
  Edge<Key<2>, blk_t> eC;

  Write_SpMatrix<> c(C, eC, keymap);
  auto &c_status = c.status();
  assert(!has_value(c_status));
  SpMM<> a_times_b(ctl, eC, A, B, a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx,
                   mTiles, nTiles, kTiles, keymap, P, Q, memory, forced_split, lookahead, comm_threshold);
  TTGUNUSED(a_times_b);

  auto connected = make_graph_executable(&control, a_times_b.get_reada(), a_times_b.get_readb());
  assert(connected);
  TTGUNUSED(connected);

  MPI_Barrier(MPI_COMM_WORLD);
  struct timeval start {
    0
  }, end{0}, diff{0};
  gettimeofday(&start, nullptr);
  // ready, go! need only 1 kick, so must be done by 1 thread only
  if (ttg_default_execution_context().rank() == 0) control.start(a_times_b.initbound());
  ttg_fence(ttg_default_execution_context());
  gettimeofday(&end, nullptr);
  timersub(&end, &start, &diff);
  double tc = (double)diff.tv_sec + (double)diff.tv_usec / 1e6;
#if defined(TTG_USE_MADNESS)
  std::string rt("MAD");
#elif defined(TTG_USE_PARSEC)
  std::string rt("PARSEC");
#else
  std::string rt("Unkown???");
#endif
  if (ttg_default_execution_context().rank() == 0) {
    double avg, stdev;
    std::tie(avg, stdev) = a_times_b.gemmsperrankperphase();

    std::cout << "TTG-" << rt << " PxQxg=   " << P << " " << Q << " 1 average_NB= " << avg_nb << " M= " << M
              << " N= " << N << " K= " << K << " Tiling= " << tiling_type << " A_density= " << Adensity
              << " B_density= " << Bdensity << " gflops= " << gflops << " seconds= " << tc
              << " gflops/s= " << gflops / tc << " nb_phases= " << a_times_b.nbphases() << " lookahead= " << lookahead
              << " average_nb_gemm_per_rank_per_phase= " << avg << " stdev_nb_gemm_per_rank_per_phase= " << stdev
              << std::endl;
  }

  return C;
}

#if !defined(BLOCK_SPARSE_GEMM)
static void make_rowidx_to_colidx_from_eigen(const SpMatrix<> &mat, std::vector<std::vector<long>> &r2c) {
  for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
    for (typename SpMatrix<blk_t>::InnerIterator it(mat, k); it; ++it) {
      const long row = it.row();
      const long col = it.col();
      if (row >= r2c.size()) r2c.resize(row + 1);
      r2c[row].push_back(col);
    }
  }
  // Sort each vector of column indices, as we pushed them in an arbitrary order
  for (auto &row : r2c) {
    std::sort(row.begin(), row.end());
  }
}

static void make_colidx_to_rowidx_from_eigen(const SpMatrix<> &mat, std::vector<std::vector<long>> &c2r) {
  for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
    for (typename SpMatrix<blk_t>::InnerIterator it(mat, k); it; ++it) {
      const long row = it.row();
      const long col = it.col();

      if (col >= c2r.size()) c2r.resize(col + 1);
      c2r[col].push_back(row);
    }
    // Sort each vector of row indices, as we pushed them in an arbitrary order
    for (auto &col : c2r) {
      std::sort(col.begin(), col.end());
    }
  }
}
#endif  // !defined(BLOCK_SPARSE_GEMM)

static double compute_gflops(const std::vector<std::vector<long>> &a_r2c, const std::vector<std::vector<long>> &b_r2c,
                             const std::vector<long> &mTiles, const std::vector<long> &nTiles,
                             const std::vector<long> &kTiles) {
  unsigned long flops = 0;
  for (auto i = 0; i < a_r2c.size(); i++) {
    for (auto kk = 0; kk < a_r2c[i].size(); kk++) {
      auto k = a_r2c[i][kk];
      if (k >= b_r2c.size()) continue;
      for (auto jj = 0; jj < b_r2c[k].size(); jj++) {
        auto j = b_r2c[k][jj];
        flops += mTiles[i] * nTiles[j] * kTiles[k];
      }
    }
  }
  return 2.0 * (double)flops / 1e9;
}

int main(int argc, char **argv) {
  bool timing;
  double gflops;

  int cores = -1;
  std::string nbCoreStr(getCmdOption(argv, argv + argc, "-c"));
  cores = parseOption(nbCoreStr, cores);

  if (int dashdash = cmdOptionIndex(argv, argv + argc, "--") > -1) {
    ttg_initialize(argc - dashdash, argv + dashdash, cores);
  } else {
    ttg_initialize(1, argv, cores);
  }

  std::string debugStr(getCmdOption(argv, argv + argc, "-d"));
  auto debug = (unsigned int)parseOption(debugStr, 0);

  if (debug & (1 << 1)) {
    using mpqc::Debugger;
    auto debugger = std::make_shared<Debugger>();
    Debugger::set_default_debugger(debugger);
    debugger->set_exec(argv[0]);
    debugger->set_prefix(ttg_default_execution_context().rank());
    // debugger->set_cmd("lldb_xterm");
    debugger->set_cmd("gdb_xterm");
  }

  int mpi_size = ttg_default_execution_context().size();
  int mpi_rank = ttg_default_execution_context().rank();
  int best_pq = mpi_size;
  int P, Q;
  for (int p = 1; p <= (int)sqrt(mpi_size); p++) {
    if ((mpi_size % p) == 0) {
      int q = mpi_size / p;
      if (abs(p - q) < best_pq) {
        best_pq = abs(p - q);
        P = p;
        Q = q;
      }
    }
  }

  // ttg::launch_gdb(ttg_default_execution_context().rank(), argv[0]);

  {
    if (debug & (1 << 0)) {
      ttg::trace_on();
      OpBase::set_trace_all(true);
    }

    SpMatrix<> A, B, C, Aref, Bref;
    std::stringstream tiling_type;
    int M = -1, N = -1, K = -1;

    double avg_nb = nan("undefined");
    double Adensity = nan("undefined");
    double Bdensity = nan("undefined");

    std::string PStr(getCmdOption(argv, argv + argc, "-P"));
    P = parseOption(PStr, P);
    std::string QStr(getCmdOption(argv, argv + argc, "-Q"));
    Q = parseOption(QStr, Q);

    if (P * Q != mpi_size) {
      if (!cmdOptionExists(argv, argv + argc, "-Q") && (mpi_size % P) == 0)
        Q = mpi_size / P;
      else if (!cmdOptionExists(argv, argv + argc, "-P") && (mpi_size % Q) == 0)
        P = mpi_size / Q;
      else {
        if (0 == mpi_rank) {
          std::cerr << P << "x" << Q << " is not a valid process grid -- bailing out" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
      }
    }

    size_t memory = 64 * 1024 * 1024;
    long lookahead = 1;
    long forced_split = 0;
    long comm_threshold = 8;

    std::string LookaheadStr(getCmdOption(argv, argv + argc, "-l"));
    lookahead = parseOption(LookaheadStr, lookahead);
    std::string MemoryStr(getCmdOption(argv, argv + argc, "-mem"));
    memory = parseOption(MemoryStr, memory);
    std::string ForcedSplitStr(getCmdOption(argv, argv + argc, "-dim"));
    forced_split = parseOption(ForcedSplitStr, forced_split);
    std::string CommThresholdStr(getCmdOption(argv, argv + argc, "-com"));
    comm_threshold = parseOption(CommThresholdStr, comm_threshold);

    // block-cyclic map
    auto bc_keymap = [P, Q](const Key<2> &key) {
      int i = (int)key[0];
      int j = (int)key[1];
      int pq = tile2rank(i, j, P, Q);
      return pq;
    };

    std::string seedStr(getCmdOption(argv, argv + argc, "-s"));
    unsigned long seed = parseOption(seedStr, 0l);
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
      if (0 == ttg_default_execution_context().rank()) std::cerr << "#Random seeded with " << seed << std::endl;
    }
    ttg_broadcast(ttg_default_execution_context(), seed, 0);

    std::vector<long> mTiles;
    std::vector<long> nTiles;
    std::vector<long> kTiles;
    std::vector<std::vector<long>> a_rowidx_to_colidx;
    std::vector<std::vector<long>> a_colidx_to_rowidx;
    std::vector<std::vector<long>> b_rowidx_to_colidx;
    std::vector<std::vector<long>> b_colidx_to_rowidx;

    std::string checkStr(getCmdOption(argv, argv + argc, "-x"));
    int check = parseOption(checkStr, !(argc >= 2));
    timing = (check == 0);

#ifndef BLOCK_SPARSE_GEMM
    if (cmdOptionExists(argv, argv + argc, "-mm")) {
      char *filename = getCmdOption(argv, argv + argc, "-mm");
      tiling_type << filename;
      initSpMatrixMarket(bc_keymap, filename, A, B, C, M, N, K);
    } else if (cmdOptionExists(argv, argv + argc, "-rmat")) {
      char *opt = getCmdOption(argv, argv + argc, "-rmat");
      tiling_type << "RandomSparseMatrix";
      initSpRmat(bc_keymap, opt, A, B, C, M, N, K, seed);
    } else {
      tiling_type << "HardCodedSparseMatrix";
      initSpHardCoded(bc_keymap, A, B, C, M, N, K);
    }

    if (check) {
      // We don't generate the sparse matrices in distributed, so Aref and Bref can
      // just point to the same matrix, or be a local copy.
      Aref = A;
      Bref = B;
    }

    // We still need to build the metadata from the  matrices.
    make_rowidx_to_colidx_from_eigen(A, a_rowidx_to_colidx);
    make_colidx_to_rowidx_from_eigen(A, a_colidx_to_rowidx);
    make_rowidx_to_colidx_from_eigen(B, b_rowidx_to_colidx);
    make_colidx_to_rowidx_from_eigen(B, b_colidx_to_rowidx);
    // This is only needed to compute the flops
    for (int mt = 0; mt < M; mt++) mTiles.emplace_back(1);
    for (int nt = 0; nt < N; nt++) nTiles.emplace_back(1);
    for (int kt = 0; kt < K; kt++) kTiles.emplace_back(1);
#else  // !defined(BLOCK_SPARSE_GEMM)
    if (argc >= 2) {
#ifndef BSPMM_HAS_LIBINT
      std::string Mstr(getCmdOption(argv, argv + argc, "-M"));
      M = parseOption(Mstr, 1200);
      std::string Nstr(getCmdOption(argv, argv + argc, "-N"));
      N = parseOption(Nstr, 1200);
      std::string Kstr(getCmdOption(argv, argv + argc, "-K"));
      K = parseOption(Kstr, 1200);
      std::string minTsStr(getCmdOption(argv, argv + argc, "-t"));
      int minTs = parseOption(minTsStr, 32);
      std::string maxTsStr(getCmdOption(argv, argv + argc, "-T"));
      int maxTs = parseOption(maxTsStr, 256);
      std::string avgStr(getCmdOption(argv, argv + argc, "-a"));
      double avg = parseOption(avgStr, 0.3);
      timing = (check == 0);
      tiling_type << "RandomIrregularTiling";
      initBlSpRandom(bc_keymap, M, N, K, minTs, maxTs, avg, A, B, Aref, Bref, check, mTiles, nTiles, kTiles,
                     a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, avg_nb, Adensity,
                     Bdensity, seed, P, Q);
      C.resize((long)mTiles.size(), (long)nTiles.size());
#else
      std::string xyz_filename(getCmdOption(argv, argv + argc, "-y"));
      if (xyz_filename.empty()) throw std::runtime_error("missing -y argument to the libint2-based bspmm example");
      std::ifstream xyz_file(xyz_filename);
      auto atoms = libint2::read_dotxyz(xyz_file);
      std::string basis_name(getCmdOption(argv, argv + argc, "-b"));
      if (basis_name.empty()) basis_name = "cc-pvdz";
      std::string op_param_str(getCmdOption(argv, argv + argc, "-p"));
      auto op_param = parseOption(op_param_str, 1.);
      std::string maxTsStr(getCmdOption(argv, argv + argc, "-T"));
      int maxTs = parseOption(maxTsStr, 256);
      std::string eps_param_str(getCmdOption(argv, argv + argc, "-e"));
      double tile_perelem_2norm_threshold = parseOption(eps_param_str, 1e-5);
      std::cerr << "#Generating matrices with Libint2 on " << xyz_filename << " and " << cores << " cores" << std::endl;
      auto start = std::chrono::high_resolution_clock::now();
      initBlSpLibint2(libint2::Operator::yukawa, libint2::any{op_param}, atoms, basis_name,
                      tile_perelem_2norm_threshold, bc_keymap, maxTs, cores == -1 ? 1 : cores, A, B,
                      Aref, Bref, check, mTiles, nTiles,kTiles, a_rowidx_to_colidx,
                      a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, avg_nb,Adensity,
                      Bdensity);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = duration_cast<std::chrono::microseconds>(end-start);
      std::cerr << "#Generation done (" << duration.count()/1000000. << "s)" << std::endl;
      tiling_type << xyz_filename << "_" << basis_name << "_" << tile_perelem_2norm_threshold << "_" << op_param;
#endif
      C.resize(A.rows(), B.cols());
    } else {
      tiling_type << "HardCodedBlockSparseMatrix";
      initBlSpHardCoded(bc_keymap, A, B, C, Aref, Bref, true, mTiles, nTiles, kTiles, a_rowidx_to_colidx,
                        a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, M, N, K);
    }
#endif  // !defined(BLOCK_SPARSE_GEMM)

    if (M == -1) M = std::accumulate(mTiles.begin(), mTiles.end(), 0);
    if (N == -1) N = std::accumulate(nTiles.begin(), nTiles.end(), 0);
    if (K == -1) K = std::accumulate(kTiles.begin(), kTiles.end(), 0);

    gflops = compute_gflops(a_rowidx_to_colidx, b_rowidx_to_colidx, mTiles, nTiles, kTiles);

    std::string nbrunStr(getCmdOption(argv, argv + argc, "-n"));
    int nb_runs = parseOption(nbrunStr, 1);

    if (timing) {
      SpMatrix<> C;  // store the result in case need to use it

      // Start up engine
      ttg_execute(ttg_default_execution_context());
      for (int nrun = 0; nrun < nb_runs; nrun++) {
        C = timed_measurement(A, B, bc_keymap, tiling_type.str(), gflops, avg_nb, Adensity, Bdensity, a_rowidx_to_colidx,
                          a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, mTiles, nTiles, kTiles, M, N, K,
                          P, Q, memory, forced_split, lookahead, comm_threshold);
      }

#ifdef BSPMM_BUILD_TA_TEST
      {
        // prelims
        auto MT = mTiles.size();
        auto NT = nTiles.size();
        auto KT = kTiles.size();
        auto& mad_world = ttg_default_execution_context().impl().impl();

        // make tranges
        auto make_trange1 = [](const auto& tile_sizes) {
          std::vector<int64_t> hashes; hashes.reserve(tile_sizes.size()+1);
          hashes.push_back(0);
          for(auto& tile_size: tile_sizes) { hashes.push_back(hashes.back() + tile_size); }
          return TiledArray::TiledRange1(hashes.begin(), hashes.end());
        };
        auto mtr1 = make_trange1(mTiles);
        auto ntr1 = make_trange1(nTiles);
        auto ktr1 = make_trange1(kTiles);
        TA::TiledRange A_trange({mtr1, ktr1});
        TA::TiledRange B_trange({ktr1, ntr1});
        TA::TiledRange C_trange({mtr1, ntr1});

        // make shapes
        auto make_shape = [&mad_world](const SpMatrix<>& mat, const auto& trange) {
          TA::Tensor<float> norms(TA::Range(mat.rows(), mat.cols()), 0.);
          for (int k=0; k<mat.outerSize(); ++k) {
            for (SpMatrix<>::InnerIterator it(mat, k); it; ++it) {
              auto r = it.row();  // row index
              auto c = it.col();  // col index (here it is equal to k)
              const auto& v = it.value();
              norms(r, c) = std::sqrt(btas::dot(v, v));
            }
          }
          return TA::SparseShape<float>(mad_world, norms, trange);
        };
        auto A_shape = make_shape(A, A_trange);
        auto B_shape = make_shape(B, B_trange);
        auto C_shape = make_shape(C, C_trange);

        // make pmaps
        auto A_pmap = std::make_shared<TA::detail::UserPmap>(mad_world, MT*KT, [&](size_t mk) -> size_t { auto [m, k] = std::div((long)mk, (long)KT); return tile2rank(m, k, P, Q); } );
        auto B_pmap = std::make_shared<TA::detail::UserPmap>(mad_world, KT*NT, [&](size_t kn) -> size_t { auto [k, n] = std::div((long)kn, (long)NT); return tile2rank(k, n, P, Q); } );
        auto C_pmap = std::make_shared<TA::detail::UserPmap>(mad_world, MT*NT, [&](size_t mn) -> size_t { auto [m, n] = std::div((long)mn, (long)NT); return tile2rank(m, n, P, Q); } );

        // make distarrays
        auto make_ta = [&mad_world](const SpMatrix<>& mat, const auto& trange, const auto& shape, const auto& pmap) {
          TA::TSpArrayD mat_ta(mad_world, trange, shape, pmap);
          for (int k=0; k<mat.outerSize(); ++k) {
            for (SpMatrix<>::InnerIterator it(mat, k); it; ++it) {
              auto r = it.row();  // row index
              auto c = it.col();  // col index (here it is equal to k)
              assert(mat_ta.is_local({r, c}) && !mat_ta.is_zero({r, c}));
              mat_ta.set({r, c}, TA::Tensor<double>(it.value()));
            }
          }
          return mat_ta;
        };
        auto A_ta = make_ta(A, A_trange, A_shape, A_pmap);
        auto B_ta = make_ta(B, B_trange, B_shape, B_pmap);

        for (int nrun = 0; nrun < nb_runs; nrun++) {
          auto start = std::chrono::high_resolution_clock::now();
          TA::TSpArrayD C_ta;
          C_ta("m,n") = (A_ta("m,k") * B_ta("k,n")).set_shape(C_shape);
          C_ta.world().gop.fence();
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = duration_cast<std::chrono::microseconds>(end-start);
          std::cout << "Time to compute C=A*B in TiledArray = " << duration.count()/1000000. << std::endl;
          auto print = [](const auto& label, const SpMatrix<>& mat) {
            for (int k=0; k<mat.outerSize(); ++k) {
              for (SpMatrix<>::InnerIterator it(mat, k); it; ++it) {
                auto r = it.row();  // row index
                auto c = it.col();  // col index (here it is equal to k)
                std::cout << label << ": {" << r << "," << c << "}: " << it.value() << std::endl;
              }
            }
          };
//          print("A", A);
//          print("C", C);
//          std::cout << "A_ta = " << A_ta << std::endl;
//          std::cout << "C_ta = " << C_ta << std::endl;
        }
      }
#endif

    } else {
      // flow graph needs to exist on every node
      auto keymap_write = [](const Key<2> &key) { return 0; };
      Edge<Key<2>, Control> ctl("StartupControl");
      StartupControl control(ctl);
      Edge<Key<2>, blk_t> eC;
      Write_SpMatrix<> c(C, eC, keymap_write);
      auto &c_status = c.status();
      assert(!has_value(c_status));
      //  SpMM a_times_b(world, eA, eB, eC, A, B);
      SpMM<> a_times_b(ctl, eC, A, B, a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx,
                       mTiles, nTiles, kTiles, bc_keymap, P, Q, memory, forced_split, lookahead, comm_threshold);

      if (get_default_world().rank() == 0)
        std::cout << Dot{}(a_times_b.get_reada(), a_times_b.get_readb()) << std::endl;

      // ready to run!
      auto connected = make_graph_executable(&control, a_times_b.get_reada(), a_times_b.get_readb());
      assert(connected);
      TTGUNUSED(connected);

      // ready, go! need only 1 kick, so must be done by 1 thread only
      if (ttg_default_execution_context().rank() == 0) control.start(a_times_b.initbound());

      ttg_execute(ttg_default_execution_context());
      ttg_fence(ttg_default_execution_context());

      // validate C=A*B against the reference output
      assert(has_value(c_status));
      if (ttg_default_execution_context().rank() == 0) {
        std::cout << "Product done, computing locally with Eigen to check" << std::endl;

        SpMatrix<> Cref = Aref * Bref;

        double norm_2_square, norm_inf;
        std::tie(norm_2_square, norm_inf) = norms<blk_t>(Cref - C);
        std::cout << "||Cref - C||_2      = " << std::sqrt(norm_2_square) << std::endl;
        std::cout << "||Cref - C||_\\infty = " << norm_inf << std::endl;
        if (norm_inf > 1e-9) {
          if(Cref.nonZeros() < 100) {
            std::cout << "Cref:\n" << Cref << std::endl;
            std::cout << "C:\n" << C << std::endl;
          }
          ttg_abort();
        }
      }

      // validate Acopy=A against the reference output
      //      assert(has_value(copy_status));
      //      if (ttg_default_execution_context().rank() == 0) {
      //        double norm_2_square, norm_inf;
      //        std::tie(norm_2_square, norm_inf) = norms<blk_t>(Acopy - A);
      //        std::cout << "||Acopy - A||_2      = " << std::sqrt(norm_2_square) << std::endl;
      //        std::cout << "||Acopy - A||_\\infty = " << norm_inf << std::endl;
      //        if (::ttg::tracing()) {
      //          std::cout << "Acopy (" << static_cast<void *>(&Acopy) << "):\n" << Acopy << std::endl;
      //          std::cout << "A (" << static_cast<void *>(&A) << "):\n" << A << std::endl;
      //        }
      //        if (norm_inf != 0) {
      //          ttg_abort();
      //        }
      //      }
    }
  }

  ttg_finalize();

  return 0;
}
