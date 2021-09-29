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
#if __has_include(<btas/features.h>)
#include <btas/features.h>
#ifdef BTAS_IS_USABLE
#include <btas/btas.h>
#include <btas/optimize/contract.h>
#include <btas/util/mohndle.h>
#else
#warning "found btas/features.h but Boost.Iterators is missing, hence BTAS is unusable ... add -I/path/to/boost"
#endif
#endif

#include <sys/time.h>
#include <boost/graph/rmat_graph_generator.hpp>
#if !defined(BLOCK_SPARSE_GEMM)
#include <boost/graph/directed_graph.hpp>
#include <boost/random/linear_congruential.hpp>
#include <unsupported/Eigen/SparseExtra>
#endif

#include "ttg.h"

using namespace ttg;

#include "ttg/util/future.h"

#include "ttg/util/bug.h"

#include "active-set-strategy.h"

#if defined(BLOCK_SPARSE_GEMM) && defined(BTAS_IS_USABLE)
using blk_t = btas::Tensor<double, btas::DEFAULT::range, btas::mohndle<btas::varray<double>, btas::Handle::shared_ptr>>;

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

#if defined(BLOCK_SPARSE_GEMM) && defined(BTAS_IS_USABLE)

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
    btas::contract_222(1.0, A, array{1, 2}, B, array{2, 3}, 1.0, C, array{1, 3}, false, false);
    return std::move(C);
  }
}  // namespace btas
#endif  // BTAS_IS_USABLE
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
  static constexpr const long max_index_square = max_index * max_index;
  Key() = default;
  template <typename Integer>
  Key(std::initializer_list<Integer> ilist) {
    std::copy(ilist.begin(), ilist.end(), this->begin());
    assert(valid());
  }
  explicit Key(std::size_t hash) {
    static_assert(Rank == 2 || Rank == 3, "Key<Rank>::Key(hash) only implemented for Rank={2,3}");
    if (Rank == 2) {
      (*this)[0] = hash / max_index;
      (*this)[1] = hash % max_index;
    } else if (Rank == 3) {
      (*this)[0] = hash / max_index_square;
      (*this)[1] = (hash % max_index_square) / max_index;
      (*this)[2] = hash % max_index;
    }
  }
  std::size_t hash() const {
    static_assert(Rank == 2 || Rank == 3, "Key<Rank>::hash only implemented for Rank={2,3}");
    return Rank == 2 ? (*this)[0] * max_index + (*this)[1]
                     : ((*this)[0] * max_index + (*this)[1]) * max_index + (*this)[2];
  }

 private:
  bool valid() {
    bool result = true;
    for (auto &idx : *this) {
      result = result && (idx < max_index);
    }
    return result;
  }
};

template <std::size_t Rank>
std::ostream &operator<<(std::ostream &os, const Key<Rank> &key) {
  os << "{";
  for (size_t i = 0; i != Rank; ++i) os << key[i] << (i + 1 != Rank ? "," : "");
  os << "}";
  return os;
}

inline int tile2rank(int i, int j, int P, int Q) {
  int p = (i % P);
  int q = (j % Q);
  int r = (q * P) + p;
  return r;
}

// flow (move?) data into an existing SpMatrix on rank 0
template <typename Blk = blk_t>
class Write_SpMatrix : public Op<Key<2>, std::tuple<>, Write_SpMatrix<Blk>, Blk> {
 public:
  using baseT = Op<Key<2>, std::tuple<>, Write_SpMatrix<Blk>, Blk>;

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
  SpMM(Edge<Key<3>, Control> &progress_ctl, Edge<Key<2>, Blk> &c_flow, const SpMatrix<Blk> &a_mat,
       const SpMatrix<Blk> &b_mat, const std::vector<std::vector<long>> &a_rowidx_to_colidx,
       const std::vector<std::vector<long>> &a_colidx_to_rowidx,
       const std::vector<std::vector<long>> &b_rowidx_to_colidx,
       const std::vector<std::vector<long>> &b_colidx_to_rowidx, const std::vector<long> &mTiles,
       const std::vector<long> &nTiles, const std::vector<long> &kTiles, const Keymap &keymap, const long P,
       const long Q, size_t memory, const long forced_split, const long lookahead)
      : a_flow_(), b_flow_(), a_ctl_(), b_ctl_(), c2c_ctl_(), a_ijk_(), b_ijk_(), c_ijk_(), plan_(nullptr) {
    plan_ = std::make_shared<Plan>(a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx,
                                   mTiles, nTiles, kTiles, keymap, P, Q, memory, forced_split, lookahead);

    coordinator_ = std::make_unique<Coordinator>(progress_ctl, a_ctl_, b_ctl_, c2c_ctl_, plan_, keymap);
    read_a_ = std::make_unique<Read_SpMatrix>("A", a_mat, a_ctl_, a_flow_, plan_, keymap);
    read_b_ = std::make_unique<Read_SpMatrix>("B", b_mat, b_ctl_, b_flow_, plan_, keymap);
    bcast_a_ = std::make_unique<BcastA>(a_flow_, a_ijk_, plan_, keymap);
    bcast_b_ = std::make_unique<BcastB>(b_flow_, b_ijk_, plan_, keymap);
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

   private:
    using gemmset_t = std::set<std::tuple<long, long, long>>;
    using bcastset_t = std::set<std::tuple<long, long>>;
    using step_t = std::vector<std::tuple<gemmset_t, long, bcastset_t, bcastset_t>>;

    const step_t steps_;

   public:
    Plan(const std::vector<std::vector<long>> &a_rowidx_to_colidx,
         const std::vector<std::vector<long>> &a_colidx_to_rowidx,
         const std::vector<std::vector<long>> &b_rowidx_to_colidx,
         const std::vector<std::vector<long>> &b_colidx_to_rowidx, const std::vector<long> &mTiles,
         const std::vector<long> &nTiles, const std::vector<long> &kTiles, const Keymap &keymap, const long P,
         const long Q, const size_t memory, const long forced_split, const long lookahead)
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
        , steps_(strategy_selector(memory, forced_split)) {}

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
      step_t steps;
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

      for (long mm = 0; mm < mns; mm++) {
        for (long nn = 0; nn < nns; nn++) {
          for (long kk = 0; kk < kns; kk++) {
            gemmset_t gemms;
            long nb_local_gemms = 0;
            bcastset_t a_sent;
            bcastset_t b_sent;
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
                  if (keymap_(Key<2>({m, n})) == rank) nb_local_gemms++;
                  gemms.insert({m, n, k});
                  if (a_sent.find({m, k}) == a_sent.end()) {
                    a_sent.insert({m, k});
                  }
                  if (b_sent.find({k, n}) == b_sent.end()) {
                    b_sent.insert({k, n});
                  }
                }
              }
            }
            if (tracing()) {
              if (gemms.size() < 30) {
                std::ostringstream dbg;
                dbg << "On rank " << ttg_default_execution_context().rank() << ", Step " << steps.size() << " is ";
                for (auto it : gemms) {
                  dbg << "(" << std::get<0>(it) << "," << std::get<1>(it) << "," << std::get<2>(it) << ") ";
                }
                ttg::print(dbg.str());
              } else {
                ttg::print("On rank ", ttg_default_execution_context().rank(),
                           ", plan is not displayed because it is too large");
              }
            }
            steps.emplace_back(std::make_tuple(gemms, nb_local_gemms, a_sent, b_sent));
          }
        }
      }
      return steps;
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
      auto a_iter = std::find(a_k_range.begin(), a_iter_fence, k);
      assert(a_iter != a_iter_fence);
      const auto &b_k_range = b_colidx_to_rowidx_.at(j);
      auto b_iter_fence = b_k_range.end();
      auto b_iter = std::find(b_k_range.begin(), b_iter_fence, k);
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

    long nb_steps() const { return steps_.size(); }

    std::tuple<long, long, long> gemm_coordinates(long i, long j, long k) const {
      long p = i % this->p();
      long q = j % this->q();
      for (long s = 0l; s < steps_.size(); s++) {
        const gemmset_t *gs = &std::get<0>(steps_[s]);
        if (gs->find({i, j, k}) != gs->end()) {
          return std::make_tuple(p, q, s);
        }
      }
      abort();
      return std::make_tuple(p, q, -1);
    }

    struct GemmCoordinate {
      long r_;
      long c_;
      const Blk v_;

      long row() { return r_; }
      long col() { return c_; }
      const Blk &value() { return v_; }
    };

    std::vector<GemmCoordinate> bcast_in_step(long s, bool is_a, const SpMatrix<Blk> &matrix) const {
      std::vector<GemmCoordinate> res;
      const bcastset_t *bset;
      if (is_a) {
        bset = &std::get<2>(steps_[s]);
      } else {
        bset = &std::get<3>(steps_[s]);
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

    const gemmset_t &gemms(long s) const { return std::get<0>(steps_[s]); }

    long nb_local_gemms(long s) const { return std::get<1>(steps_[s]); }

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
  class Coordinator : public Op<Key<3>, std::tuple<Out<Key<3>, Control>, Out<Key<3>, Control>, Out<Key<3>, Control>>,
                                Coordinator, Control> {
   public:
    using baseT =
        Op<Key<3>, std::tuple<Out<Key<3>, Control>, Out<Key<3>, Control>, Out<Key<3>, Control>>, Coordinator, Control>;

    Coordinator(Edge<Key<3>, Control> progress_ctl, Edge<Key<3>, Control> &a_ctl, Edge<Key<3>, Control> &b_ctl,
                Edge<Key<3>, Control> &c2c_ctl, std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(fuse(progress_ctl, c2c_ctl)), edges(a_ctl, b_ctl, c2c_ctl), std::string("SpMM::Coordinator"),
                {"progress_ctl"}, {"a_ctl", "b_ctl", "c2c_ctl"},
                [keymap](const Key<3> &key) {
                  Key<2> k{key[0], key[1]};
                  return keymap(k);
                })
        , plan_(plan) {
      baseT::template set_input_reducer<0>([](Control &&a, const Control &&b) { return a; });
      for (long p = 0; p < plan_->p(); p++) {
        for (long q = 0; q < plan_->q(); q++) {
          if (keymap(Key<2>({p, q})) == ttg_default_execution_context().rank()) {
            for (long l = 0; l < plan_->lookahead_ && l < plan_->nb_steps(); l++) {
              if (tracing())
                ttg::print("On rank ", ttg_default_execution_context().rank(),
                           " : at bootstrap, setting the number of"
                           " local GEMMS to trigger step ",
                           l, " to 1");
              baseT::template set_argstream_size<0>(Key<3>{p, q, l}, 1);
            }
          }
        }
      }
    }

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&input,
            std::tuple<Out<Key<3>, Control>, Out<Key<3>, Control>, Out<Key<3>, Control>> &out) {
      auto p = key[0];
      auto q = key[1];
      auto s = key[2];
      if (s + plan_->lookahead_ < plan_->nb_steps()) {
        auto nb = plan_->nb_local_gemms(s);
        if (nb > 0) {  // We set the number of reductions before triggering the first GEMM (through trigger of bcasts)
          if (tracing())
            ttg::print("Coordinator(", p, ", ", q, ", ", s, "): setting the number of local GEMMS to trigger step ",
                       s + plan_->lookahead_, " to ", nb);
          baseT::template set_argstream_size<0>(Key<3>({p, q, s + plan_->lookahead_}), nb);
        } else {
          if (tracing())
            ttg::print("Coordinator(", p, ", ", q, ", ", s, "): there are 0 local GEMMS in step ", s,
                       " ; triggering next coordinator step ", s + plan_->lookahead_);
          baseT::template set_argstream_size<0>(Key<3>({p, q, s + plan_->lookahead_}), 1);
          ::send<2>(Key<3>({p, q, s + plan_->lookahead_}), Control{}, out);
        }
      }
      if (tracing())
        ttg::print("On rank ", ttg_default_execution_context().rank(), "Coordinator(", p, ", ", q, ", ", s,
                   "): Sending control to broadcast A and B for step ", s);
      ::send<0>(Key<3>({p, q, s}), Control{}, out);
      ::send<1>(Key<3>({p, q, s}), Control{}, out);
    }

   private:
    std::shared_ptr<const Plan> plan_;
  };

  // flow data from an existing SpMatrix
  class Read_SpMatrix : public Op<Key<3>, std::tuple<Out<Key<3>, Blk>>, Read_SpMatrix, Control> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<3>, Blk>>, Read_SpMatrix, Control>;

    Read_SpMatrix(const char *label, const SpMatrix<Blk> &matrix, Edge<Key<3>, Control> &ctl, Edge<Key<3>, Blk> &out,
                  std::shared_ptr<const Plan> plan, const Keymap &keymap)
        : baseT(edges(ctl), edges(out), std::string("read_spmatrix(") + label + ")", {"ctl"},
                {std::string(label) + "pqs"},
                [keymap](const Key<3> &key) {
                  Key<2> k{key[0], key[1]};
                  return keymap(k);
                })
        , matrix_(matrix)
        , plan_(plan)
        , is_a_(is_label_a(label)) {}

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&inputs, std::tuple<Out<Key<3>, Blk>> &out) {
      auto rank = ttg_default_execution_context().rank();
      auto p = key[0];
      auto q = key[1];
      auto step = key[2];
      for (auto x : plan_->bcast_in_step(step, is_a_, matrix_)) {
        if (rank == this->get_keymap()(Key<3>({x.row(), x.col(), step}))) {
          if (tracing())
            ttg::print("On rank ", rank, " Read_SpMatrix", (is_a_ ? "(A)" : "(B)"), " (", p, ", ", q, ",", step,
                       "): send block to Bcast (", x.row(), ", ", x.col(), ", ", step, ")");
          ::send<0>(Key<3>({x.row(), x.col(), step}), x.value(), out);
        }
      }
    }

   private:
    const SpMatrix<Blk> &matrix_;
    std::shared_ptr<const Plan> plan_;
    const bool is_a_;

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
  class BcastA : public Op<Key<3>, std::tuple<Out<Key<3>, Blk>>, BcastA, Blk> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<3>, Blk>>, BcastA, Blk>;

    BcastA(Edge<Key<3>, Blk> &a, Edge<Key<3>, Blk> &a_ijk, std::shared_ptr<const Plan> plan, Keymap keymap)
        : baseT(edges(a), edges(a_ijk), "SpMM::bcast_a", {"a_iks"}, {"a_ijk"},
                [keymap](const Key<3> &key) {
                  Key<2> k{key[0], key[1]};
                  return keymap(k);
                })
        , plan_(plan) {}

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&a_ik, std::tuple<Out<Key<3>, Blk>> &a_ijk) {
      auto world = get_default_world();
      const auto i = key[0];
      const auto k = key[1];
      const auto s = key[2];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank ", rank, "BcastA(", i, ", ", k, ",", s, ")");
      // broadcast a_ik to existing {i,j,k} that belongs to step s of the plan
      std::vector<Key<3>> ijk_keys;
      for (auto ijk : plan_->gemms(s)) {
        long ii, jj, kk;
        std::tie(ii, jj, kk) = ijk;
        if (ii != i) {
          if (tracing())
            ttg::print("Rank ", ttg_default_execution_context().rank(), " during step ", s, " NOT Broadcasting A[", i,
                       "][", k, "] to GEMM(", i, ", ", jj, ", ", k, ") -- ii == ", ii, " != ", i, " == i");
          continue;
        }
        if (kk != k) {
          if (tracing())
            ttg::print("Rank ", ttg_default_execution_context().rank(), "during step ", s, " NOT Broadcasting A[", i,
                       "][", k, "] to GEMM(", ii, ", ", jj, ", ", kk, ") -- kk == ", kk, " != ", k, " == k");
          continue;
        }
        assert(k < plan_->b_rowidx_to_colidx_.size());
        assert(std::find(plan_->b_rowidx_to_colidx_[k].begin(), plan_->b_rowidx_to_colidx_[k].end(), jj) !=
               plan_->b_rowidx_to_colidx_[k].end());
        if (tracing())
          ttg::print("Rank ", ttg_default_execution_context().rank(), " during step ", s, " Broadcasting A[", i, "][",
                     k, "] to GEMM(", ii, ", ", jj, ", ", kk, ")");
        ijk_keys.emplace_back(Key<3>({ii, jj, kk}));
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(a_ik), a_ijk);
    }

   private:
    std::shared_ptr<const Plan> plan_;
  };  // class BcastA

  /// broadcast B[k][j] to all {i,j,k} such that A[i][k] exists
  class BcastB : public Op<Key<3>, std::tuple<Out<Key<3>, Blk>>, BcastB, Blk> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<3>, Blk>>, BcastB, Blk>;

    BcastB(Edge<Key<3>, Blk> &b, Edge<Key<3>, Blk> &b_ijk, std::shared_ptr<const Plan> plan, Keymap keymap)
        : baseT(edges(b), edges(b_ijk), "SpMM::bcast_b", {"b_kjs"}, {"b_ijk"},
                [keymap](const Key<3> &key) {
                  Key<2> k{key[0], key[1]};
                  return keymap(k);
                })
        , plan_(plan) {}

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&b_kj, std::tuple<Out<Key<3>, Blk>> &b_ijk) {
      const auto k = key[0];
      const auto j = key[1];
      const auto s = key[2];
      auto rank = ttg_default_execution_context().rank();
      if (tracing()) ttg::print("On rank ", rank, " BcastB(", k, ", ", j, ",", s, ")");
      std::vector<Key<3>> ijk_keys;
      for (auto ijk : plan_->gemms(s)) {
        long ii, jj, kk;
        std::tie(ii, jj, kk) = ijk;
        if (jj != j) {
          if (tracing())
            ttg::print("Rank ", ttg_default_execution_context().rank(), " NOT Broadcasting B[", k, "][", j,
                       "]"
                       " to GEMM(",
                       ii, ", ", jj, ", ", kk, ") -- jj == ", jj, " != ", j, " == j");
          continue;
        }
        if (kk != k) {
          if (tracing())
            ttg::print("Rank ", ttg_default_execution_context().rank(), " NOT Broadcasting A[", k, "][", j,
                       "]"
                       " to GEMM(",
                       ii, ", ", jj, ", ", kk, ") -- kk == ", kk, " != ", k, " == k");
          continue;
        }
        assert(k < plan_->b_rowidx_to_colidx_.size());
        assert(std::find(plan_->b_rowidx_to_colidx_[k].begin(), plan_->b_rowidx_to_colidx_[k].end(), jj) !=
               plan_->b_rowidx_to_colidx_[k].end());
        if (tracing())
          ttg::print("Rank ", ttg_default_execution_context().rank(), " Broadcasting B[", k, "][", j,
                     "]"
                     " to GEMM(",
                     ii, ", ", jj, ", ", kk, ")");
        ijk_keys.emplace_back(Key<3>({ii, jj, kk}));
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(b_kj), b_ijk);
    }

   private:
    std::shared_ptr<const Plan> plan_;
  };  // class BcastA

  /// multiply task has 3 input flows: a_ijk, b_ijk, and c_ijk, c_ijk contains the running total
  class MultiplyAdd : public Op<Key<3>, std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>, Out<Key<3>, Control>>,
                                MultiplyAdd, const Blk, const Blk, Blk> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>, Out<Key<3>, Control>>, MultiplyAdd,
                     const Blk, const Blk, Blk>;

    MultiplyAdd(Edge<Key<3>, Blk> &a_ijk, Edge<Key<3>, Blk> &b_ijk, Edge<Key<3>, Blk> &c_ijk, Edge<Key<2>, Blk> &c,
                Edge<Key<3>, Control> &progress_ctl, std::shared_ptr<const Plan> plan, Keymap keymap)
        : baseT(edges(a_ijk, b_ijk, c_ijk), edges(c, c_ijk, progress_ctl), "SpMM::MultiplyAdd",
                {"a_ijk", "b_ijk", "c_ijk"}, {"c_ij", "c_ijk", "progress_ctl"},
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
                  ttg::print("On rank ", owner, " Initializing C[", i, "][", j, "] to zero (first k ", k, ")");
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
            std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>, Out<Key<3>, Control>> &result) {
      const auto i = key[0];
      const auto j = key[1];
      const auto k = key[2];
      long p, q, s;
      std::tie(p, q, s) = plan_->gemm_coordinates(i, j, k);
      long next_k;
      bool have_next_k;
      std::tie(next_k, have_next_k) = plan_->compute_next_k(i, j, k);
      if (tracing()) {
        ttg::print("Rank ", ttg_default_execution_context().rank(), " step ", s, " : C[", i, "][", j, "]  += A[", i,
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
          ttg::print("Rank ", ttg_default_execution_context().rank(), " step ", s, " Notifying coordinator of step ",
                     s + plan_->lookahead_, " that GEMM(", i, ", ", j, ", ", k, ") completed");
        }
        ::send<2>(Key<3>({p, q, s + plan_->lookahead_}), Control{}, result);
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
  Edge<Key<3>, Blk> a_flow_;
  Edge<Key<3>, Blk> b_flow_;
  Edge<Key<3>, Control> a_ctl_;
  Edge<Key<3>, Control> b_ctl_;
  Edge<Key<3>, Control> c2c_ctl_;
  std::unique_ptr<Coordinator> coordinator_;
  std::unique_ptr<BcastA> bcast_a_;
  std::unique_ptr<BcastB> bcast_b_;
  std::unique_ptr<MultiplyAdd> multiplyadd_;
  std::unique_ptr<Read_SpMatrix> read_a_;
  std::unique_ptr<Read_SpMatrix> read_b_;
  std::shared_ptr<Plan> plan_;
};

class StartupControl : public Op<void, std::tuple<Out<Key<3>, Control>>, StartupControl> {
  using baseT = Op<void, std::tuple<Out<Key<3>, Control>>, StartupControl>;
  long P;
  long Q;
  long initbound;

 public:
  explicit StartupControl(Edge<Key<3>, Control> &ctl)
      : baseT(edges(), edges(ctl), "StartupControl", {}, {"ctl"}), P(0), Q(0), initbound(0) {}

  void op(std::tuple<Out<Key<3>, Control>> &out) const {
    for (long i = 0; i < P; i++) {
      for (long j = 0; j < Q; j++) {
        for (long l = 0; l < initbound; l++) {
          Key<3> k{i, j, l};
          if (ttg::tracing()) ttg::print("StartupControl: enable {", i, ", ", j, ", 0}");
          ::send<0>(k, Control{}, out);
        }
      }
    }
  }

  void start(const long _p, const long _q, const long _b) {
    P = _p;
    Q = _q;
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
#endif

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
#else
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
#endif
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
#else
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
#endif
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
  TTGUNUSED(rank);

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
    auto blksize = {kTiles[kt], nTiles[nt]};
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

  average_tile_size = (double)avg_nb / avg_nb_nb;
}
#endif

#endif

static void timed_measurement(SpMatrix<> &A, SpMatrix<> &B, const std::function<int(const Key<2> &)> &keymap,
                              const std::string &tiling_type, double gflops, double avg_nb, double Adensity,
                              double Bdensity, const std::vector<std::vector<long>> &a_rowidx_to_colidx,
                              const std::vector<std::vector<long>> &a_colidx_to_rowidx,
                              const std::vector<std::vector<long>> &b_rowidx_to_colidx,
                              const std::vector<std::vector<long>> &b_colidx_to_rowidx, std::vector<long> &mTiles,
                              std::vector<long> &nTiles, std::vector<long> &kTiles, int M, int N, int K, int P, int Q,
                              size_t memory, const long forced_split, long lookahead) {
  int MT = (int)A.rows();
  int NT = (int)B.cols();
  int KT = (int)A.cols();
  assert(KT == B.rows());

  SpMatrix<> C;
  C.resize(MT, NT);

  // flow graph needs to exist on every node
  Edge<Key<3>, Control> ctl("StartupControl");
  StartupControl control(ctl);
  Edge<Key<2>, blk_t> eC;

  Write_SpMatrix<> c(C, eC, keymap);
  auto &c_status = c.status();
  assert(!has_value(c_status));
  SpMM<> a_times_b(ctl, eC, A, B, a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx,
                   mTiles, nTiles, kTiles, keymap, P, Q, memory, forced_split, lookahead);
  TTGUNUSED(a_times_b);

  auto connected = make_graph_executable(&control);
  assert(connected);
  TTGUNUSED(connected);

  struct timeval start {
    0
  }, end{0}, diff{0};
  gettimeofday(&start, nullptr);
  // ready, go! need only 1 kick, so must be done by 1 thread only
  if (ttg_default_execution_context().rank() == 0) control.start(P, Q, a_times_b.initbound());
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
#endif

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
    std::string tiling_type;
    int M = 0, N = 0, K = 0;

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

    std::string LookaheadStr(getCmdOption(argv, argv + argc, "-l"));
    lookahead = parseOption(LookaheadStr, lookahead);
    std::string MemoryStr(getCmdOption(argv, argv + argc, "-mem"));
    memory = parseOption(MemoryStr, memory);
    std::string ForcedSplitStr(getCmdOption(argv, argv + argc, "-dim"));
    forced_split = parseOption(ForcedSplitStr, forced_split);

    const auto &keymap = [P, Q](const Key<2> &key) {
      int i = (int)key[0];
      int j = (int)key[1];
      int r = tile2rank(i, j, P, Q);
      return r;
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

#if !defined(BLOCK_SPARSE_GEMM)
    if (cmdOptionExists(argv, argv + argc, "-mm")) {
      char *filename = getCmdOption(argv, argv + argc, "-mm");
      tiling_type = filename;
      initSpMatrixMarket(keymap, filename, A, B, C, M, N, K);
    } else if (cmdOptionExists(argv, argv + argc, "-rmat")) {
      char *opt = getCmdOption(argv, argv + argc, "-rmat");
      tiling_type = "RandomSparseMatrix";
      initSpRmat(keymap, opt, A, B, C, M, N, K, seed);
    } else {
      tiling_type = "HardCodedSparseMatrix";
      initSpHardCoded(keymap, A, B, C, M, N, K);
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
#else
    if (argc >= 2) {
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
      tiling_type = "RandomIrregularTiling";
      initBlSpRandom(keymap, M, N, K, minTs, maxTs, avg, A, B, Aref, Bref, check, mTiles, nTiles, kTiles,
                     a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, avg_nb, Adensity,
                     Bdensity, seed, P, Q);
      C.resize((long)mTiles.size(), (long)nTiles.size());
    } else {
      tiling_type = "HardCodedBlockSparseMatrix";
      initBlSpHardCoded(keymap, A, B, C, Aref, Bref, true, mTiles, nTiles, kTiles, a_rowidx_to_colidx,
                        a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, M, N, K);
    }
#endif  // !defined(BLOCK_SPARSE_GEMM)

    gflops = compute_gflops(a_rowidx_to_colidx, b_rowidx_to_colidx, mTiles, nTiles, kTiles);

    std::string nbrunStr(getCmdOption(argv, argv + argc, "-n"));
    int nb_runs = parseOption(nbrunStr, 1);

    if (timing) {
      // Start up engine
      ttg_execute(ttg_default_execution_context());
      for (int nrun = 0; nrun < nb_runs; nrun++) {
        timed_measurement(A, B, keymap, tiling_type, gflops, avg_nb, Adensity, Bdensity, a_rowidx_to_colidx,
                          a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, mTiles, nTiles, kTiles, M, N, K,
                          P, Q, memory, forced_split, lookahead);
      }
    } else {
      // flow graph needs to exist on every node
      auto keymap_write = [](const Key<2> &key) { return 0; };
      Edge<Key<3>, Control> ctl("StartupControl");
      StartupControl control(ctl);
      Edge<Key<2>, blk_t> eC;
      Write_SpMatrix<> c(C, eC, keymap_write);
      auto &c_status = c.status();
      assert(!has_value(c_status));
      //  SpMM a_times_b(world, eA, eB, eC, A, B);
      SpMM<> a_times_b(ctl, eC, A, B, a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx,
                       mTiles, nTiles, kTiles, keymap, P, Q, memory, forced_split, lookahead);

      if (get_default_world().rank() == 0)
        std::cout << Dot{}(a_times_b.get_reada(), a_times_b.get_readb()) << std::endl;

      // ready to run!
      auto connected = make_graph_executable(&control);
      assert(connected);
      TTGUNUSED(connected);

      // ready, go! need only 1 kick, so must be done by 1 thread only
      if (ttg_default_execution_context().rank() == 0) control.start(P, Q, a_times_b.initbound());

      ttg_execute(ttg_default_execution_context());
      ttg_fence(ttg_default_execution_context());

      // validate C=A*B against the reference output
      assert(has_value(c_status));
      if (ttg_default_execution_context().rank() == 0) {
        SpMatrix<> Cref = Aref * Bref;

        double norm_2_square, norm_inf;
        std::tie(norm_2_square, norm_inf) = norms<blk_t>(Cref - C);
        std::cout << "||Cref - C||_2      = " << std::sqrt(norm_2_square) << std::endl;
        std::cout << "||Cref - C||_\\infty = " << norm_inf << std::endl;
        if (norm_inf > 1e-9) {
          std::cout << "Cref:\n" << Cref << std::endl;
          std::cout << "C:\n" << C << std::endl;
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
