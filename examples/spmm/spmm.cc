#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if __has_include(<btas/features.h>)
#pragma message("C Preprocessor got here!")
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

#include "ttg/util/multiindex.h"

#include "ttg/util/bug.h"

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

// template <typename _Scalar, int _Options, typename _StorageIndex>
// struct colmajor_layout;
// template <typename _Scalar, typename _StorageIndex>
// struct colmajor_layout<_Scalar, Eigen::ColMajor, _StorageIndex> : public std::true_type {};
// template <typename _Scalar, typename _StorageIndex>
// struct colmajor_layout<_Scalar, Eigen::RowMajor, _StorageIndex> : public std::false_type {};

template <std::size_t Rank>
using Key = MultiIndex<Rank>;

/// maps {i,j} to rank within first (R=0) layer of the 3-d process grid
inline int ij2rank(int i, int j, int P, int Q) {
  std::vector<int> vec;
  int p = (i % P);
  int q = (j % Q);
  int rank = (q * P) + p;
  return rank;
}

/// maps {i,j,k} to rank within a 3-d process grid
inline int ijk2rank(int i, int j, int k, int P, int Q, int R) {
  std::vector<int> vec;
  int p = (i % P);
  int q = (j % Q);
  int l = (k % R);
  int rank = (l * P * Q) + (q * P) + p;
  return rank;
}

// flow data from an existing SpMatrix on rank 0
template <typename Blk = blk_t, typename Keymap = std::function<int(const Key<2> &)>>
class Read_SpMatrix : public TT<Key<2>, std::tuple<Out<Key<2>, Blk>>, Read_SpMatrix<Blk, Keymap>, ttg::typelist<void>> {
 public:
  using baseT = typename Read_SpMatrix::ttT;
  Read_SpMatrix(const char *label, const SpMatrix<Blk> &matrix, Edge<Key<2>> &ctl, Edge<Key<2>, Blk> &out,
                Keymap &ij_keymap)
      : baseT(edges(ctl), edges(out), std::string("read_spmatrix(") + label + ")", {"ctl"}, {std::string(label) + "ij"},
              ij_keymap)
      , matrix_(matrix) {}

  void op(const Key<2> &, std::tuple<Out<Key<2>, Blk>> &out) {
    auto rank = ttg::default_execution_context().rank();
    for (int k = 0; k < matrix_.outerSize(); ++k) {
      for (typename SpMatrix<Blk>::InnerIterator it(matrix_, k); it; ++it) {
        if (rank == this->get_keymap()(Key<2>(std::initializer_list<long>({it.row(), it.col()}))))
          ::send<0>(Key<2>(std::initializer_list<long>({it.row(), it.col()})), it.value(), out);
      }
    }
  }

 private:
  const SpMatrix<Blk> &matrix_;
};

// flow (move?) data into an existing SpMatrix on rank 0
template <typename Blk = blk_t>
class Write_SpMatrix : public TT<Key<2>, std::tuple<>, Write_SpMatrix<Blk>, ttg::typelist<Blk>> {
 public:
  using baseT = typename Write_SpMatrix::ttT;

  template <typename Keymap2>
  Write_SpMatrix(SpMatrix<Blk> &matrix, Edge<Key<2>, Blk> &in, Keymap2 &&ij_keymap)
      : baseT(edges(in), edges(), "write_spmatrix", {"Cij"}, {}, ij_keymap), matrix_(matrix) {}

  void op(const Key<2> &key, typename baseT::input_values_tuple_type &&elem, std::tuple<> &) {
    std::lock_guard<std::mutex> lock(mtx_);
    ttg::trace("rank =", default_execution_context().rank(),
               "/ thread_id =", reinterpret_cast<std::uintptr_t>(pthread_self()), "spmm.cc Write_SpMatrix wrote {",
               key[0], ",", key[1], "} = ", baseT::template get<0>(elem), " in ", static_cast<void *>(&matrix_),
               " with mutex @", static_cast<void *>(&mtx_), " for object @", static_cast<void *>(this));
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

/// sparse mm via 2.5D SUMMA

/// @tparam KeyMap2 maps {i,j} to processor
/// @tparam KeyMap3 maps {i,j,k} to processor
template <typename Keymap2 = std::function<int(const Key<2> &)>, typename Keymap3 = std::function<int(const Key<3> &)>,
          typename Blk = blk_t>
class SpMM25D {
 public:
  /// @param ij_keymap maps {i,j} to process, specifies distribution of tiles of A, B, and C
  /// @param ijk_keymap maps {i,j,k} to process, controls distribution of tasks performing C[i][j] += A[i][k]*B[k][j]
  /// @param R the number of "layers" in the 3-D process grid
  SpMM25D(Edge<Key<2>, Blk> &a, Edge<Key<2>, Blk> &b, Edge<Key<2>, Blk> &c, const SpMatrix<Blk> &a_mat,
          const SpMatrix<Blk> &b_mat, const std::vector<std::vector<long>> &a_rowidx_to_colidx,
          const std::vector<std::vector<long>> &a_colidx_to_rowidx,
          const std::vector<std::vector<long>> &b_rowidx_to_colidx,
          const std::vector<std::vector<long>> &b_colidx_to_rowidx, const std::vector<int> &mTiles,
          const std::vector<int> &nTiles, const std::vector<int> &kTiles, Keymap2 ij_keymap, Keymap3 ijk_keymap, long R)
      : a_rowidx_to_colidx_(a_rowidx_to_colidx)
      , b_colidx_to_rowidx_(b_colidx_to_rowidx)
      , a_colidx_to_rowidx_(a_colidx_to_rowidx)
      , b_rowidx_to_colidx_(b_rowidx_to_colidx)
      , ij_keymap_(std::move(ij_keymap))
      , ijk_keymap_(std::move(ijk_keymap)) {
    bcast_a_ = std::make_unique<BcastA>(a, local_a_ijk_, b_rowidx_to_colidx_, ij_keymap_, ijk_keymap_);
    local_bcast_a_ = std::make_unique<LocalBcastA>(local_a_ijk_, a_ijk_, b_rowidx_to_colidx_, ijk_keymap_);
    bcast_b_ = std::make_unique<BcastB>(b, local_b_ijk_, a_colidx_to_rowidx_, ij_keymap_, ijk_keymap_);
    local_bcast_b_ = std::make_unique<LocalBcastB>(local_b_ijk_, b_ijk_, a_colidx_to_rowidx_, ijk_keymap_);
    multiplyadd_ = std::make_unique<MultiplyAdd>(a_ijk_, b_ijk_, c_ijk_, c_ij_p_, a_rowidx_to_colidx_,
                                                 b_colidx_to_rowidx_, mTiles, nTiles, ijk_keymap_);
    reduce_c_ = std::make_unique<ReduceC>(c_ij_p_, c, ij_keymap_);
    reduce_c_->template set_input_reducer<0>([](Blk &c_ij, const Blk &c_ij_p) { c_ij = c_ij + c_ij_p; });
    // compute how many contributions each C[i][j] should expect ... MultiplyAdd already does this, but need a way to
    // send message from each process p to the process owning C[i][j] to expect a contribution from it for now replicate
    // this logic ...
    // TODO: do this in MultiplyAdd (need to allreduce this info so that everyone has it)
    // N.B. only need to set stream size on the rank that will accumulate the C[i][j] contribution
    const auto my_rank = ttg::default_execution_context().rank();
    for (auto i = 0ul; i != a_rowidx_to_colidx_.size(); ++i) {
      if (a_rowidx_to_colidx_[i].empty()) continue;
      for (auto j = 0ul; j != b_colidx_to_rowidx_.size(); ++j) {
        if (b_colidx_to_rowidx_[j].empty()) continue;

        if (ij_keymap_(Key<2>{i, j}) == my_rank) {
          decltype(i) k;
          bool have_k;
          std::tie(k, have_k) = multiplyadd_->compute_first_k(i, j);
          std::vector<bool> c_ij_procmask(R, false);
          if (have_k) {
            const auto pR = k % R;  // k values are distributed round-robin among the layers of the 3-D grid
            assert(pR < c_ij_procmask.size());
            c_ij_procmask[pR] = true;
            while (have_k) {
              std::tie(k, have_k) = multiplyadd_->compute_next_k(i, j, k);
              if (have_k) {
                const auto pR = k % R;
                assert(pR < c_ij_procmask.size());
                c_ij_procmask[pR] = true;
              }
            }
          }
          const auto c_ij_nprocs = std::count_if(c_ij_procmask.begin(), c_ij_procmask.end(), [](bool b) { return b; });
          if (c_ij_nprocs > 0) reduce_c_->template set_argstream_size<0>(Key<2>{i, j}, c_ij_nprocs);
        }
      }
    }

    TTGUNUSED(bcast_a_);
    TTGUNUSED(bcast_b_);
    TTGUNUSED(multiplyadd_);
    TTGUNUSED(reduce_c_);
  }

  /// Locally broadcast `A[i][k]` assigned to this processor `p` to matmul tasks `{i,j,k}` for all `j` such that
  /// `B[k][j]` exists AND `k` contribution to `C[i][j]` is assigned to this processor
  class LocalBcastA : public TT<Key<3>, std::tuple<Out<Key<3>, Blk>>, LocalBcastA, ttg::typelist<Blk>> {
   public:
    using baseT = typename LocalBcastA::ttT;

    LocalBcastA(Edge<Key<3>, Blk> &a, Edge<Key<3>, Blk> &a_ijk,
                const std::vector<std::vector<long>> &b_rowidx_to_colidx, const Keymap3 &ijk_keymap)
        : baseT(edges(a), edges(a_ijk), "SpMM25D::local_bcast_a", {"a_ikp"}, {"a_ijk"},
                [](const Key<3> &ikp) { return ikp[2]; })
        , b_rowidx_to_colidx_(b_rowidx_to_colidx)
        , ijk_keymap_(ijk_keymap) {}

    void op(const Key<3> &ikp, typename baseT::input_values_tuple_type &&a_ik, std::tuple<Out<Key<3>, Blk>> &a_ijk) {
      const auto i = ikp[0];
      const auto k = ikp[1];
      const auto p = ikp[2];

      auto world = default_execution_context();
      assert(p == world.rank());
      ttg::trace("LocalBcastA(", i, ", ", k, ", ", p, ")");
      if (k >= b_rowidx_to_colidx_.size()) return;
      // local broadcast a_ik to all {i,j,k} such that b_kj exists
      std::vector<Key<3>> ijk_keys;
      for (auto &j : b_rowidx_to_colidx_[k]) {
        if (ijk_keymap_(Key<3>({i, j, k})) == world.rank()) {
          ttg::trace("Broadcasting A[", i, "][", k, "] on proc ", p, " to j=", j);
          ijk_keys.emplace_back(Key<3>({i, j, k}));
        }
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(a_ik), a_ijk);
    }

   private:
    const std::vector<std::vector<long>> &b_rowidx_to_colidx_;
    const Keymap3 &ijk_keymap_;
  };  // class LocalBcastA

  /// broadcast `A[i][k]` to all processors which will contain at least one `C[i][j]` such that `B[k][j]` exists
  class BcastA : public TT<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastA, ttg::typelist<Blk>> {
   public:
    using baseT = typename BcastA::ttT;

    BcastA(Edge<Key<2>, Blk> &a_ik, Edge<Key<3>, Blk> &a_ikp, const std::vector<std::vector<long>> &b_rowidx_to_colidx,
           const Keymap2 &ij_keymap, const Keymap3 &ijk_keymap)
        : baseT(edges(a_ik), edges(a_ikp), "SpMM25D::bcast_a", {"a_ik"}, {"a_ikp"}, ij_keymap)
        , b_rowidx_to_colidx_(b_rowidx_to_colidx)
        , ijk_keymap_(ijk_keymap) {}

    void op(const Key<2> &ik, typename baseT::input_values_tuple_type &&a_ik, std::tuple<Out<Key<3>, Blk>> &a_ikp) {
      const auto i = ik[0];
      const auto k = ik[1];
      ttg::trace("BcastA(", i, ", ", k, ")");
      std::vector<Key<3>> ikp_keys;

      if (k >= b_rowidx_to_colidx_.size()) return;
      auto world = default_execution_context();
      std::vector<bool> procmap(world.size());
      for (auto &j : b_rowidx_to_colidx_[k]) {
        const long p = ijk_keymap_(Key<3>(
            {i, j, k}));  // N.B. in 2.5D SUMMA different k contributions to C[i][j] are computed on different nodes
        if (!procmap[p]) {
          ttg::trace("Broadcasting A[", i, "][", k, "] to proc ", p);
          ikp_keys.emplace_back(Key<3>({i, k, p}));
          procmap[p] = true;
        }
      }
      ::broadcast<0>(ikp_keys, baseT::template get<0>(a_ik), a_ikp);
    }

   private:
    const std::vector<std::vector<long>> &b_rowidx_to_colidx_;
    const Keymap3 &ijk_keymap_;
  };  // class BcastA

  /// Locally broadcast `B[k][j]` assigned to this processor `p` to matmul tasks `{i,j,k}` for all `k` such that
  /// `A[i][k]` exists AND `k` contribution to `C[i][j]` is assigned to this processor
  class LocalBcastB : public TT<Key<3>, std::tuple<Out<Key<3>, Blk>>, LocalBcastB, ttg::typelist<Blk>> {
   public:
    using baseT = typename LocalBcastB::ttT;

    LocalBcastB(Edge<Key<3>, Blk> &b_kjp, Edge<Key<3>, Blk> &b_ijk,
                const std::vector<std::vector<long>> &a_colidx_to_rowidx, const Keymap3 &ijk_keymap)
        : baseT(edges(b_kjp), edges(b_ijk), "SpMM25D::local_bcast_b", {"b_kjp"}, {"b_ijk"},
                [](const Key<3> &kjp) { return kjp[2]; })
        , a_colidx_to_rowidx_(a_colidx_to_rowidx)
        , ijk_keymap_(ijk_keymap) {}

    void op(const Key<3> &kjp, typename baseT::input_values_tuple_type &&b_kj, std::tuple<Out<Key<3>, Blk>> &b_ijk) {
      const auto k = kjp[0];
      const auto j = kjp[1];
      const auto p = kjp[2];
      auto world = default_execution_context();
      assert(p == world.rank());
      ttg::trace("BcastB(", k, ", ", j, ", ", p, ")");
      if (k >= a_colidx_to_rowidx_.size()) return;
      // broadcast b_kj to all ijk for which c_ij is on this processor and a_ik exists
      std::vector<Key<3>> ijk_keys;
      for (auto &i : a_colidx_to_rowidx_[k]) {
        if (ijk_keymap_(Key<3>({i, j, k})) == world.rank()) {
          ttg::trace("Broadcasting B[", k, "][", j, "] on proc ", p, " to i=", i);
          ijk_keys.emplace_back(Key<3>({i, j, k}));
        }
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(b_kj), b_ijk);
    }

   private:
    const std::vector<std::vector<long>> &a_colidx_to_rowidx_;
    const Keymap3 &ijk_keymap_;
  };  // class LocalBcastB

  /// broadcast `B[k][j]` to all processors which will contain at least one `C[i][j]` such that `A[i][k]` exists
  class BcastB : public TT<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastB, ttg::typelist<Blk>> {
   public:
    using baseT = typename BcastB::ttT;

    BcastB(Edge<Key<2>, Blk> &b_kj, Edge<Key<3>, Blk> &b_kjp, const std::vector<std::vector<long>> &a_colidx_to_rowidx,
           const Keymap2 &ij_keymap, const Keymap3 &ijk_keymap)
        : baseT(edges(b_kj), edges(b_kjp), "SpMM25D::bcast_b", {"b_kj"}, {"b_kjp"}, ij_keymap)
        , a_colidx_to_rowidx_(a_colidx_to_rowidx)
        , ijk_keymap_(ijk_keymap) {}

    void op(const Key<2> &kj, typename baseT::input_values_tuple_type &&b_kj, std::tuple<Out<Key<3>, Blk>> &b_kjp) {
      const auto k = kj[0];
      const auto j = kj[1];
      // broadcast b_kj to all processors which will contain at least one c_ij such that a_ik exists
      std::vector<Key<3>> kjp_keys;
      ttg::trace("BcastB(", k, ", ", j, ")");
      if (k >= a_colidx_to_rowidx_.size()) return;
      auto world = default_execution_context();
      std::vector<bool> procmap(world.size());
      for (auto &i : a_colidx_to_rowidx_[k]) {
        long p = ijk_keymap_(Key<3>({i, j, k}));
        if (!procmap[p]) {
          ttg::trace("Broadcasting B[", k, "][", j, "] to proc ", p);
          kjp_keys.emplace_back(Key<3>({k, j, p}));
          procmap[p] = true;
        }
      }
      ::broadcast<0>(kjp_keys, baseT::template get<0>(b_kj), b_kjp);
    }

   private:
    const std::vector<std::vector<long>> &a_colidx_to_rowidx_;
    const Keymap3 &ijk_keymap_;
  };  // class BcastB

  /// multiply task has 3 input flows: a_ijk, b_ijk, and c_ijk, c_ijk contains the running total for this kayer of the
  /// 3-D process grid only
  class MultiplyAdd : public TT<Key<3>, std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>>, MultiplyAdd,
                                ttg::typelist<const Blk, const Blk, Blk>> {
   public:
    using baseT = typename MultiplyAdd::ttT;

    MultiplyAdd(Edge<Key<3>, Blk> &a_ijk, Edge<Key<3>, Blk> &b_ijk, Edge<Key<3>, Blk> &c_ijk, Edge<Key<2>, Blk> &c,
                const std::vector<std::vector<long>> &a_rowidx_to_colidx,
                const std::vector<std::vector<long>> &b_colidx_to_rowidx, const std::vector<int> &mTiles,
                const std::vector<int> &nTiles, const Keymap3 &ijk_keymap)
        : baseT(edges(a_ijk, b_ijk, c_ijk), edges(c, c_ijk), "SpMM25D::MultiplyAdd", {"a_ijk", "b_ijk", "c_ijk"},
                {"c_ij", "c_ijk"}, ijk_keymap)
        , a_rowidx_to_colidx_(a_rowidx_to_colidx)
        , b_colidx_to_rowidx_(b_colidx_to_rowidx) {
      this->set_priomap([=](const Key<3> &ijk) { return this->prio(ijk); });  // map a key to an integral priority value

      // for each {i,j} determine first k that contributes AND belongs to this node,
      // initialize input {i,j,first_k} flow to 0
      for (auto i = 0ul; i != a_rowidx_to_colidx_.size(); ++i) {
        if (a_rowidx_to_colidx_[i].empty()) continue;
        for (auto j = 0ul; j != b_colidx_to_rowidx_.size(); ++j) {
          if (b_colidx_to_rowidx_[j].empty()) continue;

          const auto p = ttg::default_execution_context().rank();
          decltype(i) k;
          bool have_k;
          std::tie(k, have_k) = compute_first_k(i, j, p);
          if (have_k) {
            ttg::trace("Initializing C[", i, "][", j, "] on process ", p, " to zero");
#if BLOCK_SPARSE_GEMM
            Blk zero(btas::Range(mTiles[i], nTiles[j]), 0.0);
#else
            Blk zero{0.0};
#endif
            this->template in<2>()->send(Key<3>({i, j, k}), zero);
          } else {
            if (tracing() && a_rowidx_to_colidx_.size() * b_colidx_to_rowidx_.size() < 400)
              ttg::print("C[", i, "][", j, "] is empty");
          }
        }
      }
    }

    void op(const Key<3> &ijk, typename baseT::input_values_tuple_type &&_ijk,
            std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>> &result) {
      const auto i = ijk[0];
      const auto j = ijk[1];
      const auto k = ijk[2];  // k==l same because 000 will always be on layer 0, 001 will be accessed on layer 1
      const auto p = ttg::default_execution_context().rank();
      long next_k;
      bool have_next_k;
      std::tie(next_k, have_next_k) = compute_next_k(i, j, k, p);
      ttg::trace("Rank ", ttg::default_execution_context().rank(),
                 " :"
                 " C[",
                 i, "][", j, "]  += A[", i, "][", k, "] by B[", k, "][", j, "],  next_k? ",
                 (have_next_k ? std::to_string(next_k) : "does not exist"));
      // compute the contrib, pass the running total to the next flow, if needed
      // otherwise write to the result flow
      if (have_next_k) {
        ::send<1>(
            Key<3>({i, j, next_k}),
            gemm(std::move(baseT::template get<2>(_ijk)), baseT::template get<0>(_ijk), baseT::template get<1>(_ijk)),
            result);
      } else {  // done with all local contributions to C[i][j], reduce with others on the process to which C[i][j]
                // belongs
        ::send<0>(
            Key<2>({i, j}),
            gemm(std::move(baseT::template get<2>(_ijk)), baseT::template get<0>(_ijk), baseT::template get<1>(_ijk)),
            result);
      }
    }

   private:
    const std::vector<std::vector<long>> &a_rowidx_to_colidx_;
    const std::vector<std::vector<long>> &b_colidx_to_rowidx_;

    /* Compute the length of the remaining sequence on that tile */
    int32_t prio(const Key<3> &key) const {
      const auto i = key[0];
      const auto j = key[1];
      const auto k = key[2];
      int32_t len = -1;  // will be incremented at least once
      long next_k = k;
      bool have_next_k;
      do {
        std::tie(next_k, have_next_k) = compute_next_k(i, j, next_k);  // here I know how many 'k' I have with same ij
        ++len;
      } while (have_next_k);
      return len;
    }

   public:  // to be able to reuse this logic in SpMM25D
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
        auto a_colidx = *a_iter;  // pointing to next kth element
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
        return std::make_tuple(a_colidx, true);  // returned true for kth element exist and also returns next k since
                                                 // a_colidx points to ++a_iter,  if not reaches to fence
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
      ttg::abort();  // unreachable
      return std::make_tuple(0, false);
    }

    // given {i,j} return first k such that A[i][k] and B[k][j] exist AND ijk_keymap_(i,j,k) == p
    std::tuple<long, bool> compute_first_k(long i, long j, long p) const {
      long first_k = 0;
      bool have_k = false;
      std::tie(first_k, have_k) = compute_first_k(i, j);
      while (have_k) {
        if (this->get_keymap()(Key<3>{i, j, first_k}) == p)
          return {first_k, true};
        else
          std::tie(first_k, have_k) = compute_next_k(i, j, first_k);
      }
      return {0, false};
    }

    // given {i,j,k} such that A[i][k] and B[k][j] exist
    // return next k such that this condition holds AND ijk_keymap_(i,j,k) == p
    std::tuple<long, bool> compute_next_k(long i, long j, long k, long p) const {
      long next_k = 0;
      bool have_k = false;
      std::tie(next_k, have_k) = compute_next_k(i, j, k);
      while (have_k) {
        if (this->get_keymap()(Key<3>{i, j, next_k}) == p)
          return {next_k, true};
        else
          std::tie(next_k, have_k) = compute_next_k(i, j, next_k);
      }
      return {0, false};
    }

  };  // MultiplyAdd

  /// reduces contributions to `C[i][j]` produced on different layers of the 3-d process grid
  class ReduceC : public TT<Key<2>, std::tuple<Out<Key<2>, Blk>>, ReduceC, ttg::typelist<Blk>> {
   public:
    using baseT = typename ReduceC::ttT;

    ReduceC(Edge<Key<2>, Blk> &c_ij_p, Edge<Key<2>, Blk> &c_ij, const Keymap2 &ij_keymap)
        : baseT(edges(c_ij_p), edges(c_ij), "SpMM25D::reduce_c", {"c_ij(p)"}, {"c_ij"}, ij_keymap) {}

    void op(const Key<2> &ij, typename baseT::input_values_tuple_type &&c_ij_p, std::tuple<Out<Key<2>, Blk>> &c_ij) {
      ttg::trace("ReduceC(", ij[0], ", ", ij[1], ")");
      ::send<0>(ij, baseT::template get<0>(c_ij_p), c_ij);
    }
  };  // class ReduceC

 private:
  Edge<Key<3>, Blk> a_ijk_;
  Edge<Key<3>, Blk> local_a_ijk_;
  Edge<Key<3>, Blk> b_ijk_;
  Edge<Key<3>, Blk> local_b_ijk_;
  Edge<Key<3>, Blk> c_ijk_;
  Edge<Key<2>, Blk> c_ij_p_;
  const std::vector<std::vector<long>> &a_rowidx_to_colidx_;
  const std::vector<std::vector<long>> &b_colidx_to_rowidx_;
  const std::vector<std::vector<long>> &a_colidx_to_rowidx_;
  const std::vector<std::vector<long>> &b_rowidx_to_colidx_;
  std::unique_ptr<BcastA> bcast_a_;
  std::unique_ptr<LocalBcastA> local_bcast_a_;
  std::unique_ptr<BcastB> bcast_b_;
  std::unique_ptr<LocalBcastB> local_bcast_b_;
  std::unique_ptr<MultiplyAdd> multiplyadd_;
  std::unique_ptr<ReduceC> reduce_c_;
  Keymap2 ij_keymap_;
  Keymap3 ijk_keymap_;
};

class Control : public TT<void, std::tuple<Out<Key<2>>>, Control> {
  using baseT = typename Control::ttT;
  int P;
  int Q;

 public:
  explicit Control(Edge<Key<2>> &ctl) : baseT(edges(), edges(ctl), "Control", {}, {"ctl"}), P(0), Q(0) {}

  void op(std::tuple<Out<Key<2>>> &out) const {
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < Q; q++) {
        ttg::trace("Control: start computing on process {", p, ", ", q, "}");
        ::sendk<0>(Key<2>{p, q}, out);
      }
    }
  }

  void start(const int _p, const int _q) {
    P = _p;
    Q = _q;
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
  std::vector<int> sizes;
  // We load the entire matrix on each rank, but we only use the local part for the GEMM
  // loadMarket() is the eigan fuction to load matrix from a file
  if (!loadMarket(A, filename)) {
    std::cerr << "Failed to load " << filename << ", bailing out..." << std::endl;
    ttg::ttg_abort();
  }
  if (0 == ttg::default_execution_context().rank()) {
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

  if (ttg::default_execution_context().rank() == 0) {
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

  if (ttg::default_execution_context().rank() == 0) {
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
                              std::vector<int> &mTiles, std::vector<int> &nTiles, std::vector<int> &kTiles,
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

  int rank = ttg::default_execution_context().rank();

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
  std::cout << "A_elements.begin()" << A_elements.begin() << "A_elements.end()" << A_elements.end() << "\n";

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
static void initBlSpRandom(const std::function<int(const Key<2> &)> &keymap, size_t M, size_t N, size_t K, int minTs,
                           int maxTs, double avgDensity, SpMatrix<> &A, SpMatrix<> &B, SpMatrix<> &Aref,
                           SpMatrix<> &Bref, bool buildRefs, std::vector<int> &mTiles, std::vector<int> &nTiles,
                           std::vector<int> &kTiles, std::vector<std::vector<long>> &a_rowidx_to_colidx,
                           std::vector<std::vector<long>> &a_colidx_to_rowidx,
                           std::vector<std::vector<long>> &b_rowidx_to_colidx,
                           std::vector<std::vector<long>> &b_colidx_to_rowidx, double &average_tile_size,
                           double &Adensity, double &Bdensity, unsigned int seed) {
  int rank = ttg::default_execution_context().rank();

  int ts;
  std::mt19937 gen(seed);
  std::mt19937 genv(seed + 1);

  std::uniform_int_distribution<> dist(minTs, maxTs);  // randomly pick any value in the range minTs, maxTs
  using triplet_t = Eigen::Triplet<blk_t>;
  std::vector<triplet_t> A_elements;
  std::vector<triplet_t> B_elements;
  std::vector<triplet_t> Aref_elements;
  std::vector<triplet_t> Bref_elements;

  for (int m = 0; m < M; m += ts) {
    ts = dist(gen);
    if (ts > M - m) ts = M - m;
    mTiles.push_back(ts);
  }
  for (int n = 0; n < N; n += ts) {
    ts = dist(gen);
    if (ts > N - n) ts = N - n;
    nTiles.push_back(ts);
  }
  for (int k = 0; k < K; k += ts) {
    ts = dist(gen);
    if (ts > K - k) ts = K - k;
    kTiles.push_back(ts);
  }

  A.resize(mTiles.size(), kTiles.size());
  B.resize(kTiles.size(), nTiles.size());
  if (buildRefs) {
    Aref.resize(mTiles.size(), kTiles.size());
    Bref.resize(kTiles.size(), nTiles.size());
  }

  std::uniform_int_distribution<> mDist(0, mTiles.size() - 1);
  std::uniform_int_distribution<> nDist(0, nTiles.size() - 1);
  std::uniform_int_distribution<> kDist(0, kTiles.size() - 1);
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

  average_tile_size = (double)avg_nb / avg_nb_nb;
}
#endif

#endif

static void timed_measurement(SpMatrix<> &A, SpMatrix<> &B, const std::function<int(const Key<2> &)> &ij_keymap,
                              const std::function<int(const Key<3> &)> &ijk_keymap, const std::string &tiling_type,
                              double gflops, double avg_nb, double Adensity, double Bdensity,
                              const std::vector<std::vector<long>> &a_rowidx_to_colidx,
                              const std::vector<std::vector<long>> &a_colidx_to_rowidx,
                              const std::vector<std::vector<long>> &b_rowidx_to_colidx,
                              const std::vector<std::vector<long>> &b_colidx_to_rowidx, std::vector<int> &mTiles,
                              std::vector<int> &nTiles, std::vector<int> &kTiles, int M, int N, int K, int minTs,
                              int maxTs, int P, int Q, int R) {
  int MT = (int)A.rows();
  int NT = (int)B.cols();
  int KT = (int)A.cols();
  assert(KT == B.rows());

  SpMatrix<> C;
  C.resize(MT, NT);

  // flow graph needs to exist on every node
  Edge<Key<2>> ctl("control");
  Control control(ctl);
  Edge<Key<2>, blk_t> eA, eB;
  Edge<Key<2>, blk_t> eC;

  Read_SpMatrix a("A", A, ctl, eA, ij_keymap);
  Read_SpMatrix b("B", B, ctl, eB, ij_keymap);
  Write_SpMatrix<> c(C, eC, ij_keymap);
  auto &c_status = c.status();
  assert(!has_value(c_status));
  //  SpMM25D a_times_b(world, eA, eB, eC, A, B);
  SpMM25D<> a_times_b(eA, eB, eC, A, B, a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx,
                      mTiles, nTiles, kTiles, ij_keymap, ijk_keymap, R);
  TTGUNUSED(a);
  TTGUNUSED(b);
  TTGUNUSED(a_times_b);

  auto connected = make_graph_executable(&control);
  assert(connected);
  TTGUNUSED(connected);

  struct timeval start {
    0
  }, end{0}, diff{0};
  gettimeofday(&start, nullptr);
  // ready, go! need only 1 kick, so must be done by 1 thread only
  if (ttg::default_execution_context().rank() == 0) control.start(P, Q);
  fence();
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
  if (ttg::default_execution_context().rank() == 0) {
    std::cout << "TTG-" << rt << " PxQxR=   " << P << " " << Q << " " << R << " 1 average_NB= " << avg_nb << " M= " << M
              << " N= " << N << " K= " << K << " t= " << minTs << " T=" << maxTs << " Tiling= " << tiling_type
              << " A_density= " << Adensity << " B_density= " << Bdensity << " gflops= " << gflops << " seconds= " << tc
              << " gflops/s= " << gflops / tc << std::endl;
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
                             const std::vector<int> &mTiles, const std::vector<int> &nTiles,
                             const std::vector<int> &kTiles) {
  unsigned long flops = 0;
  for (auto i = 0; i < a_r2c.size(); i++) {
    for (auto kk = 0; kk < a_r2c[i].size(); kk++) {
      auto k = a_r2c[i][kk];
      if (k > b_r2c.size()) continue;
      for (auto jj = 0; jj < b_r2c[k].size(); jj++) {
        auto j = b_r2c[k][jj];
        flops += static_cast<long>(mTiles[i]) * nTiles[j] * kTiles[k];
      }
    }
  }
  return 2.0 * (double)flops / 1e9;
}

int main(int argc, char **argv) {
  bool timing;
  double gflops;

  // warm up silicon by calling gemm a few times
#ifdef BTAS_IS_USABLE
  for (int i = 0; i < 20; i++) {
    using baseT = typename btas::Tensor<double>;
    btas::Tensor<double, btas::Range, std::vector<double>> At(30, 30);
    btas::Tensor<double, btas::Range, std::vector<double>> Bt(30, 30);
    btas::Tensor<double, btas::Range, std::vector<double>> Ct(30, 30);
    At.fill(1.0);
    Bt.fill(2.0);
    Ct.fill(3.0);
    btas::gemm(std::move(Ct), Bt, At);
  }
#endif  // BTAS_IS_USABLE

  int cores = -1;
  std::string nbCoreStr(getCmdOption(argv, argv + argc, "-c"));
  cores = parseOption(nbCoreStr, cores);

  if (int dashdash = cmdOptionIndex(argv, argv + argc, "--") > -1) {
    initialize(argc - dashdash, argv + dashdash, cores);
  } else {
    initialize(1, argv, cores);
  }

  std::string debugStr(getCmdOption(argv, argv + argc, "-d"));
  auto debug = (unsigned int)parseOption(debugStr, 0);

  if (debug & (1 << 1)) {
    using ttg::Debugger;
    auto debugger = std::make_shared<Debugger>();
    Debugger::set_default_debugger(debugger);
    debugger->set_exec(argv[0]);
    debugger->set_prefix(ttg::default_execution_context().rank());
    // debugger->set_cmd("lldb_xterm");
    debugger->set_cmd("gdb_xterm");
  }

  int mpi_size = ttg::default_execution_context().size();
  int mpi_rank = ttg::default_execution_context().rank();
  int best_pqc = mpi_size;
  int P, Q, R;
  for (int c = 1; c <= (int)cbrt(mpi_size); c++) {
    for (int p = 1; p <= (int)sqrt(mpi_size / c); p++) {
      if ((mpi_size % (p * c)) == 0) {
        int q = mpi_size / (p * c);
        if (abs(c - p - q) <= best_pqc) {
          best_pqc = abs(c - p - q);
          P = p;
          Q = q;
          R = c;
        }
      }
    }
    // ttg::launch_lldb(ttg::default_execution_context().rank(), argv[0]);

    {
      if (debug & (1 << 0)) {
        ttg::trace_on();
        TTBase::set_trace_all(true);
      }

      SpMatrix<> A, B, C, Aref, Bref;
      std::string tiling_type;
      int M = 0, N = 0, K = 0;
      int minTs = 0, maxTs = 0;

      double avg_nb = nan("undefined");
      double Adensity = nan("undefined");
      double Bdensity = nan("undefined");

      std::string PStr(getCmdOption(argv, argv + argc, "-P"));
      P = parseOption(PStr, P);
      std::string QStr(getCmdOption(argv, argv + argc, "-Q"));
      Q = parseOption(QStr, Q);
      // to make code behave like 2D summa if R not given
      std::string RStr(getCmdOption(argv, argv + argc, "-R"));
      R = parseOption(RStr, 1);

      if (P * Q * R != mpi_size) {
        if (!cmdOptionExists(argv, argv + argc, "-Q") && (mpi_size % (P * R) == 0))
          Q = mpi_size / (P * R);
        else if (!cmdOptionExists(argv, argv + argc, "-P") && (mpi_size % (Q * R)) == 0)
          P = mpi_size / (Q * R);
        else if (!cmdOptionExists(argv, argv + argc, "-R") && (mpi_size % (Q * P)) == 0)
          R = mpi_size / (Q * P);
        else {
          if (0 == mpi_rank) {
            std::cerr << P << "x" << Q << "x" << R << " is not a valid process grid -- bailing out" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
          }
        }
      }

      auto ij_keymap = [P, Q](const Key<2> &ij) {
        int i = (int)ij[0];
        int j = (int)ij[1];
        int r = ij2rank(i, j, P, Q);
        return r;
      };

      auto ijk_keymap = [P, Q, R](const Key<3> &ijk) {
        int i = (int)ijk[0];
        int j = (int)ijk[1];
        int k = (int)ijk[2];
        int r = ijk2rank(i, j, k, P, Q, R);
        return r;
      };

      std::string seedStr(getCmdOption(argv, argv + argc, "-s"));
      unsigned int seed = parseOption(seedStr, 0);
      if (seed == 0) {
        std::random_device rd;
        seed = rd();
        if (0 == ttg::default_execution_context().rank()) std::cerr << "#Random seeded with " << seed << std::endl;
      }
      ttg_broadcast(ttg::default_execution_context(), seed, 0);

      std::vector<int> mTiles;
      std::vector<int> nTiles;
      std::vector<int> kTiles;
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
        initSpMatrixMarket(ij_keymap, filename, A, B, C, M, N, K);
      } else if (cmdOptionExists(argv, argv + argc, "-rmat")) {
        char *opt = getCmdOption(argv, argv + argc, "-rmat");
        tiling_type = "RandomSparseMatrix";
        initSpRmat(ij_keymap, opt, A, B, C, M, N, K, seed);
      } else {
        tiling_type = "HardCodedSparseMatrix";
        initSpHardCoded(ij_keymap, A, B, C, M, N, K);
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
        minTs = parseOption(minTsStr, 32);
        std::string maxTsStr(getCmdOption(argv, argv + argc, "-T"));
        maxTs = parseOption(maxTsStr, 256);
        std::string avgStr(getCmdOption(argv, argv + argc, "-a"));
        double avg = parseOption(avgStr, 0.3);
        timing = (check == 0);
        tiling_type = "RandomIrregularTiling";
        initBlSpRandom(ij_keymap, M, N, K, minTs, maxTs, avg, A, B, Aref, Bref, check, mTiles, nTiles, kTiles,
                       a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, avg_nb, Adensity,
                       Bdensity, seed);

        C.resize(mTiles.size(), nTiles.size());
      } else {
        tiling_type = "HardCodedBlockSparseMatrix";
        initBlSpHardCoded(ij_keymap, A, B, C, Aref, Bref, true, mTiles, nTiles, kTiles, a_rowidx_to_colidx,
                          a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, M, N, K);
      }
#endif  // !defined(BLOCK_SPARSE_GEMM)

      gflops = compute_gflops(a_rowidx_to_colidx, b_rowidx_to_colidx, mTiles, nTiles, kTiles);

      std::string nbrunStr(getCmdOption(argv, argv + argc, "-n"));
      int nb_runs = parseOption(nbrunStr, 1);

      if (timing) {
        // Start up engine
        execute();
        for (int nrun = 0; nrun < nb_runs; nrun++) {
          timed_measurement(A, B, ij_keymap, ijk_keymap, tiling_type, gflops, avg_nb, Adensity, Bdensity,
                            a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx, b_colidx_to_rowidx, mTiles,
                            nTiles, kTiles, M, N, K, minTs, maxTs, P, Q, R);
        }
      } else {
        // flow graph needs to exist on every node
        // N.B. to validate C we need it on node 0!
        auto keymap_write = [](const Key<2> &key) { return 0; };
        Edge<Key<2>> ctl("control");
        Control control(ctl);
        Edge<Key<2>, blk_t> eA, eB, eC;
        Read_SpMatrix a("A", A, ctl, eA, ij_keymap);
        Read_SpMatrix b("B", B, ctl, eB, ij_keymap);
        Write_SpMatrix<> c(C, eC, keymap_write);
        auto &c_status = c.status();
        assert(!has_value(c_status));
        //  SpMM25D a_times_b(world, eA, eB, eC, A, B);
        SpMM25D<> a_times_b(eA, eB, eC, A, B, a_rowidx_to_colidx, a_colidx_to_rowidx, b_rowidx_to_colidx,
                            b_colidx_to_rowidx, mTiles, nTiles, kTiles, ij_keymap, ijk_keymap, R);
        TTGUNUSED(a_times_b);
        // calling the Dot constructor with 'true' argument disables the type
        if (default_execution_context().rank() == 0) std::cout << Dot{/*disable_type=*/true}(&control) << std::endl;

        // ready to run!
        auto connected = make_graph_executable(&control);
        assert(connected);
        TTGUNUSED(connected);

        // ready, go! need only 1 kick, so must be done by 1 thread only
        if (ttg::default_execution_context().rank() == 0) control.start(P, Q);

        execute();
        fence();

        // validate C=A*B against the reference output
        assert(has_value(c_status));
        if (ttg::default_execution_context().rank() == 0) {
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
        //      if (ttg::default_execution_context().rank() == 0) {
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
}
