#include <array>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/SparseCore>
#if __has_include(<btas/features.h>)
#include <btas/features.h>
#ifdef BTAS_IS_USABLE
#include <btas/btas.h>
#include <btas/optimize/contract.h>
#include <btas/serialization.h>
#else
#warning "found btas/features.h but Boost.Iterators is missing, hence BTAS is unusable ... add -I/path/to/boost"
#endif
#endif

#include <sys/time.h>
#include <unsupported/Eigen/SparseExtra>

#include <boost/graph/directed_graph.hpp>
#include <boost/graph/rmat_graph_generator.hpp>
#include <boost/random/linear_congruential.hpp>

#include "ttg.h"

using namespace ttg;

#include "ttg/util/future.h"

#if defined(BLOCK_SPARSE_GEMM) && defined(BTAS_IS_USABLE)
using blk_t = btas::Tensor<double>;

// declare btas::Tensor serializable by Boost
#include "ttg/serialization/backends/boost.h"
namespace ttg::detail {
  template <typename Archive>
  inline static constexpr bool is_boost_serializable_v<Archive, blk_t> = is_boost_archive_v<Archive>;
}

#else
using blk_t = double;
#endif
template <typename T = blk_t>
using SpMatrix = Eigen::SparseMatrix<T>;

#if defined(BLOCK_SPARSE_GEMM) && defined(BTAS_IS_USABLE)

#if __has_include(<madness/world/archive.h>)

#include <madness/world/archive.h>

#endif  // __has_include(<madness/world/archive.h>)

namespace btas {
  template <typename _T, class _Range, class _Store>
  inline btas::Tensor<_T, _Range, _Store> operator*(const btas::Tensor<_T, _Range, _Store> &A,
                                                    const btas::Tensor<_T, _Range, _Store> &B) {
    btas::Tensor<_T, _Range, _Store> C;
    btas::contract(1.0, A, {1, 2}, B, {2, 3}, 0.0, C, {1, 3});
    return C;
  }

  template <typename _T, class _Range, class _Store>
  btas::Tensor<_T, _Range, _Store> gemm(btas::Tensor<_T, _Range, _Store> &&C, const btas::Tensor<_T, _Range, _Store> &A,
                                        const btas::Tensor<_T, _Range, _Store> &B) {
    using array = btas::DEFAULT::index<int>;
    if (C.empty()) {
      C = btas::Tensor<_T, _Range, _Store>(btas::Range(A.range().extent(0), B.range().extent(1)), 0.0);
    }
    btas::contract_222(1.0, A, array{1, 2}, B, array{2, 3}, 1.0, C, array{1, 3}, false, false);
    return std::move(C);
  }
}  // namespace btas
#endif  // BTAS_IS_USABLE
double gemm(double C, double A, double B) { return C + A * B; }
/////////////////////////////////////////////

template <typename _Scalar, int _Options, typename _StorageIndex>
struct colmajor_layout;
template <typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::ColMajor, _StorageIndex> : public std::true_type {};
template <typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::RowMajor, _StorageIndex> : public std::false_type {};

template <std::size_t Rank>
struct Key : public std::array<long, Rank> {
  static constexpr const long max_index_width = 21;
  static constexpr const long max_index = 1 << 21;
  static constexpr const long max_index_square = max_index * max_index;
  Key() = default;
  template <typename Integer>
  Key(std::initializer_list<Integer> ilist) {
    std::copy(ilist.begin(), ilist.end(), this->begin());
    assert(valid());
  }
  Key(std::size_t hash) {
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

// flow data from an existing SpMatrix on rank 0
template <typename Blk = blk_t>
class Read_SpMatrix : public Op<void, std::tuple<Out<Key<2>, Blk>>, Read_SpMatrix<Blk>, void> {
 public:
  using baseT = Op<void, std::tuple<Out<Key<2>, Blk>>, Read_SpMatrix<Blk>, void>;

  Read_SpMatrix(const char *label, const SpMatrix<Blk> &matrix, Edge<> &ctl, Edge<Key<2>, Blk> &out)
      : baseT(edges(ctl), edges(out), std::string("read_spmatrix(") + label + ")", {"ctl"}, {std::string(label) + "ij"},
              []() { return 0; })
      , matrix_(matrix) {}

  void op(std::tuple<Out<Key<2>, Blk>> &out) {
    for (int k = 0; k < matrix_.outerSize(); ++k) {
      for (typename SpMatrix<Blk>::InnerIterator it(matrix_, k); it; ++it) {
        ::send<0>(Key<2>({it.row(), it.col()}), it.value(), out);
      }
    }
  }

 private:
  const SpMatrix<Blk> &matrix_;
};

// flow (move?) data into an existing SpMatrix on rank 0
template <typename Blk = blk_t>
class Write_SpMatrix : public Op<Key<2>, std::tuple<>, Write_SpMatrix<Blk>, Blk> {
 public:
  using baseT = Op<Key<2>, std::tuple<>, Write_SpMatrix<Blk>, Blk>;

  Write_SpMatrix(SpMatrix<Blk> &matrix, Edge<Key<2>, Blk> &in)
      : baseT(edges(in), edges(), "write_spmatrix", {"Cij"}, {}, [](auto key) { return 0; }), matrix_(matrix) {}

  void op(const Key<2> &key, typename baseT::input_values_tuple_type &&elem, std::tuple<> &) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (ttg::tracing()) {
      auto &w = get_default_world();
      ttg::print(w.rank(), "/", reinterpret_cast<std::uintptr_t>(pthread_self()), "spmm.impl.h Write_SpMatrix wrote {",
                 key[0], ",", key[1], "} = ", baseT::template get<0>(elem), " in ", static_cast<void *>(&matrix_),
                 " with mutex @", static_cast<void *>(&mtx_), " for object @", static_cast<void *>(this));
    }
    matrix_.insert(key[0], key[1]) = baseT::template get<0>(elem);
  }

  /// grab completion status as a future<void>
  /// \note cannot be called once this is executable
  const std::shared_future<void> &status() const {
    assert(!this->is_executable());
    if (!completion_status_) {
      auto promise = std::make_shared<std::promise<void>>();
      completion_status_ = std::make_shared<std::shared_future<void>>(promise->get_future());
      ttg_register_status(this->get_world(), std::move(promise));
    }
    return *completion_status_.get();
  }

 private:
  std::mutex mtx_;
  SpMatrix<Blk> &matrix_;
  mutable std::shared_ptr<std::shared_future<void>> completion_status_;
};

// sparse mm
template <typename Blk = blk_t>
class SpMM {
 public:
  SpMM(Edge<Key<2>, Blk> &a, Edge<Key<2>, Blk> &b, Edge<Key<2>, Blk> &c, const SpMatrix<Blk> &a_mat,
       const SpMatrix<Blk> &b_mat)
      : a_ijk_()
      , b_ijk_()
      , c_ijk_()
      , a_rowidx_to_colidx_(make_rowidx_to_colidx(a_mat))
      , b_colidx_to_rowidx_(make_colidx_to_rowidx(b_mat))
      , a_colidx_to_rowidx_(make_colidx_to_rowidx(a_mat))
      , b_rowidx_to_colidx_(make_rowidx_to_colidx(b_mat)) {
    // data is on rank 0, broadcast metadata from there
    int root = 0;
    ttg_broadcast(ttg_default_execution_context(), a_rowidx_to_colidx_, root);
    ttg_broadcast(ttg_default_execution_context(), b_rowidx_to_colidx_, root);
    ttg_broadcast(ttg_default_execution_context(), a_colidx_to_rowidx_, root);
    ttg_broadcast(ttg_default_execution_context(), b_colidx_to_rowidx_, root);

    bcast_a_ = std::make_unique<BcastA>(a, a_ijk_, b_rowidx_to_colidx_);
    bcast_b_ = std::make_unique<BcastB>(b, b_ijk_, a_colidx_to_rowidx_);
    multiplyadd_ = std::make_unique<MultiplyAdd>(a_ijk_, b_ijk_, c_ijk_, c, a_rowidx_to_colidx_, b_colidx_to_rowidx_);
  }

  /// broadcast A[i][k] to all {i,j,k} such that B[j][k] exists
  class BcastA : public Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastA, Blk> {
   public:
    using baseT = Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastA, Blk>;

    BcastA(Edge<Key<2>, Blk> &a, Edge<Key<3>, Blk> &a_ijk, const std::vector<std::vector<long>> &b_rowidx_to_colidx)
        : baseT(edges(a), edges(a_ijk), "SpMM::bcast_a", {"a_ik"}, {"a_ijk"})
        , b_rowidx_to_colidx_(b_rowidx_to_colidx) {}

    void op(const Key<2> &key, typename baseT::input_values_tuple_type &&a_ik, std::tuple<Out<Key<3>, Blk>> &a_ijk) {
      const auto i = key[0];
      const auto k = key[1];
      // broadcast a_ik to all existing {i,j,k}
      std::vector<Key<3>> ijk_keys;
      for (auto &j : b_rowidx_to_colidx_[k]) {
        if (tracing()) ttg::print("Broadcasting A[", i, "][", k, "] to j=", j);
        ijk_keys.emplace_back(Key<3>({i, j, k}));
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(a_ik), a_ijk);
    }

   private:
    const std::vector<std::vector<long>> &b_rowidx_to_colidx_;
  };  // class BcastA

  /// broadcast B[k][j] to all {i,j,k} such that A[i][k] exists
  class BcastB : public Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastB, Blk> {
   public:
    using baseT = Op<Key<2>, std::tuple<Out<Key<3>, Blk>>, BcastB, Blk>;

    BcastB(Edge<Key<2>, Blk> &b, Edge<Key<3>, Blk> &b_ijk, const std::vector<std::vector<long>> &a_colidx_to_rowidx)
        : baseT(edges(b), edges(b_ijk), "SpMM::bcast_b", {"b_kj"}, {"b_ijk"})
        , a_colidx_to_rowidx_(a_colidx_to_rowidx) {}

    void op(const Key<2> &key, typename baseT::input_values_tuple_type &&b_kj, std::tuple<Out<Key<3>, Blk>> &b_ijk) {
      const auto k = key[0];
      const auto j = key[1];
      // broadcast b_kj to *jk
      std::vector<Key<3>> ijk_keys;
      for (auto &i : a_colidx_to_rowidx_[k]) {
        if (tracing()) ttg::print("Broadcasting B[", k, "][", j, "] to i=", i);
        ijk_keys.emplace_back(Key<3>({i, j, k}));
      }
      ::broadcast<0>(ijk_keys, baseT::template get<0>(b_kj), b_ijk);
    }

   private:
    const std::vector<std::vector<long>> &a_colidx_to_rowidx_;
  };  // class BcastA

  /// multiply task has 3 input flows: a_ijk, b_ijk, and c_ijk, c_ijk contains the running total
  class MultiplyAdd : public Op<Key<3>, std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>>, MultiplyAdd, Blk, Blk, Blk> {
   public:
    using baseT = Op<Key<3>, std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>>, MultiplyAdd, Blk, Blk, Blk>;

    MultiplyAdd(Edge<Key<3>, Blk> &a_ijk, Edge<Key<3>, Blk> &b_ijk, Edge<Key<3>, Blk> &c_ijk, Edge<Key<2>, Blk> &c,
                const std::vector<std::vector<long>> &a_rowidx_to_colidx,
                const std::vector<std::vector<long>> &b_colidx_to_rowidx)
        : baseT(edges(a_ijk, b_ijk, c_ijk), edges(c, c_ijk), "SpMM::MultiplyAdd", {"a_ijk", "b_ijk", "c_ijk"},
                {"c_ij", "c_ijk"})
        , a_rowidx_to_colidx_(a_rowidx_to_colidx)
        , b_colidx_to_rowidx_(b_colidx_to_rowidx) {
      auto &keymap = this->get_keymap();

      // for each i and j that belongs to this node
      // determine first k that contributes, initialize input {i,j,first_k} flow to 0
      for (auto i = 0ul; i != a_rowidx_to_colidx_.size(); ++i) {
        if (a_rowidx_to_colidx_[i].empty()) continue;
        for (auto j = 0ul; j != b_colidx_to_rowidx_.size(); ++j) {
          if (b_colidx_to_rowidx_[j].empty()) continue;

          // assuming here {i,j,k} for all k map to same node
          auto owner = keymap(Key<3>({i, j, 0ul}));
          if (owner == ttg_default_execution_context().rank()) {
            if (true) {
              decltype(i) k;
              bool have_k;
              std::tie(k, have_k) = compute_first_k(i, j);
              if (have_k) {
                if (tracing()) ttg::print("Initializing C[", i, "][", j, "] to zero");
                this->template in<2>()->send(Key<3>({i, j, k}), Blk(0));
                // this->set_arg<2>(Key<3>({i,j,k}), Blk(0));
              } else {
                if (tracing()) ttg::print("C[", i, "][", j, "] is empty");
              }
            }
          }
        }
      }
    }

    void op(const Key<3> &key, typename baseT::input_values_tuple_type &&_ijk,
            std::tuple<Out<Key<2>, Blk>, Out<Key<3>, Blk>> &result) {
      const auto i = key[0];
      const auto j = key[1];
      const auto k = key[2];
      long next_k;
      bool have_next_k;
      std::tie(next_k, have_next_k) = compute_next_k(i, j, k);
      if (tracing()) {
        ttg::print("C[", i, "][", j, "]  += A[", i, "][", k, "] by B[", k, "][", j, "],  next_k? ",
                   (have_next_k ? std::to_string(next_k) : "does not exist"));
      }
      // compute the contrib, pass the running total to the next flow, if needed
      // otherwise write to the result flow
      if (have_next_k) {
        ::send<1>(
            Key<3>({i, j, next_k}),
            gemm(std::move(baseT::template get<2>(_ijk)), baseT::template get<0>(_ijk), baseT::template get<1>(_ijk)),
            result);
      } else
        ::send<0>(
            Key<2>({i, j}),
            gemm(std::move(baseT::template get<2>(_ijk)), baseT::template get<0>(_ijk), baseT::template get<1>(_ijk)),
            result);
    }

   private:
    const std::vector<std::vector<long>> &a_rowidx_to_colidx_;
    const std::vector<std::vector<long>> &b_colidx_to_rowidx_;

    // given {i,j} return first k such that A[i][k] and B[k][j] exist
    std::tuple<long, bool> compute_first_k(long i, long j) {
      auto a_iter_fence = a_rowidx_to_colidx_[i].end();
      auto a_iter = a_rowidx_to_colidx_[i].begin();
      if (a_iter == a_iter_fence) return std::make_tuple(-1, false);
      auto b_iter_fence = b_colidx_to_rowidx_[j].end();
      auto b_iter = b_colidx_to_rowidx_[j].begin();
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
    std::tuple<long, bool> compute_next_k(long i, long j, long k) {
      auto a_iter_fence = a_rowidx_to_colidx_[i].end();
      auto a_iter = std::find(a_rowidx_to_colidx_[i].begin(), a_iter_fence, k);
      assert(a_iter != a_iter_fence);
      auto b_iter_fence = b_colidx_to_rowidx_[j].end();
      auto b_iter = std::find(b_colidx_to_rowidx_[j].begin(), b_iter_fence, k);
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
  };

 private:
  Edge<Key<3>, Blk> a_ijk_;
  Edge<Key<3>, Blk> b_ijk_;
  Edge<Key<3>, Blk> c_ijk_;
  std::vector<std::vector<long>> a_rowidx_to_colidx_;
  std::vector<std::vector<long>> b_colidx_to_rowidx_;
  std::vector<std::vector<long>> a_colidx_to_rowidx_;
  std::vector<std::vector<long>> b_rowidx_to_colidx_;
  std::unique_ptr<BcastA> bcast_a_;
  std::unique_ptr<BcastB> bcast_b_;
  std::unique_ptr<MultiplyAdd> multiplyadd_;

  // result[i][j] gives the j-th nonzero row for column i in matrix mat
  std::vector<std::vector<long>> make_colidx_to_rowidx(const SpMatrix<Blk> &mat) {
    std::vector<std::vector<long>> colidx_to_rowidx;
    for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
      for (typename SpMatrix<Blk>::InnerIterator it(mat, k); it; ++it) {
        const std::size_t row = it.row();
        const std::size_t col = it.col();
        if (col >= colidx_to_rowidx.size()) colidx_to_rowidx.resize(col + 1);
        // in either case (col- or row-major) row index increasing for the given col
        colidx_to_rowidx[col].push_back(row);
      }
    }
    return colidx_to_rowidx;
  }
  // result[i][j] gives the j-th nonzero column for row i in matrix mat
  std::vector<std::vector<long>> make_rowidx_to_colidx(const SpMatrix<Blk> &mat) {
    std::vector<std::vector<long>> rowidx_to_colidx;
    for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
      for (typename SpMatrix<Blk>::InnerIterator it(mat, k); it; ++it) {
        const std::size_t row = it.row();
        const std::size_t col = it.col();
        if (row >= rowidx_to_colidx.size()) rowidx_to_colidx.resize(row + 1);
        // in either case (col- or row-major) col index increasing for the given row
        rowidx_to_colidx[row].push_back(col);
      }
    }
    return rowidx_to_colidx;
  }
};

class Control : public Op<void, std::tuple<Out<>>, Control> {
  using baseT = Op<void, std::tuple<Out<>>, Control>;

 public:
  Control(Edge<> &ctl) : baseT(edges(), edges(ctl), "Control", {}, {"ctl"}) {}

  void op(std::tuple<Out<>> &out) { ::send<0>(out); }

  void start() { invoke(); }
};

#ifdef BTAS_IS_USABLE
template <typename _T, class _Range, class _Store>
std::tuple<_T, _T> norms(const btas::Tensor<_T, _Range, _Store> &t) {
  _T norm_2_square = 0.0;
  _T norm_inf = 0.0;
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

#include "../ttg_matrix.h"

char *getCmdOption(char **begin, char **end, const std::string &option) {
  static char *empty = "";
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) return *itr;
  return empty;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

int cmdOptionIndex(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end) return itr - begin;
  return -1;
}

static int parseOption(std::string &option, int default_value) {
  size_t pos;
  std::string token;
  int N = default_value;
  if (option.length() == 0) return N;
  pos = option.find(":");
  if (pos == std::string::npos) {
    pos = option.length();
  }
  token = option.substr(0, pos);
  N = std::stoi(token);
  option.erase(0, pos + 1);
  return N;
}

static double parseOption(std::string &option, double default_value) {
  size_t pos;
  std::string token;
  double N = default_value;
  if (option.length() == 0) return N;
  pos = option.find(":");
  if (pos == std::string::npos) {
    pos = option.length();
  }
  token = option.substr(0, pos);
  N = std::stod(token);
  option.erase(0, pos + 1);
  return N;
}

#if !defined(BLOCK_SPARSE_GEMM)
static void initSpMatrixMarket(const char *filename, SpMatrix<> &A, SpMatrix<> &B, SpMatrix<> &C) {
  std::vector<int> sizes;
  // rank 0 only: initialize inputs (these will become shapes when switch to blocks)
  if (ttg_default_execution_context().rank() == 0) {
    if (!loadMarket(A, filename)) {
      std::cerr << "Failed to load " << filename << ", bailing out..." << std::endl;
      ttg::ttg_abort();
    }
    std::cout << "##MatrixMarket file " << filename << " -- " << A.rows() << " x " << A.cols() << " -- " << A.nonZeros()
              << " nnz (density: " << (float)A.nonZeros() / (float)A.rows() / (float)A.cols() << ")" << std::endl;
    sizes[0] = A.rows();
    sizes[1] = A.cols();
  }
  ttg_broadcast(ttg_default_execution_context(), sizes, 0);
  if (ttg_default_execution_context().rank() == 0) {
    A.resize(sizes[0], sizes[1]);
  }
  if (A.rows() != A.cols()) {
    B = A.transpose();
  } else {
    B = A;
  }
  C.resize(A.rows(), B.cols());
}

static void initSpRmat(const char *opt, SpMatrix<> &A, SpMatrix<> &B, SpMatrix<> &C) {
  int N, E = -1;
  double a = 0.25, b = 0.25, c = 0.25, d = 0.25;
  size_t nnz = 0;

  if (nullptr == opt) {
    std::cerr << "Usage: -rmat <#nodes>[:<#edges>[:<a>[:<b>:[<c>[:<d>]]]]]" << std::endl;
    exit(1);
  }
  std::string token;
  std::string option = std::string(opt);
  N = parseOption(option, -1);

  A.resize(N, N);

  if (ttg_default_execution_context().rank() == 0) {
    E = parseOption(option, (int)(0.01 * N * N));
    a = parseOption(option, 0.25);
    b = parseOption(option, 0.25);
    c = parseOption(option, 0.25);
    d = parseOption(option, 0.25);

    std::cout << "#R-MAT: " << N << " nodes, " << E << " edges, a/b/c/d = " << a << "/" << b << "/" << c << "/" << d
              << std::endl;

    boost::minstd_rand gen;
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
  }

  B = A;
  C.resize(N, N);

  std::cout << "#R-MAT: " << E << " nonzero elements, density: " << (double)nnz / (double)N / (double)N << std::endl;
}

static void initSpHardCoded(SpMatrix<> &A, SpMatrix<> &B, SpMatrix<> &C) {
  const int n = 2;
  const int m = 3;
  const int k = 4;

  std::cout << "#HardCoded A, B, C" << std::endl;
  A.resize(n, k);
  B.resize(k, m);
  C.resize(n, m);
  // rank 0 only: initialize inputs (these will become shapes when switch to blocks)
  if (ttg_default_execution_context().rank() == 0) {
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
}
#else
static void initBlSpHardCoded(SpMatrix<> &A, SpMatrix<> &B, SpMatrix<> &C) {
  const int n = 2;
  const int m = 3;
  const int k = 4;

  std::cout << "#HardCoded A, B, C" << std::endl;
  A.resize(n, k);
  B.resize(k, m);
  C.resize(n, m);
  // rank 0 only: initialize inputs (these will become shapes when switch to blocks)
  if (ttg_default_execution_context().rank() == 0) {
    using triplet_t = Eigen::Triplet<blk_t>;
    std::vector<triplet_t> A_elements;
#if defined(BTAS_IS_USABLE)
    auto A_blksize = {128, 256};
    A_elements.emplace_back(0, 1, blk_t(btas::Range(A_blksize), 12.3));
    A_elements.emplace_back(0, 2, blk_t(btas::Range(A_blksize), 10.7));
    A_elements.emplace_back(0, 3, blk_t(btas::Range(A_blksize), -2.3));
    A_elements.emplace_back(1, 0, blk_t(btas::Range(A_blksize), -0.3));
    A_elements.emplace_back(1, 2, blk_t(btas::Range(A_blksize), 1.2));
#else
    A_elements.emplace_back(0, 1, 12.3);
    A_elements.emplace_back(0, 2, 10.7);
    A_elements.emplace_back(0, 3, -2.3);
    A_elements.emplace_back(1, 0, -0.3);
    A_elements.emplace_back(1, 2, .2);
#endif
    A.setFromTriplets(A_elements.begin(), A_elements.end());

    std::vector<triplet_t> B_elements;
#if defined(BTAS_IS_USABLE)
    auto B_blksize = {256, 196};
    B_elements.emplace_back(0, 0, blk_t(btas::Range(B_blksize), 12.3));
    B_elements.emplace_back(1, 0, blk_t(btas::Range(B_blksize), 10.7));
    B_elements.emplace_back(3, 0, blk_t(btas::Range(B_blksize), -2.3));
    B_elements.emplace_back(1, 1, blk_t(btas::Range(B_blksize), -0.3));
    B_elements.emplace_back(1, 2, blk_t(btas::Range(B_blksize), 1.2));
    B_elements.emplace_back(2, 2, blk_t(btas::Range(B_blksize), 7.2));
    B_elements.emplace_back(3, 2, blk_t(btas::Range(B_blksize), 0.2));
#else
    B_elements.emplace_back(0, 0, 12.3);
    B_elements.emplace_back(1, 0, 10.7);
    B_elements.emplace_back(3, 0, -2.3);
    B_elements.emplace_back(1, 1, -0.3);
    B_elements.emplace_back(1, 2, 1.2);
    B_elements.emplace_back(2, 2, 7.2);
    B_elements.emplace_back(3, 2, 0.2);
#endif
    B.setFromTriplets(B_elements.begin(), B_elements.end());
  }
}

#if defined(BTAS_IS_USABLE)
static void initBlSpRandom(int M, int N, int K, int minTs, int maxTs, double avgDensity, SpMatrix<> &A, SpMatrix<> &B,
                           SpMatrix<> &C) {
  A.resize(M, K);
  B.resize(K, N);
  C.resize(M, N);

  if (ttg_default_execution_context().rank() == 0) {
    float density = 0.0;
    int ts;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(minTs, maxTs);
    std::vector<int> mTiles, nTiles, kTiles;
    std::map<std::tuple<int, int>, bool> filling;
    using triplet_t = Eigen::Triplet<blk_t>;
    std::vector<triplet_t> A_elements;
    std::vector<triplet_t> B_elements;

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

    std::uniform_int_distribution<> mDist(0, mTiles.size() - 1);
    std::uniform_int_distribution<> nDist(0, nTiles.size() - 1);
    std::uniform_int_distribution<> kDist(0, kTiles.size() - 1);

    density = 0.0;
    filling.clear();
    while (density < avgDensity) {
      int mt = mDist(gen);
      int kt = kDist(gen);
      int m = 0, k = 0;
      if (filling.count({mt, kt}) > 0) continue;
      filling[{mt, kt}] = true;
      density += (float)(mTiles[mt] * kTiles[kt]) / (float)(M * K);
      auto blksize = {mTiles[mt], kTiles[kt]};
      for (int i = 0; i < mt; i++) m += mTiles[i];
      for (int i = 0; i < kt; i++) k += kTiles[i];
      A_elements.emplace_back(m, k, blk_t(btas::Range(blksize), (float)std::rand() / (float)RAND_MAX - 0.5));
    }
    A.setFromTriplets(A_elements.begin(), A_elements.end());
    std::cout << "#RandomBlockSparse-matrixT: A = " << M << " x " << K << " minTs " << minTs << " maxTs " << maxTs
              << " density " << density << std::endl;

    density = 0.0;
    filling.clear();
    while (density < avgDensity) {
      int nt = nDist(gen);
      int kt = kDist(gen);
      int n = 0, k = 0;
      if (filling.count({kt, nt}) > 0) continue;
      filling[{kt, nt}] = true;
      density += (float)(kTiles[kt] * nTiles[nt]) / (float)(K * N);
      auto blksize = {kTiles[kt], nTiles[nt]};
      for (int i = 0; i < nt; i++) n += nTiles[i];
      for (int i = 0; i < kt; i++) k += kTiles[i];
      B_elements.emplace_back(k, n, blk_t(btas::Range(blksize), (float)std::rand() / (float)RAND_MAX - 0.5));
    }
    B.setFromTriplets(B_elements.begin(), B_elements.end());
    std::cout << "#RandomBlockSparse-matrixT: B = " << K << " x " << N << " minTs " << minTs << " maxTs " << maxTs
              << " density " << density << std::endl;
  }
}
#endif

#endif

int main(int argc, char **argv) {
  bool timing = false;

  if (int dashdash = cmdOptionIndex(argv, argv + argc, "--") > -1) {
    ttg_initialize(argc - dashdash, argv + dashdash, -1);
  } else {
    ttg_initialize(1, argv, -1);
  }

  //  using mpqc::Debugger;
  //  auto debugger = std::make_shared<Debugger>();
  //  Debugger::set_default_debugger(debugger);
  //  debugger->set_exec(argv[0]);
  //  debugger->set_prefix(ttg_default_execution_context().rank());
  //  debugger->set_cmd("lldb_xterm");
  //
  //  initialize_watchpoints();

  {
    // ttg::trace_on();
    // OpBase::set_trace_all(true);

    SpMatrix<> A, B, C;

#if !defined(BLOCK_SPARSE_GEMM)
    if (cmdOptionExists(argv, argv + argc, "-mm")) {
      char *filename = getCmdOption(argv, argv + argc, "-mm");
      timing = true;
      initSpMatrixMarket(filename, A, B, C);
    } else if (cmdOptionExists(argv, argv + argc, "-rmat")) {
      char *opt = getCmdOption(argv, argv + argc, "-rmat");
      timing = true;
      initSpRmat(opt, A, B, C);
    } else {
      initSpHardCoded(A, B, C);
    }
#else
    if (argc >= 1) {
      std::string Mstr(getCmdOption(argv, argv + argc, "-M"));
      int M = parseOption(Mstr, 1200);
      std::string Nstr(getCmdOption(argv, argv + argc, "-N"));
      int N = parseOption(Nstr, 1200);
      std::string Kstr(getCmdOption(argv, argv + argc, "-K"));
      int K = parseOption(Kstr, 1200);
      std::string minTsStr(getCmdOption(argv, argv + argc, "-t"));
      int minTs = parseOption(minTsStr, 32);
      std::string maxTsStr(getCmdOption(argv, argv + argc, "-T"));
      int maxTs = parseOption(maxTsStr, 256);
      std::string avgStr(getCmdOption(argv, argv + argc, "-a"));
      double avg = parseOption(avgStr, 0.3);
      timing = true;
      initBlSpRandom(M, N, K, minTs, maxTs, avg, A, B, C);
    } else {
      initBlSpHardCoded(A, B, C);
    }
#endif  // !defined(BLOCK_SPARSE_GEMM)

    // flow graph needs to exist on every node
    Edge<> ctl("control");
    Control control(ctl);
    Edge<Key<2>, blk_t> eA, eB, eC;
    Read_SpMatrix<> a("A", A, ctl, eA);
    Read_SpMatrix<> b("B", B, ctl, eB);
    Write_SpMatrix<> c(C, eC);
    auto c_status = c.status();
    assert(!has_value(c_status));
    //  SpMM a_times_b(world, eA, eB, eC, A, B);
    SpMM<> a_times_b(eA, eB, eC, A, B);

    std::cout << Dot{}(&a, &b) << std::endl;

    // ready to run!
    auto connected = make_graph_executable(&control);
    assert(connected);
    TTGUNUSED(connected);

    if (timing) {
      struct timeval start, end, diff;
      gettimeofday(&start, NULL);
      // ready, go! need only 1 kick, so must be done by 1 thread only
      if (ttg_default_execution_context().rank() == 0) control.start();
      ttg_execute(ttg_default_execution_context());
      ttg_fence(ttg_default_execution_context());
      gettimeofday(&end, NULL);
      timersub(&end, &start, &diff);
      std::cout << "Time to completion: " << diff.tv_sec + (double)diff.tv_usec / 1e6 << "s" << std::endl;
    } else {
      // ready, go! need only 1 kick, so must be done by 1 thread only
      if (ttg_default_execution_context().rank() == 0) control.start();
    }

    ///////////////////////////////////////////////////////////////////////////
    // copy matrix using ttg::Matrix
    Matrix<blk_t> aflow;
    aflow << A;
    SpMatrix<> Acopy(A.rows(), A.cols());  // resizing will be automatic in the future when shape computation is
                                           // complete .. see Matrix::operator>>
    auto copy_status = aflow >> Acopy;
    assert(!has_value(copy_status));
    aflow.pushall();
    Control control2(ttg_ctl_edge(ttg_default_execution_context()));
    {
      // std::cout << "matrix copy using ttg::Matrix" << std::endl;
      //      if (ttg_default_execution_context().rank() == 0) std::cout << Dot{}(&control2) << std::endl;

      // ready to run!
      auto connected = make_graph_executable(&control2);
      assert(connected);
      TTGUNUSED(connected);

      // ready, go! need only 1 kick, so must be done by 1 thread only
      if (ttg_default_execution_context().rank() == 0) control2.start();
    }
    //////////////////////////////////////////////////////////////////////////

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());

    // validate C=A*B against the reference output
    assert(has_value(c_status));
    if (ttg_default_execution_context().rank() == 0) {
      SpMatrix<> Cref = A * B;

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
    assert(has_value(copy_status));
    if (ttg_default_execution_context().rank() == 0) {
      double norm_2_square, norm_inf;
      std::tie(norm_2_square, norm_inf) = norms<blk_t>(Acopy - A);
      std::cout << "||Acopy - A||_2      = " << std::sqrt(norm_2_square) << std::endl;
      std::cout << "||Acopy - A||_\\infty = " << norm_inf << std::endl;
      if (::ttg::tracing()) {
        std::cout << "Acopy (" << static_cast<void *>(&Acopy) << "):\n" << Acopy << std::endl;
        std::cout << "A (" << static_cast<void *>(&A) << "):\n" << A << std::endl;
      }
      if (norm_inf != 0) {
        ttg_abort();
      }
    }
  }

  ttg_finalize();

  return 0;
}
