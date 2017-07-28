#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <array>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include <Eigen/SparseCore>
#if __has_include(<btas/features.h>)
# include <btas/features.h>
# ifdef BTAS_IS_USABLE
#  include <btas/btas.h>
# else
#  warning "found btas/features.h but Boost.Iterators is missing, hence BTAS is unusable ... add -I/path/to/boost"
# endif
#endif

#include "madness/ttg.h"

using namespace madness;
using namespace madness::ttg;
using namespace ::ttg;

#if defined(BLOCK_SPARSE_GEMM) && defined(BTAS_IS_USABLE)
using blk_t = btas::Tensor<double>;
#else
using blk_t = double;
#endif
using SpMatrix = Eigen::SparseMatrix<blk_t>;

/////////////////////////////////////////////
// additional ops are needed to make Eigen::SparseMatrix<btas::Tensor> possible
#ifdef BTAS_IS_USABLE
namespace madness {
  namespace archive {

    template<class Archive, typename T>
    struct ArchiveLoadImpl<Archive, btas::varray<T> > {
        static inline void load(const Archive& ar, btas::varray<T>& x) {
          typename btas::varray<T>::size_type n;
          ar & n;
          x.resize(n);
          for (typename btas::varray<T>::value_type& xi : x)
            ar & xi;
        }
    };

    template<class Archive, typename T>
    struct ArchiveStoreImpl<Archive, btas::varray<T> > {
        static inline void store(const Archive& ar, const btas::varray<T>& x) {
          ar & x.size();
          for (const typename btas::varray<T>::value_type& xi : x)
            ar & xi;
        }
    };

    template<class Archive, typename _T, class _Range, class _Store>
    struct ArchiveSerializeImpl<Archive, btas::Tensor<_T, _Range, _Store> > {
        static inline void serialize(const Archive& ar,
                                     btas::Tensor<_T, _Range, _Store>& t) {
        }
    };

  }
}
#endif  // BTAS_IS_USABLE
/////////////////////////////////////////////

template<typename _Scalar, int _Options, typename _StorageIndex>
struct colmajor_layout;
template<typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::ColMajor, _StorageIndex> : public std::true_type {
};
template<typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::RowMajor, _StorageIndex> : public std::false_type {
};

template <std::size_t Rank>
struct Key : public std::array<long, Rank> {
  static constexpr const long max_key = 1<<20;
  Key() = default;
  template <typename Integer> Key(std::initializer_list<Integer> ilist) {
    std::copy(ilist.begin(), ilist.end(), this->begin());
  }
  template <typename Archive>
  void serialize(Archive& ar) {ar & madness::archive::wrap((unsigned char*) this, sizeof(*this));}
  madness::hashT hash() const {
    static_assert(Rank == 2 || Rank == 3, "Key<Rank>::hash only implemented for Rank={2,3}");
    return Rank == 2 ? (*this)[0] * max_key + (*this)[1] : ((*this)[0] * max_key + (*this)[1]) * max_key + (*this)[2];
  }
};

template <std::size_t Rank>
std::ostream&
operator<<(std::ostream& os, const Key<Rank>& key) {
  os << "{";
  for(size_t i=0; i!=Rank; ++i)
    os << key[i] << (i+1!=Rank ? "," : "");
  os << "}";
  return os;
}

// flow data from existing data structure
class Read_SpMatrix : public Op<int, std::tuple<Out<Key<2>,blk_t>>, Read_SpMatrix, int> {
 public:
  using baseT =              Op<int, std::tuple<Out<Key<2>,blk_t>>, Read_SpMatrix, int>;

  Read_SpMatrix(const SpMatrix& matrix, Edge<int,int>& ctl, Edge<Key<2>,blk_t>& out):
    baseT(edges(ctl), edges(out), "read_spmatrix", {"ctl"}, {"Mij"}),
    matrix_(matrix) {}

  void op(const int& key, const std::tuple<int>& junk, std::tuple<Out<Key<2>,blk_t>>& out) {
    for (int k=0; k<matrix_.outerSize(); ++k) {
      for (SpMatrix::InnerIterator it(matrix_,k); it; ++it)
      {
        ::send<0>(Key<2>({it.row(),it.col()}), it.value(), out);
      }
    }
  }
 private:
  const SpMatrix& matrix_;
};

// flow (move?) data into a data structure
class Write_SpMatrix : public Op<Key<2>, std::tuple<>, Write_SpMatrix, blk_t> {
 public:
  using baseT =               Op<Key<2>, std::tuple<>, Write_SpMatrix, blk_t>;

  Write_SpMatrix(SpMatrix& matrix, Edge<Key<2>,blk_t>& in):
    baseT(edges(in), edges(), "write_spmatrix", {"Cij"}, {}),
    matrix_(matrix) {}

  void op(const Key<2>& key, const std::tuple<blk_t>& elem, std::tuple<>&) {
    matrix_.insert(key[0], key[1]) = std::get<0>(elem);
  }

 private:
  SpMatrix& matrix_;
};

// sparse mm
class SpMM {
 public:
  SpMM(Edge<Key<2>,blk_t>& a, Edge<Key<2>,blk_t>& b, Edge<Key<2>,blk_t>& c,
       const SpMatrix& a_mat, const SpMatrix& b_mat) :
    a_(a), b_(b), c_(c), a_ijk_(), b_ijk_(), c_ijk_(),
    a_rowidx_to_colidx_(make_rowidx_to_colidx(a_mat)),
    b_colidx_to_rowidx_(make_colidx_to_rowidx(b_mat)),
    bcast_a_(a, a_ijk_, make_rowidx_to_colidx(b_mat)),
    bcast_b_(b, b_ijk_, make_colidx_to_rowidx(a_mat)),
    multiplyadd_(a_ijk_, b_ijk_, c_ijk_, c, a_rowidx_to_colidx_, b_colidx_to_rowidx_)
 {
 }

  /// broadcast A[i][k] to all {i,j,k} such that B[j][k] exists
  class BcastA : public Op<Key<2>, std::tuple<Out<Key<3>,blk_t>>, BcastA, blk_t> {
   public:
    using baseT =       Op<Key<2>, std::tuple<Out<Key<3>,blk_t>>, BcastA, blk_t>;

    BcastA(Edge<Key<2>,blk_t>& a, Edge<Key<3>,blk_t>& a_ijk, std::vector<std::vector<long>>&& b_rowidx_to_colidx) :
      baseT(edges(a), edges(a_ijk),"SpMM::bcast_a", {"a_ik"}, {"a_ijk"}),
      b_rowidx_to_colidx_(std::move(b_rowidx_to_colidx)) {}

    void op(const Key<2>& key, const std::tuple<blk_t>& a_ik, std::tuple<Out<Key<3>,blk_t>>& a_ijk) {
      const auto i = key[0];
      const auto k = key[1];
      // broadcast a_ik to all existing {i,j,k}
      std::vector<Key<3>> ijk_keys;
      for(auto& j: b_rowidx_to_colidx_[k]) {
//        std::cout << "Broadcasting A[" << i << "][" << k << "] to j=" << j << std::endl;
        ijk_keys.emplace_back(Key<3>({i,j,k}));
      }
      ::broadcast<0>(ijk_keys, std::get<0>(a_ik), a_ijk);
    }

   private:
    std::vector<std::vector<long>> b_rowidx_to_colidx_;
  };  // class BcastA

  /// broadcast B[k][j] to all {i,j,k} such that A[i][k] exists
  class BcastB : public Op<Key<2>, std::tuple<Out<Key<3>,blk_t>>, BcastB, blk_t> {
   public:
    using baseT =       Op<Key<2>, std::tuple<Out<Key<3>,blk_t>>, BcastB, blk_t>;

    BcastB(Edge<Key<2>,blk_t>& b, Edge<Key<3>,blk_t>& b_ijk, std::vector<std::vector<long>>&& a_colidx_to_rowidx) :
      baseT(edges(b), edges(b_ijk),"SpMM::bcast_b", {"b_kj"}, {"b_ijk"}),
      a_colidx_to_rowidx_(std::move(a_colidx_to_rowidx)) {}

    void op(const Key<2>& key, const std::tuple<blk_t>& b_kj, std::tuple<Out<Key<3>,blk_t>>& b_ijk) {
      const auto k = key[0];
      const auto j = key[1];
      // broadcast b_kj to *jk
      std::vector<Key<3>> ijk_keys;
      for(auto& i: a_colidx_to_rowidx_[k]) {
//        std::cout << "Broadcasting B[" << k << "][" << j << "] to i=" << i << std::endl;
        ijk_keys.emplace_back(Key<3>({i,j,k}));
      }
      ::broadcast<0>(ijk_keys, std::get<0>(b_kj), b_ijk);
    }

   private:
    std::vector<std::vector<long>> a_colidx_to_rowidx_;
  };  // class BcastA

  /// multiply task has 3 input flows: a_ijk, b_ijk, and c_ijk, c_ijk contains the running total
  class MultiplyAdd : public Op<Key<3>, std::tuple<Out<Key<2>,blk_t>, Out<Key<3>,blk_t>>, MultiplyAdd, blk_t, blk_t, blk_t> {
   public:
    using baseT =            Op<Key<3>, std::tuple<Out<Key<2>,blk_t>, Out<Key<3>,blk_t>>, MultiplyAdd, blk_t, blk_t, blk_t>;

    MultiplyAdd(Edge<Key<3>,blk_t>& a_ijk, Edge<Key<3>,blk_t>& b_ijk,
                Edge<Key<3>,blk_t>& c_ijk,
                Edge<Key<2>,blk_t>& c,
                const std::vector<std::vector<long>>& a_rowidx_to_colidx,
                const std::vector<std::vector<long>>& b_colidx_to_rowidx) :
      baseT(edges(a_ijk,b_ijk,c_ijk), edges(c, c_ijk), "SpMM::Multiply", {"a_ijk", "b_ijk", "c_ijk"}, {"c_ij", "c_ijk"}),
      c_ijk_(c_ijk),
      a_rowidx_to_colidx_(a_rowidx_to_colidx), b_colidx_to_rowidx_(b_colidx_to_rowidx)
      {
        // for each i and j determine first k that contributes, initialize input {i,j,first_k} flow to 0
        for(long i=0; i!=a_rowidx_to_colidx_.size(); ++i) {
          if (a_rowidx_to_colidx_[i].empty()) continue;
          for(long j=0; j!=b_colidx_to_rowidx_.size(); ++j) {
            if (b_colidx_to_rowidx_[j].empty()) continue;

            long k;
            bool have_k;
            std::tie(k,have_k) = compute_first_k(i,j);
            if (have_k) {
//              std::cout << "Initializing C[" << i << "][" << j << "] to zero" << std::endl;
              ::send(Key<3>({i,j,k}), 0, c_ijk_.in());
            }
            else {
//              std::cout << "C[" << i << "][" << j << "] is empty" << std::endl;
            }
          }
        }
      }

    void op(const Key<3>& key, const std::tuple<blk_t, blk_t, blk_t>& _ijk,
            std::tuple<Out<Key<2>,blk_t>,Out<Key<3>,blk_t>>& result) {
      const auto i = key[0];
      const auto j = key[1];
      const auto k = key[2];
      long next_k;
      bool have_next_k;
      std::tie(next_k,have_next_k) = compute_next_k(i,j,k);
//      std::cout << "Multiplying A[" << i << "][" << k << "] by B[" << k << "][" << j << "]" << std::endl;
//      std::cout << "  next_k? " << (have_next_k ? std::to_string(next_k) : "does not exist") << std::endl;
      // compute the contrib, pass the running total to the next flow, if needed
      // otherwise write to the result flow
      if (have_next_k) {
        ::send(Key<3>({i,j,next_k}), std::get<2>(_ijk) + std::get<0>(_ijk) * std::get<1>(_ijk), c_ijk_.in());
      }
      else
        ::send<0>(Key<2>({i,j}), std::get<2>(_ijk) + std::get<0>(_ijk) * std::get<1>(_ijk), result);
    }
   private:
    Edge<Key<3>,blk_t> c_ijk_;
    const std::vector<std::vector<long>>& a_rowidx_to_colidx_;
    const std::vector<std::vector<long>>& b_colidx_to_rowidx_;

    // given {i,j} return first k such that A[i][k] and B[k][j] exist
    std::tuple<long,bool> compute_first_k(long i, long j) {
      auto a_iter_fence = a_rowidx_to_colidx_[i].end();
      auto a_iter = a_rowidx_to_colidx_[i].begin();
      if (a_iter == a_iter_fence)
        return std::make_tuple(-1,false);
      auto b_iter_fence = b_colidx_to_rowidx_[j].end();
      auto b_iter = b_colidx_to_rowidx_[j].begin();
      if (b_iter == b_iter_fence)
        return std::make_tuple(-1,false);

      {
        auto a_colidx = *a_iter;
        auto b_rowidx = *b_iter;
        while (a_colidx != b_rowidx) {
          if (a_colidx < b_rowidx) {
            ++a_iter;
            if (a_iter == a_iter_fence)
              return std::make_tuple(-1,false);
            a_colidx = *a_iter;
          }
          else {
            ++b_iter;
            if (b_iter == b_iter_fence)
              return std::make_tuple(-1,false);
            b_rowidx = *b_iter;
          }
        }
        return std::make_tuple(a_colidx,true);
      }
      assert(false);
    }

    // given {i,j,k} such that A[i][k] and B[k][j] exist
    // return next k such that this condition holds
    std::tuple<long,bool> compute_next_k(long i, long j, long k) {
      auto a_iter_fence = a_rowidx_to_colidx_[i].end();
      auto a_iter = std::find(a_rowidx_to_colidx_[i].begin(), a_iter_fence, k);
      assert(a_iter != a_iter_fence);
      auto b_iter_fence = b_colidx_to_rowidx_[j].end();
      auto b_iter = std::find(b_colidx_to_rowidx_[j].begin(), b_iter_fence, k);
      assert(b_iter != b_iter_fence);
      while (a_iter != a_iter_fence && b_iter != b_iter_fence) {
        ++a_iter;
        ++b_iter;
        if (a_iter == a_iter_fence || b_iter == b_iter_fence)
          return std::make_tuple(-1,false);
        auto a_colidx = *a_iter;
        auto b_rowidx = *b_iter;
        while (a_colidx != b_rowidx) {
          if (a_colidx < b_rowidx) {
            ++a_iter;
            if (a_iter == a_iter_fence)
              return std::make_tuple(-1,false);
            a_colidx = *a_iter;
          }
          else {
            ++b_iter;
            if (b_iter == b_iter_fence)
              return std::make_tuple(-1,false);
            b_rowidx = *b_iter;
          }
        }
        return std::make_tuple(a_colidx,true);
      }
    }

  };

 private:
  Edge<Key<2>,blk_t>& a_;
  Edge<Key<2>,blk_t>& b_;
  Edge<Key<2>,blk_t>& c_;
  Edge<Key<3>,blk_t> a_ijk_;
  Edge<Key<3>,blk_t> b_ijk_;
  Edge<Key<3>,blk_t> c_ijk_;
  std::vector<std::vector<long>> a_rowidx_to_colidx_;
  std::vector<std::vector<long>> b_colidx_to_rowidx_;
  BcastA bcast_a_;
  BcastB bcast_b_;
  MultiplyAdd multiplyadd_;

  // result[i][j] gives the j-th nonzero row for column i in matrix mat
  std::vector<std::vector<long>> make_colidx_to_rowidx(const SpMatrix& mat) {
    std::vector<std::vector<long>> colidx_to_rowidx;
    for (int k=0; k<mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
      for (SpMatrix::InnerIterator it(mat,k); it; ++it) {
        auto row = it.row();
        auto col = it.col();
        if (col >= colidx_to_rowidx.size())
          colidx_to_rowidx.resize(col+1);
        // in either case (col- or row-major) row index increasing for the given col
        colidx_to_rowidx[col].push_back(row);
      }
    }
    return colidx_to_rowidx;
  }
  // result[i][j] gives the j-th nonzero column for row i in matrix mat
  std::vector<std::vector<long>> make_rowidx_to_colidx(const SpMatrix& mat) {
    std::vector<std::vector<long>> rowidx_to_colidx;
    for (int k=0; k<mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
      for (SpMatrix::InnerIterator it(mat,k); it; ++it) {
        auto row = it.row();
        auto col = it.col();
        if (row >= rowidx_to_colidx.size())
          rowidx_to_colidx.resize(row+1);
        // in either case (col- or row-major) col index increasing for the given row
        rowidx_to_colidx[row].push_back(col);
      }
    }
    return rowidx_to_colidx;
  }

};

class Control : public Op<int, std::tuple<Out<int,int>>, Control> {
    using baseT =      Op<int, std::tuple<Out<int,int>>, Control>;

 public:
    Control(Edge<int,int>& ctl)
        : baseT(edges(),edges(ctl),"Control",{},{"ctl"})
        {}

    void op(const int& key, const std::tuple<>&, std::tuple<Out<int,int>>& out) {
        ::send<0>(0,0,out);
    }

    void start() {
        invoke(0);
    }
};

int main(int argc, char** argv) {
  World& world = madness::initialize(argc, argv);
  world.gop.fence();

  if (world.size() > 1) {
      madness::error("Currently only works with one MPI process (multiple threads OK)");
  }

  //OpBase::set_trace_all(true);

  const int n = 2;
  const int m = 3;
  const int k = 4;

  // initialize inputs (these will become shapes when switch to blocks)
  SpMatrix A(n,k), B(k,m);
  {
    using triplet_t = Eigen::Triplet<blk_t>;
    std::vector<triplet_t> A_elements;
    A_elements.emplace_back(0,1,12.3);
    A_elements.emplace_back(0,2,10.7);
    A_elements.emplace_back(0,3,-2.3);
    A_elements.emplace_back(1,0,-0.3);
    A_elements.emplace_back(1,2,1.2);
    A.setFromTriplets(A_elements.begin(), A_elements.end());

    std::vector<triplet_t> B_elements;
    B_elements.emplace_back(0,0,12.3);
    B_elements.emplace_back(1,0,10.7);
    B_elements.emplace_back(3,0,-2.3);
    B_elements.emplace_back(1,1,-0.3);
    B_elements.emplace_back(1,2,1.2);
    B_elements.emplace_back(2,2,7.2);
    B_elements.emplace_back(3,2,0.2);
    B.setFromTriplets(B_elements.begin(), B_elements.end());
  }

  Edge<int, int> ctl("control");
  Control control(ctl);
  Edge<Key<2>,blk_t> eA, eB, eC;
  Read_SpMatrix a(A, ctl, eA);
  Read_SpMatrix b(B, ctl, eB);

  // set up matmul flow
  SpMatrix C(n,m);
  Write_SpMatrix c(C, eC);
  SpMM a_times_b(eA, eB, eC, A, B);

  // execute the flow
  control.start();

  // no way to check for completion yet
  world.gop.fence();
  world.gop.fence();

  // validate against the reference output
  SpMatrix Cref = A * B;
  std::cout << "Cref-C=" << Cref-C << std::endl;

  madness::finalize();

  return 0;
}
