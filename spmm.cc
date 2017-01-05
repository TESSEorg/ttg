#include <array>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#include <Eigen/SparseCore>

using blk_t = double;  // replace with btas::Tensor to have a block-sparse matrix
using SpMatrix = Eigen::SparseMatrix<blk_t>;

template<typename _Scalar, int _Options, typename _StorageIndex>
struct colmajor_layout;
template<typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::ColMajor, _StorageIndex> : public std::true_type {
};
template<typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::RowMajor, _StorageIndex> : public std::false_type {
};


#include <flow.h>

template <std::size_t Rank>
struct Key : public std::array<long, Rank> {
  Key() = default;
  template <typename Integer> Key(std::initializer_list<Integer> ilist) {
    std::copy(ilist.begin(), ilist.end(), this->begin());
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

struct Control {};
using ControlFlow = Flow<Key<0>, Control>;
using SpMatrixFlow = Flow<Key<2>,blk_t>;

// flow data from existing data structure
class Flow_From_SpMatrix : public Op<ControlFlow, SpMatrixFlow, Flow_From_SpMatrix> {
 public:
  using baseT = Op<ControlFlow, SpMatrixFlow, Flow_From_SpMatrix>;
  Flow_From_SpMatrix(const SpMatrix& matrix, SpMatrixFlow flow, ControlFlow ctl):
    baseT(ctl, flow, "spmatrix_to_flow"),
    matrix_(matrix) {}
  void op(const Key<0>& key, const Control&, SpMatrixFlow& flow) {
    for (int k=0; k<matrix_.outerSize(); ++k) {
      for (SpMatrix::InnerIterator it(matrix_,k); it; ++it)
      {
        flow.send<0>(Key<2>({it.row(),it.col()}), it.value());
      }
    }
  }
 private:
  const SpMatrix& matrix_;
  SpMatrixFlow flow_;
};
// flow (move?) data into a data structure
class Flow_To_SpMatrix : public Op<SpMatrixFlow, ControlFlow, Flow_To_SpMatrix> {
 public:
  using baseT = Op<SpMatrixFlow, ControlFlow, Flow_To_SpMatrix>;

  Flow_To_SpMatrix(SpMatrix& matrix, SpMatrixFlow flow, ControlFlow ctl):
    baseT(flow, ctl, "flow_to_spmatrix"),
    matrix_(matrix) {}
  void op(const Key<2>& key, const blk_t& elem, ControlFlow) {
    matrix_.insert(key[0], key[1]) = elem;
  }

 private:
  SpMatrix& matrix_;
};

// sparse mm
class SpMM {
 public:
  SpMM(SpMatrixFlow a, SpMatrixFlow b, SpMatrixFlow c,
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
  class BcastA : public Op<SpMatrixFlow, FlowArray<Key<3>,blk_t>, BcastA> {
   public:
    using baseT = Op<SpMatrixFlow, FlowArray<Key<3>,blk_t>, BcastA>;
    BcastA(SpMatrixFlow a, FlowArray<Key<3>,blk_t> a_ijk, std::vector<std::vector<long>>&& b_rowidx_to_colidx) :
      baseT(a, a_ijk,"SpMM::bcast_a"), b_rowidx_to_colidx_(std::move(b_rowidx_to_colidx)) {}
    void op(const Key<2>& key, const blk_t& a_ik, FlowArray<Key<3>,blk_t> a_ijk) {
      const auto i = key[0];
      const auto k = key[1];
      // broadcast a_ik to all existing {i,j,k}
      std::vector<Key<3>> ijk_keys;
      for(auto& j: b_rowidx_to_colidx_[k]) {
//        std::cout << "Broadcasting A[" << i << "][" << k << "] to j=" << j << std::endl;
        ijk_keys.emplace_back(Key<3>({i,j,k}));
      }
      a_ijk.broadcast<0>(ijk_keys, a_ik);
    }
   private:
    std::vector<std::vector<long>> b_rowidx_to_colidx_;
  };  // class BcastA

  /// broadcast B[k][j] to all {i,j,k} such that A[i][k] exists
  class BcastB : public Op<SpMatrixFlow, FlowArray<Key<3>,blk_t>, BcastB> {
   public:
    using baseT = Op<SpMatrixFlow, FlowArray<Key<3>,blk_t>, BcastB>;
    BcastB(SpMatrixFlow b, FlowArray<Key<3>,blk_t> b_ijk, std::vector<std::vector<long>>&& a_colidx_to_rowidx) :
      baseT(b, b_ijk,"SpMM::bcast_b"), a_colidx_to_rowidx_(std::move(a_colidx_to_rowidx)) {}
    void op(const Key<2>& key, const blk_t& b_kj, FlowArray<Key<3>,blk_t> b_ijk) {
      const auto k = key[0];
      const auto j = key[1];
      // broadcast b_kj to *jk
      std::vector<Key<3>> ijk_keys;
      for(auto& i: a_colidx_to_rowidx_[k]) {
//        std::cout << "Broadcasting B[" << k << "][" << j << "] to i=" << i << std::endl;
        ijk_keys.emplace_back(Key<3>({i,j,k}));
      }
      b_ijk.broadcast<0>(ijk_keys, b_kj);
    }
   private:
    std::vector<std::vector<long>> a_colidx_to_rowidx_;
  };  // class BcastA

  /// multiply task has 3 input flows: a_ijk, b_ijk, and c_ijk, c_ijk contains the running total
  class MultiplyAdd : public Op<FlowArray<Key<3>,blk_t,blk_t,blk_t>, SpMatrixFlow, MultiplyAdd> {
   public:
    using baseT = Op<FlowArray<Key<3>,blk_t,blk_t,blk_t>, SpMatrixFlow, MultiplyAdd>;
    MultiplyAdd(FlowArray<Key<3>,blk_t> a_ijk, FlowArray<Key<3>,blk_t> b_ijk,
             FlowArray<Key<3>,blk_t> c_ijk,
             SpMatrixFlow c,
             const std::vector<std::vector<long>>& a_rowidx_to_colidx,
             const std::vector<std::vector<long>>& b_colidx_to_rowidx) :
      baseT(make_flows(a_ijk.get<0>(),b_ijk.get<0>(),c_ijk.get<0>()), c, "SpMM::Multiply"),
      c_ijk_(c_ijk), a_rowidx_to_colidx_(a_rowidx_to_colidx), b_colidx_to_rowidx_(b_colidx_to_rowidx)
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
              c_ijk_.send<0>(Key<3>({i,j,k}),0);
            }
            else {
//              std::cout << "C[" << i << "][" << j << "] is empty" << std::endl;
            }
          }
        }
      }
    void op(const Key<3>& key, const blk_t& a_ijk, const blk_t& b_ijk, const blk_t& c_ijk,
            SpMatrixFlow result) {
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
        c_ijk_.send<0>(Key<3>({i,j,next_k}),c_ijk + a_ijk * b_ijk);
      }
      else
        result.send<0>(Key<2>({i,j}), c_ijk + a_ijk * b_ijk);
    }
   private:
    FlowArray<Key<3>,blk_t> c_ijk_;
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
  SpMatrixFlow a_;
  SpMatrixFlow b_;
  SpMatrixFlow c_;
  FlowArray<Key<3>,blk_t> a_ijk_;
  FlowArray<Key<3>,blk_t> b_ijk_;
  FlowArray<Key<3>,blk_t> c_ijk_;
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

int main(int argc, char* argv[]) {

  BaseOp::set_trace(false);

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

  ControlFlow ctl; // this is just a kick to jumpstart the computation
  SpMatrixFlow A_flow, B_flow;
  Flow_From_SpMatrix a(A, A_flow, ctl);
  Flow_From_SpMatrix b(B, B_flow, ctl);

  // set up matmul flow
  SpMatrix C(n,m);
  SpMatrixFlow C_flow;
  Flow_To_SpMatrix c(C, C_flow, ctl);
  SpMM a_times_b(A_flow, B_flow, C_flow, A, B);

  // execute the flow
  ctl.send<0>(Key<0>(),Control());

  // no way to check for completion yet
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(2s);

  // validate against the reference output
  SpMatrix Cref = A * B;
  std::cout << "Cref-C=" << Cref-C << std::endl;

  return 0;
}
