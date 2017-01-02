#include <array>
#include <iostream>
#include <vector>

#include <Eigen/SparseCore>

using blk_t = double;  // replace with btas::Tensor to have a block-sparse matrix
using SpMatrix = Eigen::SparseMatrix<blk_t>;

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
using ControlFlow = Flows<Key<0>, Control>;
using SpMatrixFlow = Flows<Key<2>,blk_t>;

// flow data from existing data structure
// @todo extend Flow to fit the concept expected by Op (and satisfied by Flows), e.g. provide size()
class Flow_From_SpMatrix : public Op<Key<0>, ControlFlow, SpMatrixFlow, Flow_From_SpMatrix> {
 public:
  using baseT = Op<Key<0>, ControlFlow, SpMatrixFlow, Flow_From_SpMatrix>;
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
class Flow_To_SpMatrix : public Op<Key<2>, SpMatrixFlow, ControlFlow, Flow_To_SpMatrix> {
 public:
  using baseT = Op<Key<2>, SpMatrixFlow, ControlFlow, Flow_To_SpMatrix>;

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
  SpMM(SpMatrixFlow a, SpMatrixFlow b, SpMatrixFlow c) :
    a_(a), b_(b), c_(c) {}

  /// broadcast A[i][k] to all {i,j,k} such that B[j][k] exists
  class BcastA : public Op<Key<2>, SpMatrixFlow, Flows<Key<3>,blk_t>, BcastA> {
   public:
    using baseT = Op<Key<2>, SpMatrixFlow, Flows<Key<3>,blk_t>, BcastA>;
    BcastA(SpMatrixFlow a, Flows<Key<3>,blk_t> a_repl, const std::vector<std::vector<long>>& b_rowidx_to_colidx) :
      baseT(a, a_repl,"SpMM::bcast_a"), b_rowidx_to_colidx_(b_rowidx_to_colidx) {}
    void op(const Key<2>& key, const blk_t& a_ik, Flows<Key<3>,blk_t> bcast) {
      const auto i = key[0];
      const auto k = key[1];
      // broadcast a_ik to all existing {i,j,k}
      std::vector<Key<3>> ijk_keys;
      for(auto& j: b_rowidx_to_colidx_[k])
        ijk_keys.emplace_back(Key<3>({i,j,k}));
      bcast.broadcast<0>(ijk_keys, a_ik);
    }
   private:
    const std::vector<std::vector<long>>& b_rowidx_to_colidx_;
  };  // class BcastA

  /// broadcast B[k][j] to all {i,j,k} such that A[i][k] exists
  class BcastB : public Op<Key<2>, SpMatrixFlow, Flows<Key<3>,blk_t>, BcastB> {
   public:
    using baseT = Op<Key<2>, SpMatrixFlow, Flows<Key<3>,blk_t>, BcastB>;
    BcastB(SpMatrixFlow b, Flows<Key<3>,blk_t> b_repl, const std::vector<std::vector<long>>& a_colidx_to_rowidx) :
      baseT(b, b_repl,"SpMM::bcast_b"), a_colidx_to_rowidx_(a_colidx_to_rowidx) {}
    void op(const Key<2>& key, const blk_t& b_kj, Flows<Key<3>,blk_t> bcast) {
      const auto k = key[0];
      const auto j = key[1];
      // broadcast b_kj to *jk
      std::vector<Key<3>> ijk_keys;
      for(auto& i: a_colidx_to_rowidx_[k])
        ijk_keys.emplace_back(Key<3>({i,j,k}));
      bcast.broadcast<0>(ijk_keys, b_kj);
    }
   private:
    const std::vector<std::vector<long>>& a_colidx_to_rowidx_;
  };  // class BcastA

  /// multiply task has 3 input flows: a_ijk, b_ijk, and c_ijk, c_ijk contains the running total
  class Multiply : public Op<Key<3>, Flows<Key<3>,blk_t,blk_t,blk_t>, SpMatrixFlow, Multiply> {
   public:
    using baseT = Op<Key<3>, Flows<Key<3>,blk_t,blk_t,blk_t>, SpMatrixFlow, Multiply>;
    Multiply(Flows<Key<3>,blk_t> a_ijk, Flows<Key<3>,blk_t> b_ijk,
             Flows<Key<3>,blk_t> c_ijk,
             SpMatrixFlow c) :
      baseT(make_flows(a_ijk.get<0>(),b_ijk.get<0>(),c_ijk.get<0>()), c, "SpMM::Multiply"),
      c_ijk_(c_ijk)
      {
      // for each i and j determine first k that contributes, initialize input {i,j,first_k} flow to 0
      }
    void op(const Key<3>& key, const blk_t& a_ijk, const blk_t& b_ijk, const blk_t& c_ijk,
            Flows<Key<2>,blk_t> result) {
      const auto i = key[0];
      const auto j = key[1];
      const auto k = key[2];
      long next_k;
      bool have_next_k;
      std::tie(next_k,have_next_k) = compute_next_k(i,j,k);
      // compute the contrib, pass the running total to the next flow, if needed
      // otherwise write to the result flow
      if (have_next_k) {
        // need Op::inputs()!
        // N.B.initial c_ijk was zeroed out above
        c_ijk_.send<0>(Key<3>({i,j,next_k}),c_ijk + a_ijk * b_ijk);
      }
      else
        result.send<0>(Key<2>({i,j}), c_ijk + a_ijk * b_ijk);
    }
   private:
    Flows<Key<3>,blk_t> c_ijk_;
  };

 private:
  SpMatrixFlow a_;
  SpMatrixFlow b_;
  SpMatrixFlow c_;
};

int main(int argc, char* argv[]) {

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
  SpMM a_times_b(A_flow, B_flow, C_flow);

  // execute the flow
  ctl.send<0>(Key<0>(),Control());

  // validate against the reference output
  SpMatrix Cref = A * B;
  std::cout << "Cref-C=" << Cref-C << std::endl;

  return 0;
}
