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
    matrix_(matrix), flow_(flow), ctl_(ctl) {}
  void op(const Key<0>& key, const Control&, SpMatrixFlow&) {
    for (int k=0; k<matrix_.outerSize(); ++k) {
      for (SpMatrix::InnerIterator it(matrix_,k); it; ++it)
      {
        flow_.send<0>(Key<2>({it.row(),it.col()}), it.value());
      }
    }
  }
 private:
  const SpMatrix& matrix_;
  SpMatrixFlow flow_;
  ControlFlow ctl_;
};
// flow (move?) data into a data structure
class Flow_To_SpMatrix : public Op<Key<2>, SpMatrixFlow, ControlFlow, Flow_To_SpMatrix> {
 public:
  using baseT = Op<Key<2>, SpMatrixFlow, ControlFlow, Flow_To_SpMatrix>;

  Flow_To_SpMatrix(SpMatrix& matrix, SpMatrixFlow flow, ControlFlow ctl):
    baseT(flow, ctl, "flow_to_spmatrix"),
    matrix_(matrix), flow_(flow), ctl_(ctl) {}
  void op(const Key<2>& key, const blk_t& elem, ControlFlow) {
    matrix_.insert(key[0], key[1]) = elem;
  }

 private:
  SpMatrix& matrix_;
  SpMatrixFlow flow_;
  ControlFlow ctl_;
};

// sparse mm
class SpMM: public Op<Key<3>, Flows<Key<3>,blk_t,blk_t>, SpMatrixFlow, SpMM> {
 public:
  using baseT = Op<Key<3>, Flows<Key<3>,blk_t,blk_t>, SpMatrixFlow, SpMM>;
  SpMM(SpMatrixFlow a, SpMatrixFlow b, SpMatrixFlow c) :
    baseT(make_flows(a.get<0>(),b.get<0>()),c,"SpMM"),
    a_(a), b_(b), c_(c) {}
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
