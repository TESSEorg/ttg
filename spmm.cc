#include <array>
#include <iostream>
#include <vector>

#include <Eigen/SparseCore>

using real_t = double;
using SpMatrix = Eigen::SparseMatrix<real_t>;

#include <flow.h>

template <std::size_t Rank>
using Key = std::array<long, Rank>;

struct Control {};
using ControlFlow = Flow<Key<0>, Control>;
using SpMatrixFlow = Flow<Key<2>,real_t>;

// flow data from existing data structure
// @todo extend Flow to fit the concept expected by Op (and satisfied by Flows), e.g. provide size()
class Flow_From_SpMatrix : Op<Key<2>, ControlFlow, SpMatrixFlow, Flow_From_SpMatrix> {
 public:
  Flow_From_SpMatrix(const SpMatrix& matrix, const SpMatrixFlow& flow, const ControlFlow& ctl):
    matrix_(matrix), flow_(flow), ctl_(ctl) {}
 private:
  const SpMatrix& matrix_;
  const SpMatrixFlow& flow_;
  const ControlFlow& ctl_;
};
// flow (move?) data into a data structure
class Flow_To_SpMatrix : Op<Key<2>, SpMatrixFlow, ControlFlow, Flow_To_SpMatrix> {
 public:
  Flow_To_SpMatrix(const SpMatrix& matrix, const SpMatrixFlow& flow, const ControlFlow& ctl):
    matrix_(matrix), flow_(flow), ctl_(ctl) {}
 private:
  const SpMatrix& matrix_;
  const SpMatrixFlow& flow_;
  const ControlFlow& ctl_;
};

// sparse mm
class SpMM: Op<Key<3>, Flows<Key<2>,real_t,real_t>, SpMatrixFlow, SpMM> {
 public:
  SpMM(const SpMatrixFlow& a, const SpMatrixFlow& b, const SpMatrixFlow& c);
};

int main(int argc, char* argv[]) {

  const int n = 2;
  const int m = 3;
  const int k = 4;

  // initialize inputs (these will become shapes when switch to blocks)
  SpMatrix A(n,k), B(k,m);
  {
    using triplet_t = Eigen::Triplet<real_t>;
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

  ControlFlow ctl;
  SpMatrixFlow A_flow, B_flow;
  Flow_From_SpMatrix a(A, A_flow, ctl);
  Flow_From_SpMatrix b(B, B_flow, ctl);

  // set up matmul flow
  SpMatrix C(n,m);
  SpMatrixFlow C_flow;
  Flow_To_SpMatrix c(C, C_flow, ctl);
  SpMM a_times_b(A_flow, B_flow, C_flow);

  // execute the flow
  ctl.send(Key<0>{},Control());

  // validate against the reference output
  SpMatrix Cref = A * B;
  std::cout << "Cref-C=" << Cref-C << std::endl;

  return 0;
}
