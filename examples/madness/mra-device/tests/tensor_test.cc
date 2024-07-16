#include <cassert>
#include <ttg.h>

#include "tensor.h"



int main(int argc, char **argv) {
  ttg::initialize(argc, argv);

  using matrix_type = mra::Tensor<double, 2>;
  using matrixview_type = typename matrix_type::view_type;
  matrix_type m1 = matrix_type(2, 2); // 2x2 matrix
  matrix_type m2 = matrix_type(4, 4); // 4x4 matrix
  assert(m1.size() == 4);
  assert(m2.size() == 16);

  matrixview_type m2v = m2.current_view();

  for (int i = 0; i < m2v.dim(0); ++i) {
    for (int j = 0; j < m2v.dim(1); ++j) {
      m2v(i, j) = 1.0;
    }
  }

  m1 = std::move(m2); // move m2 into m1
  assert(m1.size() == 16);

  // check m1, should all be 1
  matrixview_type m1v = m1.current_view();
  for (int i = 0; i < m2v.dim(0); ++i) {
    for (int j = 0; j < m2v.dim(1); ++j) {
      assert(m1v(i, j) = 1.0);
    }
  }


  ttg::finalize();
}