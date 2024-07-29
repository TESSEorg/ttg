#include <cassert>
#include <ttg.h>

#include "../tensor.h"



int main(int argc, char **argv) {

  ttg::initialize(argc, argv);

  using matrix_type = mra::Tensor<double, 2>;
  using matrixview_type = typename matrix_type::view_type;
  matrix_type m1 = matrix_type(2, 2); // 2x2 matrix
  matrix_type m2 = matrix_type(4, 4); // 4x4 matrix
  assert(m1.size() == 4);
  assert(m2.size() == 16);

  matrixview_type m2v = m2.current_view();

  /* explicit iteration */
  for (int i = 0; i < m2v.dim(0); ++i) {
    for (int j = 0; j < m2v.dim(1); ++j) {
      m2v(i, j) = 1.0;
    }
  }

  m1 = std::move(m2); // move m2 into m1
  assert(m1.size() == 16);

  // check m1, should all be 1
  matrixview_type m1v = m1.current_view();
  for (int i = 0; i < m1v.dim(0); ++i) {
    for (int j = 0; j < m1v.dim(1); ++j) {
      assert(m1v(i, j) == 1.0);
    }
  }


  // check tensor slices: slice 2x2 matrix from the center, scale by 2, and check
  std::array<mra::Slice, 2> slices = {mra::Slice(1, 3), mra::Slice(1, 3)};
  auto s1 = mra::TensorSlice(m1v, slices);
  s1 *= 2;
  for (int i = 0; i < m1v.dim(0); ++i) {
    for (int j = 0; j < m1v.dim(1); ++j) {
      if ((i < 1 || i > 2) || (j < 1 || j > 2)) {
        assert(m1v(i, j) == 1.0);
      } else {
        assert(m1v(i, j) == 2.0);
      }
    }
  }

  // create new tensor, set to 1.0 and assign a slice
  m2 = matrix_type(4, 4); // 4x4 matrix
  m2v = m2.current_view();
  m2v = 1.0; // set all elements to 1.0
  for (int i = 0; i < m2v.dim(0); ++i) {
    for (int j = 0; j < m2v.dim(1); ++j) {
      assert(m2v(i, j) == 1.0);
    }
  }
  auto s2 = mra::TensorSlice(m2v, slices);
  s2 = s1; // assign slice
  assert(m2.size() == 16);
  for (int i = 0; i < m2v.dim(0); ++i) {
    for (int j = 0; j < m2v.dim(1); ++j) {
      if ((i < 1 || i > 2) || (j < 1 || j > 2)) {
        assert(m2v(i, j) == 1.0);
      } else {
        assert(m2v(i, j) == 2.0);
      }
    }
  }

  ttg::finalize();

}