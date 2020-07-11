#include <algorithm>  // for std::max
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include "blockmatrix.h"

/*!
        \file wavefront_df.impl.h
        \brief Wavefront computation on distributed memory
        \defgroup
        ingroup examples

        \par Points of interest
        - dynamic recursive DAG.
*/

using Key = std::pair<int, int>;

// An empty class used for pure control flows
struct Control {};

template <typename T>
inline BlockMatrix<T> stencil_computation(int i, int j, int M, int N, BlockMatrix<T> bm, BlockMatrix<T> left, 
                          BlockMatrix<T> top, BlockMatrix<T> right, BlockMatrix<T> bottom) {
  // i==0 -> no top block
  // j==0 -> no left block
  // i==M-1 -> no bottom block
  // j==N-1 -> no right block
  int MB = bm.rows();
  int NB = bm.cols();
  BlockMatrix<T> current = bm;
  std::cout << i << " " << j << std::endl;
  for (int ii = 0; ii < MB; ++ii) {
    for (int jj = 0; jj < NB; ++jj) {
      current(ii,jj,(current(ii,jj) + ((ii == 0) ? (i > 0 ? top(MB - 1, jj) : 0.0) : current(ii - 1, jj))));
      current(ii,jj,(current(ii,jj) + ((ii == MB - 1) ? (i < M - 1 ? bottom(0, jj) : 0.0) : current(ii + 1, jj))));
      current(ii,jj,(current(ii,jj) + ((jj == 0) ? (j > 0 ? left(ii, NB - 1) : 0.0) : current(ii, jj - 1))));
      current(ii,jj,(current(ii,jj) + ((jj == NB - 1) ? (j < N - 1 ? right(ii, 0) : 0.0) : current(ii, jj + 1))));
      current(ii,jj, current(ii,jj) * 0.25);
    }
  }
  return current;
}

// serial implementation of wavefront computation.
template <typename T>
void wavefront_serial(Matrix<T>* m, Matrix<T>* result, int n_brows, int n_bcols) {
  for (int i = 0; i < n_brows; i++) {
    for (int j = 0; j < n_bcols; j++) {
      BlockMatrix<T> left, top, right, bottom;

      if (i < n_brows - 1) bottom = ((*m)(i + 1, j));
      if (j < n_bcols - 1) right = ((*m)(i, j + 1));
      if (j > 0) left = ((*m)(i, j - 1));
      if (i > 0) top = ((*m)(i - 1, j));
      
      (*result)(i,j,stencil_computation(i, j, n_brows, n_bcols, ((*m)(i,j)), (left), (top), (right), (bottom)));
    }
  }
}

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

// Method to generate  wavefront tasks with two inputs.
template <typename funcT, typename T>
auto make_wavefront2(const funcT& func, Matrix<T>* m, Edge<Key, BlockMatrix<T>>& input1,
                     Edge<Key, BlockMatrix<T>>& input2, Edge<Key, BlockMatrix<T>>& result) {
  
  auto f = [m, func](const Key& key, BlockMatrix<T>&& bm1, BlockMatrix<T>&& bm2,
                     std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    BlockMatrix<T> left, top, right, bottom;
    left = (bm2);
    top = (bm1);

    if (j < m->cols() - 1) {
      right = ((*m)(i, j + 1));
    }
    if (i < m->rows() - 1) {
      bottom = ((*m)(i + 1, j));
    }

    BlockMatrix<T> res = func(i, j, m->rows(), m->cols(), ((*m)(i, j)), (left), (top), (right), (bottom));
    //Processing finished for this block, so send it to output Op
    send<2>(Key(i,j), res, out);

    if (next_i < m->rows()) {
      if (j == 0)
        send<0>(Key(next_i, j), res, out);
      else
        send<0>(Key(next_i, j), res, out);
    }

    if (next_j < m->cols()) {
      if (i == 0)
        send<0>(Key(i, next_j), res, out);
      else
        send<1>(Key(i, next_j), res, out);
    }
  };

  return wrap(f, edges(input1, input2), edges(input1, input2, result), "wavefront2", {"first", "second"},
              {"output1", "output2", "result"});
}

// Method to generate wavefront task with single input.
template <typename funcT, typename T>
auto make_wavefront(const funcT& func, Matrix<T>* m, Edge<Key, BlockMatrix<T>>& input1,
                    Edge<Key, BlockMatrix<T>>& input2, Edge<Key, BlockMatrix<T>>& result) {
  auto f = [m, func](const Key& key, BlockMatrix<T>&& bm,
                     std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    // This handles the cases where i=0 or j=0,
    // since these are the only tasks that have single input dependency.
    BlockMatrix<T> left, top, right, bottom;

    if (i == 0) {
      left = (bm);
      bottom = ((*m)(i + 1, j));
    }
    if (j == 0) {
      top = (bm);
      right = ((*m)(i, j + 1));
    }

    if (j < m->cols() - 1) right = ((*m)(i, j + 1));
    if (i < m->rows() - 1) bottom = ((*m)(i + 1, j));

    BlockMatrix<T> res = func(i, j, m->rows(), m->cols(), ((*m)(i, j)), (left), (top), (right), (bottom));
    //Processing finished for this block, so send it to output Op
    send<3>(Key(i,j), res, out);

    if (next_i < m->rows()) {
      if (j == 0)  // Single predecessor
        send<0>(Key(next_i, j), res, out);
      else  // Two predecessors
        send<1>(Key(next_i, j), res, out);
    }
    if (next_j < m->cols()) {
      if (i == 0)  // Single predecessor
        send<0>(Key(i, next_j), res, out);
      else  // Two predecessors
        send<2>(Key(i, next_j), res, out);
    }
  };

  Edge<Key, BlockMatrix<T>> recur("recur");
  return wrap(f, edges(recur), edges(recur, input1, input2, result), "wavefront", {"control"}, {"recur", "output1", "output2", "result"});
}

template <typename T>
auto make_result(Matrix<T> *r, const Edge<Key, BlockMatrix<T>>& result) {
  auto f = [r](const Key& key, BlockMatrix<T>&& bm, std::tuple<>& out) {
    auto [i,j] = key;
    (*r)(i,j,bm);
  };

  return wrap(f, edges(result), edges(), "Final Output", {"result"}, {});
}

int main(int argc, char** argv) {
  int n_rows, n_cols, B;
  int n_brows, n_bcols;

  n_rows = n_cols = 16384;
  B = 128;

  n_brows = (n_rows / B) + (n_rows % B > 0);
  n_bcols = (n_cols / B) + (n_cols % B > 0);

  Matrix<double>* m = new Matrix<double>(n_brows, n_bcols, B, B);
  Matrix<double>* m2 = new Matrix<double>(n_brows, n_bcols, B, B);

  Matrix<double>* r = new Matrix<double>(n_brows, n_bcols, B, B);
  Matrix<double>* r2 = new Matrix<double>(n_brows, n_bcols, B, B);

  m->fill();
  m2->fill();

  Edge<Key, BlockMatrix<double>> parent1("parent1"), parent2("parent2"), result("result");;
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  ttg_initialize(argc, argv, -1);
  {
    auto s = make_wavefront(stencil_computation<double>, m, parent1, parent2, result);
    auto s2 = make_wavefront2(stencil_computation<double>, m, parent1, parent2, result);
    auto res = make_result(r, result);

    auto connected = make_graph_executable(s.get());
    assert(connected);
    TTGUNUSED(connected);
    std::cout << "Graph is connected.\n";

    if (ttg_default_execution_context().rank() == 0) {
      std::cout << "==== begin dot ====\n";
      std::cout << Dot()(s.get()) << std::endl;
      std::cout << "==== end dot ====\n";

      beg = std::chrono::high_resolution_clock::now();
      s->in<0>()->send(Key(0, 0), (*m)(0, 0));
      // This doesn't work!
      // s->send<0>(Key(0,0), Control());
    }

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;
  }

  ttg_finalize();

  std::cout << "Computing using serial version....";
  beg = std::chrono::high_resolution_clock::now();
  wavefront_serial(m2, r2, n_brows, n_bcols);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "....done!" << std::endl;
  std::cout << "Serial Execution Time (milliseconds) : "
            << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1e3 << std::endl;

  /*m->print();
  std::cout << std::endl << std::endl;
  r->print();
  std::cout << std::endl << std::endl;
  m2->print();
  std::cout << std::endl << std::endl;
  r2->print();*/
  /*r->print();
  std::cout << std::endl << std::endl;
  m2->print();*/

  std::cout << "Verifying the result....";
  bool success = true;
  for (int i = 0; i < n_brows; i++) {
    for (int j = 0; j < n_bcols; j++) {
      if ((*r)(i, j) != (*r2)(i, j)) {
        success = false;
        std::cout << "ERROR in block (" << i << "," << j << ")\n";
      }
    }
  }

  std::cout << "....done!" << std::endl;
  std::cout << (success ? "SUCCESS!!!" : "FAILED!!!") << std::endl;

  delete m;
  delete m2;
  delete r;
  return 0;
}
