#include <algorithm>  // for std::max
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <utility>

#include "ttg/serialization/std/pair.h"
#include "ttg/util/hash/std/pair.h"

#include "ttg.h"

using namespace ttg;

int M, N, B;
int MB, NB;

std::shared_ptr<double> matrix;
std::shared_ptr<double> matrix2;

using Key = std::pair<int, int>;  // I, J
namespace std {
  std::ostream& operator<<(std::ostream& out, Key const& k) {
    out << "Key(" << k.first << ", " << k.second << ")";
    return out;
  }
}  // namespace std

// initialize the matrix
inline void init_matrix() {
  matrix = std::shared_ptr<double>(new double[M * N], [](double* p) { delete[] p; });
  matrix2 = std::shared_ptr<double>(new double[M * N], [](double* p) { delete[] p; });

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      matrix.get()[i * N + j] = 1;  // i*N + j;
      matrix2.get()[i * N + j] = 1;
    }
  }
}

// print the matrix
inline void print_matrix() {
  std::cout << "Output Matrix of TTG version" << std::endl;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) std::cout << matrix.get()[i * N + j] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;
  std::cout << "Output Matrix of Serial version" << std::endl;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) std::cout << matrix2.get()[i * N + j] << " ";
    std::cout << std::endl;
  }
}

// destroy the matrix
inline void delete_matrix() {
  delete[] matrix.get();
  delete matrix2.get();
}

inline void stencil_computation(int i, int j, std::shared_ptr<double> m) {
  int start_i = i * B;
  int end_i = (i * B + B > M) ? M : i * B + B;
  int start_j = j * B;
  int end_j = (j * B + B > N) ? N : j * B + B;

  for (int ii = start_i; ii < end_i; ++ii) {
    for (int jj = start_j; jj < end_j; ++jj) {
      double& val = m.get()[ii * N + jj];
      val += ii == 0 ? 0.0 : m.get()[(ii - 1) * N + jj];
      val += ii >= M - 1 ? 0.0 : m.get()[(ii + 1) * N + jj];
      val += jj == 0 ? 0.0 : m.get()[ii * N + jj - 1];
      val += jj >= N - 1 ? 0.0 : m.get()[ii * N + jj + 1];
      val *= 0.25;
    }
  }
}

// serial implementation of wavefront computation.
void wavefront_serial() {
  for (int i = 0; i < MB; i++) {
    for (int j = 0; j < NB; j++) {
      stencil_computation(i, j, matrix2);
    }
  }
}

// Method to generate  wavefront tasks with two inputs.
template <typename funcT>
auto make_wavefront2(std::shared_ptr<double> m, const funcT& func, Edge<Key, void>& input1, Edge<Key, void>& input2) {
  auto f = [m, func](const Key& key, std::tuple<Out<Key, void>, Out<Key, void>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    func(i, j, m);

    if (next_i < MB) {
      if (j == 0)
        sendk<0>(Key(next_i, j), out);
      else
        sendk<0>(Key(next_i, j), out);
    }

    if (next_j < NB) {
      if (i == 0)
        sendk<0>(Key(i, next_j), out);
      else
        sendk<1>(Key(i, next_j), out);
    }
  };

  return make_tt(f, edges(input1, input2), edges(input1, input2), "wavefront2", {"first", "second"},
                 {"output1", "output2"});
}

// Method to generate wavefront task with single input.
template <typename funcT>
auto make_wavefront(std::shared_ptr<double> m, const funcT& func, Edge<Key, void>& input1, Edge<Key, void>& input2) {
  auto f = [m, func](const Key& key, std::tuple<Out<Key, void>, Out<Key, void>, Out<Key, void>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    func(i, j, m);

    if (next_i < MB) {
      if (j == 0)  // Single predecessor
        sendk<0>(Key(next_i, j), out);
      else  // Two predecessors
        sendk<1>(Key(next_i, j), out);
    }
    if (next_j < NB) {
      if (i == 0)  // Single predecessor
        sendk<0>(Key(i, next_j), out);
      else  // Two predecessors
        sendk<2>(Key(i, next_j), out);
    }
  };

  Edge<Key, void> recur("recur");
  return make_tt(f, edges(recur), edges(recur, input1, input2), "wavefront", {"control"},
                 {"recur", "output1", "output2"});
}

int main(int argc, char** argv) {
  initialize(argc, argv, -1);
  if (ttg::default_execution_context().size() > 1) {
    std::cout << "This is a shared memory version of Wavefront. Please run it on a single process.\n";
    ttg_abort();
  }
  M = N = 2048;
  B = 64;

  MB = (M / B) + (M % B > 0);
  NB = (N / B) + (N % B > 0);

  init_matrix();

  Edge<Key, void> parent1("parent1"), parent2("parent2");
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  auto s = make_wavefront(matrix, stencil_computation, parent1, parent2);
  auto s2 = make_wavefront2(matrix, stencil_computation, parent1, parent2);

  auto connected = make_graph_executable(s.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected.\n";

  if (ttg::default_execution_context().rank() == 0) {
    // std::cout << "==== begin dot ====\n";
    // std::cout << Dot()(s.get()) << std::endl;
    // std::cout << "==== end dot ====\n";

    beg = std::chrono::high_resolution_clock::now();
    s->in<0>()->sendk(Key(0, 0));
  }

  execute();
  fence();

  if (ttg::default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;
  }

  ttg_finalize();

  std::cout << "Computing using serial version....";
  beg = std::chrono::high_resolution_clock::now();
  wavefront_serial();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "....done!" << std::endl;
  std::cout << "Serial Execution Time (milliseconds) : "
            << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1e3 << std::endl;

  // print_matrix();
  std::cout << "Verifying the result....";
  bool success = true;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (matrix.get()[i * N + j] - matrix2.get()[i * N + j] != 0) {
        success = false;
        std::cout << "ERROR (" << i << "," << j << ")\n";
      }
    }
  }

  std::cout << "....done!" << std::endl;
  std::cout << (success ? "SUCCESS!!!" : "FAILED!!!") << std::endl;

  // delete_matrix();

  return 0;
}
