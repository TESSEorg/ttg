/*
./fw-apsp-mad 1024 16 iterative verify-results
*/
#include <stdlib.h> /* atoi, srand, rand */
#include <time.h>   /* time */
#include <chrono>
#include <cmath>    /* fabs */
#include <iostream> /* cout, boolalpha */
#include <limits>   /* numeric_limits<double>::epsilon() */
#include <memory>
#include <string> /* string */
#include <tuple>
#include <utility>
#include <algorithm>

#if __has_include(<execution>)
#include <execution>
#define HAS_EXECUTION_HEADER
#endif

#include "blockmatrix.h"
#include "../ttg/util/bug.h"

//#include <omp.h> //

#include "FW-APSP/FloydIterativeKernelDF.h"        // contains the iterative kernel
#include "FW-APSP/FloydRecursiveSerialKernelDF.h"  // contains the recursive but serial kernels
// #include "FloydRecursiveParallelKernel.h" // contains the recursive and parallel kernels

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

struct Key {
  // ((I, J), K) where (I, J) is the tile coordiante and K is the iteration number
  std::pair<std::pair<int, int>, int> execution_info;

  bool operator==(const Key& b) const {
    if (this == &b) return true;
    return execution_info.first.first == b.execution_info.first.first &&
           execution_info.first.second == b.execution_info.first.second &&
           execution_info.second == b.execution_info.second;
  }

  bool operator!=(const Key& b) const { return !((*this) == b); }

  madness::hashT hash_val;

  Key() : execution_info(std::make_pair(std::make_pair(0, 0), 0)) { rehash(); }
  Key(const std::pair<std::pair<int, int>, int>& e) : execution_info(e) { rehash(); }
  Key(int e_f_f, int e_f_s, int e_s) : execution_info(std::make_pair(std::make_pair(e_f_f, e_f_s), e_s)) { rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    std::hash<int> int_hasher;
    hash_val = int_hasher(execution_info.first.first) ^ int_hasher(execution_info.first.second) ^
               int_hasher(execution_info.second);
  }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar& madness::archive::wrap((unsigned char*)this, sizeof(*this));
  }
};

namespace std {
  // specialize std::hash for Key
  template <>
  struct hash<Key> {
    std::size_t operator()(const Key& s) const noexcept { return s.hash(); }
  };
}  // namespace std

std::ostream& operator<<(std::ostream& s, const Key& key) {
  s << "Key((" << key.execution_info.first.first << "," << key.execution_info.first.second << "), "
    << key.execution_info.second << ")";
  return s;
}

// An empty class used for pure control flows
class Control {
 public:
  template <typename Archive>
  void serialize(Archive& ar) {}
};

std::ostream& operator<<(std::ostream& s, const Control ctl) {
  s << "Ctl";
  return s;
}

template <typename T>
class Initiator : public Op<int,
                            std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                            Out<Key, BlockMatrix<T>>>,
                            Initiator<T>> {
  using baseT = Op<int,
                   std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                              Out<Key, BlockMatrix<T>>>,
                   Initiator>;
  Matrix<T>* adjacency_matrix_ttg;

 public:
  Initiator(Matrix<T>* adjacency_matrix_ttg, const std::string& name)
      : baseT(name, {}, {"outA", "outB", "outC", "outD"}
      ), adjacency_matrix_ttg(adjacency_matrix_ttg) {}
  Initiator(Matrix<T>* adjacency_matrix_ttg, const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(edges(), outedges, name, {}, {"outA", "outB", "outC", "outD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg) {}

  ~Initiator() {}

  void op(const int& iterations, typename baseT::output_terminals_type& out) {
    // making x_ready for all the blocks (for function calls A, B, C, and D)
    // This triggers for the immediate execution of function A at tile [0, 0]. But
    // functions B, C, and D have other dependencies to meet before execution; They wait
#ifdef HAS_EXECUTION_HEADER
    std::for_each(std::execution::par, adjacency_matrix_ttg->get().begin(), adjacency_matrix_ttg->get().end(),
      [&out](const std::pair<std::pair<int,int>, BlockMatrix<T>>& kv)
#else
    std::for_each(adjacency_matrix_ttg->get().begin(), adjacency_matrix_ttg->get().end(),
      [&out](const std::pair<std::pair<int,int>, BlockMatrix<T>>& kv)
#endif
    {
        auto [i,j] = kv.first;
        if (i == 0 && j == 0) {  // A function call
          ::send<0>(Key(std::make_pair(std::make_pair(i, j), 0)), kv.second, out);
        } else if (i == 0) {  // B function call
          ::send<1>(Key(std::make_pair(std::make_pair(i, j), 0)), kv.second, out);
        } else if (j == 0) {  // C function call
          ::send<2>(Key(std::make_pair(std::make_pair(i, j), 0)), kv.second, out);
        } else {  // D function call
          ::send<3>(Key(std::make_pair(std::make_pair(i, j), 0)), kv.second, out);
        }
    });
  }
};

template <typename T>
class Finalizer : public Op<Key, std::tuple<>, Finalizer<T>, BlockMatrix<T>> {
  using baseT = Op<Key, std::tuple<>, Finalizer<T>, BlockMatrix<T>>;
  Matrix<T>* result_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;
  bool verify_results;
  T* adjacency_matrix_serial;

 public:
  Finalizer(Matrix<T>* result_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
            int recursive_fan_out, int base_size, const std::string& name,
            T* adjacency_matrix_serial, bool verify_results = false)
      : baseT(name, {"input"}, {})
      , result_matrix_ttg(result_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size)
      , verify_results(verify_results)
      , adjacency_matrix_serial(adjacency_matrix_serial) {}

  Finalizer(Matrix<T>* result_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
            int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
            const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"input"}, {})
      , result_matrix_ttg(result_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  // BlockMatrix<T> &&in -- use direct arguments
  void op(const Key& key, const std::tuple<BlockMatrix<T>>&& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int block_size = problem_size / blocking_factor;

    BlockMatrix<T> bm = get<0>(t);
    //cout << "[" << I << "," << J << "] : " << bm << endl;
    bool equal = true;
    if (verify_results) {
      int block_size = problem_size / blocking_factor;
      for (int i = 0; i < problem_size; ++i) {
        int row = i * problem_size;
        int blockX = i / block_size;
        if (blockX == I) {
          int x = i % block_size;
          for (int j = 0; j < problem_size; ++j) {
            int blockY = j / block_size;
            if (blockY == J) {
              int y = j % block_size;
              double v1 = bm(x,y);
              double v2 = adjacency_matrix_serial[row + j];
              if (fabs(v1 - v2) > numeric_limits<double>::epsilon()) {
                cout << "ERROR in block [" << I << "," << J << "], element[" << i << ", " << j << "]: " << v2 << "!=" << v1 << endl;
                equal = false;
                break;
              }
            }
          }
        }
      }
    }
    // Use std::forward to forward a decay of reference types? Read about this.
    //BlockMatrix<T> bm (block_size, block_size);
    //(madness::archive::wrap(bm.get(), block_size * block_size));
    //cout << bm << endl;
    //(*result_matrix_ttg)(I, J, get<0>(t));  // Store the result block in the matrix
  }
};

template <typename T>
class FuncA : public Op<Key,
                        std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                   Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                   Out<Key, BlockMatrix<T>>>,
                        FuncA<T>, BlockMatrix<T>> {
  using baseT = Op<
      Key,
      std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                 Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
      FuncA, BlockMatrix<T>>;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncA(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready"}, {"outA", "outB", "outC", "outD", "readyB", "readyC", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncA(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready"}, {"outA", "outB", "outC", "outD", "readyB", "readyC", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<BlockMatrix<T>>&& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    BlockMatrix<T> m_ij;
    // Executing the update
    if (kernel_type == "iterative") {
      // cout << "FuncA" << I << " " << J << " " << K << endl;
      m_ij = floyd_iterative_kernel(problem_size / blocking_factor, (get<0>(t)), (get<0>(t)), (get<0>(t)));
      //cout << "A[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
    }

    // Making u_ready/v_ready for all the B/C function calls in the CURRENT iteration
    for (int l = 0; l < blocking_factor; ++l) {
      if (l != K) {
        /*if (K == 0) {
          // B calls - x_ready
          ::send<1>(Key(std::make_pair(std::make_pair(I, l), K)), (*adjacency_matrix_ttg)(I,l), out);
          // C calls - x_ready
          ::send<2>(Key(std::make_pair(std::make_pair(l, J), K)), (*adjacency_matrix_ttg)(l,J), out);
        }*/
        // B calls
        // cout << "Send " << I << " " << l << " " << K << endl;
        ::send<4>(Key(std::make_pair(std::make_pair(I, l), K)), m_ij, out);
        // C calls
        // cout << "Send " << l << " " << J << " " << K << endl;
        ::send<5>(Key(std::make_pair(std::make_pair(l, J), K)), m_ij, out);
      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
                                       // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else {  // in the next iteration, we have D function call
                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      }
    } else {
      //cout << "A[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
      ::send<6>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
    }
  }
};

template <typename T>
class FuncB : public Op<Key,
                        std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                   Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                        FuncB<T>, BlockMatrix<T>, BlockMatrix<T>> {
  using baseT = Op<Key,
                   std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                              Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                   FuncB, BlockMatrix<T>, BlockMatrix<T>>;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncB(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "u_ready"}, {"outA", "outB", "outC", "outD", "readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncB(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "u_ready"}, {"outA", "outB", "outC", "outD", "readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<BlockMatrix<T>, BlockMatrix<T>>&& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    BlockMatrix<T> m_ij;
    // Executing the update
    if (kernel_type == "iterative") {
      // cout << "FuncB" << I << " " << J << " " << K << endl;
      m_ij = floyd_iterative_kernel(problem_size / blocking_factor, (get<0>(t)), (get<1>(t)), (get<0>(t)));
      //cout << "B[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
    }

    // Making v_ready for all the D function calls in the CURRENT iteration
    for (int i = 0; i < blocking_factor; ++i) {
      if (i != I) {
        //if (K == 0)
          //::send<3>(Key(std::make_pair(std::make_pair(i, J), K)), (*adjacency_matrix_ttg)(i,J), out);
        // cout << "Send " << i << " " << J << " " << K << endl;
        ::send<4>(Key(std::make_pair(std::make_pair(i, J), K)), m_ij, out);
      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
                                       // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else {  // in the next iteration, we have D function call
                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      }
    } else {
      //cout << "B[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
      ::send<5>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
    }
  }
};

template <typename T>
class FuncC : public Op<Key,
                        std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                   Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                        FuncC<T>, BlockMatrix<T>, BlockMatrix<T>> {
  using baseT = Op<Key,
                   std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                              Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                   FuncC, BlockMatrix<T>, BlockMatrix<T>>;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncC(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "v_ready"}, {"outA", "outB", "outC", "outD", "readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncC(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "v_ready"}, {"outA", "outB", "outC", "outD", "readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<BlockMatrix<T>, BlockMatrix<T>>&& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    BlockMatrix<T> m_ij;
    // Executing the update
    if (kernel_type == "iterative") {
      // cout << "FuncC" << I << " " << J << " " << K << endl;
      m_ij = floyd_iterative_kernel(problem_size / blocking_factor, (get<0>(t)), (get<0>(t)), (get<1>(t)));
      //cout << "C[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
    }

    // Making u_ready for all the D function calls in the CURRENT iteration
    for (int j = 0; j < blocking_factor; ++j) {
      if (j != J) {
        //::send<4>(Key(std::make_pair(std::make_pair(I, j), K)), (*adjacency_matrix_ttg)(I,j), out);
        // cout << "Send " << I << " " << j << " " << K << endl;
        ::send<4>(Key(std::make_pair(std::make_pair(I, j), K)), m_ij, out);
      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
                                       // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else {  // in the next iteration, we have D function call
                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      }
    } else {
      //cout << "C[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
      ::send<5>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
    }
  }
};

template <typename T>
class FuncD : public Op<Key,
                        std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                   Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                        FuncD<T>, BlockMatrix<T>, BlockMatrix<T>, BlockMatrix<T>> {
  using baseT = Op<Key,
                   std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                              Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                   FuncD, BlockMatrix<T>, BlockMatrix<T>, BlockMatrix<T>>;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncD(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "v_ready", "u_ready"}, {"outA", "outB", "outC", "outD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncD(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "v_ready", "u_ready"}, {"outA", "outB", "outC", "outD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key,
          const std::tuple<BlockMatrix<T>, BlockMatrix<T>, BlockMatrix<T>>&& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    BlockMatrix<T> m_ij;
    // Executing the update
    if (kernel_type == "iterative") {
      // cout << "FuncD" << I << " " << J << " " << K << endl;
      m_ij = floyd_iterative_kernel(problem_size / blocking_factor, (get<0>(t)), (get<2>(t)), (get<1>(t)));
      //cout << "D[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
                                       // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
                                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else {  // in the next iteration, we have D function call
                // cout << "Send " << I << " " << J << " " << K << endl;
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      }
    } else {
      //cout << "D[" << I << "," << J << "," << K <<  "]: " << m_ij << endl;
      ::send<4>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
    }
  }
};

template <typename T>
class FloydWarshall {
  Initiator<T> initiator;
  FuncA<T> funcA;
  FuncB<T> funcB;
  FuncC<T> funcC;
  FuncD<T> funcD;
  Finalizer<T> finalizer;
  World& world;

  // Needed for Initiating the execution in Initiator data member (see the function start())
  int blocking_factor;

 public:
  FloydWarshall(Matrix<T>* adjacency_matrix_ttg, Matrix<T>* result_matrix_ttg, int problem_size, int blocking_factor,
                const std::string& kernel_type, int recursive_fan_out, int base_size, T* adjacency_matrix_serial,
                bool verify_results = false)
      : initiator(adjacency_matrix_ttg, "initiator")
      , funcA(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcA")
      , funcB(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcB")
      , funcC(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcC")
      , funcD(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcD")
      , finalizer(result_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size,
                  "finalizer", adjacency_matrix_serial, verify_results)
      , world(madness::World::get_default())
      , blocking_factor(blocking_factor) {
    initiator.template out<0>()->connect(funcA.template in<0>());
    initiator.template out<1>()->connect(funcB.template in<0>());
    initiator.template out<2>()->connect(funcC.template in<0>());
    initiator.template out<3>()->connect(funcD.template in<0>());

    funcA.template out<0>()->connect(funcA.template in<0>());
    funcA.template out<1>()->connect(funcB.template in<0>());
    funcA.template out<2>()->connect(funcC.template in<0>());
    funcA.template out<3>()->connect(funcD.template in<0>());
    funcA.template out<4>()->connect(funcB.template in<1>());
    funcA.template out<5>()->connect(funcC.template in<1>());
    funcA.template out<6>()->connect(finalizer.template in<0>());

    funcB.template out<0>()->connect(funcA.template in<0>());
    funcB.template out<1>()->connect(funcB.template in<0>());
    funcB.template out<2>()->connect(funcC.template in<0>());
    funcB.template out<3>()->connect(funcD.template in<0>());
    funcB.template out<4>()->connect(funcD.template in<1>());
    funcB.template out<5>()->connect(finalizer.template in<0>());

    funcC.template out<0>()->connect(funcA.template in<0>());
    funcC.template out<1>()->connect(funcB.template in<0>());
    funcC.template out<2>()->connect(funcC.template in<0>());
    funcC.template out<3>()->connect(funcD.template in<0>());
    funcC.template out<4>()->connect(funcD.template in<2>());
    funcC.template out<5>()->connect(finalizer.template in<0>());

    funcD.template out<0>()->connect(funcA.template in<0>());
    funcD.template out<1>()->connect(funcB.template in<0>());
    funcD.template out<2>()->connect(funcC.template in<0>());
    funcD.template out<3>()->connect(funcD.template in<0>());
    funcD.template out<4>()->connect(finalizer.template in<0>());

    if (!make_graph_executable(&initiator)) throw "should be connected";
    world.gop.fence();
  }

  void print() {}  //{Print()(&producer);}
  std::string dot() { return Dot()(&initiator); }
  void start() {
    if (world.rank() == 0) initiator.invoke(blocking_factor);
    ttg_execute(ttg_default_execution_context());
  }
  void fence() { ttg_fence(ttg_default_execution_context()); }
};

/* How to call? ./floyd
                                        <PROBLEM_SIZE? 1024>
                                        <BLOCKING_FACTOR? 16>
                                        <KERNEL_TYPE? iterative/recursive-serial/recursive-parallel>
                                        <VERIFY_RESULTS? verify-results, do-not-verify-results>
                                        <IF KERNEL_TYPE == recursive-serial or recursive-parallel -> R? 8>
                                        <IF KERNEL_TYPE == recursive-serial or recursive-parallel -> BASE_SIZE? 32>
        E.g.,:
                ./floyd 8192 16 iterative verify-results
                ./floyd 8192 16 recursive-serial verify-results 8 32
                ./floyd 8192 32 recursive-serial do-not-verify-results 2 32
*/

// Parses the input arguments received from the command line
void parse_arguments(int argc, char** argv, int& problem_size, int& blocking_factor, string& kernel_type,
                     int& recursive_fan_out, int& base_size, bool& verify_results);
// Initializes the adjacency matrices
void init_square_matrix(int problem_size, int blocking_factor, bool verify_results, double* adjacency_matrix_serial,
                        Matrix<double>* m);
// Print the matrix
void print_square_matrix(int problem_size, int blocking_factor, bool verify_results, double* adjacency_matrix_serial);
// Checking the equality of two double values
bool equals(double v1, double v2);
// Checking the equality of the two matrix after computing fw-apsp on them
bool equals(Matrix<double>* matrix1, double* matrix2, int problem_size, int blocking_factor);
// Iterative O(n^3) loop-based fw-apsp
void floyd_iterative(double* adjacency_matrix_serial, int problem_size);

int main(int argc, char** argv) {
  OpBase::set_trace_all(false);

  ttg_initialize(argc, argv);

  using mpqc::Debugger;
  auto debugger = std::make_shared<Debugger>();
  Debugger::set_default_debugger(debugger);
  debugger->set_exec(argv[0]);
  debugger->set_prefix(ttg_default_execution_context().rank());
  debugger->set_cmd("lldb_xterm");

  ttg_fence(ttg_default_execution_context());

  for (int arg = 1; arg < argc; ++arg) {
    if (strcmp(argv[arg], "-dx") == 0) madness::xterm_debug(argv[0], 0);
  }

  // NEW IMPLEMENTATION
  int problem_size;
  int blocking_factor;
  string kernel_type;
  // if the kernel_type is "recursive-serial" or "recursive-parallel"
  int recursive_fan_out;
  int base_size;
  bool verify_results;
  parse_arguments(argc, argv, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, verify_results);

  double* adjacency_matrix_serial = nullptr;  // Using for the verification (if needed)
  // double *adjacency_matrix_ttg = nullptr; // Using for running the blocked implementation of FW-APSP algorithm on ttg
  // runtime

  int block_size = problem_size / blocking_factor;
  int n_brows = (problem_size / block_size) + (problem_size % block_size > 0);
  int n_bcols = n_brows;

  Matrix<double>* m = new Matrix<double>(n_brows, n_bcols, block_size, block_size);
  Matrix<double>* r = new Matrix<double>(n_brows, n_bcols, block_size, block_size);

  if (verify_results) {
    adjacency_matrix_serial = (double*)malloc(sizeof(double) * problem_size * problem_size);
  }

  init_square_matrix(problem_size, blocking_factor, verify_results, adjacency_matrix_serial, m);

  //Run in every process to be able to verify? Is there another way?
  // Calling the iterative fw-apsp
  if (verify_results) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    floyd_iterative(adjacency_matrix_serial, problem_size);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    cout << "iterative fw-apsp took: " << duration / 1000000.0 << " seconds" << endl;
  }

  // Running the ttg version
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  // Calling the blocked implementation of FW-APSP algorithm on ttg runtime
  FloydWarshall fw_apsp(m, r, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size,
                        adjacency_matrix_serial, verify_results);
  //std::cout << fw_apsp.dot() << std::endl;
  fw_apsp.start();
  fw_apsp.fence();
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (world.rank() == 0)
    cout << "blocked ttg (data-flow) fw-apsp took: " << duration / 1000000.0 << " seconds" << endl;

  /*if (verify_results && world.rank() == 0) {
    r->print();
    if (equals(r, adjacency_matrix_serial, problem_size, blocking_factor)) {
      cout << "Serial and TTG implementation matches!" << endl;
    } else {
      cout << "Serial and TTG implementation DOESN\"T match!" << endl;
    }
  }*/

  // deallocating the allocated memories
  if (verify_results) {
    free(adjacency_matrix_serial);
  }
  delete m;

  ttg_finalize();

  return 0;
}

void floyd_iterative(double* adjacency_matrix_serial, int problem_size) {
  for (int k = 0; k < problem_size; ++k) {
    int k_row = problem_size * k;
    for (int i = 0; i < problem_size; ++i) {
      int i_row = problem_size * i;
      for (int j = 0; j < problem_size; ++j) {
        adjacency_matrix_serial[i_row + j] =
            min(adjacency_matrix_serial[i_row + j],
                adjacency_matrix_serial[i_row + k] + adjacency_matrix_serial[k_row + j]);
      }
    }
  }
}

bool equals(double v1, double v2) { return fabs(v1 - v2) < numeric_limits<double>::epsilon(); }

bool equals(Matrix<double>* matrix1, double* matrix2, int problem_size, int blocking_factor) {
  int block_size = problem_size / blocking_factor;

  for (int i = 0; i < problem_size; ++i) {
    int row = i * problem_size;
    int blockX = i / block_size;
    int x = i % block_size;
    for (int j = 0; j < problem_size; ++j) {
      int blockY = j / block_size;
      int y = j % block_size;
      double v1 = ((*matrix1)(blockX, blockY))(x, y);
      double v2 = matrix2[row + j];
      if (!equals(v1, v2)) {
        cout << "[" << i << ", " << j << "]: " << v2 << "!=" << v1 << endl;
        return false;
      }
    }
  }
  return true;
}

void init_square_matrix(int problem_size, int blocking_factor, bool verify_results, double* adjacency_matrix_serial,
                        Matrix<double>* m) {
  //srand(123);  // srand(time(nullptr));
  int block_size = problem_size / blocking_factor;
  for (int i = 0; i < problem_size; ++i) {
    int row = i * problem_size;
    int blockX = i / block_size;
    int x = i % block_size;
    for (int j = 0; j < problem_size; ++j) {
      int blockY = j / block_size;
      int y = j % block_size;
      if (i != j) {
        double value = i * blocking_factor + j; //rand() % 100 + 1;
        ((*m)(blockX, blockY))(x, y, value);
        if (verify_results) {
          adjacency_matrix_serial[row + j] = value;
        }
      } else {
        ((*m)(blockX, blockY))(x, y, 0.0);
        if (verify_results) {
          adjacency_matrix_serial[row + j] = 0.0;
        }
      }
    }
  }
}

void print_square_matrix(int problem_size, int blocking_factor, bool verify_results, double* adjacency_matrix_serial) {
  for (int i = 0; i < problem_size; ++i) {
    int row = i * problem_size;
    for (int j = 0; j < problem_size; ++j) {
      cout << adjacency_matrix_serial[row + j] << " ";
    }
    cout << endl;
  }
}

void parse_arguments(int argc, char** argv, int& problem_size, int& blocking_factor, string& kernel_type,
                     int& recursive_fan_out, int& base_size, bool& verify_results) {
  problem_size = atoi(argv[1]);     // e.g., 1024
  blocking_factor = atoi(argv[2]);  // e.g., 16
  kernel_type = argv[3];            // e.g., iterative/recursive-serial/recursive-parallel
  string verify_res(argv[4]);
  verify_results = (verify_res == "verify-results");

  cout << "Problem_size: " << problem_size << ", blocking_factor: " << blocking_factor
       << ", kernel_type: " << kernel_type << ", verify_results: " << boolalpha << verify_results;
  if (argc > 5) {
    recursive_fan_out = atoi(argv[5]);
    base_size = atoi(argv[6]);
    cout << ", recursive_fan_out: " << recursive_fan_out << ", base_size: " << base_size << endl;
  } else {
    cout << endl;
  }
}
