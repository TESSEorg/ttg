#include <stdlib.h> /* atoi, srand, rand */
#include <time.h>   /* time */
#include <cmath>    /* fabs */
#include <iostream> /* cout, boolalpha */
#include <limits>   /* numeric_limits<double>::epsilon() */
#include <memory>
#include <string> /* string */
#include <tuple>
#include <utility>
#if __has_include(<execution>)
#include <execution>
#define HAS_EXECUTION_HEADER
#endif
#include "../blockmatrix.h"

// #include <omp.h> //

#include "GE/GEIterativeKernelDF.h"  // contains the iterative kernel
//#include "GE/GERecursiveParallelKernel.h"  // contains the recursive but serial kernels
#include "GE/GERecursiveSerialKernel.h"  // contains the recursive and parallel kernels

#include "ttg.h"
using namespace ttg;

#include "ttg/serialization.h"
#include "ttg/serialization/std/pair.h"

#include <madness/world/world.h>

struct Key {
  // ((I, J), K) where (I, J) is the tile coordiante and K is the iteration number
  std::pair<std::pair<int, int>, int> execution_info;
  madness::hashT hash_val;

  Key() : execution_info(std::make_pair(std::make_pair(0, 0), 0)) { rehash(); }
  Key(const std::pair<std::pair<int, int>, int>& e) : execution_info(e) { rehash(); }
  Key(int e_f_f, int e_f_s, int e_s) : execution_info(std::make_pair(std::make_pair(e_f_f, e_f_s), e_s)) { rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    std::hash<int> int_hasher;
    hash_val = int_hasher(execution_info.first.first) * 2654435769 + int_hasher(execution_info.first.second) * 40503 +
               int_hasher(execution_info.second);
  }

  // Equality test
  bool operator==(const Key& b) const { return execution_info == b.execution_info; }

  // Inequality test
  bool operator!=(const Key& b) const { return !((*this) == b); }

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive>
  void serialize(Archive& ar) {
    ar& madness::archive::wrap((unsigned char*)this, sizeof(*this));
  }
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& execution_info;
    if constexpr (ttg::detail::is_boost_input_archive_v<Archive>) rehash();
  }
#endif

  friend std::ostream& operator<<(std::ostream& out, Key const& k) {
    out << "Key((" << k.execution_info.first.first << "," << k.execution_info.first.second << "),"
        << k.execution_info.second << ")";
    return out;
  }
};

namespace std {
  // specialize std::hash for Key
  template <>
  struct hash<Key> {
    std::size_t operator()(const Key& s) const noexcept { return s.hash(); }
  };
}  // namespace std

// An empty class used for pure control flows
class Control {
 public:
  template <typename Archive>
  void serialize(Archive& ar) {}
};

std::ostream& operator<<(std::ostream& s, const Control& ctl) {
  s << "Ctl";
  return s;
}

struct Integer {
  int value;
  madness::hashT hash_val;
  Integer() : value(0) { rehash(); }
  Integer(int v) : value(v) { rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    std::hash<int> int_hasher;
    hash_val = int_hasher(value);
  }

  // Equality test
  bool operator==(const Integer& b) const { return value == b.value; }

  // Inequality test
  bool operator!=(const Integer& b) const { return !((*this) == b); }

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive>
  void serialize(Archive& ar) {
    ar& value;
    if constexpr (madness::is_input_archive_v<Archive>) rehash();
  }
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& value;
    if constexpr (ttg::detail::is_boost_input_archive_v<Archive>) rehash();
  }
#endif
};

std::ostream& operator<<(std::ostream& s, const Integer& intVal) {
  s << "Integer-wrapper -- value: " << intVal.value;
  return s;
}

template <typename T>
class Initiator : public TT<Integer,
                            std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                       Out<Key, BlockMatrix<T>>>,
                            Initiator<T>> {
  using baseT = typename Initiator::ttT;

  Matrix<T>* adjacency_matrix_ttg;

 public:
  Initiator(Matrix<T>* adjacency_matrix_ttg, const std::string& name)
      : baseT(name, {}, {"outA", "outB", "outC", "outD"}), adjacency_matrix_ttg(adjacency_matrix_ttg) {}
  Initiator(Matrix<T>* adjacency_matrix_ttg, const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(edges(), outedges, name, {}, {"outA", "outB", "outC", "outD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg) {}

  ~Initiator() {}

  void op(const Integer& iterations, typename baseT::output_terminals_type& out) {
    // making x_ready for all the blocks (for function calls A, B, C, and D)
    // This triggers for the immediate execution of function A at tile [0, 0]. But
    // functions B, C, and D have other dependencies to meet before execution; They wait
#ifdef HAS_EXECUTION_HEADER
    std::for_each(std::execution::par, adjacency_matrix_ttg->get().begin(), adjacency_matrix_ttg->get().end(),
                  [&out](const std::pair<std::pair<int, int>, BlockMatrix<T>>& kv)
#else
    std::for_each(adjacency_matrix_ttg->get().begin(), adjacency_matrix_ttg->get().end(),
                  [&out](const std::pair<std::pair<int, int>, BlockMatrix<T>>& kv)
#endif
                  {
                    auto [i, j] = kv.first;
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
class Finalizer : public TT<Key, std::tuple<>, Finalizer<T>, ttg::typelist<BlockMatrix<T>>> {
  using baseT = typename Finalizer::ttT;
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
            int recursive_fan_out, int base_size, const std::string& name, T* adjacency_matrix_serial,
            bool verify_results = false)
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

  void op(const Key& key, const std::tuple<BlockMatrix<T>>&& t, typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int block_size = problem_size / blocking_factor;

    BlockMatrix<T> bm = get<0>(t);

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
              double v1 = bm(x, y);
              double v2 = adjacency_matrix_serial[row + j];
              if (fabs(v1 - v2) > numeric_limits<double>::epsilon()) {
                cout << "ERROR in block [" << I << "," << J << "], element[" << i << ", " << j << "]: " << v2
                     << "!=" << v1 << endl;
                equal = false;
                break;
              }
            }
          }
        }
      }
    }
    // Use std::forward to forward a decay of reference types? Read about this.
    //(*result_matrix_ttg)(I, J, get<0>(t));  // Store the result block in the matrix
  }
};

template <typename T>
class FuncA : public TT<Key,
                        std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                   Out<Key, BlockMatrix<T>>>,
                        FuncA<T>, ttg::typelist<BlockMatrix<T>>> {
  using baseT = typename FuncA::ttT;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncA(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready"}, {"readyB", "readyC", "readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncA(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready"}, {"readyB", "readyC", "readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<BlockMatrix<T>>& t, typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    // std::cout << "FuncA " << I << " " << J << " " << K << std::endl;
    BlockMatrix<T> m_ij = get<0>(t);
    // Executing the update
    if (kernel_type == "iterative") {
      // m_ij =
      ge_iterative_kernel(problem_size / blocking_factor, I, J, K, m_ij.get(), get<0>(t).get(), get<0>(t).get(),
                          get<0>(t).get());
    } /*else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      ge_recursive_serial_kernelA(adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb, k_lb, recursive_fan_out,
                                  base_size);
    } */
    else {
      //      int block_size = problem_size / blocking_factor;
      //      int i_lb = I * block_size;
      //      int j_lb = J * block_size;
      //      int k_lb = K * block_size;
      //#pragma omp prallel
      //      {
      //#pragma omp single
      //        {
      //#pragma omp task
      //          ge_recursive_parallel_kernelA(adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb, k_lb,
      //                                        recursive_fan_out, base_size);
      //        }
      //      }
    }

    // Making u_ready/v_ready for all the B/C function calls in the CURRENT iteration
    for (int l = K + 1; l < blocking_factor; ++l) {
      // B calls
      ::send<0>(Key(std::make_pair(std::make_pair(I, l), K)), m_ij, out);
      // C calls
      ::send<1>(Key(std::make_pair(std::make_pair(l, J), K)), m_ij, out);
    }
    // Making w_ready for all the D function calls in the CURRENT iteration
    for (int i = K + 1; i < blocking_factor; ++i) {
      for (int j = K + 1; j < blocking_factor; ++j) {
        // D calls
        ::send<2>(Key(std::make_pair(std::make_pair(i, j), K)), m_ij, out);
      }
    }
    // Send result
    ::send<3>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
  }
};

template <typename T>
class FuncB : public TT<Key, std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                        FuncB<T>, ttg::typelist<BlockMatrix<T>, BlockMatrix<T>>> {
  using baseT = typename FuncB::ttT;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncB(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "u_ready"}, {"readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncB(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "u_ready"}, {"readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<BlockMatrix<T>, BlockMatrix<T>>& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    // std::cout << "FuncB " << I << " " << J << " " << K << std::endl;
    BlockMatrix<T> m_ij = get<0>(t);
    // Executing the update
    if (kernel_type == "iterative") {
      // m_ij =
      ge_iterative_kernel(problem_size / blocking_factor, I, J, K, m_ij.get(), get<1>(t).get(), get<0>(t).get(),
                          get<1>(t).get());
    } /*else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      ge_recursive_serial_kernelB(adjacency_matrix_ttg, adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb,
                                  k_lb, recursive_fan_out, base_size);
    } */
    else {
      //      int block_size = problem_size / blocking_factor;
      //      int i_lb = I * block_size;
      //      int j_lb = J * block_size;
      //      int k_lb = K * block_size;
      //#pragma omp prallel
      //      {
      //#pragma omp single
      //        {
      //#pragma omp task
      //          ge_recursive_parallel_kernelB(adjacency_matrix_ttg, adjacency_matrix_ttg, problem_size, block_size,
      //          i_lb,
      //                                        j_lb, k_lb, recursive_fan_out, base_size);
      //        }
      //      }
    }

    // Making v_ready for all the D function calls in the CURRENT iteration
    for (int i = K + 1; i < blocking_factor; ++i) {
      ::send<0>(Key(std::make_pair(std::make_pair(i, J), K)), m_ij, out);
    }
    // Send result
    ::send<1>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
  }
};

template <typename T>
class FuncC : public TT<Key, std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>,
                        FuncC<T>, ttg::typelist<BlockMatrix<T>, BlockMatrix<T>>> {
  using baseT = typename FuncC::ttT;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncC(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "v_ready"}, {"readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncC(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "v_ready"}, {"readyD", "result"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<BlockMatrix<T>, BlockMatrix<T>>& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    // std::cout << "FuncC " << I << " " << J << " " << K << std::endl;
    BlockMatrix<T> m_ij = get<0>(t);

    // Executing the update
    if (kernel_type == "iterative") {
      // m_ij =
      ge_iterative_kernel(problem_size / blocking_factor, I, J, K, m_ij.get(), get<0>(t).get(), get<1>(t).get(),
                          get<1>(t).get());
    } /*else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      ge_recursive_serial_kernelC(adjacency_matrix_ttg, adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb,
                                  k_lb, recursive_fan_out, base_size);
    }*/
    else {
      //      int block_size = problem_size / blocking_factor;
      //      int i_lb = I * block_size;
      //      int j_lb = J * block_size;
      //      int k_lb = K * block_size;
      //#pragma omp prallel
      //      {
      //#pragma omp single
      //        {
      //#pragma omp task
      //          ge_recursive_parallel_kernelC(adjacency_matrix_ttg, adjacency_matrix_ttg, problem_size, block_size,
      //          i_lb,
      //                                        j_lb, k_lb, recursive_fan_out, base_size);
      //        }
      //      }
    }

    // Making u_ready for all the D function calls in the CURRENT iteration
    for (int j = K + 1; j < blocking_factor; ++j) {
      ::send<0>(Key(std::make_pair(std::make_pair(I, j), K)), m_ij, out);
    }
    // Send result
    ::send<1>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
  }
};

template <typename T>
class FuncD : public TT<Key,
                        std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
                                   Out<Key, BlockMatrix<T>>>,
                        FuncD<T>, ttg::typelist<BlockMatrix<T>, BlockMatrix<T>, BlockMatrix<T>, BlockMatrix<T>>> {
  using baseT = typename FuncD::ttT;
  Matrix<T>* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncD(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "v_ready", "u_ready", "w_ready"}, {"outA", "outB", "outC", "outD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncD(Matrix<T>* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "v_ready", "u_ready", "w_ready"}, {"outA", "outB", "outC", "outD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<BlockMatrix<T>, BlockMatrix<T>, BlockMatrix<T>, BlockMatrix<T>>&& t,
          typename baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;
    // std::cout << "FuncD " << I << " " << J << " " << K << std::endl;
    BlockMatrix<T> m_ij = get<0>(t);
    // Executing the update
    if (kernel_type == "iterative") {
      // m_ij =
      ge_iterative_kernel(problem_size / blocking_factor, I, J, K, m_ij.get(), get<2>(t).get(), get<1>(t).get(),
                          get<3>(t).get());
    } /*else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      ge_recursive_serial_kernelD(adjacency_matrix_ttg, adjacency_matrix_ttg, adjacency_matrix_ttg,
                                  adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb, k_lb, recursive_fan_out,
                                  base_size);
    }*/
    else {
      //      int block_size = problem_size / blocking_factor;
      //      int i_lb = I * block_size;
      //      int j_lb = J * block_size;
      //      int k_lb = K * block_size;
      //#pragma omp parallel
      //      {
      //#pragma omp single
      //        {
      //#pragma omp task
      //          ge_recursive_parallel_kernelD(adjacency_matrix_ttg, adjacency_matrix_ttg, adjacency_matrix_ttg,
      //                                        adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb, k_lb,
      //                                        recursive_fan_out, base_size);
      //        }
      //      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      } else {  // in the next iteration, we have D function call
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), m_ij, out);
      }
    }
    // else
    // Send result when all iterations are done.
    //::send<4>(Key(std::make_pair(std::make_pair(I, J), K)), m_ij, out);
  }
};

template <typename T>
class GaussianElimination {
  Initiator<T> initiator;
  FuncA<T> funcA;
  FuncB<T> funcB;
  FuncC<T> funcC;
  FuncD<T> funcD;
  Finalizer<T> finalizer;

  // Needed for Initiating the execution in Initiator data member (see the function start())
  int blocking_factor;

  ttg::World world;

 public:
  GaussianElimination(Matrix<T>* adjacency_matrix_ttg, Matrix<T>* result_matrix_ttg, int problem_size,
                      int blocking_factor, const std::string& kernel_type, int recursive_fan_out, int base_size,
                      T* adjacency_matrix_serial, bool verify_results = false)
      : initiator(adjacency_matrix_ttg, "initiator")
      , funcA(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcA")
      , funcB(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcB")
      , funcC(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcC")
      , funcD(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcD")
      , finalizer(result_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size,
                  "finalizer", adjacency_matrix_serial, verify_results)
      , blocking_factor(blocking_factor)
      , world(ttg::default_execution_context()) {
    initiator.template out<0>()->connect(funcA.template in<0>());
    initiator.template out<1>()->connect(funcB.template in<0>());
    initiator.template out<2>()->connect(funcC.template in<0>());
    initiator.template out<3>()->connect(funcD.template in<0>());

    funcA.template out<0>()->connect(funcB.template in<1>());
    funcA.template out<1>()->connect(funcC.template in<1>());
    funcA.template out<2>()->connect(funcD.template in<3>());
    funcA.template out<3>()->connect(finalizer.template in<0>());

    funcB.template out<0>()->connect(funcD.template in<1>());
    funcC.template out<0>()->connect(funcD.template in<2>());
    funcB.template out<1>()->connect(finalizer.template in<0>());
    funcC.template out<1>()->connect(finalizer.template in<0>());

    funcD.template out<0>()->connect(funcA.template in<0>());
    funcD.template out<1>()->connect(funcB.template in<0>());
    funcD.template out<2>()->connect(funcC.template in<0>());
    funcD.template out<3>()->connect(funcD.template in<0>());

    if (!make_graph_executable(&initiator)) throw "should be connected";
    fence();
  }

  void print() {}  //{Print()(&producer);}
  std::string dot() { return Dot()(&initiator); }
  void start() {
    if (world.rank() == 0) initiator.invoke(Integer(blocking_factor));
    ttg_execute(world);
  }
  void fence() { ttg_fence(world); }
};

/* How to call? ./ge
                                        <PROBLEM_SIZE? 1024>
                                        <BLOCKING_FACTOR? 16>
                                        <KERNEL_TYPE? iterative/recursive-serial/recursive-parallel>
                                        <VERIFY_RESULTS? verify-results, do-not-verify-results>
                                        <IF KERNEL_TYPE == recursive-serial or recursive-parallel -> R? 8>
                                        <IF KERNEL_TYPE == recursive-serial or recursive-parallel -> BASE_SIZE? 32>
        E.g.,:
                ./ge 8192 16 iterative verify-results
                ./ge 8192 16 recursive-serial verify-results 8 32
                ./ge 8192 32 recursive-serial do-not-verify-results 2 32
*/

// Parses the input arguments received from the command line
void parse_arguments(int argc, char** argv, int& problem_size, int& blocking_factor, string& kernel_type,
                     int& recursive_fan_out, int& base_size, bool& verify_results);
// Initializes the adjacency matrices
void init_square_matrix(int problem_size, int blocking_factor, bool verify_results, double* adjacency_matrix_serial,
                        Matrix<double>* m);
// Checking the equality of two double values
bool equals(double v1, double v2);
// Checking the equality of the two matrix after computing ge on them
bool equals(Matrix<double>* matrix1, double* matrix2, int problem_size, int blocking_factor);
// Iterative O(n^3) loop-based ge
void ge_iterative(double* adjacency_matrix_serial, int problem_size);

int main(int argc, char** argv) {
  initialize(argc, argv);
  ttg::TTBase::set_trace_all(false);

  auto world = ttg::default_execution_context();

  // world.taskq.add(world.rank(), hi);
  ttg_fence(world);

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
  if (argc != 5) {
    std::cout << "Usage: ./ge-df-<runtime - mad/parsec> <problem size> <blocking factor> <kernel type - "
                 "iterative/recursive-serial/recursive-parallel> verify-results/do-not-verify-results\n";
    problem_size = 2048;
    blocking_factor = 32;
    kernel_type = "iterative";
    verify_results = false;
    std::cout << "Running with problem size: " << problem_size << ", blocking factor: " << blocking_factor
              << ", kernel type: " << kernel_type << ", verify results: " << verify_results << std::endl;
  } else {
    parse_arguments(argc, argv, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size,
                    verify_results);
  }

  double* adjacency_matrix_serial;  // Using for the verification (if needed)
  // double* adjacency_matrix_ttg;     // Using for running the blocked implementation of GE algorithm on ttg runtime
  if (verify_results) {
    adjacency_matrix_serial = (double*)malloc(sizeof(double) * problem_size * problem_size);
  }
  // adjacency_matrix_ttg = (double*)malloc(sizeof(double) * problem_size * problem_size);
  int block_size = problem_size / blocking_factor;
  int n_brows = (problem_size / block_size) + (problem_size % block_size > 0);
  int n_bcols = n_brows;

  Matrix<double>* m = new Matrix<double>(n_brows, n_bcols, block_size, block_size);
  Matrix<double>* r = new Matrix<double>(n_brows, n_bcols, block_size, block_size);

  init_square_matrix(problem_size, blocking_factor, verify_results, adjacency_matrix_serial, m);
  // m->print();
  // Calling the iterative ge
  if (verify_results) {
    // std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    ge_iterative(adjacency_matrix_serial, problem_size);
    // std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    // cout << problem_size << " " << blocking_factor << " " << duration / 1000000.0 << endl;
    // cout << "iterative ge took: " << duration / 1000000.0 << " seconds" << endl;
  }

  // Running the ttg version
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  // Calling the blocked implementation of GE algorithm on ttg runtime
  GaussianElimination<double> ge(m, r, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size,
                                 adjacency_matrix_serial, verify_results);
  // std::cout << ge.dot() << std::endl;
  ge.start();
  ge.fence();
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (world.rank() == 0) cout << problem_size << " " << blocking_factor << " " << duration / 1000000.0 << endl;
  /*if (verify_results && world.rank() == 0) {
    if (equals(r, adjacency_matrix_serial, problem_size, blocking_factor)) {
      cout << "Serial and TTG implementation matches!" << endl;
    } else {
      cout << "Serial and TTG implementation DOESN\"T matches!" << endl;
    }
  }*/
  // deallocating the allocated memories
  // free(adjacency_matrix_ttg);
  delete m;
  delete r;

  if (verify_results) {
    free(adjacency_matrix_serial);
  }

  ttg_finalize();

  return 0;
}

void ge_iterative(double* adjacency_matrix_serial, int problem_size) {
  for (int k = 0; k < problem_size - 1; ++k) {
    int k_row = problem_size * k;
    for (int i = 0; i < problem_size; ++i) {
      int i_row = problem_size * i;
      for (int j = 0; j < problem_size; ++j) {
        if (i > k && j >= k) {
          adjacency_matrix_serial[i_row + j] -=
              (adjacency_matrix_serial[i_row + k] * adjacency_matrix_serial[k_row + j]) /
              adjacency_matrix_serial[k_row + k];
        }
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
  // srand(123);  // srand(time(nullptr));
  int block_size = problem_size / blocking_factor;
  for (int i = 0; i < problem_size; ++i) {
    int row = i * problem_size;
    int blockX = i / block_size;
    int x = i % block_size;
    for (int j = 0; j < problem_size; ++j) {
      int blockY = j / block_size;
      int y = j % block_size;
      double value = i * blocking_factor + j;  // rand() % 100 + 1;
      ((*m)(blockX, blockY))(x, y, value);
      if (verify_results) {
        adjacency_matrix_serial[row + j] = value;
      }
    }
  }
}

void parse_arguments(int argc, char** argv, int& problem_size, int& blocking_factor, string& kernel_type,
                     int& recursive_fan_out, int& base_size, bool& verify_results) {
  problem_size = atoi(argv[1]);     // e.g., 1024
  blocking_factor = atoi(argv[2]);  // e.g., 16
  kernel_type = argv[3];            // e.g., iterative/recursive-serial/recursive-parallel
  string verify_res(argv[4]);
  verify_results = (verify_res == "verify-results");

  // cout << "Problem_size: " << problem_size << ", blocking_factor: " << blocking_factor
  //     << ", kernel_type: " << kernel_type << ", verify_results: " << boolalpha << verify_results;
  if (argc > 5) {
    recursive_fan_out = atoi(argv[5]);
    base_size = atoi(argv[6]);
    cout << ", recursive_fan_out: " << recursive_fan_out << ", base_size: " << base_size << endl;
  } else {
    cout << endl;
  }
}
