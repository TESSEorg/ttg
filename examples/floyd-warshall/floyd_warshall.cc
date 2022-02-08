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

// #include <omp.h> //

#include "FW-APSP/FloydIterativeKernel.h"        // contains the iterative kernel
#include "FW-APSP/FloydRecursiveSerialKernel.h"  // contains the recursive but serial kernels
// #include "FloydRecursiveParallelKernel.h" // contains the recursive and parallel kernels

#include "ttg.h"
using namespace ttg;

#include "ttg/serialization.h"
#include "ttg/serialization/std/pair.h"

struct Key {
  // ((I, J), K) where (I, J) is the tile coordinate and K is the iteration number
  std::pair<std::pair<int, int>, int> execution_info;

  bool operator==(const Key& b) const {
    if (this == &b) return true;
    return execution_info.first.first == b.execution_info.first.first &&
           execution_info.first.second == b.execution_info.first.second &&
           execution_info.second == b.execution_info.second;
  }

  bool operator!=(const Key& b) const { return !((*this) == b); }

  std::size_t hash_val;

  Key() : execution_info(std::make_pair(std::make_pair(0, 0), 0)) { rehash(); }
  Key(const std::pair<std::pair<int, int>, int>& e) : execution_info(e) { rehash(); }
  Key(int e_f_f, int e_f_s, int e_s) : execution_info(std::make_pair(std::make_pair(e_f_f, e_f_s), e_s)) { rehash(); }

  std::size_t hash() const { return hash_val; }

  void rehash() {
    std::hash<int> int_hasher;
    hash_val = int_hasher(execution_info.first.first) * 2654435769 + int_hasher(execution_info.first.second) * 40503 +
               int_hasher(execution_info.second);
  }

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

class Initiator : public TT<int, std::tuple<Out<Key, Control>, Out<Key, Control>, Out<Key, Control>, Out<Key, Control>>,
                            Initiator> {
  using baseT = typename Initiator::ttT;

 public:
  Initiator(const std::string& name) : baseT(name, {}, {"outA", "outB", "outC", "outD"}) {}
  Initiator(const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(edges(), outedges, name, {}, {"outA", "outB", "outC", "outD"}) {}

  ~Initiator() { std::cout << "Initiator destructor\n"; }

  void op(const int& iterations, baseT::output_terminals_type& out) {
    // making x_ready for all the blocks (for function calls A, B, C, and D)
    // This triggers for the immediate execution of function A at tile [0, 0]. But
    // functions B, C, and D have other dependencies to meet before execution; They wait
    for (int i = 0; i < iterations; ++i) {
      for (int j = 0; j < iterations; ++j) {
        if (i == 0 && j == 0) {  // A function call
          ::send<0>(Key(std::make_pair(std::make_pair(i, j), 0)), Control(), out);
        } else if (i == 0) {  // B function call
          ::send<1>(Key(std::make_pair(std::make_pair(i, j), 0)), Control(), out);
        } else if (j == 0) {  // C function call
          ::send<2>(Key(std::make_pair(std::make_pair(i, j), 0)), Control(), out);
        } else {  // D function call
          ::send<3>(Key(std::make_pair(std::make_pair(i, j), 0)), Control(), out);
        }
      }
    }
  }
};

class FuncA : public TT<Key,
                        std::tuple<Out<Key, Control>, Out<Key, Control>, Out<Key, Control>, Out<Key, Control>,
                                   Out<Key, Control>, Out<Key, Control>>,
                        FuncA, ttg::typelist<Control>> {
  using baseT = typename FuncA::ttT;
  double* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncA(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready"}, {"outA", "outB", "outC", "outD", "readyB", "readyC"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncA(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready"}, {"outA", "outB", "outC", "outD", "readyB", "readyC"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<Control>& t, baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    // Executing the update
    if (kernel_type == "iterative") {
      // std::cout << "FuncA: " << K << " " << I << " " << J << std::endl;
      floyd_iterative_kernel(problem_size, blocking_factor, I, J, K, adjacency_matrix_ttg);
    } else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      floyd_recursive_serial_kernelA(adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb, k_lb,
                                     recursive_fan_out, base_size);
    } else {
      // int block_size = problem_size/blocking_factor;
      // int i_lb = I * block_size;
      // int j_lb = J * block_size;
      // int k_lb = K * block_size;
      // #pragma omp prallel
      // {
      //        #pragma omp single
      //         {
      //                 #pragma omp task
      //                 floyd_recursive_serial_kernelA(adjacency_matrix_ttg, problem_size,
      //                                                                    block_size, i_lb, j_lb, k_lb,
      //                                                                    recursive_fan_out, base_size);
      //         }
      // }
    }

    // Making u_ready/v_ready for all the B/C function calls in the CURRENT iteration
    for (int l = 0; l < blocking_factor; ++l) {
      if (l != K) {
        // B calls
        ::send<4>(Key(std::make_pair(std::make_pair(I, l), K)), Control(), out);
        // C calls
        ::send<5>(Key(std::make_pair(std::make_pair(l, J), K)), Control(), out);
      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else {  // in the next iteration, we have D function call
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      }
    }
  }
};

class FuncB
    : public TT<
          Key,
          std::tuple<Out<Key, Control>, Out<Key, Control>, Out<Key, Control>, Out<Key, Control>, Out<Key, Control>>,
          FuncB, ttg::typelist<Control, Control>> {
  using baseT = typename FuncB::ttT;
  double* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncB(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "u_ready"}, {"outA", "outB", "outC", "outD", "readyD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncB(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "u_ready"}, {"outA", "outB", "outC", "outD", "readyD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<Control, Control>& t, baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    // std::cout << "FuncB: " << K << " " << I << " " << J << std::endl;
    // Executing the update
    if (kernel_type == "iterative") {
      floyd_iterative_kernel(problem_size, blocking_factor, I, J, K, adjacency_matrix_ttg);
    } else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      floyd_recursive_serial_kernelB(adjacency_matrix_ttg, adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb,
                                     k_lb, recursive_fan_out, base_size);
    } else {
      // int block_size = problem_size/blocking_factor;
      // int i_lb = I * block_size;
      // int j_lb = J * block_size;
      // int k_lb = K * block_size;
      // #pragma omp prallel
      // {
      //         #pragma omp single
      //         {
      //                 #pragma omp task
      //                 floyd_recursive_serial_kernelB(adjacency_matrix_ttg, adjacency_matrix_ttg,
      //                                                                            problem_size, block_size, i_lb,
      //                                                                            j_lb,
      // k_lb,                                                                            recursive_fan_out, base_size);
      //         }
      // }
    }

    // Making v_ready for all the D function calls in the CURRENT iteration
    for (int i = 0; i < blocking_factor; ++i) {
      if (i != I) {
        ::send<4>(Key(std::make_pair(std::make_pair(i, J), K)), Control(), out);
      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else {  // in the next iteration, we have D function call
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      }
    }
  }
};

class FuncC
    : public TT<
          Key,
          std::tuple<Out<Key, Control>, Out<Key, Control>, Out<Key, Control>, Out<Key, Control>, Out<Key, Control>>,
          FuncC, ttg::typelist<Control, Control>> {
  using baseT = typename FuncC::ttT;
  double* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncC(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "v_ready"}, {"outA", "outB", "outC", "outD", "readyD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncC(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "v_ready"}, {"outA", "outB", "outC", "outD", "readyD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<Control, Control>& t, baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;

    // std::cout << "FuncC: " << K << " " << I << " " << J << std::endl;
    // Executing the update
    if (kernel_type == "iterative") {
      floyd_iterative_kernel(problem_size, blocking_factor, I, J, K, adjacency_matrix_ttg);
    } else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      floyd_recursive_serial_kernelC(adjacency_matrix_ttg, adjacency_matrix_ttg, problem_size, block_size, i_lb, j_lb,
                                     k_lb, recursive_fan_out, base_size);
    } else {
      // int block_size = problem_size/blocking_factor;
      // int i_lb = I * block_size;
      // int j_lb = J * block_size;
      // int k_lb = K * block_size;
      // #pragma omp prallel
      // {
      //         #pragma omp single
      //         {
      //                 #pragma omp task
      //                 floyd_recursive_serial_kernelC(adjacency_matrix_ttg, adjacency_matrix_ttg,
      //                                                                            problem_size, block_size, i_lb,
      //                                                                            j_lb,
      // k_lb,                                                                            recursive_fan_out, base_size);
      //         }
      // }
    }

    // Making u_ready for all the D function calls in the CURRENT iteration
    for (int j = 0; j < blocking_factor; ++j) {
      if (j != J) {
        ::send<4>(Key(std::make_pair(std::make_pair(I, j), K)), Control(), out);
      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else {  // in the next iteration, we have D function call
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      }
    }
  }
};

class FuncD : public TT<Key, std::tuple<Out<Key, Control>, Out<Key, Control>, Out<Key, Control>, Out<Key, Control>>,
                        FuncD, ttg::typelist<Control, Control, Control>> {
  using baseT = typename FuncD::ttT;
  double* adjacency_matrix_ttg;
  int problem_size;
  int blocking_factor;
  std::string kernel_type;
  int recursive_fan_out;
  int base_size;

 public:
  FuncD(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const std::string& name)
      : baseT(name, {"x_ready", "v_ready", "u_ready"}, {"outA", "outB", "outC", "outD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  FuncD(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
        int recursive_fan_out, int base_size, const typename baseT::input_edges_type& inedges,
        const typename baseT::output_edges_type& outedges, const std::string& name)
      : baseT(inedges, outedges, name, {"x_ready", "v_ready", "u_ready"}, {"outA", "outB", "outC", "outD"})
      , adjacency_matrix_ttg(adjacency_matrix_ttg)
      , problem_size(problem_size)
      , blocking_factor(blocking_factor)
      , kernel_type(kernel_type)
      , recursive_fan_out(recursive_fan_out)
      , base_size(base_size) {}

  void op(const Key& key, const std::tuple<Control, Control, Control>& t, baseT::output_terminals_type& out) {
    int I = key.execution_info.first.first;
    int J = key.execution_info.first.second;
    int K = key.execution_info.second;
    // std::cout << "FuncD: " << K << " " << I << " " << J << std::endl;
    // Executing the update
    if (kernel_type == "iterative") {
      floyd_iterative_kernel(problem_size, blocking_factor, I, J, K, adjacency_matrix_ttg);
    } else if (kernel_type == "recursive-serial") {
      int block_size = problem_size / blocking_factor;
      int i_lb = I * block_size;
      int j_lb = J * block_size;
      int k_lb = K * block_size;
      floyd_recursive_serial_kernelD(adjacency_matrix_ttg, adjacency_matrix_ttg, adjacency_matrix_ttg, problem_size,
                                     block_size, i_lb, j_lb, k_lb, recursive_fan_out, base_size);
    } else {
      //      int block_size = problem_size / blocking_factor;
      //      int i_lb = I * block_size;
      //      int j_lb = J * block_size;
      //      int k_lb = K * block_size;
      //#pragma omp prallel
      //      {
      //#pragma omp single
      //        {
      //#pragma omp task
      //          floyd_recursive_serial_kernelD(adjacency_matrix_ttg, adjacency_matrix_ttg, adjacency_matrix_ttg,
      //          problem_size,
      //                                         block_size, i_lb, j_lb, k_lb, recursive_fan_out, base_size);
      //        }
      //      }
    }

    // making x_ready for the computation on the SAME block in the NEXT iteration
    if (K < (blocking_factor - 1)) {   // if there is a NEXT iteration
      if (I == K + 1 && J == K + 1) {  // in the next iteration, we have A function call
        ::send<0>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (I == K + 1) {  // in the next iteration, we have B function call
        ::send<1>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else if (J == K + 1) {  // in the next iteration, we have C function call
        ::send<2>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      } else {  // in the next iteration, we have D function call
        ::send<3>(Key(std::make_pair(std::make_pair(I, J), K + 1)), Control(), out);
      }
    }
  }
};

class FloydWarshall {
  Initiator initiator;
  FuncA funcA;
  FuncB funcB;
  FuncC funcC;
  FuncD funcD;
  ttg::World world;

  // Needed for Initiating the execution in Initiator data member (see the function start())
  int blocking_factor;

 public:
  FloydWarshall(double* adjacency_matrix_ttg, int problem_size, int blocking_factor, const std::string& kernel_type,
                int recursive_fan_out, int base_size)
      : initiator("initiator")
      , funcA(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcA")
      , funcB(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcB")
      , funcC(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcC")
      , funcD(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size, "funcD")
      , world(ttg::default_execution_context())
      , blocking_factor(blocking_factor) {
    initiator.out<0>()->connect(funcA.in<0>());
    initiator.out<1>()->connect(funcB.in<0>());
    initiator.out<2>()->connect(funcC.in<0>());
    initiator.out<3>()->connect(funcD.in<0>());

    funcA.out<0>()->connect(funcA.in<0>());
    funcA.out<1>()->connect(funcB.in<0>());
    funcA.out<2>()->connect(funcC.in<0>());
    funcA.out<3>()->connect(funcD.in<0>());
    funcA.out<4>()->connect(funcB.in<1>());
    funcA.out<5>()->connect(funcC.in<1>());

    funcB.out<0>()->connect(funcA.in<0>());
    funcB.out<1>()->connect(funcB.in<0>());
    funcB.out<2>()->connect(funcC.in<0>());
    funcB.out<3>()->connect(funcD.in<0>());
    funcB.out<4>()->connect(funcD.in<1>());

    funcC.out<0>()->connect(funcA.in<0>());
    funcC.out<1>()->connect(funcB.in<0>());
    funcC.out<2>()->connect(funcC.in<0>());
    funcC.out<3>()->connect(funcD.in<0>());
    funcC.out<4>()->connect(funcD.in<2>());

    funcD.out<0>()->connect(funcA.in<0>());
    funcD.out<1>()->connect(funcB.in<0>());
    funcD.out<2>()->connect(funcC.in<0>());
    funcD.out<3>()->connect(funcD.in<0>());

    if (!make_graph_executable(&initiator)) throw "should be connected";
    ttg_fence(world);
  }

  void print() {}  //{Print()(&producer);}
  std::string dot() { return Dot()(&initiator); }
  // OLD CODE
  // void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
  // void fence() { world.gop.fence(); }
  // END OLD CODE
  // NEW CODE
  void start() {
    if (world.rank() == 0) initiator.invoke(blocking_factor);
    ttg_execute(world);
  }
  void fence() { ttg_fence(world); }
  // END NEW CODE
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
void init_square_matrix(double* adjacency_matrix_ttg, int problem_size, bool verify_results,
                        double* adjacency_matrix_serial);
// Checking the equality of two double values
bool equals(double v1, double v2);
// Checking the equality of the two matrix after computing fw-apsp on them
bool equals(double* matrix1, double* matrix2, int problem_size);
// Iterative O(n^3) loop-based fw-apsp
void floyd_iterative(double* adjacency_matrix_serial, int problem_size);

int main(int argc, char** argv) {
  initialize(argc, argv);
  fence();

  ttg::TTBase::set_trace_all(false);

  // NEW IMPLEMENTATION
  int problem_size;
  int blocking_factor;
  string kernel_type;
  // if the kernel_type is "recursive-serial" or "recursive-parallel"
  int recursive_fan_out;
  int base_size;
  bool verify_results;
  if (argc != 5) {
    std::cout << "Usage: ./fw-apsp-<runtime - mad/parsec> <problem size> <blocking factor> <kernel type - "
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

  double* adjacency_matrix_serial = nullptr;  // Using for the verification (if needed)
  double* adjacency_matrix_ttg =
      nullptr;  // Using for running the blocked implementation of FW-APSP algorithm on ttg runtime
  if (verify_results) {
    adjacency_matrix_serial = (double*)malloc(sizeof(double) * problem_size * problem_size);
  }
  adjacency_matrix_ttg = (double*)malloc(sizeof(double) * problem_size * problem_size);
  init_square_matrix(adjacency_matrix_ttg, problem_size, verify_results, adjacency_matrix_serial);

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
  FloydWarshall fw_apsp(adjacency_matrix_ttg, problem_size, blocking_factor, kernel_type, recursive_fan_out, base_size);
  fw_apsp.start();
  fw_apsp.fence();
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  cout << "blocked ttg (data-flow) fw-apsp took: " << duration / 1000000.0 << " seconds" << endl;

  if (verify_results) {
    if (equals(adjacency_matrix_ttg, adjacency_matrix_serial, problem_size)) {
      cout << "Serial and TTG implementation matches!" << endl;
    } else {
      cout << "Serial and TTG implementation DOESN\"T matches!" << endl;
    }
  }
  // deallocating the allocated memories
  free(adjacency_matrix_ttg);
  if (verify_results) {
    free(adjacency_matrix_serial);
  }

  finalize();

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

bool equals(double* matrix1, double* matrix2, int problem_size) {
  for (int i = 0; i < problem_size; ++i) {
    int row = i * problem_size;
    for (int j = 0; j < problem_size; ++j) {
      double v1 = matrix1[row + j];
      double v2 = matrix2[row + j];
      if (!equals(v1, v2)) {
        cout << "matrix1[" << i << ", " << j << "]: " << v1 << endl;
        cout << "matrix2[" << i << ", " << j << "]: " << v2 << endl;
        return false;
      }
    }
  }
  return true;
}

void init_square_matrix(double* adjacency_matrix_ttg, int problem_size, bool verify_results,
                        double* adjacency_matrix_serial) {
  srand(123);  // srand(time(nullptr));
  for (int i = 0; i < problem_size; ++i) {
    int row = i * problem_size;
    for (int j = 0; j < problem_size; ++j) {
      if (i != j) {
        double value = rand() % 100 + 1;
        adjacency_matrix_ttg[row + j] = value;
        if (verify_results) {
          adjacency_matrix_serial[row + j] = value;
        }
      } else {
        adjacency_matrix_ttg[row + j] = 0.0;
        if (verify_results) {
          adjacency_matrix_serial[row + j] = 0.0;
        }
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
