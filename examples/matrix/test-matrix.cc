#include <array>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "ttg.h"

#include <Eigen/SparseCore>
#if __has_include(<btas/features.h>)
#include <btas/features.h>
#ifdef BTAS_IS_USABLE
#include <btas/btas.h>
#include <btas/optimize/contract.h>
#else
#warning "found btas/features.h but Boost.Iterators is missing, hence BTAS is unusable ... add -I/path/to/boost"
#endif
#endif

using namespace ttg;

#include "ttg/util/future.h"

#if defined(BLOCK_SPARSE_GEMM)
using blk_t = btas::Tensor<double>;
#else
using blk_t = double;
#endif
template <typename T = blk_t>
using SpMatrix = Eigen::SparseMatrix<T>;

template <typename _T, class _Range, class _Store>
std::tuple<_T, _T> norms(const btas::Tensor<_T, _Range, _Store> &t) {
  _T norm_2_square = 0.0;
  _T norm_inf = 0.0;
  for (auto k : t) {
    norm_2_square += k * k;
    norm_inf = std::max(norm_inf, std::abs(k));
  }
  return std::make_tuple(norm_2_square, norm_inf);
}

std::tuple<double, double> norms(double t) { return std::make_tuple(t * t, std::abs(t)); }

template <typename Blk = blk_t>
std::tuple<double, double> norms(const SpMatrix<Blk> &A) {
  double norm_2_square = 0.0;
  double norm_inf = 0.0;
  for (int i = 0; i < A.outerSize(); ++i) {
    for (typename SpMatrix<Blk>::InnerIterator it(A, i); it; ++it) {
      //  cout << 1+it.row() << "\t"; // row index
      //  cout << 1+it.col() << "\t"; // col index (here it is equal to k)
      //  cout << it.value() << endl;
      auto elem = it.value();
      double elem_norm_2_square, elem_norm_inf;
      std::tie(elem_norm_2_square, elem_norm_inf) = norms(elem);
      norm_2_square += elem_norm_2_square;
      norm_inf = std::max(norm_inf, elem_norm_inf);
    }
  }
  return std::make_tuple(norm_2_square, norm_inf);
}

template <typename _Scalar, int _Options, typename _StorageIndex>
struct colmajor_layout;
template <typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::ColMajor, _StorageIndex> : public std::true_type {};
template <typename _Scalar, typename _StorageIndex>
struct colmajor_layout<_Scalar, Eigen::RowMajor, _StorageIndex> : public std::false_type {};

template <std::size_t Rank>
struct Key : public std::array<long, Rank> {
  static constexpr const long max_index_width = 21;
  static constexpr const long max_index = 1 << 21;
  static constexpr const long max_index_square = max_index * max_index;
  Key() = default;
  template <typename Integer>
  Key(std::initializer_list<Integer> ilist) {
    std::copy(ilist.begin(), ilist.end(), this->begin());
    assert(valid());
  }
  Key(std::size_t hash) {
    static_assert(Rank == 2 || Rank == 3, "Key<Rank>::Key(hash) only implemented for Rank={2,3}");
    if (Rank == 2) {
      (*this)[0] = hash / max_index;
      (*this)[1] = hash % max_index;
    } else if (Rank == 3) {
      (*this)[0] = hash / max_index_square;
      (*this)[1] = (hash % max_index_square) / max_index;
      (*this)[2] = hash % max_index;
    }
  }
  std::size_t hash() const {
    static_assert(Rank == 2 || Rank == 3, "Key<Rank>::hash only implemented for Rank={2,3}");
    return Rank == 2 ? (*this)[0] * max_index + (*this)[1]
                     : ((*this)[0] * max_index + (*this)[1]) * max_index + (*this)[2];
  }

 private:
  bool valid() {
    bool result = true;
    for (auto &idx : *this) {
      result = result && (idx < max_index);
    }
    return result;
  }
};

template <std::size_t Rank>
std::ostream &operator<<(std::ostream &os, const Key<Rank> &key) {
  os << "{";
  for (size_t i = 0; i != Rank; ++i) os << key[i] << (i + 1 != Rank ? "," : "");
  os << "}";
  return os;
}

#include "../matrix/ttg_matrix.h"

int main(int argc, char **argv) {
  ttg_initialize(argc, argv, 4);

  //  using mpqc::Debugger;
  //  auto debugger = std::make_shared<Debugger>();
  //  Debugger::set_default_debugger(debugger);
  //  debugger->set_exec(argv[0]);
  //  debugger->set_prefix(ttg_default_execution_context().rank());
  //  debugger->set_cmd("lldb_xterm");
  //
  //  initialize_watchpoints();

  {
    // ttg::trace_on();
    // OpBase::set_trace_all(true);

    const int n = 2;
    const int m = 3;
    const int k = 4;
    SpMatrix<> A(n, k), B(k, m), C(n, m);

    // rank 0 only: initialize inputs (these will become shapes when switch to blocks)
    if (ttg_default_execution_context().rank() == 0) {
      using triplet_t = Eigen::Triplet<blk_t>;
      std::vector<triplet_t> A_elements;
#if defined(BLOCK_SPARSE_GEMM) && defined(BTAS_IS_USABLE)
      auto A_blksize = {128, 256};
      A_elements.emplace_back(0, 1, blk_t(btas::Range(A_blksize), 12.3));
      A_elements.emplace_back(0, 2, blk_t(btas::Range(A_blksize), 10.7));
      A_elements.emplace_back(0, 3, blk_t(btas::Range(A_blksize), -2.3));
      A_elements.emplace_back(1, 0, blk_t(btas::Range(A_blksize), -0.3));
      A_elements.emplace_back(1, 2, blk_t(btas::Range(A_blksize), 1.2));
#else
      A_elements.emplace_back(0, 1, 12.3);
      A_elements.emplace_back(0, 2, 10.7);
      A_elements.emplace_back(0, 3, -2.3);
      A_elements.emplace_back(1, 0, -0.3);
      A_elements.emplace_back(1, 2, 1.2);
#endif
      A.setFromTriplets(A_elements.begin(), A_elements.end());

      std::vector<triplet_t> B_elements;
#if defined(BLOCK_SPARSE_GEMM) && defined(BTAS_IS_USABLE)
      auto B_blksize = {256, 196};
      B_elements.emplace_back(0, 0, blk_t(btas::Range(B_blksize), 12.3));
      B_elements.emplace_back(1, 0, blk_t(btas::Range(B_blksize), 10.7));
      B_elements.emplace_back(3, 0, blk_t(btas::Range(B_blksize), -2.3));
      B_elements.emplace_back(1, 1, blk_t(btas::Range(B_blksize), -0.3));
      B_elements.emplace_back(1, 2, blk_t(btas::Range(B_blksize), 1.2));
      B_elements.emplace_back(2, 2, blk_t(btas::Range(B_blksize), 7.2));
      B_elements.emplace_back(3, 2, blk_t(btas::Range(B_blksize), 0.2));
#else
      B_elements.emplace_back(0, 0, 12.3);
      B_elements.emplace_back(1, 0, 10.7);
      B_elements.emplace_back(3, 0, -2.3);
      B_elements.emplace_back(1, 1, -0.3);
      B_elements.emplace_back(1, 2, 1.2);
      B_elements.emplace_back(2, 2, 7.2);
      B_elements.emplace_back(3, 2, 0.2);
#endif
      B.setFromTriplets(B_elements.begin(), B_elements.end());
    }

    ///////////////////////////////////////////////////////////////////////////
    // copy matrix using ttg::Matrix
    Matrix<blk_t> aflow;
    aflow << A;
    SpMatrix<> Acopy(A.rows(), A.cols());  // resizing will be automatic in the future when shape computation is
                                           // complete .. see Matrix::operator>>
    auto copy_status = aflow >> Acopy;
    assert(!has_value(copy_status));
    aflow.pushall();
    Control control2(ttg_ctl_edge(ttg_default_execution_context()));
    {
      std::cout << "matrix copy using ttg::Matrix" << std::endl;
      if (ttg_default_execution_context().rank() == 0) std::cout << Dot{}(&control2) << std::endl;

      // ready to run!
      auto connected = make_graph_executable(&control2);
      assert(connected);
      TTGUNUSED(connected);

      // ready, go! need only 1 kick, so must be done by 1 thread only
      if (ttg_default_execution_context().rank() == 0) control2.start();
    }
    //////////////////////////////////////////////////////////////////////////

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());

    // validate Acopy=A against the reference output
    assert(has_value(copy_status));
    if (ttg_default_execution_context().rank() == 0) {
      double norm_2_square, norm_inf;
      std::tie(norm_2_square, norm_inf) = norms<blk_t>(Acopy - A);
      std::cout << "||Acopy - A||_2      = " << std::sqrt(norm_2_square) << std::endl;
      std::cout << "||Acopy - A||_\\infty = " << norm_inf << std::endl;
      if (::ttg::tracing()) {
        std::cout << "Acopy (" << static_cast<void *>(&Acopy) << "):\n" << Acopy << std::endl;
        std::cout << "A (" << static_cast<void *>(&A) << "):\n" << A << std::endl;
      }
      if (norm_inf != 0) {
        ttg_abort();
      }
    }
  }

  ttg_finalize();

  return 0;
}
