#define TTG_USE_PARSEC 1

#include <ttg.h>
//#include <madness.h>
#include "../blockmatrix.h"

#include <lapack.hh>

#include "core_dplgsy.h"

// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>

struct Key {
  // ((I, J), K) where (I, J) is the tile coordiante and K is the iteration number
  int I = 0, J = 0, K = 0;
  madness::hashT hash_val;

  Key() { rehash(); }
  Key(int I, int J, int K) : I(I), J(J), K(K) { rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    hash_val = (static_cast<madness::hashT>(I) << 48)
             ^ (static_cast<madness::hashT>(J) << 32)
             ^ (K << 16);
  }

  // Equality test
  bool operator==(const Key& b) const { return I == b.I && J == b.J && K == b.K; }

  // Inequality test
  bool operator!=(const Key& b) const { return !((*this) == b); }

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
  s << "Key(" << key.I << "," << key.J << "," << key.K << ")";
  return s;
}

void plgsy(Matrix<double>* A)
{
  auto bump = A->rows();
  auto seed = 42;
  for (int i = 0; i < A->rows(); ++i) {
    for (int j = 0; j < A->cols(); ++j) {
      auto tile = (*A)(i, j);
      CORE_dplgsy(bump, tile.rows(), tile.cols(), tile.get(), tile.rows(),
                  A->rows(), i*tile.rows(), j*tile.cols(), seed);
    }
  }
}

template <typename T>
auto make_potrf(Matrix<T>* A,
                ttg::Edge<Key, BlockMatrix<T>>& input,
                ttg::Edge<Key, BlockMatrix<T>>& output_trsm,
                ttg::Edge<Key, BlockMatrix<T>>& output_result)
{
  auto f = [=](const Key& key,
               BlockMatrix<T>&& tile_kk,
               std::tuple<ttg::Out<Key, BlockMatrix<T>>,
                          ttg::Out<Key, BlockMatrix<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I == J);
    assert(I == K);

    lapack::potrf(lapack::Uplo::Lower, tile_kk.rows(), tile_kk.get(), tile_kk.rows());

    //std::cout << "POTRF(" << key << ")" << std::endl;

    /* tile is done */
    ttg::send<0>(key, tile_kk, out);

    /* send the tile to outputs */
    for (int m = I+1; m < A->rows(); ++m) {
      /* send tile to trsm */
      ttg::send<1>(Key(m, J, K), tile_kk, out);
    }
  };
  return ttg::wrap(f, ttg::edges(input), ttg::edges(output_result, output_trsm), "POTRF", {"tile_kk"}, {"output_result", "output_trsm"});
}

template <typename T>
auto make_trsm(Matrix<T>* A,
               ttg::Edge<Key, BlockMatrix<T>>& input_kk,
               ttg::Edge<Key, BlockMatrix<T>>& input_mk,
               ttg::Edge<Key, BlockMatrix<T>>& output_diag,
               ttg::Edge<Key, BlockMatrix<T>>& output_row,
               ttg::Edge<Key, BlockMatrix<T>>& output_col,
               ttg::Edge<Key, BlockMatrix<T>>& output_result)
{
  auto f = [=](const Key& key,
               const BlockMatrix<T>&  tile_kk,
                     BlockMatrix<T>&& tile_mk,
                     std::tuple<ttg::Out<Key, BlockMatrix<T>>,
                                ttg::Out<Key, BlockMatrix<T>>,
                                ttg::Out<Key, BlockMatrix<T>>,
                                ttg::Out<Key, BlockMatrix<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I > K); // we're below (k, k) in row i, column j [k+1 .. NB, k]

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_kk.rows());
    assert(tile_mk.cols() == tile_kk.cols());

    auto m = tile_mk.rows();

    blas::trsm(blas::Layout::RowMajor,
               blas::Side::Right,
               lapack::Uplo::Lower,
               blas::Op::Trans,
               blas::Diag::NonUnit,
               tile_kk.rows(), m, 1.0,
               tile_kk.get(), m,
               tile_mk.get(), m);

    //std::cout << "TRSM(" << key << ")" << std::endl;

    /* tile is done */
    ttg::send<0>(key, tile_kk, out);

    //if (I+1 < A->rows()) {
      /* send tile to syrk on diagonal */
      //std::cout << "TRSM(" << key << "): sending output to diag " << Key{I, I, K} << std::endl;
      ttg::send<1>(Key(I, I, K), tile_mk, out);

      /* send the tile to all gemms across in row i */
      for (int n = J+1; n < I; ++n) {
        //std::cout << "TRSM(" << key << "): sending output to row " << Key{I, n, K} << std::endl;
        ttg::send<2>(Key(I, n, K), tile_mk, out);
      }

      /* send the tile to all gemms down in column i */
      for (int m = I+1; m < A->rows(); ++m) {
        //std::cout << "TRSM(" << key << "): sending output to col " << Key{m, I, K} << std::endl;
        ttg::send<3>(Key(m, I, K), tile_mk, out);
      }
    //}
  };
  return ttg::wrap(f, ttg::edges(input_kk, input_mk), ttg::edges(output_result, output_diag, output_row, output_col),
                   "TRSM", {"tile_kk", "tile_mk"}, {"output_result", "output_diag", "output_row", "output_col"});
}


template <typename T>
auto make_syrk(Matrix<T>* A,
               ttg::Edge<Key, BlockMatrix<T>>& input_mk,
               ttg::Edge<Key, BlockMatrix<T>>& input_mm,
               ttg::Edge<Key, BlockMatrix<T>>& output_potrf,
               ttg::Edge<Key, BlockMatrix<T>>& output_syrk)
{
  auto f = [=](const Key& key,
               const BlockMatrix<T>&  tile_mk,
                     BlockMatrix<T>&& tile_mm,
                     std::tuple<ttg::Out<Key, BlockMatrix<T>>,
                                ttg::Out<Key, BlockMatrix<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I == J);
    assert(I > K);

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_mm.rows());
    assert(tile_mk.cols() == tile_mm.cols());

    auto m = tile_mk.rows();

    blas::syrk(blas::Layout::RowMajor,
               lapack::Uplo::Lower,
               blas::Op::NoTrans,
               tile_mk.rows(), m, -1.0,
               tile_mk.get(), m, 1.0,
               tile_mm.get(), m);

    //std::cout << "SYRK(" << key << ")" << std::endl;

    if (I == K+1) {
      /* send the tile to potrf */
      //std::cout << "SYRK(" << key << "): sending output to POTRF " << Key{I, I, K+1} << std::endl;
      ttg::send<0>(Key(I, I, K+1), tile_mm, out);
    } else {
      /* send output to next syrk */
      //std::cout << "SYRK(" << key << "): sending output to SYRK " << Key{I, I, K+1} << std::endl;
      ttg::send<1>(Key(I, I, K+1), tile_mm, out);
    }

  };
  return ttg::wrap(f,
                   ttg::edges(input_mk, input_mm),
                   ttg::edges(output_potrf, output_syrk), "SYRK",
                   {"tile_mk", "tile_mm"}, {"output_potrf", "output_syrk"});
}


template <typename T>
auto make_gemm(Matrix<T>* A,
               ttg::Edge<Key, BlockMatrix<T>>& input_nk,
               ttg::Edge<Key, BlockMatrix<T>>& input_mk,
               ttg::Edge<Key, BlockMatrix<T>>& input_nm,
               ttg::Edge<Key, BlockMatrix<T>>& output_trsm,
               ttg::Edge<Key, BlockMatrix<T>>& output_gemm)
{
  auto f = [](const Key& key,
              const BlockMatrix<T>& tile_nk,
              const BlockMatrix<T>& tile_mk,
                    BlockMatrix<T>&& tile_nm,
                    std::tuple<ttg::Out<Key, BlockMatrix<T>>,
                               ttg::Out<Key, BlockMatrix<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I != J && I > K && J > K);

    /* No support for different tile sizes yet */
    assert(tile_nk.rows() == tile_mk.rows() && tile_nk.rows() == tile_nm.rows());
    assert(tile_nk.cols() == tile_mk.cols() && tile_nk.cols() == tile_nm.cols());

    auto m = tile_nk.rows();

    blas::gemm(blas::Layout::RowMajor,
               blas::Op::NoTrans,
               blas::Op::Trans,
               m, m, m, -1.0,
               tile_nk.get(), m,
               tile_mk.get(), m, 1.0,
               tile_nm.get(), m);

    //std::cout << "GEMM(" << key << ")" << std::endl;

    /* send the tile to output */
    if (J == K+1) {
      /* send the tile to trsm */
      //std::cout << "GEMM(" << key << "): sending output to TRSM " << Key{I, J, K+1} << std::endl;
      ttg::send<0>(Key(I, J, K+1), tile_nm, out);
    } else {
      /* send the tile to the next gemm */
      //std::cout << "GEMM(" << key << "): sending output to GEMM " << Key{I, J, K+1} << std::endl;
      ttg::send<1>(Key(I, J, K+1), tile_nm, out);
    }
  };
  return ttg::wrap(f,
                   ttg::edges(input_nk, input_mk, input_nm),
                   ttg::edges(output_trsm, output_gemm), "GEMM",
                   {"input_nk", "input_mk", "input_nm"},
                   {"output_trsm", "outout_gemm"});
}

template<typename T>
auto initiator(Matrix<T>* A,
               ttg::Edge<Key, BlockMatrix<T>>& syrk_potrf,
               ttg::Edge<Key, BlockMatrix<T>>& gemm_trsm,
               ttg::Edge<Key, BlockMatrix<T>>& syrk_syrk,
               ttg::Edge<Key, BlockMatrix<T>>& gemm_gemm)
{
  auto f = [=](const Key&,
               std::tuple<ttg::Out<Key, BlockMatrix<T>>,
                          ttg::Out<Key, BlockMatrix<T>>,
                          ttg::Out<Key, BlockMatrix<T>>,
                          ttg::Out<Key, BlockMatrix<T>>>& out){
    /* kick off first POTRF */
    ttg::send<0>(Key{0, 0, 0}, (*A)(0, 0), out);
    for (int i = 1; i < A->rows(); i++) {
      /* send gemm input to TRSM */
      ttg::send<1>(Key{i, 0, 0}, (*A)(i, 0), out);
      /* send syrk to SYRK */
      ttg::send<2>(Key{i, i, 0}, (*A)(i, i), out);
      for (int j = 1; j < i; j++) {
        /* send gemm to GEMM */
        ttg::send<3>(Key{i, j, 0}, (*A)(i, j), out);
      }
    }
  };

  return ttg::wrap<Key>(f, ttg::edges(), ttg::edges(syrk_potrf, gemm_trsm, syrk_syrk, gemm_gemm), "INITIATOR");
}

template <typename T>
auto make_result(Matrix<T> *A, const ttg::Edge<Key, BlockMatrix<T>>& result) {
  auto f = [](const Key& key, BlockMatrix<T>&& tile, std::tuple<>& out) {
    /* TODO: is this node actually needed? */
    //std::cout << "FINAL " << key << std::endl;
  };

  return ttg::wrap(f, ttg::edges(result), ttg::edges(), "Final Output", {"result"}, {});
}


int main(int argc, char **argv)
{

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  int N = 1024;
  int M = N;
  int NB = 128;
  ttg::ttg_initialize(argc, argv, 2);

  if (argc > 1) {
    N = M = atoi(argv[1]);
  }

  if (argc > 2) {
    NB = atoi(argv[2]);
  }

  int n_rows = (N / NB) + (N % NB > 0);
  int n_cols = (M / NB) + (M % NB > 0);

  Matrix<double>* A = new Matrix<double>(n_rows, n_cols, NB, NB);

  ttg::Edge<Key, BlockMatrix<double>> potrf_trsm("potrf_trsm"),
                                      trsm_syrk("trsm_syrk"),
                                      syrk_potrf("syrk_potrf"),
                                      syrk_syrk("syrk_syrk"),
                                      gemm_gemm("gemm_gemm"),
                                      gemm_trsm("gemm_trsm"),
                                      trsm_gemm_row("trsm_gemm_row"),
                                      trsm_gemm_col("trsm_gemm_col"),
                                      result("result");

  /* initialize the matrix */
  plgsy(A);

  auto op_init  = initiator(A, syrk_potrf, gemm_trsm, syrk_syrk, gemm_gemm);
  auto op_potrf = make_potrf(A, syrk_potrf, potrf_trsm, result);
  auto op_trsm  = make_trsm(A,
                            potrf_trsm, gemm_trsm,
                            trsm_syrk, trsm_gemm_row, trsm_gemm_col, result);
  auto op_syrk  = make_syrk(A, trsm_syrk, syrk_syrk, syrk_potrf, syrk_syrk);
  auto op_gemm  = make_gemm(A,
                            trsm_gemm_row, trsm_gemm_col, gemm_gemm,
                            gemm_trsm, gemm_gemm);
  auto op_result = make_result(A, result);

  auto connected = make_graph_executable(op_init.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected: " << connected << std::endl;

  auto world = ttg::ttg_default_execution_context();

  if (world.rank() == 0) {
    std::cout << "==== begin dot ====\n";
    std::cout << ttg::Dot()(op_init.get()) << std::endl;
    std::cout << "==== end dot ====\n";

    beg = std::chrono::high_resolution_clock::now();
    op_init->invoke(Key{0, 0, 0});
  }

  ttg::ttg_execute(world);
  ttg::ttg_fence(world);
  if (world.rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;
  }

  delete A;
  ttg::ttg_finalize();
  return 0;
}
