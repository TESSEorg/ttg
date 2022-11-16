#pragma once

#include <parsec.h>
#include <parsec/data_dist/matrix/matrix.h>
#include <parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h>
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>
#include <parsec/data_internal.h>

#include <iomanip>
#include "../matrixtile.h"

#include <ttg/util/multiindex.h>

using Key1 = ttg::MultiIndex<1>;
using Key2 = ttg::MultiIndex<2>;
using Key3 = ttg::MultiIndex<3>;

/* C++ type to PaRSEC's matrix_type mapping */
template<typename T>
struct type2matrixtype
{ };

template<>
struct type2matrixtype<float>
{
    static constexpr const parsec_matrix_type_t value = parsec_matrix_type_t::PARSEC_MATRIX_FLOAT;
};

template<>
struct type2matrixtype<double>
{
    static constexpr const parsec_matrix_type_t value = parsec_matrix_type_t::PARSEC_MATRIX_DOUBLE;
};

template<typename PaRSECMatrixT, typename ValueT>
class PaRSECMatrixWrapper {
  PaRSECMatrixT* pm;

public:
 using element_type = ValueT;

  PaRSECMatrixWrapper(PaRSECMatrixT* dc) : pm(dc)
  {
    //std::cout << "PaRSECMatrixWrapper of matrix with " << rows() << "x" << cols() << " tiles " << std::endl;
    //for (int i = 0; i < rows(); ++i) {
    //  for (int j = 0; j <= i; ++j) {
    //    std::cout << "Tile [" << i << ", " << j << "] is at rank " << rank_of(i, j) << std::endl;
    //  }
    //}
  }

  MatrixTile<ValueT> operator()(int row, int col) const {
    ValueT* ptr = static_cast<ValueT*>(parsec_data_copy_get_ptr(
                      parsec_data_get_copy(pm->super.super.data_of(&pm->super.super, row, col), 0)));
    auto mb = (row < pm->super.mt - 1) ? pm->super.mb : pm->super.m - row * pm->super.mb;
    auto nb = (col < pm->super.nt - 1) ? pm->super.nb : pm->super.n - col * pm->super.nb;
    return MatrixTile<ValueT>{mb, nb, ptr, pm->super.mb};
  }

  /** Number of tiled rows **/
  int rows(void) const {
    return pm->super.mt;
  }

  /** Number of rows in tile */
  int rows_in_tile(void) const {
    return pm->super.mb;
  }

  /** Number of rows in the matrix */
  int rows_in_matrix(void) const {
    return pm->super.m;
  }

  /** Number of tiled columns **/
  int cols(void) const {
    return pm->super.nt;
  }

  /** Number of columns in tile */
  int cols_in_tile(void) const {
    return pm->super.nb;
  }

  /** Number of columns in the matrix */
  int cols_in_matrix(void) const {
    return pm->super.n;
  }

  /* The rank storing the tile at {row, col} */
  int rank_of(int row, int col) const {
    return pm->super.super.rank_of(&pm->super.super, row, col);
  }

  bool is_local(int row, int col) const {
    return ttg::default_execution_context().rank() == rank_of(row, col);
  }

  bool in_matrix(int row, int col) const {
    return (pm->uplo == PARSEC_MATRIX_LOWER && col <= row) ||
           (pm->uplo == PARSEC_MATRIX_UPPER && col >= row);
  }

  PaRSECMatrixT* parsec() {
    return pm;
  }

  const PaRSECMatrixT* parsec() const {
    return pm;
  }

  /* Copy entire input matrix (which is local) into a single LAPACK format matrix */
  ValueT *getLAPACKMatrix() const {
    ValueT *ret = new ValueT[rows_in_matrix()*cols_in_matrix()];
    for(auto i = 0; i < rows_in_matrix(); i++) {
      for(auto j = 0; j < cols_in_matrix(); j++) {
        if( in_matrix(i/rows_in_tile(), j/cols_in_tile()) ) {
          auto m = i/rows_in_tile();
          auto n = j/cols_in_tile();
          auto tile = this->operator()(m, n);
          auto it = i%rows_in_tile();
          auto jt = j%cols_in_tile();
          ret[i + j*rows_in_matrix()] = tile.data()[it + jt*tile.lda()];
        } else {
          ret[i + j*rows_in_matrix()] = 0.0;
        }
      }
    }
    return ret;
  }
};

template<typename ValueT>
using MatrixT = PaRSECMatrixWrapper<sym_two_dim_block_cyclic_t, ValueT>;

static auto make_load_tt(MatrixT<double> &A, ttg::Edge<Key2, MatrixTile<double>> &toop, bool defer_write)
{
  auto load_tt = ttg::make_tt<void>([&](std::tuple<ttg::Out<Key2, MatrixTile<double>>>& out) {
      for(int i = 0; i < A.rows(); i++) {
        for(int j = 0; j < A.cols() && A.in_matrix(i, j); j++) {
          if(A.is_local(i, j)) {
            if(ttg::tracing()) ttg::print("load(", Key2{i, j}, ")");
            ttg::send<0>(Key2{i, j}, std::move(A(i, j)), out);
          }
        }
      }
    }, ttg::edges(), ttg::edges(toop), "Load Matrix", {}, {"To Op"});
  load_tt->set_keymap([]() {return ttg::ttg_default_execution_context().rank();});
  load_tt->set_defer_writer(defer_write);

  return std::move(load_tt);
}

static void print_LAPACK_matrix( const double *A, int N, const char *label)
{
  std::cout << label << std::endl;
  for(int i = 0; i < N; i++) {
    std::cout << " ";
    for(int j = 0; j < N; j++) {
      std::cout << std::setw(11) << std::setprecision(5) << A[i+j*N] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
