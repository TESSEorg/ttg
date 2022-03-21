#pragma once

#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/data_dist/matrix/matrix.h>
#include <parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h>
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>

#include "../matrixtile.h"

struct Key2 {
  // ((I, J), K) where (I, J) is the tile coordiante and K is the iteration number
  int I = 0, J = 0;
  madness::hashT hash_val;

  Key2() { rehash(); }
  Key2(int I, int J) : I(I), J(J){ rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    hash_val = (static_cast<madness::hashT>(I) << 32)
             ^ (static_cast<madness::hashT>(J));
  }

  // Equality test
  bool operator==(const Key2& b) const { return I == b.I && J == b.J; }

  // Inequality test
  bool operator!=(const Key2& b) const { return !((*this) == b); }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar& madness::archive::wrap((unsigned char*)this, sizeof(*this));
  }
};


namespace std {
  // specialize std::hash for Key

  template <>
  struct hash<Key2> {
    std::size_t operator()(const Key2& s) const noexcept { return s.hash(); }
  };

  std::ostream& operator<<(std::ostream& s, const Key2& key) {
    s << "Key(" << key.I << "," << key.J << ")";
    return s;
  }

}  // namespace std

/* C++ type to PaRSEC's matrix_type mapping */
template<typename T>
struct type2matrixtype
{ };

template<>
struct type2matrixtype<float>
{
    static constexpr const matrix_type value = matrix_type::matrix_RealFloat;
};

template<>
struct type2matrixtype<double>
{
    static constexpr const matrix_type value = matrix_type::matrix_RealDouble;
};

template<typename PaRSECMatrixT, typename ValueT>
class PaRSECMatrixWrapper {
  PaRSECMatrixT* pm;

public:
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
    return MatrixTile<ValueT>{pm->super.mb, pm->super.nb, ptr};
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

  PaRSECMatrixT* parsec() {
    return pm;
  }

  const PaRSECMatrixT* parsec() const {
    return pm;
  }

};

template<typename ValueT>
using MatrixT = PaRSECMatrixWrapper<sym_two_dim_block_cyclic_t, ValueT>;
