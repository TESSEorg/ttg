//#define TTG_USE_PARSEC 1

#ifdef TTG_USE_PARSEC
// tell TTG/PARSEC that we know what we are doing (TM)
#define TTG_USE_USER_TERMDET 1
#endif // TTG_USE_PARSEC

#define USE_PARSEC_PROF_API 0

#include <ttg.h>
#include <ttg/serialization.h>
#include <ttg/serialization/splitmd_data_descriptor.h>
//#include <madness.h>
#include "../blockmatrix.h"

#include "lapack.hh"

#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/data_dist/matrix/matrix.h>
#include <parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h>
#include <parsec/data_dist/matrix/two_dim_rectangle_cyclic.h>
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>

#include <dplasma.h>
#include <parsec/profiling.h>

#define USE_DPLASMA

//#define PRINT_TILES

static void
dplasma_dprint_tile( int m, int n,
                     const parsec_tiled_matrix_dc_t* descA,
                     const double *M );

/* FLOP macros taken from DPLASMA */
#define FMULS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (     FMULS_POTRF((__n)) +       FADDS_POTRF((__n)) )

static thread_local parsec_profiling_stream_t *prof = nullptr;
static int event_trsm_startkey, event_trsm_endkey;
static int event_syrk_startkey, event_syrk_endkey;
static int event_potrf_startkey, event_potrf_endkey;
static bool profiling_enabled = false;
#define EVENT_B_INFO_CONVERTER "I{int};J{int}"
#define EVENT_PO_INFO_CONVERTER "I{int}"

static void init_prof_thread()
{
#if USE_PARSEC_PROF_API
  if (nullptr == prof) {
    prof = parsec_profiling_stream_init(4096, "PaRSEC thread");
  }
#endif // USE_PARSEC_PROF_API
}

/**
 * Wrapper around parsec_profiling_trace_flags to enable/disable at will
 */
int potrf_parsec_profiling_trace_flags(parsec_profiling_stream_t* context, int key,
                                       uint64_t event_id, uint32_t taskpool_id,
                                       const void *info, uint16_t flags )
{
  int rc = 0;
#if USE_PARSEC_PROF_API
  if (profiling_enabled) {
    init_prof_thread();
    rc = parsec_profiling_trace_flags(context, key, event_id, taskpool_id, info, flags);
  }
#endif // USE_PARSEC_PROF_API
  return rc;
}



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


struct Key1 {
  // ((I, J), K) where (I, J) is the tile coordiante and K is the iteration number
  int K = 0;
  madness::hashT hash_val;

  Key1() { rehash(); }
  Key1(int K) : K(K){ rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    hash_val = K;
  }

  // Equality test
  bool operator==(const Key1& b) const { return K == b.K; }

  // Inequality test
  bool operator!=(const Key1& b) const { return !((*this) == b); }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar& madness::archive::wrap((unsigned char*)this, sizeof(*this));
  }
};


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


struct Key3 {
  // ((I, J), K) where (I, J) is the tile coordiante and K is the iteration number
  int I = 0, J = 0, K = 0;
  madness::hashT hash_val;

  Key3() { rehash(); }
  Key3(int I, int J, int K) : I(I), J(J), K(K) { rehash(); }

  madness::hashT hash() const { return hash_val; }
  void rehash() {
    hash_val = (static_cast<madness::hashT>(I) << 48)
             ^ (static_cast<madness::hashT>(J) << 32)
             ^ (K << 16);
  }

  // Equality test
  bool operator==(const Key3& b) const { return I == b.I && J == b.J && K == b.K; }

  // Inequality test
  bool operator!=(const Key3& b) const { return !((*this) == b); }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar& madness::archive::wrap((unsigned char*)this, sizeof(*this));
  }
};

namespace std {
  // specialize std::hash for Key

  template <>
  struct hash<Key1> {
    std::size_t operator()(const Key1& s) const noexcept { return s.hash(); }
  };

  template <>
  struct hash<Key2> {
    std::size_t operator()(const Key2& s) const noexcept { return s.hash(); }
  };

  template <>
  struct hash<Key3> {
    std::size_t operator()(const Key3& s) const noexcept { return s.hash(); }
  };

  std::ostream& operator<<(std::ostream& s, const Key1& key) {
    s << "Key(" << key.K << ")";
    return s;
  }

  std::ostream& operator<<(std::ostream& s, const Key2& key) {
    s << "Key(" << key.I << "," << key.J << ")";
    return s;
  }

  std::ostream& operator<<(std::ostream& s, const Key3& key) {
    s << "Key(" << key.I << "," << key.J << "," << key.K << ")";
    return s;
  }
}  // namespace std

template<typename T>
class MatrixTile {

public:
  using metadata_t = typename std::pair<int, int>;

  using pointer_t  = typename std::shared_ptr<T>;

private:
  pointer_t _data;
  int _rows = 0, _cols = 0;

  // (Re)allocate the tile memory
  void realloc() {
    //std::cout << "Reallocating new tile" << std::endl;
    _data = std::shared_ptr<T>(new T[_rows * _cols], [](T* p) { delete[] p; });
  }

public:

  MatrixTile()
  { }


  MatrixTile(int rows, int cols) : _rows(rows), _cols(cols)
  {
    realloc();
  }

  MatrixTile(const metadata_t& metadata)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata))
  { }

  MatrixTile(int rows, int cols, pointer_t data)
  : _data(data), _rows(rows), _cols(cols)
  { }

  MatrixTile(const metadata_t& metadata, pointer_t data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), std::forward(data))
  { }

  /**
   * Constructor with outside memory. The tile will *not* delete this memory
   * upon destruction.
   */
  MatrixTile(int rows, int cols, T* data)
  : _data(data, [](T*){}), _rows(rows), _cols(cols)
  { }

  MatrixTile(const metadata_t& metadata, T* data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), data)
  { }


#if 0
  /* Copy dtor and operator with a static_assert to catch unexpected copying */
  MatrixTile(const MatrixTile& other) {
    static_assert("Oops, copy ctor called?!");
  }

  MatrixTile& operator=(const MatrixTile& other) {
    static_assert("Oops, copy ctor called?!");
  }
#endif


  MatrixTile(MatrixTile<T>&& other)  = default;

  MatrixTile& operator=(MatrixTile<T>&& other)  = default;


  /* Defaulted copy ctor and op for shallow copies, see comment below */
  MatrixTile(const MatrixTile<T>& other)  = default;

  MatrixTile& operator=(const MatrixTile<T>& other)  = default;

  /* Deep copy ctor und op are not needed for PO since tiles will never be read
   * and written concurrently. Hence shallow copies are enough, will all
   * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
   * can handle data sharing without excessive copying */
#if 0
  MatrixTile(const MatrixTile<T>& other)
  : _rows(other._rows), _cols(other._cols)
  {
    this->realloc();
    std::copy_n(other.data(), _rows*_cols, this->data());
  }

  MatrixTile& operator=(const MatrixTile<T>& other) {
    this->_rows = other._rows;
    this->_cols = other._cols;
    this->realloc();
    std::copy_n(other.data(), _rows*_cols, this->data());
  }
#endif // 0

  void set_metadata(metadata_t meta) {
    _rows = std::get<0>(meta);
    _cols = std::get<1>(meta);
  }

  metadata_t get_metadata(void) const {
    return metadata_t{_rows, _cols};
  }

  // Accessing the raw data
  T* data(){
    return _data.get();
  }

  const T* data() const {
    return _data.get();
  }

  size_t size() const {
    return _cols*_rows;
  }

  int rows() const {
    return _rows;
  }

  int cols() const {
    return _cols;
  }
};

namespace ttg {

  template<typename T>
  struct SplitMetadataDescriptor<MatrixTile<T>>
  {

    auto get_metadata(const MatrixTile<T>& t)
    {
      return t.get_metadata();
    }

    auto get_data(MatrixTile<T>& t)
    {
      return std::array<iovec, 1>({t.size()*sizeof(T), t.data()});
    }

    auto create_from_metadata(const typename MatrixTile<T>::metadata_t& meta)
    {
      return MatrixTile<T>(meta);
    }
  };

} // namespace ttg


#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
namespace madness {
  namespace archive {
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, MatrixTile<T>> {
      static inline void store(const Archive& ar, const MatrixTile<T>& tile) {
        ar << tile.rows() << tile.cols();
        ar << wrap(tile.data(), tile.rows() * tile.cols());
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, MatrixTile<T>> {
      static inline void load(const Archive& ar, MatrixTile<T>& tile) {
        int rows, cols;
        ar >> rows >> cols;
        tile = MatrixTile<T>(rows, cols);
        ar >> wrap(tile.data(), tile.rows() * tile.cols());  // MatrixTile<T>(bm.rows(), bm.cols());
      }
    };
  }  // namespace archive
}  // namespace madness

static_assert(madness::is_serializable_v<madness::archive::BufferOutputArchive, MatrixTile<float>>);

#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS


#if 0
namespace ttg {
  namespace iovec_archive {
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, MatrixTile<T>> {
      static inline void store(const Archive& ar, const MatrixTile<T>& tile) {
        // store metadata
        ar << tile.get_metadata();
      }
    };

    template <class Archive, typename T>
    struct ArchiveIovecImpl<Archive, MatrixTile<T>> {
      static inline void iovec(const Archive& ar, MatrixTile<T>& tile) {
        // iovec of the tile (either source or newly constructed)
        ar << ttg::iovec(tile.get_data(), tile.size());
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, MatrixTile<T>> {
      static inline void load(const Archive& ar, MatrixTile<T>& tile) {
        typename MatrixTile<T>::metadata_t md;
        // load metadata
        ar >> md;
        tile.set_metadata(md);
      }
    };
  }
}
#endif // 0


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

  /** Number of tiled columns **/
  int cols(void) const {
    return pm->super.nt;
  }

  /* The rank storing the tile at {row, col} */
  int rank_of(int row, int col) const {
    return pm->super.super.rank_of(&pm->super.super, row, col);
  }

  bool is_local(int row, int col) const {
    return ttg::ttg_default_execution_context().rank() == rank_of(row, col);
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


template <typename T>
auto make_potrf(MatrixT<T>& A,
                ttg::Edge<Key1, MatrixTile<T>>& input,
                ttg::Edge<Key2, MatrixTile<T>>& output_trsm,
                ttg::Edge<Key2, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key1& key,
               MatrixTile<T>&& tile_kk,
               std::tuple<ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>>& out){
    const int K = key.K;

    //std::cout << "POTF " << key << std::endl;
    potrf_parsec_profiling_trace_flags(prof, event_potrf_startkey, K, PROFILE_OBJECT_ID_NULL,
                                       &key, PARSEC_PROFILING_EVENT_HAS_INFO);

    lapack::potrf(lapack::Uplo::Lower, tile_kk.rows(), tile_kk.data(), tile_kk.rows());

    potrf_parsec_profiling_trace_flags(prof, event_potrf_endkey, K, PROFILE_OBJECT_ID_NULL, NULL, 0);

    /* send the tile to outputs */
    std::vector<Key2> keylist;
    keylist.reserve(A.rows() - K);
    /* TODO: reverse order of arrays */
    for (int m = K+1; m < A.rows(); ++m) {
      /* send tile to trsm */
      keylist.push_back(Key2(m, K));
    }
    ttg::broadcast<0, 1>(std::make_tuple(std::array<Key2, 1>{Key2(K, K)}, keylist), std::move(tile_kk), out);
  };
  return ttg::wrap(f, ttg::edges(input), ttg::edges(output_result, output_trsm), "POTRF", {"tile_kk"}, {"output_result", "output_trsm"});
}

template <typename T>
auto make_trsm(MatrixT<T>& A,
               ttg::Edge<Key2, MatrixTile<T>>& input_kk,    // from POTRF
               ttg::Edge<Key2, MatrixTile<T>>& input_mk,    // from previous GEMM
               ttg::Edge<Key2, MatrixTile<T>>& output_diag, // to SYRK
               ttg::Edge<Key3, MatrixTile<T>>& output_row,   // to GEMM
               ttg::Edge<Key3, MatrixTile<T>>& output_col,   // to GEMM
               ttg::Edge<Key2, MatrixTile<T>>& output_result)
{
  auto f = [=](const Key2& key,
               const MatrixTile<T>&  tile_kk,
                     MatrixTile<T>&& tile_mk,
                     std::tuple<ttg::Out<Key2, MatrixTile<T>>,
                                ttg::Out<Key2, MatrixTile<T>>,
                                ttg::Out<Key3, MatrixTile<T>>,
                                ttg::Out<Key3, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.J; // the column equals the outer most look K (same as PO)

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_kk.rows());
    assert(tile_mk.cols() == tile_kk.cols());

    auto m = tile_mk.rows();

#ifdef PRINT_TILES
    std::cout << "TRSM BEFORE: kk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_kk.data());
    std::cout << "TRSM BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_mk.data());
#endif // PRINT_TILES

    potrf_parsec_profiling_trace_flags(prof, event_trsm_startkey, J, PROFILE_OBJECT_ID_NULL,
                                       &key, PARSEC_PROFILING_EVENT_HAS_INFO);
    blas::trsm(blas::Layout::ColMajor,
               blas::Side::Right,
               lapack::Uplo::Lower,
               blas::Op::Trans,
               blas::Diag::NonUnit,
               tile_kk.rows(), m, 1.0,
               tile_kk.data(), m,
               tile_mk.data(), m);

    potrf_parsec_profiling_trace_flags(prof, event_trsm_endkey, J, PROFILE_OBJECT_ID_NULL, NULL, 0);

#ifdef PRINT_TILES
    std::cout << "TRSM AFTER: kk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_kk.data());
    std::cout << "TRSM AFTER: mk" << std::endl;
    dplasma_dprint_tile(I, J, &A.parsec()->super, tile_mk.data());
#endif // PRINT_TILES

    //std::cout << "TRSM(" << key << ")" << std::endl;

    std::vector<Key3> keylist_row;
    keylist_row.reserve(I-J-1);
    std::vector<Key3> keylist_col;
    keylist_col.reserve(A.rows()-I-1);

    /* tile is done */
    //ttg::send<0>(key, std::move(tile_mk), out);

    /* send tile to syrk on diagonal */
    //std::cout << "TRSM(" << key << "): sending output to diag " << Key{I, I, K} << std::endl;
    //ttg::send<1>(Key(I, I, K), tile_mk, out);

    /* send the tile to all gemms across in row i */
    for (int n = J+1; n < I; ++n) {
      //std::cout << "TRSM(" << key << "): sending output to row " << Key{I, n, K} << std::endl;
      //ttg::send<2>(Key(I, n, K), tile_mk, out);
      keylist_row.push_back(Key3(I, n, K));
    }

    /* send the tile to all gemms down in column i */
    for (int m = I+1; m < A.rows(); ++m) {
      //std::cout << "TRSM(" << key << "): sending output to col " << Key{m, I, K} << std::endl;
      //ttg::send<3>(Key(m, I, K), tile_mk, out);
      keylist_col.push_back(Key3(m, I, K));
    }

    ttg::broadcast<0, 1, 2, 3>(std::make_tuple(std::array<Key2, 1>{key},
                                               std::array<Key2, 1>{Key2(I, K)},
                                               keylist_row, keylist_col),
                            std::move(tile_mk), out);
  };
  return ttg::wrap(f, ttg::edges(input_kk, input_mk), ttg::edges(output_result, output_diag, output_row, output_col),
                   "TRSM", {"tile_kk", "tile_mk"}, {"output_result", "output_diag", "output_row", "output_col"});
}


template <typename T>
auto make_syrk(MatrixT<T>& A,
               ttg::Edge<Key2, MatrixTile<T>>& input_mk,    // from TRSM
               ttg::Edge<Key2, MatrixTile<T>>& input_mm,    // from SYRK
               ttg::Edge<Key1, MatrixTile<T>>& output_potrf,// to POTRF
               ttg::Edge<Key2, MatrixTile<T>>& output_syrk)
{
  auto f = [=](const Key2& key,
               const MatrixTile<T>&  tile_mk,
                     MatrixTile<T>&& tile_mm,
                     std::tuple<ttg::Out<Key1, MatrixTile<T>>,
                                ttg::Out<Key2, MatrixTile<T>>>& out){
    const int I = key.I;
    const int K = key.J;

    /* No support for different tile sizes yet */
    assert(tile_mk.rows() == tile_mm.rows());
    assert(tile_mk.cols() == tile_mm.cols());

    auto m = tile_mk.rows();

#ifdef PRINT_TILES
    std::cout << "SYRK BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "SYRK BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mm.data());
#endif // PRINT_TILES

    //std::cout << "SYRK " << key << std::endl;

    potrf_parsec_profiling_trace_flags(prof, event_syrk_startkey, I, PROFILE_OBJECT_ID_NULL,
                                       &key, PARSEC_PROFILING_EVENT_HAS_INFO);

    blas::syrk(blas::Layout::ColMajor,
               lapack::Uplo::Lower,
               blas::Op::NoTrans,
               tile_mk.rows(), m, -1.0,
               tile_mk.data(), m, 1.0,
               tile_mm.data(), m);

    potrf_parsec_profiling_trace_flags(prof, event_syrk_endkey, I, PROFILE_OBJECT_ID_NULL, NULL, 0);

#ifdef PRINT_TILES
    std::cout << "SYRK AFTER: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "SYRK AFTER: nk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mm.data());
    std::cout << "SYRK(" << key << ")" << std::endl;
#endif // PRINT_TILES

    if (I == K+1) {
      /* send the tile to potrf */
      //std::cout << "SYRK(" << key << "): sending output to POTRF " << Key{I, I, K+1} << std::endl;
      ttg::send<0>(Key1(K+1), std::move(tile_mm), out);
    } else {
      /* send output to next syrk */
      //std::cout << "SYRK(" << key << "): sending output to SYRK " << Key{I, I, K+1} << std::endl;
      ttg::send<1>(Key2(I, K+1), std::move(tile_mm), out);
    }

  };
  return ttg::wrap(f,
                   ttg::edges(input_mk, input_mm),
                   ttg::edges(output_potrf, output_syrk), "SYRK",
                   {"tile_mk", "tile_mm"}, {"output_potrf", "output_syrk"});
}


template <typename T>
auto make_gemm(MatrixT<T>& A,
               ttg::Edge<Key3, MatrixTile<T>>& input_nk,    // from TRSM
               ttg::Edge<Key3, MatrixTile<T>>& input_mk,    // from TRSM
               ttg::Edge<Key3, MatrixTile<T>>& input_nm,    // from TRSM
               ttg::Edge<Key2, MatrixTile<T>>& output_trsm, // to TRSM
               ttg::Edge<Key3, MatrixTile<T>>& output_gemm)
{
  auto f = [=](const Key3& key,
              const MatrixTile<T>& tile_nk,
              const MatrixTile<T>& tile_mk,
                    MatrixTile<T>&& tile_nm,
                    std::tuple<ttg::Out<Key2, MatrixTile<T>>,
                               ttg::Out<Key3, MatrixTile<T>>>& out){
    const int I = key.I;
    const int J = key.J;
    const int K = key.K;
    assert(I != J && I > K && J > K);

    /* No support for different tile sizes yet */
    assert(tile_nk.rows() == tile_mk.rows() && tile_nk.rows() == tile_nm.rows());
    assert(tile_nk.cols() == tile_mk.cols() && tile_nk.cols() == tile_nm.cols());

    auto m = tile_nk.rows();

#ifdef PRINT_TILES
    std::cout << "GEMM BEFORE: nk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nk.data());
    std::cout << "GEMM BEFORE: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "GEMM BEFORE: nm" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nm.data());
#endif // PRINT_TILES

    //std::cout << "GEMM " << key << std::endl;

    blas::gemm(blas::Layout::ColMajor,
               blas::Op::NoTrans,
               blas::Op::Trans,
               m, m, m, -1.0,
               tile_nk.data(), m,
               tile_mk.data(), m, 1.0,
               tile_nm.data(), m);

#ifdef PRINT_TILES
    std::cout << "GEMM AFTER: nk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nk.data());
    std::cout << "GEMM AFTER: mk" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_mk.data());
    std::cout << "GEMM AFTER: nm" << std::endl;
    dplasma_dprint_tile(I, I, &A.parsec()->super, tile_nm.data());
    std::cout << "GEMM(" << key << ")" << std::endl;
#endif // PRINT_TILES

    /* send the tile to output */
    if (J == K+1) {
      /* send the tile to trsm */
      //std::cout << "GEMM(" << key << "): sending output to TRSM " << Key{I, J, K+1} << std::endl;
      ttg::send<0>(Key2(I, J), std::move(tile_nm), out);
    } else {
      /* send the tile to the next gemm */
      //std::cout << "GEMM(" << key << "): sending output to GEMM " << Key{I, J, K+1} << std::endl;
      ttg::send<1>(Key3(I, J, K+1), std::move(tile_nm), out);
    }
  };
  return ttg::wrap(f,
                   ttg::edges(input_nk, input_mk, input_nm),
                   ttg::edges(output_trsm, output_gemm), "GEMM",
                   {"input_nk", "input_mk", "input_nm"},
                   {"output_trsm", "outout_gemm"});
}

template<typename T>
auto initiator(MatrixT<T>& A,
               ttg::Edge<Key1, MatrixTile<T>>& syrk_potrf, // to POTRF
               ttg::Edge<Key2, MatrixTile<T>>& gemm_trsm,  // to TRSM
               ttg::Edge<Key2, MatrixTile<T>>& syrk_syrk,  // TO SYRK
               ttg::Edge<Key3, MatrixTile<T>>& gemm_gemm)
{
  auto f = [=](const Key3& key,
               std::tuple<ttg::Out<Key1, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key2, MatrixTile<T>>,
                          ttg::Out<Key3, MatrixTile<T>>>& out){
    /* kick off first POTRF */
    //std::cout << "Initiator called with " << key << std::endl;
    if (A.is_local(0, 0)) {
      ttg::send<0>(Key1(0), std::move(A(0, 0)), out);
    }
    for (int i = 1; i < A.rows(); i++) {
      /* send gemm input to TRSM */
      if (A.is_local(i, 0)) {
        //std::cout << "Initiating TRSM " << Key{i, 0, 0} << std::endl;
        ttg::send<1>(Key2(i, 0), std::move(A(i, 0)), out);
      }
      /* send syrk to SYRK */
      if (A.is_local(i, i)) {
        ttg::send<2>(Key2(i, 0), std::move(A(i, i)), out);
      }
      for (int j = 1; j < i; j++) {
        /* send gemm to GEMM */
        if (A.is_local(i, j)) {
          ttg::send<3>(Key3(i, j, 0), std::move(A(i, j)), out);
        }
      }
    }
  };

  return ttg::wrap<Key3>(f, ttg::edges(), ttg::edges(syrk_potrf, gemm_trsm, syrk_syrk, gemm_gemm), "INITIATOR");
}

template <typename T>
auto make_result(MatrixT<T>& A, const ttg::Edge<Key2, MatrixTile<T>>& result) {
  auto f = [=](const Key2& key, MatrixTile<T>&& tile, std::tuple<>& out) {
    /* write back any tiles that are not in the matrix already */
    const int I = key.I;
    const int J = key.J;
    if (A(I, J).data() != tile.data()) {
      //std::cout << "Writing back tile {" << I << ", " << J << "} " << std::endl;
      std::copy_n(tile.data(), tile.rows()*tile.cols(), A(I, J).data());
    }
#ifdef TTG_USE_USER_TERMDET
    if (I == A.cols()-1 && J == A.rows()-1) {
      ttg::get_default_world().impl().final_task();
    }
#endif // TTG_USE_USER_TERMDET
  };

  return ttg::wrap(f, ttg::edges(result), ttg::edges(), "Final Output", {"result"}, {});
}


int main(int argc, char **argv)
{

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  int N = 1024;
  int M = N;
  int NB = 128;
  int check = 0;
  int nthreads = -1;
  const char* prof_filename = nullptr;

  if (argc > 1) {
    N = M = atoi(argv[1]);
  }

  if (argc > 2) {
    NB = atoi(argv[2]);
  }

  if (argc > 3) {
    check = atoi(argv[3]);
  }

  if (argc > 4) {
    nthreads = atoi(argv[4]);
  }

  if (argc > 5) {
    prof_filename = argv[5];
    profiling_enabled = true;
  }

  ttg::ttg_initialize(argc, argv, nthreads);

  auto world = ttg::ttg_default_execution_context();

#if USE_PARSEC_PROF_API
  if (nullptr != prof_filename) {
    parsec_profiling_init();

    parsec_profiling_dbp_start( prof_filename, "Cholesky Factorization" );

    parsec_profiling_add_dictionary_keyword("TRSM", "#0000FF",
                                            sizeof(int)*2, EVENT_B_INFO_CONVERTER,
                                            &event_trsm_startkey, &event_trsm_endkey);

    parsec_profiling_add_dictionary_keyword("SYRK", "#00FF00",
                                            sizeof(int)*2, EVENT_B_INFO_CONVERTER,
                                            &event_syrk_startkey, &event_syrk_endkey);

    parsec_profiling_add_dictionary_keyword("POTRF", "#FF0000",
                                            sizeof(int)*1, EVENT_PO_INFO_CONVERTER,
                                            &event_potrf_startkey, &event_potrf_endkey);
    parsec_profiling_start();
    profiling_enabled = true;
  }
#endif // USE_PARSEC_PROF_API

  int P = std::sqrt(world.size());
  int Q = (world.size() + P - 1)/P;

  static_assert(ttg::has_split_metadata<MatrixTile<double>>::value);

  std::cout << "Creating 2D block cyclic matrix with NB " << NB << " N " << N << " M " << M << " P " << P << std::endl;

  sym_two_dim_block_cyclic_t dcA;
  sym_two_dim_block_cyclic_init(&dcA, matrix_type::matrix_RealDouble,
                                world.size(), world.rank(), NB, NB, N, M,
                                0, 0, N, M, P, matrix_Lower);
  dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                 (size_t)dcA.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));
  parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, "Matrix A");

  ttg::Edge<Key1, MatrixTile<double>> syrk_potrf("syrk_potrf");

  ttg::Edge<Key2, MatrixTile<double>> potrf_trsm("potrf_trsm"),
                                      trsm_syrk("trsm_syrk"),
                                      gemm_trsm("gemm_trsm"),
                                      syrk_syrk("syrk_syrk"),
                                      result("result");
  ttg::Edge<Key3, MatrixTile<double>> gemm_gemm("gemm_gemm"),
                                      trsm_gemm_row("trsm_gemm_row"),
                                      trsm_gemm_col("trsm_gemm_col");

  //Matrix<double>* A = new Matrix<double>(n_rows, n_cols, NB, NB);
  MatrixT<double> A{&dcA};
  /* TODO: initialize the matrix */
  /* This works only with the parsec backend! */
  int random_seed = 3872;

#ifdef USE_DPLASMA
  dplasma_dplgsy( world.impl().context(), (double)(N), matrix_Lower,
                (parsec_tiled_matrix_dc_t *)&dcA, random_seed);
#endif // USE_DPLASMA

  //dplasma_dprint(world.impl().context(), matrix_Lower, dcA);
  // plgsy(A);

  auto keymap1 = [&](const Key1& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.I, key.J) << std::endl;
    return A.rank_of(key.K, key.K);
  };

  auto keymap2 = [&](const Key2& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.I, key.J) << std::endl;
    return A.rank_of(key.I, key.J);
  };

  auto keymap3 = [&](const Key3& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.I, key.J) << std::endl;
    return A.rank_of(key.I, key.J);
  };

  auto op_init  = initiator(A, syrk_potrf, gemm_trsm, syrk_syrk, gemm_gemm);
  /* op_init gets a special keymap where all keys are local */
  op_init->set_keymap([&](const Key3&){ return world.rank(); });
  auto op_potrf = make_potrf(A, syrk_potrf, potrf_trsm, result);
  op_potrf->set_keymap(keymap1);
  auto op_trsm  = make_trsm(A,
                            potrf_trsm, gemm_trsm,
                            trsm_syrk, trsm_gemm_row, trsm_gemm_col, result);
  op_trsm->set_keymap(keymap2);
  auto op_syrk  = make_syrk(A, trsm_syrk, syrk_syrk, syrk_potrf, syrk_syrk);
  op_syrk->set_keymap(keymap2);
  auto op_gemm  = make_gemm(A,
                            trsm_gemm_row, trsm_gemm_col, gemm_gemm,
                            gemm_trsm, gemm_gemm);
  op_gemm->set_keymap(keymap3);
  auto op_result = make_result(A, result);
  op_result->set_keymap(keymap2);


  /* Priorities taken from DPLASMA */
  auto nt = A.cols();
  op_potrf->set_priomap([&](const Key1& key){ return ((nt - key.K) * (nt - key.K) * (nt - key.K)); });
  op_trsm->set_priomap([&](const Key2& key) { return ((nt - key.I) * (nt - key.I) * (nt - key.I)
                                                      + 3 * ((2 * nt) - key.J - key.I - 1) * (key.I - key.J)); });
  op_syrk->set_priomap([&](const Key2& key) { return ((nt - key.I) * (nt - key.I) * (nt - key.I)
                                                      + 3 * (key.I - key.J)); });
  op_gemm->set_priomap([&](const Key3& key) { return ((nt - key.I) * (nt - key.I) * (nt - key.I)
                                                      + 3 * ((2 * nt) - key.I - key.J - 3) * (key.I - key.J)
                                                      + 6 * (key.I - key.K)); });

  auto connected = make_graph_executable(op_init.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected: " << connected << std::endl;

  if (world.rank() == 0) {
#if 0
    std::cout << "==== begin dot ====\n";
    std::cout << ttg::Dot()(op_init.get()) << std::endl;
    std::cout << "==== end dot ====\n";
#endif // 0
    beg = std::chrono::high_resolution_clock::now();
  }
  op_init->invoke(Key3{0, 0, 0});

  ttg::ttg_execute(world);
  ttg::ttg_fence(world);
  if (world.rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << elapsed / 1E3 << " : Flops " << (FLOPS_DPOTRF(N)) << " " << (FLOPS_DPOTRF(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
  }

#ifdef USE_DPLASMA
  if( check ) {
    /* Check the factorization */
    int loud = 10;
    int ret = 0;
    sym_two_dim_block_cyclic_t dcA0;
    sym_two_dim_block_cyclic_init(&dcA0, matrix_type::matrix_RealDouble,
                                  world.size(), world.rank(), NB, NB, N, M,
                                  0, 0, N, M, P, matrix_Lower);
    dcA0.mat = parsec_data_allocate((size_t)dcA0.super.nb_local_tiles *
                                  (size_t)dcA0.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcA0.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcA0, "Matrix A0");
    dplasma_dplgsy( world.impl().context(), (double)(N), matrix_Lower,
                  (parsec_tiled_matrix_dc_t *)&dcA0, random_seed);

    ret |= check_dpotrf( world.impl().context(), (world.rank() == 0) ? loud : 0, matrix_Lower,
                          (parsec_tiled_matrix_dc_t *)&dcA,
                          (parsec_tiled_matrix_dc_t *)&dcA0);

    /* Check the solution */
    two_dim_block_cyclic_t dcB;
    two_dim_block_cyclic_init(&dcB, matrix_type::matrix_RealDouble, matrix_storage::matrix_Tile,
                              world.size(), world.rank(), NB, NB, N, M,
                              0, 0, N, M, 1, 1, P);
    dcB.mat = parsec_data_allocate((size_t)dcB.super.nb_local_tiles *
                                  (size_t)dcB.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcB.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcB, "Matrix B");
    dplasma_dplrnt( world.impl().context(), 0, (parsec_tiled_matrix_dc_t *)&dcB, random_seed+1);

    two_dim_block_cyclic_t dcX;
    two_dim_block_cyclic_init(&dcX, matrix_type::matrix_RealDouble, matrix_storage::matrix_Tile,
                              world.size(), world.rank(), NB, NB, N, M,
                              0, 0, N, M, 1, 1, P);
    dcX.mat = parsec_data_allocate((size_t)dcX.super.nb_local_tiles *
                                  (size_t)dcX.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(dcX.super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)&dcX, "Matrix X");
    dplasma_dlacpy( world.impl().context(), dplasmaUpperLower,
                    (parsec_tiled_matrix_dc_t *)&dcB, (parsec_tiled_matrix_dc_t *)&dcX );

    dplasma_dpotrs(world.impl().context(), matrix_Lower,
                    (parsec_tiled_matrix_dc_t *)&dcA,
                    (parsec_tiled_matrix_dc_t *)&dcX );

    ret |= check_daxmb( world.impl().context(), (world.rank() == 0) ? loud : 0, matrix_Lower,
                        (parsec_tiled_matrix_dc_t *)&dcA0,
                        (parsec_tiled_matrix_dc_t *)&dcB,
                        (parsec_tiled_matrix_dc_t *)&dcX);

    /* Cleanup */
    parsec_data_free(dcA0.mat); dcA0.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA0 );
    parsec_data_free(dcB.mat); dcB.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB );
    parsec_data_free(dcX.mat); dcX.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcX );
  }
#endif // USE_DPLASMA

  //delete A;
  /* cleanup allocated matrix before shutting down PaRSEC */
  parsec_data_free(dcA.mat); dcA.mat = NULL;
  parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);

#if USE_PARSEC_PROF_API
  /** Finalize profiling */
  if (profiling_enabled) {
    parsec_profiling_dbp_dump();
    parsec_profiling_fini();
  }
#endif // USE_PARSEC_PROF_API

  ttg::ttg_finalize();
  return 0;
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_double_check
 *
 * check_dpotrf - Check the correctness of the Cholesky factorization computed
 * Cholesky functions with the following criteria:
 *
 *    \f[ ||L'L-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  or
 *
 *    \f[ ||UU'-A||_oo/(||A||_oo.N.eps) < 60. \f]
 *
 *  where A is the original matrix, and L, or U, the result of the Cholesky
 *  factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A and A0 are referenced;
 *          = dplasmaLower: Lower triangle of A and A0 are referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == dplasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == dplasmaLower, the
 *          lower part is referenced.
 *
 * @param[in] A0
 *          Descriptor of the original distributed matrix A before
 *          factorization. If uplo == dplasmaUpper, the only the upper part is
 *          referenced, otherwise if uplo == dplasmaLower, the lower part is
 *          referenced.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_dpotrf( parsec_context_t *parsec, int loud,
                  dplasma_enum_t uplo,
                  parsec_tiled_matrix_dc_t *A,
                  parsec_tiled_matrix_dc_t *A0 )
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A0;
    two_dim_block_cyclic_t LLt;
    int info_factorization;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double result = 0.0;
    int M = A->m;
    int N = A->n;
    double eps = std::numeric_limits<double>::epsilon();
    dplasma_enum_t side;

    two_dim_block_cyclic_init(&LLt, matrix_RealDouble, matrix_Tile,
                              ttg::ttg_default_execution_context().size(), twodA->grid.rank,
                              A->mb, A->nb,
                              M, N,
                              0, 0,
                              M, N,
                              twodA->grid.krows, twodA->grid.kcols,
                              twodA->grid.rows /*twodA->grid.ip, twodA->grid.jq*/);

    LLt.mat = parsec_data_allocate((size_t)LLt.super.nb_local_tiles *
                                  (size_t)LLt.super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(LLt.super.mtype));

    dplasma_dlaset( parsec, dplasmaUpperLower, 0., 0.,(parsec_tiled_matrix_dc_t *)&LLt );
    dplasma_dlacpy( parsec, uplo, A, (parsec_tiled_matrix_dc_t *)&LLt );

    /* Compute LL' or U'U  */
    side = (uplo == dplasmaUpper ) ? dplasmaLeft : dplasmaRight;
    dplasma_dtrmm( parsec, side, uplo, dplasmaTrans, dplasmaNonUnit, 1.0,
                   A, (parsec_tiled_matrix_dc_t*)&LLt);

    /* compute LL' - A or U'U - A */
    dplasma_dtradd( parsec, uplo, dplasmaNoTrans,
                    -1.0, A0, 1., (parsec_tiled_matrix_dc_t*)&LLt);

    Anorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo, A0);
    Rnorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo,
                           (parsec_tiled_matrix_dc_t*)&LLt);

    result = Rnorm / ( Anorm * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Cholesky factorization \n");

        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||L'L-A||_oo = %e\n", Anorm, Rnorm );

        printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n", result);
    }

    if ( std::isnan(Rnorm)  || std::isinf(Rnorm)  ||
         std::isnan(result) || std::isinf(result) ||
         (result > 60.0) )
    {
        if( loud ) printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else
    {
        if( loud ) printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    parsec_data_free(LLt.mat); LLt.mat = NULL;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&LLt);

    return info_factorization;
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_double_check
 *
 * check_daxmb - Returns the result of the following test
 *
 *    \f[ (|| A x - b ||_oo / ((||A||_oo * ||x||_oo + ||b||_oo) * N * eps) ) < 60. \f]
 *
 *  where A is the original matrix, b the original right hand side, and x the
 *  solution computed through any factorization.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] loud
 *          The level of verbosity required.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of A is referenced;
 *          = dplasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A result of the Cholesky
 *          factorization. Holds L or U. If uplo == dplasmaUpper, the only the
 *          upper part is referenced, otherwise if uplo == dplasmaLower, the
 *          lower part is referenced.
 *
 * @param[in,out] b
 *          Descriptor of the original distributed right hand side b.
 *          On exit, b is overwritten by (b - A * x).
 *
 * @param[in] x
 *          Descriptor of the solution to the problem, x.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 1, if the result is incorrect
 *          \retval 0, if the result is correct
 *
 ******************************************************************************/
int check_daxmb( parsec_context_t *parsec, int loud,
                 dplasma_enum_t uplo,
                 parsec_tiled_matrix_dc_t *A,
                 parsec_tiled_matrix_dc_t *b,
                 parsec_tiled_matrix_dc_t *x )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int N = b->m;
    double eps = std::numeric_limits<double>::epsilon();

    Anorm = dplasma_dlansy(parsec, dplasmaInfNorm, uplo, A);
    Bnorm = dplasma_dlange(parsec, dplasmaInfNorm, b);
    Xnorm = dplasma_dlange(parsec, dplasmaInfNorm, x);

    /* Compute b - A*x */
    dplasma_dsymm( parsec, dplasmaLeft, uplo, -1.0, A, x, 1.0, b);

    Rnorm = dplasma_dlange(parsec, dplasmaInfNorm, b);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * N * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );

        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (std::isnan(Xnorm) || std::isinf(Xnorm) || std::isnan(result) || std::isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}

static void
dplasma_dprint_tile( int m, int n,
                     const parsec_tiled_matrix_dc_t* descA,
                     const double *M )
{
    int tempmm = ( m == descA->mt-1 ) ? descA->m - m*descA->mb : descA->mb;
    int tempnn = ( n == descA->nt-1 ) ? descA->n - n*descA->nb : descA->nb;
    int ldam = BLKLDD( descA, m );

    int ii, jj;

    fflush(stdout);
    for(ii=0; ii<tempmm; ii++) {
        if ( ii == 0 )
            fprintf(stdout, "(%2d, %2d) :", m, n);
        else
            fprintf(stdout, "          ");
        for(jj=0; jj<tempnn; jj++) {
#if defined(PRECISION_z) || defined(PRECISION_c)
            fprintf(stdout, " (% e, % e)",
                    creal( M[jj*ldam + ii] ),
                    cimag( M[jj*ldam + ii] ));
#else
            fprintf(stdout, " % e", M[jj*ldam + ii]);
#endif
        }
        fprintf(stdout, "\n");
    }
    fflush(stdout);
    usleep(1000);
}
