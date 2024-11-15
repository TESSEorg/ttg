#ifndef TTG_EXAMPLES_MATRIX_TILE_H
#define TTG_EXAMPLES_MATRIX_TILE_H

#include <iostream>
#include <memory>
#include <optional>

#include <ttg.h>

#include <ttg/serialization/splitmd_data_descriptor.h>


#include <TiledArray/external/device.h>
#if defined(TILEDARRAY_HAS_DEVICE)
template<typename T>
using Allocator = TiledArray::device_pinned_allocator<T>;

inline void allocator_init(int argc, char **argv) {
  // initialize MADNESS so that TA allocators can be created
#if defined(TTG_PARSEC_IMPORTED)
  madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
  madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);
#endif // TTG_PARSEC_IMPORTED
}

inline void allocator_fini() {
#if defined(TTG_PARSEC_IMPORTED)
  madness::finalize();
#endif // TTG_PARSEC_IMPORTED
}
#else  // TILEDARRAY_HAS_DEVICE
template<typename T>
using Allocator = std::allocator<T>;

inline void allocator_init(int argc, char **argv) { }

inline void allocator_fini() { }

#endif // TILEDARRAY_HAS_DEVICE

template <typename T, class AllocatorT = Allocator<T>>
class MatrixTile : public ttg::TTValue<MatrixTile<T, AllocatorT>> {
 public:
  using buffer_t  = typename ttg::Buffer<T, Allocator>;
  using ttvalue_type = ttg::TTValue<MatrixTile<T, Allocator>>;

 private:
  buffer_t _buffer;
  std::size_t _rows = 0, _cols = 0, _lda = 0;
#ifdef DEBUG_TILES_VALUES
  mutable std::optional<T> _norm;
#endif // DEBUG_TILES_VALUES

  // (Re)allocate the tile memory
  void realloc() {
    // std::cout << "Reallocating new tile" << std::endl;
    _buffer.reset(_lda * _cols);
#ifdef DEBUG_TILES_VALUES
    std::fill(_buffer.host_ptr(), _lda * _cols, T{});
#endif // DEBUG_TILES_VALUES
  }

  struct non_owning_deleter {
    void operator()(T* ptr) { }
  };

 public:
  MatrixTile() {}

  MatrixTile(std::size_t rows, std::size_t cols, std::size_t lda)
  : ttvalue_type()
  , _buffer(lda*cols)
  , _rows(rows)
  , _cols(cols)
  , _lda(lda)
  { }

  /**
   * Constructor with outside memory. The tile will *not* delete this memory
   * upon destruction.
   */
  MatrixTile(std::size_t rows, std::size_t cols, T* data, std::size_t lda)
  : ttvalue_type()
  , _buffer(std::unique_ptr<T[], non_owning_deleter>(data, non_owning_deleter{}), lda*cols)
  , _rows(rows)
  , _cols(cols)
  , _lda(lda)
  { }

  MatrixTile(MatrixTile<T, AllocatorT>&& other) = default;

  MatrixTile& operator=(MatrixTile<T, AllocatorT>&& other) = default;

  /* Deep copy ctor und op are not needed for PO since tiles will never be read
   * and written concurrently. Hence shallow copies are enough, will all
   * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
   * can handle data sharing without excessive copying */
  MatrixTile(const MatrixTile<T, AllocatorT>& other)
  : ttvalue_type()
  , _buffer(other._lda*other._cols)
  , _rows(other._rows)
  , _cols(other._cols)
  , _lda(other._lda)
#ifdef DEBUG_TILES_VALUES
  , _norm(other._norm)
#endif // DEBUG_TILES_VALUES
  {
    std::copy_n(other.data(), _lda * _cols, this->data());
  }

  MatrixTile& operator=(const MatrixTile<T, AllocatorT>& other) {
    this->_rows = other._rows;
    this->_cols = other._cols;
    this->_lda = other._lda;
    this->realloc();
    std::copy_n(other.data(), _lda * _cols, this->data());
    return *this;
  }

  // Accessing the raw data
  T* data() { return _buffer.host_ptr(); }

  const T* data() const { return _buffer.host_ptr(); }

  size_t size() const { return _cols * _lda; }

  std::size_t rows() const { return _rows; }

  std::size_t cols() const { return _cols; }

  std::size_t lda() const { return _lda; }

  buffer_t& buffer() {
    return _buffer;
  }

  const buffer_t& buffer() const {
    return _buffer;
  }

  auto& fill(T value) {
    std::fill(data().get(), data().get() + size(), value);
    _buffer.set_current_device(0);
    return *this;
  }

#ifdef DEBUG_TILES_VALUES
  /* Only available if debugging is enabled. Norm must be
   * set by application and is not computed automatically. */
  T norm() const {
    if (!_norm) _norm = blas::nrm2(size(), data(), 1);
    return _norm.value();
  }

  void set_norm(T norm) {
    _norm = norm;
  }
#endif // DEBUG_TILES_VALUES

  friend std::ostream& operator<<(std::ostream& o, MatrixTile<T, AllocatorT> const& tt) {
    auto ptr = tt.data();
    o << std::endl << " ";
    o << "MatrixTile<" << typeid(T).name() << ">{ rows=" << tt.rows() << " cols=" << tt.cols() << " ld=" << tt.lda();
#if DEBUG_TILES_VALUES && 0
    o << " data=[ ";
    for (std::size_t i = 0; i < tt.rows(); i++) {
      for (std::size_t j = 0; j < tt.cols(); j++) {
        o << ptr[i + j * tt.lda()] << " ";
      }
      o << std::endl << " ";
    }
    o << " ] ";
#endif
    o << " } ";
    return o;
  }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    serialize(ar);
  }

  template<typename Archive>
  void serialize(Archive& ar) {
    ar & _rows & _cols & _lda;
    ar & buffer();
  }
};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
static_assert(madness::is_serializable_v<madness::archive::BufferOutputArchive, MatrixTile<float>>);
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

#endif  // TTG_EXAMPLES_MATRIX_TILE_H
