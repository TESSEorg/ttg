#ifndef TTG_EXAMPLES_MATRIX_TILE_H
#define TTG_EXAMPLES_MATRIX_TILE_H

#include <iostream>
#include <memory>

#include <ttg/serialization/splitmd_data_descriptor.h>

#include <TiledArray/device/allocators.h>

template <typename T, class Allocator = TiledArray::device_pinned_allocator<double>>
class MatrixTile : public ttg::TTValue<MatrixTile<T, Allocator>> {
 public:
  using metadata_t = typename std::tuple<int, int, int>;

  using buffer_t  = typename ttg::buffer<T, Allocator>;
  using ttvalue_type = ttg::TTValue<MatrixTile<T, Allocator>>;

 private:
  buffer_t _buffer;
  int _rows = 0, _cols = 0, _lda = 0;

  // (Re)allocate the tile memory
  void realloc() {
    // std::cout << "Reallocating new tile" << std::endl;
    _buffer.reset(_lda * _cols);
  }

 public:
  MatrixTile() {}

  MatrixTile(int rows, int cols, int lda)
  : ttvalue_type()
  , _buffer(lda*cols)
  , _rows(rows)
  , _cols(cols)
  , _lda(lda)
  { }

  MatrixTile(const metadata_t& metadata)
      : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), std::get<2>(metadata)) {}

  MatrixTile(const metadata_t& metadata, T* data)
      : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), std::forward(data), std::get<2>(metadata)) {}

  /**
   * Constructor with outside memory. The tile will *not* delete this memory
   * upon destruction.
   */
  MatrixTile(int rows, int cols, T* data, int lda)
  : ttvalue_type()
  , _buffer(data, lda*cols)
  , _rows(rows)
  , _cols(cols)
  , _lda(lda)
  { }

  MatrixTile(MatrixTile<T, Allocator>&& other) = default;

  MatrixTile& operator=(MatrixTile<T, Allocator>&& other) = default;

  /* Deep copy ctor und op are not needed for PO since tiles will never be read
   * and written concurrently. Hence shallow copies are enough, will all
   * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
   * can handle data sharing without excessive copying */
  MatrixTile(const MatrixTile<T, Allocator>& other)
  : ttvalue_type()
  , _buffer(other._lda*other._cols)
  , _rows(other._rows)
  , _cols(other._cols)
  , _lda(other._lda) {
    std::copy_n(other.data(), _lda * _cols, this->data());
  }

  MatrixTile& operator=(const MatrixTile<T, Allocator>& other) {
    this->_rows = other._rows;
    this->_cols = other._cols;
    this->_lda = other._lda;
    this->realloc();
    std::copy_n(other.data(), _lda * _cols, this->data());
    return *this;
  }

  void set_metadata(metadata_t meta) {
    _rows = std::get<0>(meta);
    _cols = std::get<1>(meta);
    _lda = std::get<2>(meta);
    this->realloc();
  }

  metadata_t get_metadata(void) const { return metadata_t{_rows, _cols, _lda}; }

  // Accessing the raw data
  T* data() { return _buffer.host_ptr(); }

  const T* data() const { return _buffer.host_ptr(); }

  size_t size() const { return _cols * _lda; }

  int rows() const { return _rows; }

  int cols() const { return _cols; }

  int lda() const { return _lda; }

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

  friend std::ostream& operator<<(std::ostream& o, MatrixTile<T> const& tt) {
    auto ptr = tt.data();
    o << std::endl << " ";
    o << "MatrixTile<" << typeid(T).name() << ">{ rows=" << tt.rows() << " cols=" << tt.cols() << " ld=" << tt.lda();
#if DEBUG_TILES_VALUES
    o << " data=[ " for (int i = 0; i < tt.rows(); i++) {
      for (int j = 0; j < tt.cols(); j++) {
        o << ptr[i + j * tt.lda()] << " ";
      }
      o << std::endl << " ";
    }
    o << " ] "
#endif
        o
      << " } ";
    return o;
  }
};

namespace ttg {

  template <typename T>
  struct SplitMetadataDescriptor<MatrixTile<T>> {
    auto get_metadata(const MatrixTile<T>& t) { return t.get_metadata(); }

    auto get_data(MatrixTile<T>& t) { return std::array<iovec, 1>({t.size() * sizeof(T), t.data()}); }

    auto create_from_metadata(const typename MatrixTile<T>::metadata_t& meta) { return MatrixTile<T>(meta); }
  };

}  // namespace ttg

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
namespace madness {
  namespace archive {
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, MatrixTile<T>> {
      static inline void store(const Archive& ar, const MatrixTile<T>& tile) {
        ar << tile.rows() << tile.cols() << tile.lda();
        ar << wrap(tile.data(), tile.rows() * tile.cols());
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, MatrixTile<T>> {
      static inline void load(const Archive& ar, MatrixTile<T>& tile) {
        int rows, cols, lda;
        ar >> rows >> cols >> lda;
        tile = MatrixTile<T>(rows, cols, lda);
        ar >> wrap(tile.data(), tile.rows() * tile.cols());  // MatrixTile<T>(bm.rows(), bm.cols());
      }
    };
  }  // namespace archive
}  // namespace madness

static_assert(madness::is_serializable_v<madness::archive::BufferOutputArchive, MatrixTile<float>>);

#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

#endif  // TTG_EXAMPLES_MATRIX_TILE_H
