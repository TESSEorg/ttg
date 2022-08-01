#ifndef TTG_EXAMPLES_MATRIX_TILE_H
#define TTG_EXAMPLES_MATRIX_TILE_H

#include <memory>
#include <iostream>

#include <ttg/serialization/splitmd_data_descriptor.h>

template<typename T>
class MatrixTile {

public:
  using metadata_t = typename std::tuple<int, int, int>;

  using pointer_t  = typename std::shared_ptr<T>;

private:
  pointer_t _data;
  int _rows = 0, _cols = 0, _lda = 0;

  // (Re)allocate the tile memory
  void realloc() {
    //std::cout << "Reallocating new tile" << std::endl;
    _data = std::shared_ptr<T>(new T[_lda * _cols], [](T* p) { delete[] p; });
  }

public:

  MatrixTile()
  { }


  MatrixTile(int rows, int cols, int lda) : _rows(rows), _cols(cols), _lda(lda)
  {
    realloc();
  }

  MatrixTile(const metadata_t& metadata)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), std::get<2>(metadata))
  { }

  MatrixTile(int rows, int cols, pointer_t data, int lda)
  : _data(data), _rows(rows), _cols(cols), _lda(lda)
  { }

  MatrixTile(const metadata_t& metadata, pointer_t data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), std::forward(data), std::get<2>(metadata))
  { }

  /**
   * Constructor with outside memory. The tile will *not* delete this memory
   * upon destruction.
   */
  MatrixTile(int rows, int cols, T* data, int lda)
  : _data(data, [](T*){}), _rows(rows), _cols(cols), _lda(lda)
  { }

  MatrixTile(const metadata_t& metadata, T* data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), data, std::get<2>(metadata))
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

#if 0
  /* Defaulted copy ctor and op for shallow copies, see comment below */
  MatrixTile(const MatrixTile<T>& other)  = default;

  MatrixTile& operator=(const MatrixTile<T>& other)  = default;
#endif // 0
  /* Deep copy ctor und op are not needed for PO since tiles will never be read
   * and written concurrently. Hence shallow copies are enough, will all
   * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
   * can handle data sharing without excessive copying */
#if 1
  MatrixTile(const MatrixTile<T>& other)
  : _rows(other._rows), _cols(other._cols), _lda(other._lda)
  {
    this->realloc();
    std::copy_n(other.data(), _lda*_cols, this->data());
  }

  MatrixTile& operator=(const MatrixTile<T>& other) {
    this->_rows = other._rows;
    this->_cols = other._cols;
    this->_lda  = other._lda;
    this->realloc();
    std::copy_n(other.data(), _lda*_cols, this->data());
  }
#endif // 1

  void set_metadata(metadata_t meta) {
    _rows = std::get<0>(meta);
    _cols = std::get<1>(meta);
    _lda  = std::get<2>(meta);
  }

  metadata_t get_metadata(void) const {
    return metadata_t{_rows, _cols, _lda};
  }

  // Accessing the raw data
  T* data(){
    return _data.get();
  }

  const T* data() const {
    return _data.get();
  }

  /// @return shared_ptr to data
  pointer_t data_shared() & {
    return _data;
  }

  /// @return shared_ptr to data
  pointer_t data_shared() const& {
    return _data;
  }

  /// yields data and resets this object to a default-constucted state
  pointer_t yield_data() && {
    pointer_t result = _data;
    *this = MatrixTile();
    return std::move(result);
  }

  size_t size() const {
    return _cols*_lda;
  }

  int rows() const {
    return _rows;
  }

  int cols() const {
    return _cols;
  }

  int lda() const {
    return _lda;
  }

  auto& fill(T value) {
    std::fill(_data.get(), _data.get()+size(), value);
    return *this;
  }

  friend std::ostream& operator<< (std::ostream& o, MatrixTile<T> const& tt)  {
  auto ptr = tt.data();
  o << std::endl << " ";
  for(int i = 0; i < tt.rows(); i++) {
    for(int j = 0; j < tt.cols(); j++) {
      o << ptr[i+j*tt.lda()] << " ";
    }
    o << std::endl << " ";
  }
  return o;
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

#endif // TTG_EXAMPLES_MATRIX_TILE_H

