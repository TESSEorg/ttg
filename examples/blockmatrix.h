#include <iostream>

template <typename T>
class BlockMatrix {
 private:
  int _rows;
  int _cols;
  std::shared_ptr<T> m_block;

 public:
  BlockMatrix() = default;

  BlockMatrix(int rows, int cols) : _rows(rows), _cols(cols) {
    // Deallocator is required since we are allocating a pointer
    m_block = std::shared_ptr<T>(new T[_rows * _cols], [](T* p) { delete[] p; });
  }
  
  /*//Move constructor
  BlockMatrix(BlockMatrix<T>&& other) {
    _rows = other._rows;
    _cols = other._cols;
    m_block = other.m_block;
    other.m_block = nullptr;
  }
  
  BlockMatrix<T>& operator=(BlockMatrix<T>& other) noexcept
  {
    if (this != &other) {
      _rows = other._rows;
      _cols = other._cols;
      m_block = other.m_block;
    }
    return *this;
  }*/

  ~BlockMatrix() {}

  int size() const { return _rows * _cols; }
  int rows() const { return _rows; }
  int cols() const { return _cols; }

  void fill() {
    // Initialize all elements of the matrix to 1
    for (int i = 0; i < _rows; ++i) {
      for (int j = 0; j < _cols; ++j) {
        m_block.get()[i * _cols + j] = 1;
      }
    }
  }

  bool operator==(const BlockMatrix& m) const { 
    bool equal = true;
    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        if (m_block.get()[i * _cols + j] != m.m_block.get()[i * _cols + j]) {
          equal = false;
          break;
        }
      }
    }
    return equal;
  }

  bool operator!=(const BlockMatrix& m) const {
    bool notequal = false;
    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        if (m_block.get()[i * _cols + j] != m.m_block.get()[i * _cols + j]) {
          notequal = true;
          break;
        }
      }
    }

    return notequal;
  }

  //Return by value
  T operator() (int row, int col) { return m_block.get()[row * _cols + col]; }
  
  void operator() (int row, int col, T val) {
    m_block.get()[row * _cols + col] = val;
  }
     
  template <typename Archive>
  void serialize(Archive& ar) {}
};

template <typename T>
std::ostream& operator<<(std::ostream& s, BlockMatrix<T>& m) {
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) s << m(i, j) << " ";
    s << std::endl;
  }
  return s;
}

template <typename T>
class Matrix {
 private:
  int nb_row;  //# of blocks in a row
  int nb_col;  //# of blocks in a col
  int b_rows;  //# of rows in a block
  int b_cols;  //# of cols in a block
               // Array of BlockMatrix<T>
  std::shared_ptr<BlockMatrix<T>> m;

 public:
  Matrix() = default;
  Matrix(int nb_row, int nb_col, int b_rows, int b_cols)
      : nb_row(nb_row), nb_col(nb_col), b_rows(b_rows), b_cols(b_cols) {
    m = std::shared_ptr<BlockMatrix<T>>(new BlockMatrix<T>[nb_row * nb_col], [](BlockMatrix<T>* b) { delete[] b; });

    for (int i = 0; i < nb_row; i++)
      for (int j = 0; j < nb_col; j++) 
        m.get()[i * nb_col + j] = BlockMatrix<T>(b_rows, b_cols);
  }

  ~Matrix() {}

  // Return total # of elements in the matrix
  int size() const { return (nb_row * b_rows) * (nb_col * b_cols); }
  // Return # of block rows
  int rows() const { return nb_row; }
  // Return # of block cols
  int cols() const { return nb_col; }

  void fill() {
    for (int i = 0; i < nb_row; i++)
      for (int j = 0; j < nb_col; j++) (m.get()[i * nb_col + j]).fill();
  }

  bool operator==(const Matrix& matrix) const { return (matrix.m == m); }

  bool operator!=(const Matrix& matrix) const { return (matrix.m != m); }

  //Return by value
  BlockMatrix<T> operator()(int block_row, int block_col) { return m.get()[block_row * nb_col + block_col]; }

  void operator()(int block_row, int block_col, BlockMatrix<T> val) {
    m.get()[block_row * nb_col + block_col] = val;
  }

  void print() {
    for (int i = 0; i < nb_row; i++) {
      for (int j = 0; j < nb_col; j++) {
        std::cout << m.get()[i * nb_col + j];
      }
    }
  }

  template <typename Archive>
  void serialize(Archive& ar) {}
};
