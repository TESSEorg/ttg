#include <iostream>
#include <unordered_map>
#include <madness/world/archive.h>


//std::unordered_map, container to hold all the blocks
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

  //Copy constructor
  /*BlockMatrix(const BlockMatrix<T>& other) : _rows(other._rows), _cols(other._cols),
              m_block(std::make_shared<T>(*other.m_block)) {}
  
  //Move constructor
  BlockMatrix(BlockMatrix<T>&& other) : _rows(other._rows), _cols(other._cols),
              m_block(std::make_shared<T>(*other.m_block)) //is it possible to use move instead? 
  {}
  
  BlockMatrix<T> operator=(BlockMatrix<T> other) {
    //std::shared_ptr<T>(other.get()).swap(m_block);
    std::swap(*this, other);
    return *this;
  }*/

  ~BlockMatrix() {}

  int size() const { return _rows * _cols; }
  int rows() const { return _rows; }
  int cols() const { return _cols; }
  const T* get() const { return m_block.get(); }
  T* get() { return m_block.get(); }
  
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
     
};

namespace madness {
  namespace archive {
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, BlockMatrix<T>> {
      static inline void store(const Archive& ar, const BlockMatrix<T>& bm) {
        ar << bm.rows() << bm.cols();
        ar << wrap(bm.get(), bm.rows() * bm.cols()); //BlockMatrix<T>(bm.rows(), bm.cols());
      }
    };
  
    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, BlockMatrix<T>> {
      static inline void load(const Archive& ar, BlockMatrix<T>& bm) {
        int rows, cols;
        ar >> rows >> cols;
        bm = BlockMatrix<T>(rows,cols);
        ar >> wrap(bm.get(), bm.rows() * bm.cols()); //BlockMatrix<T>(bm.rows(), bm.cols());
      }
    };
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& s, BlockMatrix<T>& m) {
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) s << m(i, j) << " ";
    s << std::endl;
  }
  return s;
}

//https://stackoverflow.com/questions/32685540/why-cant-i-compile-an-unordered-map-with-a-pair-as-key
//We need this since pair cannot be hashed by unordered_map.
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2;  
    }
};

template <typename T>
class Matrix {
  private:
  int nb_row;  //# of blocks in a row
  int nb_col;  //# of blocks in a col
  int b_rows;  //# of rows in a block
  int b_cols;  //# of cols in a block
               // Array of BlockMatrix<T>
  std::unordered_map<std::pair<int,int>, BlockMatrix<T>, pair_hash> m;

 public:
  Matrix() = default;
  Matrix(int nb_row, int nb_col, int b_rows, int b_cols)
      : nb_row(nb_row), nb_col(nb_col), b_rows(b_rows), b_cols(b_cols) {
    for (int i = 0; i < nb_row; i++)
      for (int j = 0; j < nb_col; j++) {
        m[std::make_pair(i,j)] = BlockMatrix<T>(b_rows, b_cols);
      }
  }

  ~Matrix() {}

  // Return total # of elements in the matrix
  int size() const { return (nb_row * b_rows) * (nb_col * b_cols); }
  // Return # of block rows
  int rows() const { return nb_row; }
  // Return # of block cols
  int cols() const { return nb_col; }
  std::unordered_map<std::pair<int,int>, BlockMatrix<T>, pair_hash> matrix() const { return m; }
    
  void fill() {
    for (int i = 0; i < nb_row; i++)
      for (int j = 0; j < nb_col; j++) 
        m[i * nb_col + j].fill();
  }

  bool operator==(const Matrix& matrix) const { return (matrix.m == m); }

  bool operator!=(const Matrix& matrix) const { return (matrix.m != m); }

  //Return by value
  BlockMatrix<T> operator()(int block_row, int block_col)
  {
    return m[std::make_pair(block_row,block_col)];
  }

  void operator()(int block_row, int block_col, BlockMatrix<T> val) {
    for (int i = 0; i < b_rows; i++) {
      for (int j = 0; j < b_cols; j++) {
        m[std::make_pair(block_row,block_col)] = val(i,j);
      }
    }
    //m.get()[block_row * nb_col + block_col] = val;
  }

  void print() {
    for (int i = 0; i < nb_row; i++) {
      for (int j = 0; j < nb_col; j++) {
        std::cout << m[i * nb_col + j];
      }
    }
  }
};

template <typename T>
class MatrixOld {
 private:
  int nb_row;  //# of blocks in a row
  int nb_col;  //# of blocks in a col
  int b_rows;  //# of rows in a block
  int b_cols;  //# of cols in a block
               // Array of BlockMatrix<T>
  std::shared_ptr<BlockMatrix<T>> m;
  
 public:
  MatrixOld() = default;
  MatrixOld(int nb_row, int nb_col, int b_rows, int b_cols)
      : nb_row(nb_row), nb_col(nb_col), b_rows(b_rows), b_cols(b_cols) {
    m = std::shared_ptr<BlockMatrix<T>>(new BlockMatrix<T>[nb_row * nb_col], [](BlockMatrix<T>* b) { delete[] b; });

    for (int i = 0; i < nb_row; i++)
      for (int j = 0; j < nb_col; j++) {
        m.get()[i * nb_col + j] = BlockMatrix<T>(b_rows, b_cols);
      }
  }

  ~MatrixOld() {}

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

  bool operator==(const MatrixOld& matrix) const { return (matrix.m == m); }

  bool operator!=(const MatrixOld& matrix) const { return (matrix.m != m); }

  //Return by value
  BlockMatrix<T> operator()(int block_row, int block_col) 
  { 
    return m.get()[block_row * nb_col + block_col]; 
  }

  void operator()(int block_row, int block_col, BlockMatrix<T> val) {
    for (int i = 0; i < b_rows; i++) {
      for (int j = 0; j < b_cols; j++) {
        m.get()[block_row * nb_col + block_col](i,j,val(i,j));
      }
    }
    //m.get()[block_row * nb_col + block_col] = val;
  }

  void print() {
    for (int i = 0; i < nb_row; i++) {
      for (int j = 0; j < nb_col; j++) {
        std::cout << m.get()[i * nb_col + j];
      }
    }
  }
};
