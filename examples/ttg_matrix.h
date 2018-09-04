//
// Created by Eduard Valeyev on 5/25/18.
//

#ifndef TTG_MATRIX_H
#define TTG_MATRIX_H

#include <vector>

namespace ttg {

  namespace matrix {

  // matrix shape = maps {column,row} index to {row,column} indices
  class Shape : public std::vector<std::vector<long>> {
    using base_t = std::vector<std::vector<long>>;
   public:
    enum class Type {col2row, row2col, invalid};
    Shape() = default;
    ~Shape() = default;

    template <typename T>
    Shape(const Eigen::SparseMatrix<T>& spmat, Type type = Type::col2row) :
    base_t(type == Type::col2row ? make_colidx_to_rowidx(spmat) : make_rowidx_to_colidx(spmat)), nrows_(spmat.rows()), ncols_(spmat.cols()), type_(type) {}

    long nrows() const { return nrows_; }
    long ncols() const { return ncols_; }
    Type type() const { return type_; }

    /// converts col->row <-> row->col
    Shape inverse() const {
      Shape result(*this);
      const_cast<Shape&>(result).inverse();
      return result;
    }
    /// in-place inverse
    Shape& inverse() {
      assert(type_ != Type::invalid);
      type_ = (type_ == Type::col2row) ? Type::row2col : Type::col2row;
      base_t result(type_ == Type::col2row ? ncols_ : nrows_);
      for(std::size_t i = 0; i!=size(); ++i) {
        for(const auto& j:(*this)[i]) {
          result.at(j).push_back(i);
        }
      }
      for(std::size_t i = 0; i!=size(); ++i) {
        std::sort(std::begin(result[i]), std::end(result[i]));
      }
      return *this;
    }

    template<typename Archive> void serialize(Archive &ar) {
      ar & type_ & nrows_ & ncols_ & static_cast<base_t&>(*this);
    }

    void print(std::ostream& os) const {
      os << "Shape: type=";
      switch(type()) {
        case Type::col2row: os << "col2row";
          break;
        case Type::row2col: os << "row2col";
          break;
        case Type::invalid: os << "invalid";
          break;
      }
      os << " { ";
      for(int i=0; i!=size(); ++i) {
        os << i << ":{ ";
        for(const auto j : (*this)[i]) {
          os << j << " ";
        }
        os << "} ";
      }
      os << "}";
    }

   private:
    long nrows_ = -1, ncols_ = -1;
    Type type_ = Type::invalid;

    // result[i][j] gives the j-th nonzero row for column i in matrix mat
    template <typename T> base_t make_colidx_to_rowidx(const Eigen::SparseMatrix<T> &mat) {
      base_t colidx_to_rowidx;
      for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
          auto row = it.row();
          auto col = it.col();
          if (col >= colidx_to_rowidx.size()) colidx_to_rowidx.resize(col + 1);
          // in either case (col- or row-major) row index increasing for the given col
          colidx_to_rowidx[col].push_back(row);
        }
      }
      return colidx_to_rowidx;
    }

    // result[i][j] gives the j-th nonzero column for row i in matrix mat
    template <typename T> base_t make_rowidx_to_colidx(const Eigen::SparseMatrix<T> &mat) {
      base_t rowidx_to_colidx;
      for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
          auto row = it.row();
          auto col = it.col();
          if (row >= rowidx_to_colidx.size()) rowidx_to_colidx.resize(row + 1);
          // in either case (col- or row-major) col index increasing for the given row
          rowidx_to_colidx[row].push_back(col);
        }
      }
      return rowidx_to_colidx;
    }
  };

  std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    shape.print(os);
    return os;
  }

  // flow data from an existing SpMatrix on rank 0
  // similar to Read_SpMatrix but uses tasks to read data:
  // - this allows to read some data only (imagine need to do Hadamard product of 2 sparse matrices ... only some blocks will be needed)
  //   but will only be efficient if can do random access (slow with CSC format used by Eigen matrices)
  // - this could be generalized to read efficiently from a distributed data structure
  // Use Read_SpMatrix is need to read all data from a data structure localized on 1 process
  template <typename Blk = blk_t>
  class Read : public Op<Key<2>, std::tuple<Out<Key<2>, Blk>>, Read<Blk>, void> {
   public:
    using baseT = Op<Key<2>, std::tuple<Out<Key<2>, Blk>>, Read<Blk>, void>;
    static constexpr const int owner = 0;  // where data resides

    Read(const char *label, const SpMatrix<Blk> &matrix, Edge<Key<2>, void> &in, Edge<Key<2>, Blk> &out)
        : baseT(edges(in), edges(out), std::string("read_spmatrix(") + label + ")", {"ij"}, {std::string(label) + "ij"},
                /* keymap */ [](auto key) { return owner; })
        , matrix_(matrix) {}

    void op(const Key<2>& key, std::tuple<Out<Key<2>, Blk>> &out) {
      // random access in CSC format is inefficient, this is only to demonstrate the way to go for hash-based storage
      // for whatever reason coeffRef does not work on a const SpMatrix&
      ::send<0>(key, static_cast<const Blk>(const_cast<SpMatrix<Blk>&>(matrix_).coeffRef(key[0], key[1])), out);
    }

   private:
    const SpMatrix<Blk> &matrix_;
  };

  // compute shape of an existing SpMatrix on rank 0
  template <typename Blk = blk_t>
  class ReadShape : public Op<void, std::tuple<Out<void, Shape>>, ReadShape<Blk>, void> {
   public:
    using baseT = Op<void, std::tuple<Out<void, Shape>>, ReadShape<Blk>, void>;
    static constexpr const int owner = 0;  // where data resides

    ReadShape(const char *label, const SpMatrix<Blk> &matrix, Edge<void, void> &in, Edge<void, Shape> &out)
        : baseT(edges(in), edges(out), std::string("read_spmatrix_shape(") + label + ")", {"ctl"}, {std::string("shape[") + label + "]"},
        /* keymap */ []() { return owner; })
        , matrix_(matrix) {}

    void op(std::tuple<Out<void, Shape>> &out) {
      ::sendv<0>(Shape(matrix_), out);
    }

   private:
    const SpMatrix<Blk> &matrix_;
  };

  // pushes all blocks given by the shape
  class Push : public Op<void, std::tuple<Out<Key<2>, void>>, Push, Shape> {
   public:
    using baseT = Op<void, std::tuple<Out<Key<2>, void>>, Push, Shape>;
    static constexpr const int owner = 0;  // where data resides

    Push(const char *label, Edge<void, Shape> &in, Edge<Key<2>, void> &out)
        : baseT(edges(in), edges(out), std::string("push_spmatrix(") + label + ")", {std::string("shape[") + label + "]"}, {"ij"},
        /* keymap */ []() { return owner; }) {}

    void op(typename baseT::input_values_tuple_type && ins, std::tuple<Out<Key<2>, void>> &out) {
      const auto& shape = baseT::get<0>(ins);
      assert(shape.type() == Shape::Type::col2row);
      long colidx = 0;
      for(const auto& col: shape) {
        for(const auto rowidx: col) {
          ::sendk<0>(Key<2>({rowidx, colidx}), out);
        }
        ++colidx;
      }
    }
  };

  }  // namespace matrix

#if 1
  /// Sparse matrix flow
  /// @tparam T element type
  template <typename T>
  class Matrix {
   public:
    using shape_t = matrix::Shape;
    using data_edge_t = Edge<Key<2>, T>;
    using shape_edge_t = Edge<void, shape_t>;
    using ctl_edge_t = Edge<Key<2>, void>;

    Matrix() = default;

    Matrix(shape_edge_t && shape_edge, ctl_edge_t && ctl_edge, data_edge_t && data_edge) :
    shape_edge_(std::move(shape_edge)),
    ctl_edge_(std::move(ctl_edge)),
    data_edge_(std::move(data_edge)) {
    }

    auto& data() { return data_edge_; }
    auto& shape() { return shape_edge_; }
    auto& ctl() { return ctl_edge_; }

    /// attach to the source sparse Eigen::Matrix
    void operator<<(const Eigen::SparseMatrix<T>& source_matrix) {
      // shape reader computes shape of source_matrix
      ttg_register_ptr(world_, std::make_shared<matrix::ReadShape<T>>("", source_matrix, ttg_ctl_edge(world_), shape_edge_));
      // reads data from source_matrix_ for a given key
      ttg_register_ptr(world_, std::make_shared<matrix::Read<T>>("", source_matrix, ctl_edge_, data_edge_));
    }

    /// pushes all data that exists
    void pushall() {
      // reads the shape and pulls all the data
      ttg_register_ptr(world_, std::make_shared<matrix::Push>("", shape_edge_, ctl_edge_));
    }

    /// @return an std::future<void> object indicating the status; @c destination_matrix is ready if calling has_value() on the return value
    /// of this function is true.
    /// @note up to the user to ensure completion before reading destination_matrix
    auto operator>>(SpMatrix<T>& destination_matrix) {
#if 0  // new code not ready yet
      // shape writer writes shape to destination_matrix
      // shape writer needs to control Writer also ... currently there is no way to activate flows so make control an input to every write task ...
      // this also ensures that shape and data flows are consistent
      ctl_edge_t ctl_edge;
      ttg_register_ptr(world_, std::make_shared<matrix::WriteShape<T>>("", destination_matrix, shape_edge_, ctl_edge));
      auto result = std::make_shared<matrix::Write<T>>(destination_matrix, data_edge_, ctl_edge);
#else
      auto result = std::make_shared<Write_SpMatrix<T>>(destination_matrix, data_edge_);
#endif
      ttg_register_ptr(world_, result);

      // return op status ... set to true after world fence
      return result->status();
    }

   private:
    data_edge_t data_edge_{};
    shape_edge_t shape_edge_{};
    ctl_edge_t ctl_edge_{};
    World& world_ = ttg_default_execution_context();
  };

  template <typename T>
  Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b) {
    using shape_edge_t = typename Matrix<T>::shape_edge_t;
    using data_edge_t = typename Matrix<T>::data_edge_t;
    using ctl_edge_t = typename Matrix<T>::ctl_edge_t;
    shape_edge_t shape_edge;
    data_edge_t data_edge;
    ctl_edge_t ctl_edge;

    // make a lambda that implements shape addition
    // wrap this lambda into an op, its output will be attached to shape_edge
    // don't forget to register this op

    // same for block addition: make a lambda that implements block addition
    // use ctl_edge as control of this op
    // wrap this lambda into an op, its output will be attached to data_edge
    // don't forget to register this op

    return Matrix<T>(std::move(shape_edge), std::move(ctl_edge), std::move(data_edge));
  }

#endif

}  // namespace ttg

#endif //TTG_MATRIX_H
