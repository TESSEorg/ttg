//
// Created by Eduard Valeyev on 5/25/18.
//

#ifndef TTG_MATRIX_H
#define TTG_MATRIX_H

#include <vector>

#include "ttg/serialization/std/vector.h"

namespace ttg {

  namespace matrix {

    // matrix shape = maps {column,row} index to {row,column} indices
    class Shape : public std::vector<std::vector<long>> {
      using base_t = std::vector<std::vector<long>>;

     public:
      enum class Type { col2row, row2col, invalid };
      Shape() = default;
      ~Shape() = default;

      template <typename T>
      Shape(const Eigen::SparseMatrix<T> &spmat, Type type = Type::col2row)
          : base_t(type == Type::col2row ? make_colidx_to_rowidx(spmat) : make_rowidx_to_colidx(spmat))
          , nrows_(spmat.rows())
          , ncols_(spmat.cols())
          , type_(type) {}

      Shape(long nrows, long ncols, Type type) : nrows_(nrows), ncols_(ncols), type_(type) {
        resize(type == Type::col2row ? ncols_ : (type == Type::row2col ? nrows_ : 0));
      }

      long nrows() const { return nrows_; }
      long ncols() const { return ncols_; }
      Type type() const { return type_; }

      static bool congruent(const Shape &shape1, const Shape &shape2) {
        return shape1.nrows() == shape2.nrows() && shape1.ncols() == shape2.ncols();
      }

      /// converts col->row <-> row->col
      Shape inverse() const {
        Shape result(*this);
        const_cast<Shape &>(result).inverse();
        return result;
      }
      /// in-place inverse
      Shape &inverse() {
        assert(type_ != Type::invalid);
        type_ = (type_ == Type::col2row) ? Type::row2col : Type::col2row;
        base_t result(type_ == Type::col2row ? ncols_ : nrows_);
        for (std::size_t i = 0; i != size(); ++i) {
          for (const auto &j : (*this)[i]) {
            result.at(j).push_back(i);
          }
        }
        for (std::size_t i = 0; i != size(); ++i) {
          std::sort(std::begin(result[i]), std::end(result[i]));
        }
        return *this;
      }

      static Shape add(const Shape &shape1, const Shape &shape2) {
        if (shape1.type() != shape2.type())
          throw std::logic_error("Shape::add(shape1,shape2): shape1.type() != shape2.type()");
        if (shape1.type() == Type::invalid)
          throw std::logic_error("Shape::add(shape1,shape2): shape1.type() == invalid");
        if (!congruent(shape1, shape2))
          throw std::logic_error("Shape::add(shape1,shape2): shape1 not congruent to shape2");

        Shape result(shape1.nrows(), shape1.ncols(), shape1.type());
        for (std::size_t i = 0; i != shape1.size(); ++i) {
          const auto &shape1_row_i = shape1[i];
          const auto &shape2_row_i = shape2[i];
          if (shape1_row_i.empty()) {
            if (shape2_row_i.empty()) {
              continue;  // both rows empty -> skip to next
            } else {
              result.at(i) = shape2_row_i;
            }
          } else {
            if (shape2_row_i.empty()) {
              result.at(i) = shape1_row_i;
            } else {
              // std::merge but only keeps uniques (when shapes have norms will add the norm values)
              // based on https://en.cppreference.com/w/cpp/iterator/inserter
              auto first1 = shape1_row_i.begin();
              auto last1 = shape1_row_i.end();
              auto first2 = shape2_row_i.begin();
              auto last2 = shape2_row_i.end();
              auto d_first = std::inserter(result.at(i), result.at(i).end());
              auto last_value = -1;
              for (; first1 != last1; ++d_first) {
                if (first2 == last2) {
                  std::copy_if(first1, last1, d_first, [last_value](auto v) { return v != last_value; });
                  break;
                }
                if (*first2 < *first1) {
                  if (*first2 != last_value) {
                    *d_first = *first2;
                    last_value = *first2;
                  }
                  ++first2;
                } else {
                  if (*first1 != last_value) {
                    *d_first = *first1;
                    last_value = *first1;
                  }
                  ++first1;
                }
              }
              std::copy_if(first2, last2, d_first, [last_value](auto v) { return v != last_value; });
            }
          }
        }
        return result;
      }

      // N.B. Boost expects version, default to 0
      template <typename Archive>
      void serialize(Archive &ar, const unsigned int = 0) {
        ar &type_ &nrows_ &ncols_ &static_cast<base_t &>(*this);
      }

      void print(std::ostream &os) const {
        os << "Shape: type=";
        switch (type()) {
          case Type::col2row:
            os << "col2row";
            break;
          case Type::row2col:
            os << "row2col";
            break;
          case Type::invalid:
            os << "invalid";
            break;
        }
        os << " { ";
        for (auto i = 0ul; i != size(); ++i) {
          os << i << ":{ ";
          for (const auto j : (*this)[i]) {
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
      template <typename T>
      base_t make_colidx_to_rowidx(const Eigen::SparseMatrix<T> &mat) {
        base_t colidx_to_rowidx;
        for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
          for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
            const std::size_t row = it.row();
            const std::size_t col = it.col();
            if (col >= colidx_to_rowidx.size()) colidx_to_rowidx.resize(col + 1);
            // in either case (col- or row-major) row index increasing for the given col
            colidx_to_rowidx[col].push_back(row);
          }
        }
        return colidx_to_rowidx;
      }

      // result[i][j] gives the j-th nonzero column for row i in matrix mat
      template <typename T>
      base_t make_rowidx_to_colidx(const Eigen::SparseMatrix<T> &mat) {
        base_t rowidx_to_colidx;
        for (int k = 0; k < mat.outerSize(); ++k) {  // cols, if col-major, rows otherwise
          for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
            const std::size_t row = it.row();
            const std::size_t col = it.col();
            if (row >= rowidx_to_colidx.size()) rowidx_to_colidx.resize(row + 1);
            // in either case (col- or row-major) col index increasing for the given row
            rowidx_to_colidx[row].push_back(col);
          }
        }
        return rowidx_to_colidx;
      }
    };

    std::ostream &operator<<(std::ostream &os, const Shape &shape) {
      shape.print(os);
      return os;
    }

    // compute shape of an existing SpMatrix on rank 0
    template <typename Blk = blk_t>
    class ReadShape : public TT<void, std::tuple<Out<void, Shape>>, ReadShape<Blk>, ttg::typelist<void>> {
     public:
      using baseT = typename ReadShape::ttT;
      static constexpr const int owner = 0;  // where data resides

      ReadShape(const char *label, const SpMatrix<Blk> &matrix, Edge<void, void> &in, Edge<void, Shape> &out)
          : baseT(edges(in), edges(out), std::string("read_spmatrix_shape(") + label + ")", {"ctl"},
                  {std::string("shape[") + label + "]"},
                  /* keymap */ []() { return owner; })
          , matrix_(matrix) {}

      void op(std::tuple<Out<void, Shape>> &out) { ::sendv<0>(Shape(matrix_), out); }

     private:
      const SpMatrix<Blk> &matrix_;
    };

    // flow data from an existing SpMatrix on rank 0
    // similar to Read_SpMatrix but uses tasks to read data:
    // - this allows to read some data only (imagine need to do Hadamard product of 2 sparse matrices ... only some
    // blocks will be needed)
    //   but will only be efficient if can do random access (slow with CSC format used by Eigen matrices)
    // - this could be generalized to read efficiently from a distributed data structure
    // Use Read_SpMatrix if need to read all data from a data structure localized on 1 process
    template <typename Blk = blk_t>
    class Read : public TT<Key<2>, std::tuple<Out<Key<2>, Blk>>, Read<Blk>, ttg::typelist<void>> {
     public:
      using baseT = TT<Key<2>, std::tuple<Out<Key<2>, Blk>>, Read<Blk>, void>;
      static constexpr const int owner = 0;  // where data resides

      Read(const char *label, const SpMatrix<Blk> &matrix, Edge<Key<2>, void> &in, Edge<Key<2>, Blk> &out)
          : baseT(edges(in), edges(out), std::string("read_spmatrix(") + label + ")", {"ctl[ij]"},
                  {std::string(label) + "[ij]"},
                  /* keymap */ [](auto key) { return owner; })
          , matrix_(matrix) {}

      void op(const Key<2> &key, std::tuple<Out<Key<2>, Blk>> &out) {
        // random access in CSC format is inefficient, this is only to demonstrate the way to go for hash-based storage
        // for whatever reason coeffRef does not work on a const SpMatrix&
        ::send<0>(key, static_cast<const Blk &>(const_cast<SpMatrix<Blk> &>(matrix_).coeffRef(key[0], key[1])), out);
      }

     private:
      const SpMatrix<Blk> &matrix_;
    };

    // WriteShape commits shape to an existing SpMatrix on rank 0 and sends it on
    // since SpMatrix supports random inserts there is no need to commit the shape into the matrix, other than get the
    // dimensions
    template <typename Blk = blk_t>
    class WriteShape : public TT<void, std::tuple<Out<void, Shape>>, WriteShape<Blk>, ttg::typelist<Shape>> {
     public:
      using baseT = typename WriteShape::ttT;
      static constexpr const int owner = 0;  // where data resides

      WriteShape(const char *label, SpMatrix<Blk> &matrix, Edge<void, Shape> &in, Edge<void, Shape> &out)
          : baseT(edges(in), edges(out), std::string("write_spmatrix_shape(") + label + ")",
                  {std::string("shape_in[") + label + "]"}, {std::string("shape_out[") + label + "]"},
                  /* keymap */ []() { return owner; })
          , matrix_(matrix) {}

      void op(typename baseT::input_values_tuple_type &&ins, std::tuple<Out<void, Shape>> &out) {
        const auto &shape = baseT::template get<0>(ins);
        ::ttg::trace("Resizing ", static_cast<void *>(&matrix_));
        matrix_.resize(shape.nrows(), shape.ncols());
        ::sendv<0>(shape, out);
      }

     private:
      SpMatrix<Blk> &matrix_;
    };

    // flow (move?) data into an existing SpMatrix on rank 0
    template <typename Blk = blk_t>
    class Write : public TT<Key<2>, std::tuple<>, Write<Blk>, Blk, ttg::typelist<void>> {
     public:
      using baseT = typename Write::ttT;

      Write(const char *label, SpMatrix<Blk> &matrix, Edge<Key<2>, Blk> &data_in, Edge<Key<2>, void> &ctl_in)
          : baseT(edges(data_in, ctl_in), edges(), std::string("write_spmatrix(") + label + ")",
                  {std::string(label) + "[ij]", std::string("ctl[ij]")}, {},
                  /* keymap */ [](auto key) { return 0; })
          , matrix_(matrix) {}

      void op(const Key<2> &key, typename baseT::input_values_tuple_type &&elem, std::tuple<> &) {
        std::lock_guard<std::mutex> lock(mtx_);
        ttg::trace("rank =", default_execution_context().rank(),
                   "/ thread_id =", reinterpret_cast<std::uintptr_t>(pthread_self()),
                   "ttg_matrix.h Write_SpMatrix wrote {", key[0], ",", key[1], "} = ", baseT::template get<0>(elem),
                   " in ", static_cast<void *>(&matrix_), " with mutex @", static_cast<void *>(&mtx_), " for object @",
                   static_cast<void *>(this));
        values_.emplace_back(key[0], key[1], baseT::template get<0>(elem));
        ttg::trace("rank =", default_execution_context().rank(),
                   "/ thread_id =", reinterpret_cast<std::uintptr_t>(pthread_self()),
                   "ttg_matrix.h Write::op: ttg_matrix.h matrix_\n", matrix_);
      }

      /// grab completion status as a future<void>
      /// \note cannot be called once this is executable
      const std::shared_future<void> &status() const {
        assert(!this->is_executable());
        if (!completion_status_) {  // if not done yet, register completion work with the world
          auto promise = std::make_shared<std::promise<void>>();
          completion_status_ = std::make_shared<std::shared_future<void>>(promise->get_future());
          ttg_register_status(this->get_world(), std::move(promise));
          ttg_register_callback(this->get_world(), [this]() {
            this->matrix_.setFromTriplets(this->values_.begin(), this->values_.end());
          });
        } else {  // if done already, commit the result
          this->matrix_.setFromTriplets(this->values_.begin(), this->values_.end());
        }
        return *completion_status_.get();
      }

     private:
      std::mutex mtx_;
      SpMatrix<Blk> &matrix_;
      std::vector<SpMatrixTriplet<Blk>> values_;
      mutable std::shared_ptr<std::shared_future<void>> completion_status_;
    };

    // ShapeAdd adds two Shape objects
    class ShapeAdd : public TT<void, std::tuple<Out<void, Shape>>, ShapeAdd, ttg::typelist<Shape, Shape>> {
     public:
      using baseT = typename ShapeAdd::ttT;
      static constexpr const int owner = 0;  // where data resides

      ShapeAdd(Edge<void, Shape> &in1, Edge<void, Shape> &in2, Edge<void, Shape> &out)
          : baseT(edges(in1, in2), edges(out), {}, {}, {},
                  /* keymap */ []() { return owner; }) {}

      void op(typename baseT::input_values_tuple_type &&ins, std::tuple<Out<void, Shape>> &out) {
        ::sendv<0>(Shape::add(baseT::template get<0>(ins), baseT::template get<1>(ins)), out);
      }
    };

    // pushes all blocks given by the shape
    class Push : public TT<void, std::tuple<Out<Key<2>, void>>, Push, ttg::typelist<Shape>> {
     public:
      using baseT = typename Push::ttT;
      static constexpr const int owner = 0;  // where data resides

      Push(const char *label, Edge<void, Shape> &in, Edge<Key<2>, void> &out)
          : baseT(edges(in), edges(out), std::string("push_spmatrix(") + label + ")",
                  {std::string("shape[") + label + "]"}, {"ctl[ij]"},
                  /* keymap */ []() { return owner; }) {}

      void op(typename baseT::input_values_tuple_type &&ins, std::tuple<Out<Key<2>, void>> &out) {
        const auto &shape = baseT::get<0>(ins);
        if (shape.type() == Shape::Type::col2row) {
          long colidx = 0;
          for (const auto &col : shape) {
            for (const auto rowidx : col) {
              ::sendk<0>(Key<2>({rowidx, colidx}), out);
            }
            ++colidx;
          }
        } else if (shape.type() == Shape::Type::row2col) {
          long rowidx = 0;
          for (const auto &row : shape) {
            for (const auto colidx : row) {
              ::sendk<0>(Key<2>({rowidx, colidx}), out);
            }
            ++rowidx;
          }
        } else
          throw std::logic_error("Push received Shape with invalid type");
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

    Matrix(shape_edge_t &&shape_edge, ctl_edge_t &&ctl_edge, data_edge_t &&data_edge)
        : data_edge_(std::move(data_edge)), shape_edge_(std::move(shape_edge)), ctl_edge_(std::move(ctl_edge)) {}

    auto &data() { return data_edge_; }
    auto &shape() { return shape_edge_; }
    auto &ctl() { return ctl_edge_; }

    /// attach to the source sparse Eigen::Matrix
    void operator<<(const Eigen::SparseMatrix<T> &source_matrix) {
      // shape reader computes shape of source_matrix
      ttg_register_ptr(world_, std::make_shared<matrix::ReadShape<T>>("Matrix.ReadShape", source_matrix,
                                                                      ttg_ctl_edge(world_), shape_edge_));
      // reads data from source_matrix_ for a given key
      ttg_register_ptr(world_, std::make_shared<matrix::Read<T>>("Matrix.Read", source_matrix, ctl_edge_, data_edge_));
    }

    /// pushes all data that exists
    void pushall() {
      // reads the shape and pulls all the data
      ttg_register_ptr(world_, std::make_shared<matrix::Push>("Matrix.Push1", shape_edge_, ctl_edge_));
    }

    /// @return an std::future<void> object indicating the status; @c destination_matrix is ready if calling has_value()
    /// on the return value of this function is true.
    /// @note up to the user to ensure completion before reading destination_matrix
    auto operator>>(SpMatrix<T> &destination_matrix) {
      // shape writer writes shape to destination_matrix
      ttg_register_ptr(world_, std::make_shared<matrix::WriteShape<T>>("Matrix.WriteShape", destination_matrix,
                                                                       shape_edge_, shape_writer_push_edge_));
      // this converts shape to control messages to ensure that shape and data flows are consistent (i.e. if shape says
      // there should be a block {r,c} Write will expect the data for it)
      // TODO if pushall had been called ctl_edge_ is already live, hence can just attach to it
      ctl_edge_t ctl_edge;
      if (!ctl_edge_.live())
        ttg_register_ptr(world_, std::make_shared<matrix::Push>("Matrix.Push2", shape_writer_push_edge_, ctl_edge));
      auto result = std::make_shared<matrix::Write<T>>("Matrix.Write", destination_matrix, data_edge_,
                                                       (ctl_edge_.live() ? ctl_edge_ : ctl_edge));
      ttg_register_ptr(world_, result);

      // return op status ... set to true after world fence
      return result->status();
    }

   private:
    data_edge_t data_edge_{"data_edge_"};
    shape_edge_t shape_edge_{"shape_edge_"};
    ctl_edge_t ctl_edge_{"ctl_edge_"};

    /// this is used internally for pushing shape to the writer tasks
    shape_edge_t shape_writer_push_edge_{"shape_writer_push_edge_"};
    World world_ = ttg::default_execution_context();
  };

  template <typename T>
  Matrix<T> operator+(const Matrix<T> &a, const Matrix<T> &b) {
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

#endif  // TTG_MATRIX_H
