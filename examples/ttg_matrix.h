//
// Created by Eduard Valeyev on 5/25/18.
//

#ifndef TTG_MATRIX_H
#define TTG_MATRIX_H

namespace ttg {

#if 1
  /// Sparse matrix flow
  /// @tparam T element type
  template <typename T>
  class MatrixFlow {
   public:
    auto& edge() { return edge_; }

    /// attach to the source sparse Eigen::Matrix
    MatrixFlow& operator<<(const Eigen::SparseMatrix<T>& source_matrix) {
      // variant 1: create an op for reading data ... need explicit control flow
      // need to attach to a graph to manage lifetime
      ttg_register_ptr(world_, std::make_shared<Read_SpMatrix<T>>("", source_matrix, ttg_ctl_edge(world_), edge_));

      return *this;
    }

    // up to the user to ensure completion before reading destination_matrix
    std::future<void> operator>>(SpMatrix<T>& destination_matrix) {
      auto result = std::make_shared<Write_SpMatrix<T>>(destination_matrix, edge_);
      ttg_register_ptr(world_, result);

      return result->status();
    }

#if 0  // need completion detection
    operator std::future<std::shared_ptr<SpMatrix<T>>>() {
      SpMatrix<T> destination_matrix;
      ttg_register_ptr(world_, std::make_shared<Write_SpMatrix<T>>(destination_matrix, edge_));

      return std::future<SpMatrix<T>>(destination_matrix);
    }
#endif

   private:
    Edge<Key<2>, T> edge_{};
    World& world_ = ttg_default_execution_context();
  };

#endif

}  // namespace ttg

#endif //TTG_MATRIX_H
