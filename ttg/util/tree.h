//
// Created by Eduard Valeyev on 12/29/17.
//

#ifndef TTG_TREE_H
#define TTG_TREE_H

#include <cassert>
#include <utility>

namespace ttg {

  /// @brief a binary spanning tree of integers in the @c [0,size) interval
  ///
  /// This is a binary spanning tree of the complete graph of the @c [0,size) set of <em>keys</em>,
  /// rooted at a particular key.
  class BinarySpanningTree {
   public:
    BinarySpanningTree(int size, int root) : size_(size), root_(root) {
      assert(root >= 0 && root < size);
      assert(size >= 0);
    }
    ~BinarySpanningTree() = default;

    /// @return the size of the tree
    const auto size() const { return size_; }
    /// @return the root of the tree
    const auto root() const { return root_; }

    /// @param[in] child_key the key of the child
    /// @return the parent key (-1 if there is no parent)
    int parent_key(const int child_key) const {
      const auto child_rank = (child_key + size_ - root_) % size_;  // cyclically shifted key such that root's key is 0
      const auto parent_key =
          (child_rank == 0 ? -1 : (((child_rank - 1) >> 1) + root_) % size_);  // Parent's key in binary tree
      return parent_key;
    }
    /// @param[in] parent_key the key of the parent
    /// @return the pair of child keys (-1 if there is no child)
    std::pair<int, int> child_keys(const int parent_key) const {
      const auto parent_rank =
          (parent_key + size_ - root_) % size_;     // cyclically shifted key such that root's key is 0
      int child0 = (parent_rank << 1) + 1 + root_;  // Left child
      int child1 = child0 + 1;                      // Right child
      const int size_plus_root = size_ + root_;
      if (child0 < size_plus_root)
        child0 %= size_;
      else
        child0 = -1;
      if (child1 < size_plus_root)
        child1 %= size_;
      else
        child1 = -1;
      return std::make_pair(child0, child1);
    }

   private:
    int size_;
    int root_;
  };

}  // namespace ttg

#endif  // TTG_TREE_H
