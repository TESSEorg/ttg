//
// Created by Eduard Valeyev on 12/22/17.
//

#ifndef TTG_REDUCE_H
#define TTG_REDUCE_H

#include <cassert>
#include <cstdlib>
#include <mutex>

/// @brief generic binary tree reduction operation over key-value pairs.
///
/// This reduces a set of {Key,Value} pairs using BinaryOp
/// This reduces one value for each key, using a binary spanning tree rooted at a particular key.
/// The primary use is for reducing over a World, hence the default keymap is identity (keymap(key) = key).
///
/// @note this is equivalent to MPI_Reduce; see also std::reduce for more info
///
template <typename Value, typename BinaryOp, typename OutKey>
class BinaryTreeReduce
    : public Op<int, std::tuple<Out<int, Value>, Out<int, Value>, Out<int, Value>, Out<OutKey, Value>>,
                BinaryTreeReduce<Value, BinaryOp, OutKey>, Value, Value, Value> {
 public:
  using baseT = Op<int, std::tuple<Out<int, Value>, Out<int, Value>, Out<int, Value>, Out<OutKey, Value>>,
                   BinaryTreeReduce<Value, BinaryOp, OutKey>, Value, Value, Value>;

  BinaryTreeReduce(Edge<int, Value> &in, Edge<OutKey, Value> &out, int root = 0, OutKey dest_key = OutKey(),
                   BinaryOp op = BinaryOp{}, World &world = ttg_default_execution_context(),
                   Edge<int, Value> inout = Edge<int, Value>{}, Edge<int, Value> inout_l = Edge<int, Value>{}, Edge<int, Value> inout_r = Edge<int, Value>{})
      : baseT(edges(fuse(in, inout), inout_l, inout_r), edges(inout, inout_l, inout_r, out), "BinaryTreeReduce",
              {"in+inout", "inout_l", "inout_r"}, {"inout", "inout_l", "inout_r", "out"}, world)
      , max_key_(world.size())
      , root_(root)
      , dest_key_(dest_key)
      , op_(std::move(op)) {
    assert(root_ < max_key_);
    init();
  }

  void op(const int &key, typename baseT::input_values_tuple_type &&indata,
          std::tuple<Out<int, Value>, Out<int, Value>, Out<int, Value>, Out<OutKey, Value>> &outdata) {
    assert(key < max_key_);
    /// skip stub values ... won't need this ugliness when streaming is implemented
    const auto my_key = this->get_world().rank();
    auto children = child_keys(my_key);
    Value result;
    if (children.first != -1 && children.second != -1)
      // L op This op R
      result = op_(op_(baseT::template get<0, Value &&>(indata), baseT::template get<1, Value &&>(indata)),baseT::template get<2, Value &&>(indata));
    else {
      if (children.first != -1)
        result = op_(baseT::template get<0, Value &&>(indata), baseT::template get<1, Value &&>(indata));
      else if (children.second != -1)
        result = op_(baseT::template get<0, Value &&>(indata), baseT::template get<2, Value &&>(indata));
      else
        result = baseT::template get<0, Value &&>(indata);
    }
    auto parent = parent_key(my_key);
    if (parent != -1) {
      // is this left or right child of the parent?
      bool this_is_left_child;
      {
        auto parents_children = child_keys(parent);
        assert(parents_children.first == my_key || parents_children.second == my_key);
        this_is_left_child = (parents_children.first == my_key);
      }
      if (this_is_left_child)
        send<1>(parent, std::move(result), outdata);
      else
        send<2>(parent, std::move(result), outdata);
    }
    else
      send<3>(dest_key_, std::move(result), outdata);
  }

 private:
  int max_key_;
  int root_;
  OutKey dest_key_;
  BinaryOp op_;

  /// since the # of arguments is constexpr in current TTG, some reductions will use stub values, initialize them here
  /// TODO this will initialize stream bounds when TTG supports streaming
  void init() {
    const auto my_key = this->get_world().rank();
    auto keys = child_keys(my_key);
    if (keys.first == -1) this->template set_arg<1>(my_key, Value());
    if (keys.second == -1) this->template set_arg<2>(my_key, Value());
  }

  /// @param[in] child_key the key of the child
  /// @return the parent key (-1 if there is no parent)
  int parent_key(const int child_key) const {
    const auto child_rank =
        (child_key + max_key_ - root_) % max_key_;  // cyclically shifted key such that root's key is 0
    const auto parent_key =
        (child_rank == 0 ? -1 : (((child_rank - 1) >> 1) + root_) % max_key_);  // Parent's key in binary tree
    return parent_key;
  }
  /// @param[in] parent_key the key of the parent
  /// @return the pair of child keys (-1 if there is no child)
  std::pair<int, int> child_keys(const int parent_key) const {
    const auto parent_rank =
        (parent_key + max_key_ - root_) % max_key_;  // cyclically shifted key such that root's key is 0
    int child0 = (parent_rank << 1) + 1 + root_;     // Left child
    int child1 = child0 + 1;                         // Right child
    const int max_key_plus_root = max_key_ + root_;
    if (child0 < max_key_plus_root)
      child0 %= max_key_;
    else
      child0 = -1;
    if (child1 < max_key_plus_root)
      child1 %= max_key_;
    else
      child1 = -1;
    return std::make_pair(child0, child1);
  }
};

#if 0
/// @brief generic reduction operation
///
/// This reduces a set of {Key,Value} pairs using Reducer
///
template <typename InKey, template Value, template Reducer, template OutKey>
class Reduce : public Op<InKey, std::tuple<Out<OutKey, Value>>, Reduce<InKey, Value, Reducer, OutKey>, Value> {
 public:
  using baseT = Op<InKey, std::tuple<Out<OutKey, Value>>, Reduce<InKey, Value, Reducer, OutKey>, Value>;

  Reduce(Edge<InKey, Value> &in, Edge<OutKey, Value> &out, OutKey dest_key = OutKey(), std::size_t nitems = 1,
         World &world = default_execution_context())
      : baseT(edges(in), edges(out, Edge<OutKey, Value>("reduce")), "Reduce", {"in"}, {"out"}, world)
      , dest_key_(dest_key)
      , nitems_(nitems) {}

  void op(const InKey &key, baseT::input_values_tuple_type &&indata, std::tuple<Out<OutKey, Value>> &outdata) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nitems_) {
      reducer_(value_, baseT::get<0>(indata));
      --nitems_;
    }
    if (nitems_ == 0) {
      binary_tree_reduce_.set_arg<0>(world.rank(), std::move(value_));
    }
  }

 private:
  OutKey dest_key_;
  size_t nitems_;
  std::mutex mutex_;
  Value value_;
  Reducer reducer_;
};  // class Reduce
#endif

#endif  // TTG_REDUCE_H
