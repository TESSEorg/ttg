// SPDX-License-Identifier: BSD-3-Clause
//
// Created by Eduard Valeyev on 12/22/17.
//

#ifndef TTG_REDUCE_H
#define TTG_REDUCE_H

#include <cassert>
#include <cstdlib>
#include <mutex>

#include "ttg/util/tree.h"

namespace ttg {

  /// @brief generic binary reduction of a set of key-value pairs.
  ///
  /// This reduces a set of Value objects keyed by an integer in the @c [0,max_key) interval using BinaryOp @c op .
  /// The reduction order is determined by breadth-first traversal of a
  /// binary spanning tree of the complete graph of the @c [0,max_key) set (see BinarySpanningTree)
  /// rooted at a particular key; at each node @c Node the reduction is performed
  /// as @c op(op(LeftSubTree,Node),RightSubTree) .
  /// The primary use is for reducing over a World, hence by default the keymap is identity (keymap(key) = key) and
  /// @c max_key=world.size() . The result is associated with output key @c dest_key .
  ///
  /// @note this is equivalent to MPI_Reduce; unlike std::reduce this lacks the initializer value.
  ///
  template <typename Value, typename BinaryOp, typename OutKey>
  class BinaryTreeReduce
      : public TT<int, std::tuple<Out<int, Value>, Out<int, Value>, Out<int, Value>, Out<OutKey, Value>>,
                  BinaryTreeReduce<Value, BinaryOp, OutKey>, ttg::typelist<Value, Value, Value>> {
   public:
    using baseT = typename BinaryTreeReduce::ttT;

    BinaryTreeReduce(Edge<int, Value> &in, Edge<OutKey, Value> &out, int root = 0, OutKey dest_key = OutKey(),
                     BinaryOp op = BinaryOp{}, World world = ttg::default_execution_context(), int max_key = -1,
                     Edge<int, Value> inout = Edge<int, Value>{}, Edge<int, Value> inout_l = Edge<int, Value>{},
                     Edge<int, Value> inout_r = Edge<int, Value>{})
        : baseT(edges(fuse(in, inout), inout_l, inout_r), edges(inout, inout_l, inout_r, out), "BinaryTreeReduce",
                {"in|inout", "inout_l", "inout_r"}, {"inout", "inout_l", "inout_r", "out"}, world, [](int key) { return key; })
        , tree_((max_key == -1 ? world.size() : max_key), root)
        , dest_key_(dest_key)
        , op_(std::move(op)) {
      init();
    }

    void op(const int &key, typename baseT::input_values_tuple_type &&indata,
            std::tuple<Out<int, Value>, Out<int, Value>, Out<int, Value>, Out<OutKey, Value>> &outdata) {
      assert(key < tree_.size());
      assert(key == this->get_world().rank());
      /// skip stub values ... won't need this ugliness when streaming is implemented
      auto children = tree_.child_keys(key);
      Value result;
      if (children.first != -1 && children.second != -1)
        // left-associative in-order reduction: L op This op R = ((L op This) op R)
        result = op_(op_(baseT::template get<1, Value &&>(indata), baseT::template get<0, Value &&>(indata)),
                     baseT::template get<2, Value &&>(indata));
      else {
        if (children.first != -1)
          result = op_(baseT::template get<1, Value &&>(indata), baseT::template get<0, Value &&>(indata));
        else if (children.second != -1)
          result = op_(baseT::template get<0, Value &&>(indata), baseT::template get<2, Value &&>(indata));
        else
          result = baseT::template get<0, Value &&>(indata);
      }
      auto parent = tree_.parent_key(key);
      if (parent != -1) {
        // is this left or right child of the parent?
        bool this_is_left_child;
        {
          auto parents_children = tree_.child_keys(parent);
          assert(parents_children.first == key || parents_children.second == key);
          this_is_left_child = (parents_children.first == key);
        }
        if (this_is_left_child)
          send<1>(parent, std::move(result), outdata);
        else
          send<2>(parent, std::move(result), outdata);
      } else
        send<3>(dest_key_, std::move(result), outdata);
    }

   private:
    BinarySpanningTree tree_;
    OutKey dest_key_;
    BinaryOp op_;

    /// since the # of arguments is constexpr in current TTG, some reductions will use stub values, initialize them here
    /// TODO this will initialize stream bounds when TTG supports streaming
    void init() {
      // iterate over keys that map to me ... if keys are equivalent to ranks this can be made simpler
      const auto my_rank = this->get_world().rank();
      for (auto key = 0; key != tree_.size(); ++key) {
        if (my_rank == this->get_keymap()(key)) {
          auto keys = tree_.child_keys(key);
          if (keys.first == -1) this->template set_arg<1>(key, Value());
          if (keys.second == -1) this->template set_arg<2>(key, Value());
        }
      }
    }
  };

#if 0
/// @brief generic reduction operation
///
/// This reduces a set of {Key,Value} pairs using Reducer
///
template <typename InKey, template Value, template Reducer, template OutKey>
class Reduce : public TT<InKey, std::tuple<Out<OutKey, Value>>, Reduce<InKey, Value, Reducer, OutKey>, Value> {
 public:
  using baseT = TT<InKey, std::tuple<Out<OutKey, Value>>, Reduce<InKey, Value, Reducer, OutKey>, Value>;

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

}  // namespace ttg

#endif  // TTG_REDUCE_H
