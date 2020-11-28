//
// Created by Eduard Valeyev on 12/29/17.
//

#ifndef TTG_BROADCAST_H
#define TTG_BROADCAST_H

#include <tuple>

#include "../../ttg/util/tree.h"

namespace ttg {

  /// @brief generic binary broadcast of a value to a set of {key,value} pairs
  ///
  /// This broadcasts a Value object through a binary tree of size @c max_key and at each node broadcasts
  /// the value to a set of keys of type @c OutKey . The input data is keyed by integers.
  /// The primary use is for broadcasting to a World, hence by default the keymap is identity (keymap(key) = key) and
  /// @c max_key=world.size() .
  ///
  /// @note this is equivalent to MPI_Bcast.
  ///
  template <typename Value, typename OutKey = int>
  class BinaryTreeBroadcast : public Op<int, std::tuple<Out<int, Value>, Out<int, Value>, Out<OutKey, Value>>,
                                        BinaryTreeBroadcast<Value, OutKey>, Value> {
   public:
    using baseT = Op<int, std::tuple<Out<int, Value>, Out<int, Value>, Out<OutKey, Value>>,
                     BinaryTreeBroadcast<Value, OutKey>, Value>;

    BinaryTreeBroadcast(Edge<int, Value> &in, Edge<OutKey, Value> &out, std::vector<OutKey> local_keys, int root = 0,
                        World &world = ttg_default_execution_context(), int max_key = -1,
                        Edge<int, Value> inout_l = Edge<int, Value>{}, Edge<int, Value> inout_r = Edge<int, Value>{})
        : baseT(edges(fuse(in, inout_l, inout_r)), edges(inout_l, inout_r, out), "BinaryTreeBroadcast",
                {"in|inout_l|inout_r"}, {"inout_l", "inout_r", "out"}, world, [](int key) { return key; })
        , tree_((max_key == -1 ? world.size() : max_key), root)
        , local_keys_(std::move(local_keys)) {}

    void op(const int &key, typename baseT::input_values_tuple_type &&indata,
            std::tuple<Out<int, Value>, Out<int, Value>, Out<int, Value>> &outdata) {
      assert(key < tree_.size());
      assert(key == this->get_world().rank());
      auto children = tree_.child_keys(key);
      if (children.first != -1) std::get<0>(outdata).send(children.first, this->template get<0, const Value &>(indata));
      if (children.second != -1) std::get<1>(outdata).send(children.second, this->template get<0, const Value &>(indata));
      broadcast<2>(local_keys_, this->template get<0, const Value &>(indata), outdata);
    }

   private:
    BinarySpanningTree tree_;
    std::vector<OutKey> local_keys_;
  };

}  // namespace ttg

#endif  // TTG_BROADCAST_H
