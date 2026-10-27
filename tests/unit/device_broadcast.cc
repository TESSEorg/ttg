// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include <tuple>
#include <vector>

#include "ttg.h"

#ifdef TTG_HAVE_COROUTINE

// Compile-time coverage of ttg::device::broadcast/broadcastk/broadcast_keygen keylist handling.
// Every keylist shape (lvalue/rvalue keylist, lvalue/rvalue tuple of keylists,
// explicit/implicit terminals) must instantiate. The coroutines are never
// resumed, so this test only has to compile.
namespace {

  using out_vals_t = std::tuple<ttg::Out<int, int>, ttg::Out<int, int>>;
  using out_keys_t = std::tuple<ttg::Out<int, void>, ttg::Out<int, void>>;

  ttg::device::Task broadcast_value(out_vals_t &outs, int value) {
    std::vector<int> keys1{1, 2};
    std::vector<int> keys2{3, 4};
    auto keylists = std::make_tuple(keys1, keys2);
    // lvalue tuple of keylists: one keylist per terminal, captured by reference
    co_await ttg::device::forward(ttg::device::broadcast<0, 1>(keylists, value, outs));
    co_await ttg::device::forward(ttg::device::broadcast<0, 1>(keylists, value));
    // rvalue tuple of keylists: moved into the coroutine frame
    co_await ttg::device::forward(ttg::device::broadcast<0, 1>(std::make_tuple(keys1, keys2), value, outs));
    co_await ttg::device::forward(ttg::device::broadcast<0, 1>(std::make_tuple(keys1, keys2), value));
    // single lvalue keylist
    co_await ttg::device::forward(ttg::device::broadcast<0>(keys1, value, outs));
    co_await ttg::device::forward(ttg::device::broadcast<0>(keys1, value));
    // single rvalue keylist
    co_await ttg::device::forward(ttg::device::broadcast<0>(std::vector<int>{5, 6}, value, outs));
    co_await ttg::device::forward(ttg::device::broadcast<0>(std::vector<int>{5, 6}, value));
  }

  ttg::device::Task broadcast_key_only(out_keys_t &outs) {
    std::vector<int> keys1{1, 2};
    std::vector<int> keys2{3, 4};
    auto keylists = std::make_tuple(keys1, keys2);
    co_await ttg::device::forward(ttg::device::broadcastk<0, 1>(keylists, outs));
    co_await ttg::device::forward(ttg::device::broadcastk<0, 1>(keylists));
    co_await ttg::device::forward(ttg::device::broadcastk<0, 1>(std::make_tuple(keys1, keys2), outs));
    co_await ttg::device::forward(ttg::device::broadcastk<0, 1>(std::make_tuple(keys1, keys2)));
    co_await ttg::device::forward(ttg::device::broadcastk<0>(keys1, outs));
    co_await ttg::device::forward(ttg::device::broadcastk<0>(keys1));
    co_await ttg::device::forward(ttg::device::broadcastk<0>(std::vector<int>{5}, outs));
    co_await ttg::device::forward(ttg::device::broadcastk<0>(std::vector<int>{5}));
  }

  ttg::device::Task broadcast_keygen_value(const int &key, int value) {
    // temporary key: copied into the coroutine frame before the coroutine suspends
    co_await ttg::device::forward(ttg::device::broadcast_keygen(key + 1, value));
    // lvalue key and rvalue value
    co_await ttg::device::forward(ttg::device::broadcast_keygen(key, std::move(value)));
  }

}  // namespace

TEST_CASE("device-broadcast-keylists", "[device][broadcast][compile-only]") {
  // the coroutines above are instantiated by the compiler; they must never be invoked here
  static_cast<void>(&broadcast_value);
  static_cast<void>(&broadcast_key_only);
  static_cast<void>(&broadcast_keygen_value);
  SUCCEED("ttg::device::broadcast/broadcastk/broadcast_keygen instantiate for all keylist shapes");
}

#endif  // TTG_HAVE_COROUTINE
