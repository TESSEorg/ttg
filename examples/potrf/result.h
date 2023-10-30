#pragma once

#include <ttg.h>
#include "pmw.h"
#include "util.h"

template <typename T>
auto make_result(MatrixT<T>& A, ttg::Edge<Key2, MatrixTile<T>>& result) {
  auto f = [=](const Key2& key, MatrixTile<T>&& tile, std::tuple<>& out) {
    /* write back any tiles that are not in the matrix already */
    const int I = key[0];
    const int J = key[1];
    if (ttg::tracing()) ttg::print("RESULT( ", key, ") on rank ", A.rank_of(key[0], key[1]));
#if defined(DEBUG_TILES_VALUES)
    T norm = blas::nrm2(tile.size(), tile.data(), 1);
    assert(check_norm(norm, tile.norm()));
#endif //defined(DEBUG_TILES_VALUES)
    auto atile = A(I, J);
    if (atile.data() != tile.data()) {
      if (ttg::tracing()) ttg::print("Writing back tile {", I, ",", J, "} ");
      std::copy_n(tile.data(), tile.rows() * tile.cols(), atile.data());
    }
#ifdef TTG_USE_USER_TERMDET
    if (I == A.cols() - 1 && J == A.rows() - 1) {
      ttg::default_execution_context().impl().final_task();
    }
#endif  // TTG_USE_USER_TERMDET
  };

  return ttg::make_tt(f, ttg::edges(result), ttg::edges(), "Final Output", {"result"}, {});
}

auto make_result_ttg(MatrixT<double>& A, ttg::Edge<Key2, MatrixTile<double>>& result, bool defer_write) {
  auto keymap2 = [&](const Key2& key) { return A.rank_of(key[0], key[1]); };
  auto result_tt = make_result(A, result);
  result_tt->set_keymap(keymap2);
  result_tt->set_defer_writer(defer_write);

  auto ins = std::make_tuple(result_tt->template in<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops(1);
  ops[0] = std::move(result_tt);

  return make_ttg(std::move(ops), ins, std::make_tuple(), "Result Writer");
}
