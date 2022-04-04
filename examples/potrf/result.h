#pragma once

#include <ttg.h>
#include "pmw.h"
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>

template <typename T>
auto make_result(MatrixT<T>& A, ttg::Edge<Key2, MatrixTile<T>>& result) {
  auto f = [=](const Key2& key, MatrixTile<T>&& tile, std::tuple<>& out) {
    /* write back any tiles that are not in the matrix already */
    const int I = key.I;
    const int J = key.J;
    if(ttg::tracing()) ttg::print("RESULT( ", key, ") on rank ", A.rank_of(key.I, key.J));
    if (A(I, J).data() != tile.data()) {
      //if(ttg::tracing()) ttg::print("Writing back tile {" << I << ", " << J << "} " << std::endl;
      std::copy_n(tile.data(), tile.rows()*tile.cols(), A(I, J).data());
    }
#ifdef TTG_USE_USER_TERMDET
    if (I == A.cols()-1 && J == A.rows()-1) {
      ttg::default_execution_context().impl().final_task();
    }
#endif // TTG_USE_USER_TERMDET
  };

  return ttg::make_tt(f, ttg::edges(result), ttg::edges(), "Final Output", {"result"}, {});
}

auto make_result_ttg(MatrixT<double> &A, ttg::Edge<Key2, MatrixTile<double>>&result) {
  auto keymap2 = [&](const Key2& key) {
    return A.rank_of(key.I, key.J);
  };
  auto result_tt = make_result(A, result);
  result_tt->set_keymap(keymap2);

  std::vector<std::unique_ptr<ttg::TTBase>> ops(1);
  ops[0] = std::move(result_tt);
  auto ins = std::make_tuple(result_tt->template in<0>());

  return make_ttg(std::move(ops), ins, std::make_tuple(), "Result Writer");
}
