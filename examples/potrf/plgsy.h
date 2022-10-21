#pragma once

#include <ttg.h>
#include "pmw.h"
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>
#include "core_plgsy.h"

template <typename T>
auto make_plgsy(MatrixT<T>& A, unsigned long bump, unsigned long random_seed, ttg::Edge<Key2, void>& input, ttg::Edge<Key2, MatrixTile<T>>& output) {
  auto f = [=](const Key2& key, std::tuple< ttg::Out<Key2, MatrixTile<T>> >& out) {
    /* write back any tiles that are not in the matrix already */
    const int I = key.I;
    const int J = key.J;
    if(ttg::tracing()) ttg::print("PLGSY( ", key, ") on rank ", A.rank_of(key.I, key.J));
    assert(A.is_local(I, J));

    T *a = A(I, J).data();
    int tempmm, tempnn, ldam;

    tempmm = (I==A.rows()-1) ? A.rows_in_matrix()-I*A.rows_in_tile() : A.rows_in_tile();
    tempnn = (J==A.cols()-1) ? A.cols_in_matrix()-J*A.cols_in_tile() : A.cols_in_tile();
    ldam   = A.rows_in_tile();

    CORE_plgsy((double)bump, tempmm, tempnn, a, ldam,
               A.rows_in_matrix(), I*A.rows_in_tile(), J*A.cols_in_tile(), random_seed);

    ttg::send<0>(key, std::move(A(I, J)), out);
  };

  return ttg::make_tt(f, ttg::edges(input), ttg::edges(output), "PLGSY", {"startup"}, {"output"});
}

auto make_plgsy_ttg(MatrixT<double> &A, unsigned long bump, unsigned long random_seed, ttg::Edge<Key2, void>& startup, ttg::Edge<Key2, MatrixTile<double>>&result, bool defer_write) {
  auto keymap2 = [&](const Key2& key) {
    return A.rank_of(key.I, key.J);
  };
  auto plgsy_tt = make_plgsy(A, bump, random_seed, startup, result);
  plgsy_tt->set_keymap(keymap2);
  plgsy_tt->set_defer_writer(defer_write);

  auto ins = std::make_tuple(plgsy_tt->template in<0>());
  auto outs = std::make_tuple(plgsy_tt->template out<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops(1);
  ops[0] = std::move(plgsy_tt);

  return make_ttg(std::move(ops), ins, outs, "PLGSY TTG");
}
