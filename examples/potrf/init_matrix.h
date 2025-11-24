// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <ttg.h>
#include "pmw.h"
#include "plgsy.h"
#include "result.h"


/**
 * Initialize a matrix using PLGSY.
 */
template<typename T>
void init_matrix(MatrixT<T>& A, int random_seed, bool cow_hint = false) {

  auto world = ttg::default_execution_context();
  int N = A.rows_in_matrix();
  ttg::Edge<Key2, MatrixTile<double>> startup("startup");
  ttg::Edge<Key2, MatrixTile<double>> plgsy_result("To PLGSY result");
  auto plgsy_init_tt = make_load_tt(A, startup, cow_hint);
  auto plgsy_ttg = make_plgsy_ttg(A, N, random_seed, startup, plgsy_result, cow_hint);
  auto plgsy_result_ttg = make_result_ttg(A, plgsy_result, cow_hint);

  auto connected = make_graph_executable(plgsy_init_tt.get());
  assert(connected);
  TTGUNUSED(connected);

  plgsy_init_tt->invoke();
  ttg::execute(world);
  ttg::fence(world);
}