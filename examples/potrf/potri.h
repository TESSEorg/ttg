#pragma once

#include <ttg.h>
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>
#include "pmw.h"
#include "lapack.hh"

#include "trtri_L.h"
#include "lauum.h"

namespace potri {

/* FLOP macros taken from DPLASMA */ 
double FMULS_POTRI(double __n) { return ( __n * ((2. / 3.) + __n * ((1. / 3.) * __n + 1. )) ); }
double FADDS_POTRI(double __n) { return ( __n * ((1. / 6.) + __n * ((1. / 3.) * __n - 0.5)) ); }
double FLOPS_DPOTRI(double __n) { return FMULS_POTRI(__n) + FADDS_POTRI(__n); }

auto make_potri_ttg(MatrixT<double> &A, ttg::Edge<Key2, MatrixTile<double>>&input, ttg::Edge<Key2, MatrixTile<double>>&output, bool defer_write ) {
  ttg::Edge<Key2, MatrixTile<double>> trtri_to_lauum("trtri_to_lauum");

  auto ttg_trtri = trtri_LOWER::make_trtri_ttg(A, lapack::Diag::NonUnit, input, trtri_to_lauum, defer_write);
  auto ttg_lauum = lauum::make_lauum_ttg(A, trtri_to_lauum, output, defer_write);

  auto ins = std::make_tuple(ttg_trtri->template in<0>());
  auto outs = std::make_tuple(ttg_lauum->template out<0>());
  std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
  ops[0] = std::move(ttg_trtri);
  ops[1] = std::move(ttg_lauum);

  return make_ttg(std::move(ops), ins, outs, "POTRI TTG");
}

};