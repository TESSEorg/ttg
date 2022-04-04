#pragma once

#include <ttg.h>
// needed for madness::hashT and xterm_debug
#include <madness/world/world.h>
#include "pmw.h"
#include "lapack.hh"

#include "trtri.h"
#include "lauum.h"

namespace potri {

/* FLOP macros taken from DPLASMA */
#define FMULS_POTRI(__n) ((double)(__n) * (((1. / 6.) * (double)(__n) + 0.5) * (double)(__n) + (1. / 3.)))
#define FADDS_POTRI(__n) ((double)(__n) * (((1. / 6.) * (double)(__n)      ) * (double)(__n) - (1. / 6.)))
#define FLOPS_DPOTRI(__n) (     FMULS_POTRI((__n)) +       FADDS_POTRI((__n)) )

auto make_potri_ttg(MatrixT<double> &A, ttg::Edge<Key2, MatrixTile<double>>&input, ttg::Edge<Key2, MatrixTile<double>>&output ) {
  ttg::Edge<Key2, MatrixTile<double>> trtri_to_lauum("trtri_to_lauum");

  auto ttg_trtri = trtri::make_trtri_ttg(A, lapack::Diag::NonUnit, input, trtri_to_lauum);
  auto ttg_lauum = lauum::make_lauum_ttg(A, trtri_to_lauum, output);

  std::vector<std::unique_ptr<ttg::TTBase>> ops(2);
  auto ins = std::make_tuple(ttg_trtri->template in<0>());
  auto outs = std::make_tuple(ttg_lauum->template out<0>());
  ops[0] = std::move(ttg_trtri);
  ops[1] = std::move(ttg_lauum);

  return make_ttg(std::move(ops), ins, outs, "POTRI TTG");
}

};