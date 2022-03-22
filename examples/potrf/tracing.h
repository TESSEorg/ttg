#pragma once

#include <parsec/profiling.h>

static thread_local parsec_profiling_stream_t *prof = nullptr;
static bool profiling_enabled = false;

static void init_prof_thread()
{
#if USE_PARSEC_PROF_API
  if (nullptr == prof) {
    prof = parsec_profiling_stream_init(4096, "PaRSEC thread");
  }
#endif // USE_PARSEC_PROF_API
}

static void
dplasma_dprint_tile( int m, int n,
                     const parsec_tiled_matrix_dc_t* descA,
                     const double *M );
