#ifndef TTG_PARSEC_VARS_H
#define TTG_PARSEC_VARS_H

#include <mutex>
#include <map>
#include <parsec.h>

#include "ttg/base/tt.h"

#include "ttg/parsec/task.h"

namespace ttg_parsec {
  namespace detail {

    inline thread_local parsec_execution_stream_t *parsec_ttg_es;

    inline thread_local detail::parsec_ttg_task_base_t *parsec_ttg_caller;

  } // namespace detail
} // namespace ttg_parsec

#endif // TTG_PARSEC_VARS_H
