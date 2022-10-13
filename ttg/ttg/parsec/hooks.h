#ifndef TTG_PARSEC_HOOKS_H
#define TTG_PARSEC_HOOKS_H

#include <parsec.h>

#include "ttg/parsec/vars.h"

namespace ttg_parsec {

  namespace detail {

    inline parsec_hook_return_t hook(struct parsec_execution_stream_s *es, parsec_task_t *parsec_task) {
      parsec_execution_stream_t *safe_es = parsec_ttg_es;
      parsec_ttg_es = es;
      parsec_ttg_task_base_t *me = (parsec_ttg_task_base_t *)parsec_task;
      me->function_template_class_ptr[static_cast<std::size_t>(ttg::ExecutionSpace::Host)](parsec_task);
      parsec_ttg_es = safe_es;
      return PARSEC_HOOK_RETURN_DONE;
    }

    inline parsec_hook_return_t hook_cuda(struct parsec_execution_stream_s *es, parsec_task_t *parsec_task) {
      parsec_execution_stream_t *safe_es = parsec_ttg_es;
      parsec_ttg_es = es;
      parsec_ttg_task_base_t *me = (parsec_ttg_task_base_t *)parsec_task;
      me->function_template_class_ptr[static_cast<std::size_t>(ttg::ExecutionSpace::CUDA)](parsec_task);
      parsec_ttg_es = safe_es;
      return PARSEC_HOOK_RETURN_DONE;
    }

    static parsec_key_fn_t parsec_tasks_hash_fcts = {.key_equal = parsec_hash_table_generic_64bits_key_equal,
                                                     .key_print = parsec_hash_table_generic_64bits_key_print,
                                                     .key_hash = parsec_hash_table_generic_64bits_key_hash};

  }  // namespace detail
} // namespace ttg_parsec

#endif // TTG_PARSEC_HUNKS_H
