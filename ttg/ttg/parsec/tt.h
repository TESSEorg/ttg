#ifndef TTG_PARSEC_TT_H
#define TTG_PARSEC_TT_H

#include <parsec/class/parsec_hash_table.h>
#include <parsec/parsec_internal.h>

namespace ttg_parsec::detail {

  /**
   * Base class for TT in the PaRSEC backend.
   * Contains the hash tables for task management.
   * Provides an interface for some virtual abstract functions
   * that must be type-punned away.
   */
  class TTBase {

  protected:
    parsec_hash_table_t tasks_table;
    parsec_hash_table_t task_constraint_table;
    parsec_task_class_t self;

  public:

    /* release the given task */
    virtual void release_task(parsec_ttg_task_base_t* task) = 0;

    /* broadcast the value using the registered keygen */
    virtual void broadcast_keygen(const void* key, const void* value) = 0;

    /* prepare the value for broadcasting */
    virtual void prepare_keygen(const void* key, const void* value) = 0;
  };

} // namespace ttg_parsec::detail

#endif // TTG_PARSEC_TT_H