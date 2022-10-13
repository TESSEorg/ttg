#ifndef TTG_PARSEC_TASK_H
#define TTG_PARSEC_TASK_H


#include <parsec.h>

#include "ttg/runtimes.h"
#include "ttg/util/meta.h"

namespace ttg_parsec {

  namespace detail {

    typedef void (*parsec_static_op_t)(void *);  // static_op will be cast to this type

    struct parsec_ttg_task_base_t {
      parsec_task_t parsec_task;
      int32_t in_data_count = 0;  //< number of satisfied inputs
      int32_t data_count = 0;     //< number of data elements in parsec_task.data
      parsec_hash_table_item_t tt_ht_item = {};
      parsec_static_op_t function_template_class_ptr[ttg::runtime_traits<ttg::Runtime::PaRSEC>::num_execution_spaces] =
          {nullptr};

      typedef void (release_task_fn)(parsec_ttg_task_base_t*);

      typedef struct {
        std::size_t goal;
        std::size_t size;
      } size_goal_t;

      /* Poor-mans virtual function
       * We cannot use virtual inheritance or private visibility because we
       * need offsetof for the mempool and scheduling.
       */
      release_task_fn* release_task_cb = nullptr;
      bool remove_from_hash = true;

      /*
      virtual void release_task() = 0;
      */
    //public:
      void release_task() {
        release_task_cb(this);
      }

     protected:
      /**
       * Protected constructors: this class should not be instantiated directly
       * but always be use through parsec_ttg_task_t.
       */

      parsec_ttg_task_base_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class, int data_count)
          : data_count(data_count) {
        PARSEC_LIST_ITEM_SINGLETON(&parsec_task.super);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
      }

      parsec_ttg_task_base_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                             parsec_taskpool_t *taskpool, int32_t priority, int data_count,
                             release_task_fn *release_fn)
          : data_count(data_count), release_task_cb(release_fn) {
        PARSEC_LIST_ITEM_SINGLETON(&parsec_task.super);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
        parsec_task.status = PARSEC_TASK_STATUS_HOOK;
        parsec_task.taskpool = taskpool;
        parsec_task.priority = priority;
        parsec_task.chore_id = 0;
      }
    };

    template <typename TT, bool KeyIsVoid = ttg::meta::is_void_v<typename TT::key_type>>
    struct parsec_ttg_task_t : public parsec_ttg_task_base_t {
      using key_type = typename TT::key_type;
      static constexpr size_t num_streams = TT::numins;
      TT* tt;
      key_type key;
      size_goal_t stream[num_streams] = {};

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class)
          : parsec_ttg_task_base_t(mempool, task_class, num_streams) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < num_streams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      parsec_ttg_task_t(const key_type& key, parsec_thread_mempool_t *mempool,
                        parsec_task_class_t *task_class, parsec_taskpool_t *taskpool,
                        TT *tt_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, priority,
                                   num_streams, &release_task)
          , tt(tt_ptr), key(key) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < num_streams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      static void release_task(parsec_ttg_task_base_t* task_base) {
        parsec_ttg_task_t *task = static_cast<parsec_ttg_task_t*>(task_base);
        TT *tt = task->tt;
        tt->release_task(task);
      }

      parsec_key_t pkey() { return reinterpret_cast<parsec_key_t>(&key); }
    };

    template <typename TT>
    struct parsec_ttg_task_t<TT, true> : public parsec_ttg_task_base_t {
      static constexpr size_t num_streams = TT::numins;
      TT* tt;
      size_goal_t stream[num_streams] = {};

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class)
          : parsec_ttg_task_base_t(mempool, task_class, num_streams) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < num_streams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                        parsec_taskpool_t *taskpool, TT *tt_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, priority,
                                   num_streams, &release_task)
          , tt(tt_ptr) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < num_streams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      static void release_task(parsec_ttg_task_base_t* task_base) {
        parsec_ttg_task_t *task = static_cast<parsec_ttg_task_t*>(task_base);
        TT *tt = task->tt;
        tt->release_task(task);
      }

      parsec_key_t pkey() { return 0; }
    };

  } // namespace detail
} // namespace ttg_parsec

#endif // TTG_PARSEC_TASK_H
