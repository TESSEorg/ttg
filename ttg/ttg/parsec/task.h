#ifndef TTG_PARSEC_TASK_H
#define TTG_PARSEC_TASK_H

#include "ttg/parsec/ttg_data_copy.h"

#include <parsec/parsec_internal.h>
#include <parsec/mca/device/device_gpu.h>

namespace ttg_parsec {

  namespace detail {

    struct device_ptr_t {
      parsec_gpu_task_t* gpu_task = nullptr;
      parsec_flow_t* flows = nullptr;
      parsec_gpu_exec_stream_t* stream = nullptr;
    };

    template<bool SupportDevice>
    struct device_state_t
    {
      static constexpr bool support_device = false;
      static constexpr size_t num_flows = 0;
      device_state_t()
      { }
      static constexpr device_ptr_t* dev_ptr() {
        return nullptr;
      }
    };

    template<>
    struct device_state_t<true> {
      static constexpr bool support_device = false;
      static constexpr size_t num_flows = MAX_PARAM_COUNT;
      parsec_flow_t m_flows[num_flows];
      device_ptr_t m_dev_ptr = {nullptr, &m_flows[0], nullptr}; // gpu_task will be allocated in each task
      device_ptr_t* dev_ptr() {
        return &m_dev_ptr;
      }
    };

    typedef parsec_hook_return_t (*parsec_static_op_t)(void *);  // static_op will be cast to this type

    struct parsec_ttg_task_base_t {
      parsec_task_t parsec_task;
      int32_t in_data_count = 0;   //< number of satisfied inputs
      int32_t data_count = 0;      //< number of data elements in the copies array
      ttg_data_copy_t **copies;    //< pointer to the fixed copies array of the derived task
      parsec_hash_table_item_t tt_ht_item = {};
      parsec_static_op_t function_template_class_ptr[ttg::runtime_traits<ttg::Runtime::PaRSEC>::num_execution_spaces] =
          {nullptr};

      typedef struct {
        std::size_t goal;
        std::size_t size;
      } size_goal_t;

      typedef void (release_task_fn)(parsec_ttg_task_base_t*);
      /* Poor-mans virtual function
       * We cannot use virtual inheritance or private visibility because we
       * need offsetof for the mempool and scheduling.
       */
      release_task_fn* release_task_cb = nullptr;
      device_ptr_t* dev_ptr;
      bool remove_from_hash = true;
      bool is_dummy = false;
      bool defer_writer = TTG_PARSEC_DEFER_WRITER; // whether to defer writer instead of creating a new copy


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

      parsec_ttg_task_base_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                             int data_count, ttg_data_copy_t **copies, device_ptr_t *dev_ptr,
                             bool defer_writer = TTG_PARSEC_DEFER_WRITER)
          : data_count(data_count)
          , copies(copies)
          , dev_ptr(dev_ptr)
          , defer_writer(defer_writer) {
        PARSEC_LIST_ITEM_SINGLETON(&parsec_task.super);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
        parsec_task.priority = 0;
      }

      parsec_ttg_task_base_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                             parsec_taskpool_t *taskpool, int32_t priority,
                             int data_count, ttg_data_copy_t **copies, device_ptr_t *dev_ptr,
                             release_task_fn *release_fn,
                             bool defer_writer = TTG_PARSEC_DEFER_WRITER)
          : data_count(data_count)
          , copies(copies)
          , release_task_cb(release_fn)
          , dev_ptr(dev_ptr)
          , defer_writer(defer_writer) {
        PARSEC_LIST_ITEM_SINGLETON(&parsec_task.super);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
        parsec_task.status = PARSEC_TASK_STATUS_HOOK;
        parsec_task.taskpool = taskpool;
        parsec_task.priority = priority;
        parsec_task.chore_mask = 1<<0;
      }

    public:
      void set_dummy(bool d) { is_dummy = d; }
      bool dummy() { return is_dummy; }
    };

    template <typename TT, bool KeyIsVoid = ttg::meta::is_void_v<typename TT::key_type>>
    struct parsec_ttg_task_t : public parsec_ttg_task_base_t {
      using key_type = typename TT::key_type;
      static constexpr size_t num_streams = TT::numins;
      /* device tasks may have to store more copies than it's inputs as their sends are aggregated */
      static constexpr size_t num_copies  = TT::derived_has_cuda_op() ? static_cast<size_t>(MAX_PARAM_COUNT)
                                                                      : (num_streams+1);
      TT* tt;
      key_type key;
      size_goal_t stream[num_streams] = {};
#ifdef TTG_HAS_COROUTINE
      void* suspended_task_address = nullptr;  // if not null the function is suspended
#endif
      ttg_data_copy_t *copies[num_copies] = { nullptr };  // the data copies tracked by this task
      device_state_t<TT::derived_has_cuda_op()> dev_state;

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class)
          : parsec_ttg_task_base_t(mempool, task_class, num_streams, copies, dev_state.dev_ptr()) {
        tt_ht_item.key = pkey();

        // We store the hash of the key and the address where it can be found in locals considered as a scratchpad
        *(uintptr_t*)&(parsec_task.locals[0]) = 0; //there is no key
        *(uintptr_t*)&(parsec_task.locals[2]) = 0; //there is no key
      }

      parsec_ttg_task_t(const key_type& key, parsec_thread_mempool_t *mempool,
                        parsec_task_class_t *task_class, parsec_taskpool_t *taskpool,
                        TT *tt_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, priority,
                                   num_streams, copies, dev_state.dev_ptr(),
                                   &release_task, tt_ptr->m_defer_writer)
          , tt(tt_ptr), key(key) {
        tt_ht_item.key = pkey();

        // We store the hash of the key and the address where it can be found in locals considered as a scratchpad
        uint64_t hv = ttg::hash<std::decay_t<decltype(key)>>{}(key);
        *(uintptr_t*)&(parsec_task.locals[0]) = hv;
        *(uintptr_t*)&(parsec_task.locals[2]) = reinterpret_cast<uintptr_t>(&this->key);
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
#ifdef TTG_HAS_COROUTINE
      void* suspended_task_address = nullptr;  // if not null the function is suspended
#endif
      ttg_data_copy_t *copies[num_streams+1] = { nullptr };  // the data copies tracked by this task
                                                             // +1 for the copy needed during send/bcast
      device_state_t<TT::derived_has_cuda_op()> dev_state;

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class)
          : parsec_ttg_task_base_t(mempool, task_class, num_streams, copies, dev_state.dev_ptr()) {
        tt_ht_item.key = pkey();
      }

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                        parsec_taskpool_t *taskpool, TT *tt_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, priority,
                                   num_streams, copies, dev_state.dev_ptr(),
                                   &release_task, tt_ptr->m_defer_writer)
          , tt(tt_ptr) {
        tt_ht_item.key = pkey();
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