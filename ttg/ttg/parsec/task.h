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
      parsec_device_gpu_module_t* device = nullptr;
      parsec_task_class_t task_class; // copy of the taskclass
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
      device_ptr_t m_dev_ptr = {nullptr, &m_flows[0], nullptr, nullptr}; // gpu_task will be allocated in each task
      device_ptr_t* dev_ptr() {
        return &m_dev_ptr;
      }
    };

    enum class ttg_parsec_data_flags : uint8_t {
      NONE           = 0,
      SINGLE_READER  = 1 << 0,
      MULTIPLE_READER   = 1 << 1,
      SINGLE_WRITER  = 1 << 2,
      MULTIPLE_WRITER   = 1 << 3,
      IS_MODIFIED    = 1 << 4,
      MARKED_PUSHOUT = 1 << 5
    };

    inline
    ttg_parsec_data_flags operator|(ttg_parsec_data_flags lhs, ttg_parsec_data_flags rhs) {
        using flags_type = std::underlying_type<ttg_parsec_data_flags>::type;
        return ttg_parsec_data_flags(static_cast<flags_type>(lhs) | static_cast<flags_type>(rhs));
    }

    inline
    ttg_parsec_data_flags operator|=(ttg_parsec_data_flags& lhs, ttg_parsec_data_flags rhs) {
        using flags_type = std::underlying_type<ttg_parsec_data_flags>::type;
        lhs = ttg_parsec_data_flags(static_cast<flags_type>(lhs) | static_cast<flags_type>(rhs));
        return lhs;
    }

    inline
    uint8_t operator&(ttg_parsec_data_flags lhs, ttg_parsec_data_flags rhs) {
        using flags_type = std::underlying_type<ttg_parsec_data_flags>::type;
        return static_cast<flags_type>(lhs) & static_cast<flags_type>(rhs);
    }

    inline
    ttg_parsec_data_flags operator&=(ttg_parsec_data_flags& lhs, ttg_parsec_data_flags rhs) {
        using flags_type = std::underlying_type<ttg_parsec_data_flags>::type;
        lhs = ttg_parsec_data_flags(static_cast<flags_type>(lhs) & static_cast<flags_type>(rhs));
        return lhs;
    }

    inline
    bool operator!(ttg_parsec_data_flags lhs) {
        using flags_type = std::underlying_type<ttg_parsec_data_flags>::type;
        return lhs == ttg_parsec_data_flags::NONE;
    }


    typedef parsec_hook_return_t (*parsec_static_op_t)(void *);  // static_op will be cast to this type

    struct parsec_ttg_task_base_t {
      parsec_task_t parsec_task;
      int32_t in_data_count = 0;   //< number of satisfied inputs
      int32_t data_count = 0;      //< number of data elements in the copies array
      ttg_data_copy_t **copies;    //< pointer to the fixed copies array of the derived task
      parsec_hash_table_item_t tt_ht_item = {};

      struct stream_info_t {
        std::size_t goal;
        std::size_t size;
        parsec_lifo_t reduce_copies;
        std::atomic<std::size_t> reduce_count;
      };

    protected:
      template<std::size_t i = 0, typename TT>
      void init_stream_info_impl(TT *tt, std::array<stream_info_t, TT::numins>& streams) {
        if constexpr (TT::numins > i) {
          if (std::get<i>(tt->input_reducers)) {
            streams[i].goal = tt->static_stream_goal[i];
            streams[i].size = 0;
            PARSEC_OBJ_CONSTRUCT(&streams[i].reduce_copies, parsec_lifo_t);
            streams[i].reduce_count.store(0, std::memory_order_relaxed);
          }
          /* recursion */
          if constexpr((i + 1) < TT::numins) {
            init_stream_info_impl<i+1>(tt, streams);
          }
        }
      }

      template<typename TT>
      void init_stream_info(TT *tt, std::array<stream_info_t, TT::numins>& streams) {
        init_stream_info_impl<0>(tt, streams);
      }

    public:
      typedef void (release_task_fn)(parsec_ttg_task_base_t*);
      /* Poor-mans virtual function
       * We cannot use virtual inheritance or private visibility because we
       * need offsetof for the mempool and scheduling.
       */
      release_task_fn* release_task_cb = nullptr;
      device_ptr_t* dev_ptr = nullptr;
      bool remove_from_hash = true;
      bool dummy = false;
      bool defer_writer = TTG_PARSEC_DEFER_WRITER; // whether to defer writer instead of creating a new copy
      ttg_parsec_data_flags data_flags; // HACKY: flags set by prepare_send and reset by the copy_handler

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
                             int data_count, ttg_data_copy_t **copies,
                             bool defer_writer = TTG_PARSEC_DEFER_WRITER)
          : data_count(data_count)
          , copies(copies)
          , defer_writer(defer_writer) {
        PARSEC_OBJ_CONSTRUCT(&parsec_task, parsec_task_t);
        PARSEC_LIST_ITEM_SINGLETON(&parsec_task.super);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
        parsec_task.priority = 0;

        // TODO: can we avoid this?
        for (int i = 0; i < MAX_PARAM_COUNT; ++i) {
          this->parsec_task.data[i].data_in  = nullptr;
          this->parsec_task.data[i].data_out = nullptr;
        }
      }

      parsec_ttg_task_base_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                             parsec_taskpool_t *taskpool, int32_t priority,
                             int data_count, ttg_data_copy_t **copies,
                             release_task_fn *release_fn,
                             bool defer_writer = TTG_PARSEC_DEFER_WRITER)
          : data_count(data_count)
          , copies(copies)
          , release_task_cb(release_fn)
          , defer_writer(defer_writer) {
        PARSEC_OBJ_CONSTRUCT(&parsec_task, parsec_task_t);
        PARSEC_LIST_ITEM_SINGLETON(&parsec_task.super);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
        parsec_task.status = PARSEC_TASK_STATUS_HOOK;
        parsec_task.taskpool = taskpool;
        parsec_task.priority = priority;
        parsec_task.chore_mask = 1<<0;

        // TODO: can we avoid this?
        for (int i = 0; i < MAX_PARAM_COUNT; ++i) {
          this->parsec_task.data[i].data_in  = nullptr;
          this->parsec_task.data[i].data_out = nullptr;
        }
      }

    public:
      void set_dummy(bool d) { dummy = d; }
      bool is_dummy() { return dummy; }
    };

    template <typename TT, bool KeyIsVoid = ttg::meta::is_void_v<typename TT::key_type>>
    struct parsec_ttg_task_t : public parsec_ttg_task_base_t {
      using key_type = typename TT::key_type;
      static constexpr size_t num_streams = TT::numins;
      /* device tasks may have to store more copies than # of its inputs as their sends are aggregated */
      static constexpr size_t num_copies  = TT::derived_has_device_op() ? static_cast<size_t>(MAX_PARAM_COUNT)
                                                                      : (num_streams+1);
      TT* tt = nullptr;
      key_type key;
      std::array<stream_info_t, num_streams> streams;
#ifdef TTG_HAVE_COROUTINE
      void* suspended_task_address = nullptr;  // if not null the function is suspended
      ttg::TaskCoroutineID coroutine_id = ttg::TaskCoroutineID::Invalid;
#endif
      device_state_t<TT::derived_has_device_op()> dev_state;
      ttg_data_copy_t *copies[num_copies] = { nullptr };  // the data copies tracked by this task

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class, TT *tt_ptr)
          : parsec_ttg_task_base_t(mempool, task_class, num_streams, copies)
          , tt(tt_ptr) {
        tt_ht_item.key = pkey();
        this->dev_ptr = this->dev_state.dev_ptr();
        // We store the hash of the key and the address where it can be found in locals considered as a scratchpad
        *(uintptr_t*)&(parsec_task.locals[0]) = 0; //there is no key
        *(uintptr_t*)&(parsec_task.locals[2]) = 0; //there is no key
      }

      parsec_ttg_task_t(const key_type& key, parsec_thread_mempool_t *mempool,
                        parsec_task_class_t *task_class, parsec_taskpool_t *taskpool,
                        TT *tt_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, priority,
                                   num_streams, copies,
                                   &release_task, tt_ptr->m_defer_writer)
          , tt(tt_ptr), key(key) {
        tt_ht_item.key = pkey();
        this->dev_ptr = this->dev_state.dev_ptr();

        // We store the hash of the key and the address where it can be found in locals considered as a scratchpad
        uint64_t hv = ttg::hash<std::decay_t<decltype(key)>>{}(key);
        *(uintptr_t*)&(parsec_task.locals[0]) = hv;
        *(uintptr_t*)&(parsec_task.locals[2]) = reinterpret_cast<uintptr_t>(&this->key);

        init_stream_info(tt, streams);
      }

      static void release_task(parsec_ttg_task_base_t* task_base) {
        parsec_ttg_task_t *task = static_cast<parsec_ttg_task_t*>(task_base);
        TT *tt = task->tt;
        tt->release_task(task);
      }

      template<ttg::ExecutionSpace Space>
      parsec_hook_return_t invoke_op() {
        if constexpr (Space == ttg::ExecutionSpace::Host) {
          return TT::static_op(&this->parsec_task);
        } else {
          return TT::device_static_op(&this->parsec_task);
        }
      }

      template<ttg::ExecutionSpace Space>
      parsec_hook_return_t invoke_evaluate() {
        if constexpr (Space == ttg::ExecutionSpace::Host) {
          return PARSEC_HOOK_RETURN_DONE;
        } else {
          return TT::device_static_evaluate(&this->parsec_task);
        }
      }

      parsec_key_t pkey() { return reinterpret_cast<parsec_key_t>(&key); }
    };

    template <typename TT>
    struct parsec_ttg_task_t<TT, true> : public parsec_ttg_task_base_t {
      static constexpr size_t num_streams = TT::numins;
      TT* tt = nullptr;
      std::array<stream_info_t, num_streams> streams;
#ifdef TTG_HAVE_COROUTINE
      void* suspended_task_address = nullptr;  // if not null the function is suspended
      ttg::TaskCoroutineID coroutine_id = ttg::TaskCoroutineID::Invalid;
#endif
      device_state_t<TT::derived_has_device_op()> dev_state;
      ttg_data_copy_t *copies[num_streams+1] = { nullptr };  // the data copies tracked by this task
                                                             // +1 for the copy needed during send/bcast

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class, TT *tt_ptr)
          : parsec_ttg_task_base_t(mempool, task_class, num_streams, copies)
          , tt(tt_ptr) {
        tt_ht_item.key = pkey();
        this->dev_ptr = this->dev_state.dev_ptr();
      }

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                        parsec_taskpool_t *taskpool, TT *tt_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, priority,
                                   num_streams, copies,
                                   &release_task, tt_ptr->m_defer_writer)
          , tt(tt_ptr) {
        tt_ht_item.key = pkey();
        this->dev_ptr = this->dev_state.dev_ptr();
        init_stream_info(tt, streams);
      }

      static void release_task(parsec_ttg_task_base_t* task_base) {
        parsec_ttg_task_t *task = static_cast<parsec_ttg_task_t*>(task_base);
        TT *tt = task->tt;
        tt->release_task(task);
      }

      template<ttg::ExecutionSpace Space>
      parsec_hook_return_t invoke_op() {
        if constexpr (Space == ttg::ExecutionSpace::Host) {
          return TT::static_op(&this->parsec_task);
        } else {
          return TT::device_static_op(&this->parsec_task);
        }
      }

      template<ttg::ExecutionSpace Space>
      parsec_hook_return_t invoke_evaluate() {
        if constexpr (Space == ttg::ExecutionSpace::Host) {
          return PARSEC_HOOK_RETURN_DONE;
        } else {
          return TT::device_static_evaluate(&this->parsec_task);
        }
      }

      parsec_key_t pkey() { return 0; }
    };


    /**
     * Reducer task representing one or more stream reductions.
     * A reducer task may be deferred on its first input (the object into which
     * all other inputs are folded). Once that input becomes available the task
     * is submitted and reduces all available inputs. Additional reducer tasks may
     * be submitted until all required inputs have been processed.
     */
    struct reducer_task_t : public parsec_ttg_task_base_t {
      parsec_ttg_task_base_t *parent_task;
      bool is_first;

      reducer_task_t(parsec_ttg_task_base_t* task, parsec_thread_mempool_t *mempool,
                     parsec_task_class_t *task_class, parsec_taskpool_t *taskpool,
                     int32_t priority, bool is_first)
      : parsec_ttg_task_base_t(mempool, task_class, taskpool, priority,
                               0, nullptr,
                               &release_task,
                               true /* deferred until other readers have completed */)
      , parent_task(task)
      , is_first(is_first)
      {
        /* store the first 4 integers from the parent task (needed for profiling) */
        for (int i = 0; i < 4; ++i) {
          parsec_task.locals[i] = task->parsec_task.locals[i];
        }
      }

      static void release_task(parsec_ttg_task_base_t* task_base) {
        /* reducer tasks have one mutable input so the task can be submitted on the first release */
        parsec_task_t *vp_task_rings[1] = { &task_base->parsec_task };
        parsec_execution_stream_t *es = parsec_my_execution_stream();
        __parsec_schedule_vp(es, vp_task_rings, 0);
      }
    };

  } // namespace detail

} // namespace ttg_parsec

#endif // TTG_PARSEC_TASK_H
