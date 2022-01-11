#ifndef PARSEC_TTG_H_INCLUDED
#define PARSEC_TTG_H_INCLUDED

/* set up env if this header was included directly */
#if !defined(TTG_IMPL_NAME)
#define TTG_USE_PARSEC 1
#endif  // !defined(TTG_IMPL_NAME)

#include "ttg/impl_selector.h"

/* include ttg header to make symbols available in case this header is included directly */
#include "../../ttg.h"

#include "ttg/base/keymap.h"
#include "ttg/base/tt.h"
#include "ttg/base/world.h"
#include "ttg/edge.h"
#include "ttg/execution.h"
#include "ttg/func.h"
#include "ttg/runtimes.h"
#include "ttg/terminal.h"
#include "ttg/tt.h"
#include "ttg/util/env.h"
#include "ttg/util/hash.h"
#include "ttg/util/meta.h"
#include "ttg/util/meta/callable.h"
#include "ttg/util/print.h"
#include "ttg/util/trace.h"
#include "ttg/util/typelist.h"

#include "ttg/serialization/data_descriptor.h"

#include "ttg/parsec/fwd.h"

#include <array>
#include <cassert>
#include <experimental/type_traits>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <parsec.h>
#include <parsec/class/parsec_hash_table.h>
#include <parsec/data_internal.h>
#include <parsec/execution_stream.h>
#include <parsec/interfaces/interface.h>
#include <parsec/mca/device/device.h>
#include <parsec/parsec_comm_engine.h>
#include <parsec/parsec_internal.h>
#include <parsec/scheduling.h>
#include <cstdlib>
#include <cstring>

#include "ttg/parsec/ttg_data_copy.h"

/* PaRSEC function declarations */
extern "C" {
void parsec_taskpool_termination_detected(parsec_taskpool_t *tp);
int parsec_add_fetch_runtime_task(parsec_taskpool_t *tp, int tasks);
}

namespace ttg_parsec {
  inline thread_local parsec_execution_stream_t *parsec_ttg_es;

  typedef void (*static_set_arg_fct_type)(void *, size_t, ttg::TTBase *);
  typedef std::pair<static_set_arg_fct_type, ttg::TTBase *> static_set_arg_fct_call_t;
  inline std::map<uint64_t, static_set_arg_fct_call_t> static_id_to_op_map;
  inline std::mutex static_map_mutex;
  typedef std::tuple<int, void *, size_t> static_set_arg_fct_arg_t;
  inline std::multimap<uint64_t, static_set_arg_fct_arg_t> delayed_unpack_actions;

  struct msg_header_t {
    typedef enum { MSG_SET_ARG = 0, MSG_SET_ARGSTREAM_SIZE = 1, MSG_FINALIZE_ARGSTREAM_SIZE = 2 } fn_id_t;
    uint32_t taskpool_id;
    uint64_t op_id;
    fn_id_t fn_id;
    int32_t param_id;
    int num_keys;
  };

  namespace detail {

    static int static_unpack_msg(parsec_comm_engine_t *ce, uint64_t tag, void *data, long unsigned int size,
                                 int src_rank, void *obj) {
      static_set_arg_fct_type static_set_arg_fct;
      parsec_taskpool_t *tp = NULL;
      msg_header_t *msg = static_cast<msg_header_t *>(data);
      uint64_t op_id = msg->op_id;
      tp = parsec_taskpool_lookup(msg->taskpool_id);
      assert(NULL != tp);
      static_map_mutex.lock();
      try {
        auto op_pair = static_id_to_op_map.at(op_id);
        static_map_mutex.unlock();
        tp->tdm.module->incoming_message_start(tp, src_rank, NULL, NULL, 0, NULL);
        static_set_arg_fct = op_pair.first;
        static_set_arg_fct(data, size, op_pair.second);
        tp->tdm.module->incoming_message_end(tp, NULL);
        return 0;
      } catch (const std::out_of_range &e) {
        void *data_cpy = malloc(size);
        assert(data_cpy != 0);
        memcpy(data_cpy, data, size);
        ttg::trace("ttg_parsec(", ttg_default_execution_context().rank(), ") Delaying delivery of message (", src_rank,
                   ", ", op_id, ", ", data_cpy, ", ", size, ")");
        delayed_unpack_actions.insert(std::make_pair(op_id, std::make_tuple(src_rank, data_cpy, size)));
        static_map_mutex.unlock();
        return 1;
      }
    }

    static int get_remote_complete_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                      int src, void *cb_data);

  }  // namespace detail

  class WorldImpl : public ttg::base::WorldImplBase {
    static constexpr const int _PARSEC_TTG_TAG = 10;      // This TAG should be 'allocated' at the PaRSEC level
    static constexpr const int _PARSEC_TTG_RMA_TAG = 11;  // This TAG should be 'allocated' at the PaRSEC level

    ttg::Edge<> m_ctl_edge;

    int query_comm_size() {
      int comm_size;
      MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
      return comm_size;
    }

    int query_comm_rank() {
      int comm_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
      return comm_rank;
    }

   public:
    static constexpr const int PARSEC_TTG_MAX_AM_SIZE = 1024 * 1024;
    WorldImpl(int *argc, char **argv[], int ncores) : WorldImplBase(query_comm_size(), query_comm_rank()) {
      ttg::detail::register_world(*this);
      ctx = parsec_init(ncores, argc, argv);
      es = ctx->virtual_processes[0]->execution_streams[0];

      parsec_ce.tag_register(_PARSEC_TTG_TAG, &detail::static_unpack_msg, this, PARSEC_TTG_MAX_AM_SIZE);
      parsec_ce.tag_register(_PARSEC_TTG_RMA_TAG, &detail::get_remote_complete_cb, this, 128);

      create_tpool();
    }

    void create_tpool() {
      assert(nullptr == tpool);
      tpool = (parsec_taskpool_t *)calloc(1, sizeof(parsec_taskpool_t));
      tpool->taskpool_id = -1;
      tpool->update_nb_runtime_task = parsec_add_fetch_runtime_task;
      tpool->taskpool_type = PARSEC_TASKPOOL_TYPE_TTG;
      parsec_taskpool_reserve_id(tpool);

#ifdef TTG_USE_USER_TERMDET
      parsec_termdet_open_module(tpool, "user_trigger");
#else   // TTG_USE_USER_TERMDET
      parsec_termdet_open_dyn_module(tpool);
#endif  // TTG_USE_USER_TERMDET
      tpool->tdm.module->monitor_taskpool(tpool, parsec_taskpool_termination_detected);
      // In TTG, we use the pending actions to denote that the
      // taskpool is not ready, i.e. some local tasks could still
      // be added by the main thread. It should then be initialized
      // to 0, execute will set it to 1 and mark the tpool as ready,
      // and the fence() will decrease it back to 0.
      tpool->tdm.module->taskpool_set_nb_pa(tpool, 0);
      parsec_taskpool_enable(tpool, NULL, NULL, es, size() > 1);

      // Termination detection in PaRSEC requires to synchronize the
      // taskpool enabling, to avoid a race condition that would keep
      // termination detection-related messages in a waiting queue
      // forever
      MPI_Barrier(comm());

      parsec_taskpool_started = false;
    }

    /* Deleted copy ctor */
    WorldImpl(const WorldImpl &other) = delete;

    /* Deleted move ctor */
    WorldImpl(WorldImpl &&other) = delete;

    /* Deleted copy assignment */
    WorldImpl &operator=(const WorldImpl &other) = delete;

    /* Deleted move assignment */
    WorldImpl &operator=(WorldImpl &&other) = delete;

    ~WorldImpl() { destroy(); }

    constexpr int parsec_ttg_tag() const { return _PARSEC_TTG_TAG; }
    constexpr int parsec_ttg_rma_tag() const { return _PARSEC_TTG_RMA_TAG; }

    MPI_Comm comm() const { return MPI_COMM_WORLD; }

    virtual void execute() override {
      parsec_enqueue(ctx, tpool);
      tpool->tdm.module->taskpool_addto_nb_pa(tpool, 1);
      tpool->tdm.module->taskpool_ready(tpool);
      int ret = parsec_context_start(ctx);
      parsec_taskpool_started = true;
      if (ret != 0) throw std::runtime_error("TTG: parsec_context_start failed");
    }

    void destroy_tpool() {
      parsec_taskpool_free(tpool);
      tpool = nullptr;
    }

    virtual void destroy() override {
      if (is_valid()) {
        if (parsec_taskpool_started) {
          // We are locally ready (i.e. we won't add new tasks)
          tpool->tdm.module->taskpool_addto_nb_pa(tpool, -1);
          ttg::trace("ttg_parsec(", this->rank(), "): final waiting for completion");
          parsec_context_wait(ctx);
        }
        release_ops();
        ttg::detail::deregister_world(*this);
        destroy_tpool();
        parsec_ce.tag_unregister(_PARSEC_TTG_TAG);
        parsec_ce.tag_unregister(_PARSEC_TTG_RMA_TAG);
        parsec_fini(&ctx);
        mark_invalid();
      }
    }

    ttg::Edge<> &ctl_edge() { return m_ctl_edge; }

    const ttg::Edge<> &ctl_edge() const { return m_ctl_edge; }

    auto *context() { return ctx; }
    auto *execution_stream() { return parsec_ttg_es == nullptr ? es : parsec_ttg_es; }
    auto *taskpool() { return tpool; }

    void increment_created() { taskpool()->tdm.module->taskpool_addto_nb_tasks(taskpool(), 1); }

    void increment_inflight_msg() { taskpool()->tdm.module->taskpool_addto_nb_pa(taskpool(), 1); }
    void decrement_inflight_msg() { taskpool()->tdm.module->taskpool_addto_nb_pa(taskpool(), -1); }

    virtual void final_task() override {
#ifdef TTG_USE_USER_TERMDET
      taskpool()->tdm.module->taskpool_set_nb_tasks(taskpool(), 0);
#endif  // TTG_USE_USER_TERMDET
    }

   protected:
    virtual void fence_impl(void) override {
      int rank = this->rank();
      if (!parsec_taskpool_started) {
        ttg::trace("ttg_parsec::(", rank, "): parsec taskpool has not been started, fence is a simple MPI_Barrier");
        MPI_Barrier(comm());
        return;
      }
      ttg::trace("ttg_parsec::(", rank, "): parsec taskpool is ready for completion");
      // We are locally ready (i.e. we won't add new tasks)
      tpool->tdm.module->taskpool_addto_nb_pa(tpool, -1);
      ttg::trace("ttg_parsec(", rank, "): waiting for completion");
      parsec_context_wait(ctx);

      // We need the synchronization between the end of the context and the restart of the taskpool
      // until we use parsec_taskpool_wait and implement an epoch in the PaRSEC taskpool
      // see Issue #118 (TTG)
      MPI_Barrier(comm());

      destroy_tpool();
      create_tpool();
      execute();
    }

   private:
    parsec_context_t *ctx = nullptr;
    parsec_execution_stream_t *es = nullptr;
    parsec_taskpool_t *tpool = nullptr;
    bool parsec_taskpool_started = false;
  };

  namespace detail {
    typedef void (*parsec_static_op_t)(void *);  // static_op will be cast to this type

    struct parsec_ttg_task_base_t {
      parsec_task_t parsec_task;
      int32_t in_data_count = 0;  //< number of satisfied inputs
      int32_t data_count = 0;     //< number of data elements in parsec_task.data
      parsec_hash_table_item_t tt_ht_item = {};
      parsec_static_op_t function_template_class_ptr[ttg::runtime_traits<ttg::Runtime::PaRSEC>::num_execution_spaces] =
          {nullptr};
      void *object_ptr = nullptr;
      void (*static_set_arg)(int, int) = nullptr;
      void (*deferred_release)(void *, parsec_ttg_task_base_t *) =
          nullptr;  // callback used to release the task from with the static context of complete_task_and_release
      void *tt_ptr = nullptr;  // pointer to the TT object, passed to deferred_release

     protected:
      /**
       * Protected constructors: this class should not be instantiated directly
       * but always be use through parsec_ttg_task_t.
       */

      parsec_ttg_task_base_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class, int data_count)
          : data_count(data_count) {
        PARSEC_LIST_ITEM_SINGLETON(&this->parsec_task);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
      }

      parsec_ttg_task_base_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                             parsec_taskpool_t *taskpool, void *object_ptr, int32_t priority, int data_count)
          : data_count(data_count), object_ptr(object_ptr) {
        PARSEC_LIST_ITEM_SINGLETON(&this->parsec_task);
        parsec_task.mempool_owner = mempool;
        parsec_task.task_class = task_class;
        parsec_task.status = PARSEC_TASK_STATUS_HOOK;
        parsec_task.taskpool = taskpool;
        parsec_task.priority = priority;
        parsec_task.chore_id = 0;
      }
    };

    template <typename Key, size_t NumStreams, bool KeyIsVoid = ttg::meta::is_void_v<Key>>
    struct parsec_ttg_task_t : public parsec_ttg_task_base_t {
      Key key;
      typedef struct {
        std::size_t goal;
        std::size_t size;
      } size_goal_t;
      size_goal_t stream[NumStreams] = {};

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class)
          : parsec_ttg_task_base_t(mempool, task_class, NumStreams) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < NumStreams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      parsec_ttg_task_t(Key key, parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class,
                        parsec_taskpool_t *taskpool, void *object_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, object_ptr, priority, NumStreams), key(key) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < NumStreams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      parsec_key_t pkey() { return reinterpret_cast<parsec_key_t>(&key); }
    };

    template <typename Key, size_t NumStreams>
    struct parsec_ttg_task_t<Key, NumStreams, true> : public parsec_ttg_task_base_t {
      typedef struct {
        std::size_t goal;
        std::size_t size;
      } size_goal_t;
      size_goal_t stream[NumStreams] = {};

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class)
          : parsec_ttg_task_base_t(mempool, task_class, NumStreams) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < NumStreams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      parsec_ttg_task_t(parsec_thread_mempool_t *mempool, parsec_task_class_t *task_class, parsec_taskpool_t *taskpool,
                        void *object_ptr, int32_t priority)
          : parsec_ttg_task_base_t(mempool, task_class, taskpool, object_ptr, priority, NumStreams) {
        tt_ht_item.key = pkey();

        for (int i = 0; i < NumStreams; ++i) {
          parsec_task.data[i].data_in = nullptr;
        }
      }

      parsec_key_t pkey() { return 0; }
    };

    inline ttg_data_copy_t *find_copy_in_task(parsec_ttg_task_base_t *task, const void *ptr) {
      ttg_data_copy_t *res = nullptr;
      if (task == nullptr || ptr == nullptr) {
        return res;
      }
      for (int i = 0; i < task->data_count; ++i) {
        auto copy = static_cast<ttg_data_copy_t *>(task->parsec_task.data[i].data_in);
        if (NULL != copy && copy->device_private == ptr) {
          res = copy;
          break;
        }
      }
      return res;
    }

    inline bool add_copy_to_task(ttg_data_copy_t *copy, parsec_ttg_task_base_t *task) {
      if (task == nullptr || copy == nullptr) {
        return false;
      }

      if (MAX_PARAM_COUNT < task->data_count) {
        throw std::logic_error("Too many data copies, check MAX_PARAM_COUNT!");
      }

      task->parsec_task.data[task->data_count].data_in = copy;
      task->data_count++;
      return true;
    }

    inline void remove_data_copy(ttg_data_copy_t *copy, parsec_ttg_task_base_t *task) {
      int i;
      /* find and remove entry; copies are usually appended and removed, so start from back */
      for (i = task->data_count; i >= 0; --i) {
        if (copy == task->parsec_task.data[i].data_in) {
          break;
        }
      }
      if (i < 0) return;
      /* move all following elements one up */
      for (; i < task->data_count - 1; ++i) {
        task->parsec_task.data[i].data_in = task->parsec_task.data[i + 1].data_in;
      }
      /* null last element */
      task->parsec_task.data[i].data_in = nullptr;
      task->data_count--;
    }

    template <typename Value>
    inline ttg_data_copy_t *create_new_datacopy(Value &&value) {
      using value_type = std::decay_t<Value>;
      ttg_data_copy_t *copy = new ttg_data_value_copy_t<value_type>(std::forward<Value>(value));
      return copy;
    }

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

    inline uint64_t parsec_tasks_hash_fct(parsec_key_t key, int nb_bits, void *data) {
      /* Use all the bits of the 64 bits key, project on the lowest base bits (0 <= hash < 1024) */
      int b = 0;
      uint64_t mask = ~0ULL >> (64 - nb_bits);
      uint64_t h = (uint64_t)key;
      (void)data;
      while (b < 64) {
        b += nb_bits;
        h ^= (uint64_t)key >> b;
      }
      return (uint64_t)(h & mask);
    }

    static parsec_key_fn_t parsec_tasks_hash_fcts = {.key_equal = parsec_hash_table_generic_64bits_key_equal,
                                                     .key_print = parsec_hash_table_generic_64bits_key_print,
                                                     .key_hash = parsec_hash_table_generic_64bits_key_hash};

    template <typename KeyT, typename ActivationCallbackT>
    class rma_delayed_activate {
      std::vector<KeyT> _keylist;
      std::atomic<int> _outstanding_transfers;
      ActivationCallbackT _cb;
      ttg_data_copy_t *_copy;

     public:
      rma_delayed_activate(std::vector<KeyT> &&key, ttg_data_copy_t *copy, int num_transfers, ActivationCallbackT cb)
          : _keylist(std::move(key)), _outstanding_transfers(num_transfers), _cb(cb), _copy(copy) {}

      bool complete_transfer(void) {
        int left = --_outstanding_transfers;
        if (0 == left) {
          _cb(std::move(_keylist), _copy);
          return true;
        }
        return false;
      }
    };

    template <typename ActivationT>
    static int get_complete_cb(parsec_comm_engine_t *comm_engine, parsec_ce_mem_reg_handle_t lreg, ptrdiff_t ldispl,
                               parsec_ce_mem_reg_handle_t rreg, ptrdiff_t rdispl, size_t size, int remote,
                               void *cb_data) {
      parsec_ce.mem_unregister(&lreg);
      ActivationT *activation = static_cast<ActivationT *>(cb_data);
      if (activation->complete_transfer()) {
        delete activation;
      }
      return PARSEC_SUCCESS;
    }

    static int get_remote_complete_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                      int src, void *cb_data) {
      std::intptr_t *fn_ptr = static_cast<std::intptr_t *>(msg);
      std::function<void(void)> *fn = reinterpret_cast<std::function<void(void)> *>(*fn_ptr);
      (*fn)();
      delete fn;
      return PARSEC_SUCCESS;
    }

    template <typename FuncT>
    static int invoke_get_remote_complete_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                             int src, void *cb_data) {
      std::intptr_t *iptr = static_cast<std::intptr_t *>(msg);
      FuncT *fn_ptr = reinterpret_cast<FuncT *>(*iptr);
      (*fn_ptr)();
      delete fn_ptr;
      return PARSEC_SUCCESS;
    }

    inline void release_data_copy(ttg_data_copy_t *copy) {
      if (NULL != copy->push_task) {
        /* Release the the deferred task.
         * The copy was mutable and will be mutated by the released task,
         * so simply transfer ownership.
         */
        parsec_task_t *push_task = copy->push_task;
        copy->push_task = nullptr;
        parsec_ttg_task_base_t *deferred_op = (parsec_ttg_task_base_t *)push_task;
        assert(deferred_op->deferred_release);
        deferred_op->deferred_release(deferred_op->tt_ptr, deferred_op);
      } else {
        if (copy->readers < 0) {
          /* current task mutated the data but there are no consumers so prepare
          * the copy to be freed below */
          copy->readers = 1;
        }

        int32_t readers = copy->readers;
        if (copy->readers > 1) {
          /* potentially more than one reader, decrement atomically */
          readers = parsec_atomic_fetch_dec_int32(&copy->readers);
        }
        /* if there was only one reader (the current task) we release the copy */
        if (1 == readers) {
          delete copy;
        }
      }
    }

    template <typename Value>
    inline ttg_data_copy_t *register_data_copy(ttg_data_copy_t *copy_in, parsec_ttg_task_base_t *task, bool readonly) {
      ttg_data_copy_t *copy_res = copy_in;
      bool replace = false;
      int32_t readers = copy_in->readers;

      if (readonly && copy_in->readers > 0) {
        /* simply increment the number of readers */
        readers = parsec_atomic_fetch_inc_int32(&copy_in->readers);
      }

      if (readers < 0) {
        /* someone is going to write into this copy -> we need to make a copy */
        copy_res = NULL;
        if (readonly) {
          /* we replace the copy in a deferred task if the copy will be mutated by
           * the deferred task and we are readonly.
           * That way, we can share the copy with other readonly tasks and release
           * the deferred task. */
          replace = true;
        }
      } else if (!readonly) {
        /* this task will mutate the data
         * check whether there are other readers already and potentially
         * defer the release of this task to give following readers a
         * chance to make a copy of the data before this task mutates it
         *
         * Try to replace the readers with a negative value that indicates
         * the value is mutable. If that fails we know that there are other
         * readers or writers already.
         *
         * NOTE: this check is not atomic: either there is a single reader
         *       (current task) or there are others, in which we case won't
         *       touch it.
         */
        if (1 == copy_in->readers) {
          /**
           * no other readers, mark copy as mutable and defer the release
           * of the task
           */
          copy_in->readers = INT32_MIN;
          assert(nullptr == copy_in->push_task);
          assert(nullptr != task);
          copy_in->push_task = &task->parsec_task;
        } else {
          /* there are readers of this copy already, make a copy that we can mutate */
          copy_res = NULL;
        }
      }

      if (NULL == copy_res) {
        ttg_data_copy_t *new_copy = detail::create_new_datacopy(*static_cast<Value *>(copy_in->device_private));
        if (replace && nullptr != copy_in->push_task) {
          /* replace the task that was deferred */
          parsec_ttg_task_base_t *deferred_op = (parsec_ttg_task_base_t *)copy_in->push_task;
          new_copy->readers = INT32_MIN;
          /* replace the copy in the deferred task */
          for (int i = 0; i < deferred_op->data_count; ++i) {
            if (deferred_op->parsec_task.data[i].data_in == copy_in) {
              deferred_op->parsec_task.data[i].data_in = new_copy;
              break;
            }
          }
          assert(deferred_op->deferred_release);
          deferred_op->deferred_release(deferred_op->tt_ptr, deferred_op);
          copy_in->push_task = NULL;
          copy_in->readers = 1;  // set the copy back to being read-only
          ++copy_in->readers;    // register as reader
          copy_res = copy_in;    // return the copy we were passed
        } else {
          if (!readonly) {
            new_copy->readers = INT32_MIN;
          }
          copy_res = new_copy;  // return the new copy
        }
      }
      return copy_res;
    }
  }  // namespace detail

  inline thread_local detail::parsec_ttg_task_base_t *parsec_ttg_caller;

  template <typename... RestOfArgs>
  inline void ttg_initialize(int argc, char **argv, int num_threads, RestOfArgs &&...) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (num_threads < 1) num_threads = ttg::detail::num_threads();
    auto world_ptr = new ttg_parsec::WorldImpl{&argc, &argv, num_threads};
    std::shared_ptr<ttg::base::WorldImplBase> world_sptr{static_cast<ttg::base::WorldImplBase *>(world_ptr)};
    ttg::World world{std::move(world_sptr)};
    ttg::detail::set_default_world(std::move(world));
  }
  inline void ttg_finalize() {
    ttg::detail::set_default_world(ttg::World{});  // reset the default world
    ttg::detail::destroy_worlds<ttg_parsec::WorldImpl>();
    MPI_Finalize();
  }
  inline ttg::World ttg_default_execution_context() { return ttg::get_default_world(); }
  inline void ttg_abort() { MPI_Abort(ttg_default_execution_context().impl().comm(), 1); }
  inline void ttg_execute(ttg::World world) { world.impl().execute(); }
  inline void ttg_fence(ttg::World world) { world.impl().fence(); }

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr) {
    world.impl().register_ptr(ptr);
  }

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, std::unique_ptr<T> &&ptr) {
    world.impl().register_ptr(std::move(ptr));
  }

  inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr) {
    world.impl().register_status(status_ptr);
  }

  template <typename Callback>
  inline void ttg_register_callback(ttg::World world, Callback &&callback) {
    world.impl().register_callback(std::forward<Callback>(callback));
  }

  inline ttg::Edge<> &ttg_ctl_edge(ttg::World world) { return world.impl().ctl_edge(); }

  inline void ttg_sum(ttg::World world, double &value) {
    double result = 0.0;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, world.impl().comm());
    value = result;
  }

  /// broadcast
  /// @tparam T a serializable type
  template <typename T>
  void ttg_broadcast(::ttg::World world, T &data, int source_rank) {
    int64_t BUFLEN;
    if (world.rank() == source_rank) {
      BUFLEN = ttg::default_data_descriptor<T>::payload_size(&data);
    }
    MPI_Bcast(&BUFLEN, 1, MPI_INT64_T, source_rank, world.impl().comm());

    unsigned char *buf = new unsigned char[BUFLEN];
    if (world.rank() == source_rank) {
      ttg::default_data_descriptor<T>::pack_payload(&data, BUFLEN, 0, buf);
    }
    MPI_Bcast(buf, BUFLEN, MPI_UNSIGNED_CHAR, source_rank, world.impl().comm());
    if (world.rank() != source_rank) {
      ttg::default_data_descriptor<T>::unpack_payload(&data, BUFLEN, 0, buf);
    }
    delete[] buf;
  }

  namespace detail {

    struct ParsecTTBase {
     protected:
      //  static std::map<int, ParsecBaseTT*> function_id_to_instance;
      parsec_hash_table_t tasks_table;
      parsec_task_class_t self;
    };

    struct msg_t {
      msg_header_t tt_id;
      unsigned char bytes[WorldImpl::PARSEC_TTG_MAX_AM_SIZE - sizeof(msg_header_t)];

      msg_t() = default;
      msg_t(uint64_t tt_id, uint32_t taskpool_id, msg_header_t::fn_id_t fn_id, int32_t param_id, int num_keys = 1)
          : tt_id{taskpool_id, tt_id, fn_id, param_id, num_keys} {}
    };
  }  // namespace detail

  template <typename keyT, typename output_terminalsT, typename derivedT, typename input_valueTs>
  class TT : public ttg::TTBase, detail::ParsecTTBase {
   private:
    /// preconditions
    static_assert(ttg::meta::is_typelist_v<input_valueTs>,
                  "The fourth template for ttg::TT must be a ttg::typelist containing the input types");
    // create a virtual control input if the input list is empty, to be used in invoke()
    using actual_input_tuple_type = std::conditional_t<!ttg::meta::typelist_is_empty_v<input_valueTs>,
                                                       ttg::meta::typelist_to_tuple_t<input_valueTs>, std::tuple<void>>;
    using input_tuple_type = ttg::meta::typelist_to_tuple_t<input_valueTs>;
    static_assert(ttg::meta::is_tuple_v<output_terminalsT>,
                  "Second template argument for ttg::TT must be std::tuple containing the output terminal types");
    static_assert((ttg::meta::none_has_reference_v<input_valueTs>), "Input typelist cannot contain reference types");
    static_assert(ttg::meta::is_none_Void_v<input_valueTs>, "ttg::Void is for internal use only, do not use it");

    parsec_mempool_t mempools;

    // check for a non-type member named have_cuda_op
    template <typename T>
    using have_cuda_op_non_type_t = decltype(&T::have_cuda_op);

    bool alive = true;

    static constexpr int numinedges = std::tuple_size_v<input_tuple_type>;     // number of input edges
    static constexpr int numins = std::tuple_size_v<actual_input_tuple_type>;  // number of input arguments
    static constexpr int numouts = std::tuple_size_v<output_terminalsT>;       // number of outputs
    static constexpr int numflows = std::max(numins, numouts);                 // max number of flows

    /// @return true if derivedT::have_cuda_op exists and is defined to true
    static constexpr bool derived_has_cuda_op() {
      if constexpr (ttg::meta::is_detected_v<have_cuda_op_non_type_t, derivedT>) {
        return derivedT::have_cuda_op;
      } else {
        return false;
      }
    }

   public:
    using ttT = TT;
    using input_terminals_type = ttg::detail::input_terminals_tuple_t<keyT, input_tuple_type>;
    using input_args_type = actual_input_tuple_type;
    using input_edges_type = ttg::detail::edges_tuple_t<keyT, ttg::meta::decayed_typelist_t<input_tuple_type>>;
    // if have data inputs and (always last) control input, convert last input to Void to make logic easier
    using input_values_full_tuple_type =
        ttg::meta::void_to_Void_tuple_t<ttg::meta::decayed_typelist_t<actual_input_tuple_type>>;
    using input_refs_full_tuple_type =
        ttg::meta::add_glvalue_reference_tuple_t<ttg::meta::void_to_Void_tuple_t<actual_input_tuple_type>>;
    using input_values_tuple_type = ttg::meta::drop_void_t<ttg::meta::decayed_typelist_t<input_tuple_type>>;
    using input_refs_tuple_type = ttg::meta::drop_void_t<ttg::meta::add_glvalue_reference_tuple_t<input_tuple_type>>;

    static constexpr int numinvals =
        std::tuple_size_v<input_refs_tuple_type>;  // number of input arguments with values (i.e. omitting the control
                                                   // input, if any)

    using output_terminals_type = output_terminalsT;
    using output_edges_type = typename ttg::terminals_to_edges<output_terminalsT>::type;

    template <std::size_t i, typename resultT, typename InTuple>
    static resultT get(InTuple &&intuple) {
      return static_cast<resultT>(std::get<i>(std::forward<InTuple>(intuple)));
    };
    template <std::size_t i, typename InTuple>
    static auto &get(InTuple &&intuple) {
      return std::get<i>(std::forward<InTuple>(intuple));
    };

   private:
    using task_t = detail::parsec_ttg_task_t<keyT, numins>;

    /* the offset of the key placed after the task structure in the memory from mempool */
    constexpr static const size_t task_key_offset = sizeof(task_t);

    input_terminals_type input_terminals;
    output_terminalsT output_terminals;

   protected:
    const auto &get_output_terminals() const { return output_terminals; }

   private:
    template <std::size_t... IS>
    static constexpr auto make_set_args_fcts(std::index_sequence<IS...>) {
      using resultT = decltype(set_arg_from_msg_fcts);
      return resultT{{&TT::set_arg_from_msg<IS>...}};
    }
    constexpr static std::array<void (TT::*)(void *, std::size_t), numins> set_arg_from_msg_fcts =
        make_set_args_fcts(std::make_index_sequence<numins>{});

    template <std::size_t... IS>
    static constexpr auto make_set_size_fcts(std::index_sequence<IS...>) {
      using resultT = decltype(set_argstream_size_from_msg_fcts);
      return resultT{{&TT::argstream_set_size_from_msg<IS>...}};
    }
    constexpr static std::array<void (TT::*)(void *, std::size_t), numins> set_argstream_size_from_msg_fcts =
        make_set_size_fcts(std::make_index_sequence<numins>{});

    template <std::size_t... IS>
    static constexpr auto make_finalize_argstream_fcts(std::index_sequence<IS...>) {
      using resultT = decltype(finalize_argstream_from_msg_fcts);
      return resultT{{&TT::finalize_argstream_from_msg<IS>...}};
    }
    constexpr static std::array<void (TT::*)(void *, std::size_t), numins> finalize_argstream_from_msg_fcts =
        make_finalize_argstream_fcts(std::make_index_sequence<numins>{});

    ttg::World world;
    ttg::meta::detail::keymap_t<keyT> keymap;
    ttg::meta::detail::keymap_t<keyT> priomap;
    // For now use same type for unary/streaming input terminals, and stream reducers assigned at runtime
    ttg::meta::detail::input_reducers_t<actual_input_tuple_type>
        input_reducers;  //!< Reducers for the input terminals (empty = expect single value)
    std::array<std::size_t, numins> static_stream_goal;

   public:
    ttg::World get_world() const { return world; }

   private:
    /// dispatches a call to derivedT::op if Space == Host, otherwise to derivedT::op_cuda if Space == CUDA
    template <ttg::ExecutionSpace Space, typename... Args>
    void op(Args &&...args) {
      derivedT *derived = static_cast<derivedT *>(this);
      if constexpr (Space == ttg::ExecutionSpace::Host)
        derived->op(std::forward<Args>(args)...);
      else if constexpr (Space == ttg::ExecutionSpace::CUDA)
        derived->op_cuda(std::forward<Args>(args)...);
      else
        abort();
    }

    template <std::size_t... IS>
    static input_refs_tuple_type make_tuple_of_ref_from_array(task_t *task, std::index_sequence<IS...>) {
      return input_refs_tuple_type{static_cast<std::tuple_element_t<IS, input_refs_tuple_type>>(
          *reinterpret_cast<std::remove_reference_t<std::tuple_element_t<IS, input_refs_tuple_type>> *>(
              task->parsec_task.data[IS].data_in->device_private))...};
    }

    template <ttg::ExecutionSpace Space>
    static void static_op(parsec_task_t *parsec_task) {
      task_t *task = (task_t *)parsec_task;
      ttT *baseobj = (ttT *)task->object_ptr;
      derivedT *obj = (derivedT *)task->object_ptr;
      assert(parsec_ttg_caller == NULL);
      parsec_ttg_caller = (detail::parsec_ttg_task_base_t *)parsec_task;
      if (obj->tracing()) {
        if constexpr (!ttg::meta::is_void_v<keyT>)
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : ", task->key, ": executing");
        else
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : executing");
      }

      if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
        auto input = make_tuple_of_ref_from_array(task, std::make_index_sequence<numinvals>{});
        baseobj->template op<Space>(task->key, std::move(input), obj->output_terminals);
      } else if constexpr (!ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
        baseobj->template op<Space>(task->key, obj->output_terminals);
      } else if constexpr (ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
        auto input = make_tuple_of_ref_from_array(task, std::make_index_sequence<numinvals>{});
        baseobj->template op<Space>(std::move(input), obj->output_terminals);
      } else if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
        baseobj->template op<Space>(obj->output_terminals);
      } else {
        abort();
      }
      parsec_ttg_caller = NULL;

      if (obj->tracing()) {
        if constexpr (!ttg::meta::is_void_v<keyT>)
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : ", task->key, ": done executing");
        else
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : done executing");
      }
    }

    template <ttg::ExecutionSpace Space>
    static void static_op_noarg(parsec_task_t *parsec_task) {
      task_t *task = (task_t *)parsec_task;
      ttT *baseobj = (ttT *)task->object_ptr;
      derivedT *obj = (derivedT *)task->object_ptr;
      assert(parsec_ttg_caller == NULL);
      parsec_ttg_caller = (detail::parsec_ttg_task_base_t *)parsec_task;
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        baseobj->template op<Space>(task->key, obj->output_terminals);
      } else if constexpr (ttg::meta::is_void_v<keyT>) {
        baseobj->template op<Space>(obj->output_terminals);
      } else
        abort();
      parsec_ttg_caller = NULL;
    }

   protected:
    template <typename T>
    uint64_t unpack(T &obj, void *_bytes, uint64_t pos) {
      const ttg_data_descriptor *dObj = ttg::get_data_descriptor<ttg::meta::remove_cvr_t<T>>();
      uint64_t payload_size;
      if constexpr (!ttg::default_data_descriptor<ttg::meta::remove_cvr_t<T>>::serialize_size_is_const) {
        const ttg_data_descriptor *dSiz = ttg::get_data_descriptor<uint64_t>();
        dSiz->unpack_payload(&payload_size, sizeof(uint64_t), pos, _bytes);
        pos += sizeof(uint64_t);
      } else {
        payload_size = dObj->payload_size(&obj);
      }
      dObj->unpack_payload(&obj, payload_size, pos, _bytes);
      return pos + payload_size;
    }

    template <typename T>
    uint64_t pack(T &obj, void *bytes, uint64_t pos) {
      const ttg_data_descriptor *dObj = ttg::get_data_descriptor<ttg::meta::remove_cvr_t<T>>();
      uint64_t payload_size = dObj->payload_size(&obj);
      if constexpr (!ttg::default_data_descriptor<ttg::meta::remove_cvr_t<T>>::serialize_size_is_const) {
        const ttg_data_descriptor *dSiz = ttg::get_data_descriptor<uint64_t>();
        dSiz->pack_payload(&payload_size, sizeof(uint64_t), pos, bytes);
        pos += sizeof(uint64_t);
      }
      dObj->pack_payload(&obj, payload_size, pos, bytes);
      return pos + payload_size;
    }

    static void static_set_arg(void *data, std::size_t size, ttg::TTBase *bop) {
      assert(size >= sizeof(msg_header_t) &&
             "Trying to unpack as message that does not hold enough bytes to represent a single header");
      msg_header_t *hd = static_cast<msg_header_t *>(data);
      derivedT *obj = reinterpret_cast<derivedT *>(bop);
      switch (hd->fn_id) {
        case msg_header_t::MSG_SET_ARG: {
          if (0 <= hd->param_id) {
            assert(hd->param_id >= 0);
            assert(hd->param_id < obj->set_arg_from_msg_fcts.size());
            auto member = obj->set_arg_from_msg_fcts[hd->param_id];
            (obj->*member)(data, size);
          } else {
            // there is no good reason to have negative param ids
            abort();
          }
          break;
        }
        case msg_header_t::MSG_SET_ARGSTREAM_SIZE: {
          assert(hd->param_id >= 0);
          assert(hd->param_id < obj->set_argstream_size_from_msg_fcts.size());
          auto member = obj->set_argstream_size_from_msg_fcts[hd->param_id];
          (obj->*member)(data, size);
          break;
        }
        case msg_header_t::MSG_FINALIZE_ARGSTREAM_SIZE: {
          assert(hd->param_id >= 0);
          assert(hd->param_id < obj->finalize_argstream_from_msg_fcts.size());
          auto member = obj->finalize_argstream_from_msg_fcts[hd->param_id];
          (obj->*member)(data, size);
          break;
        }
        default:
          abort();
      }
    }

    /** Returns the task memory pool owned by the calling thread */
    inline parsec_thread_mempool_t *get_task_mempool(void) {
      auto &world_impl = world.impl();
      parsec_execution_stream_s *es = world_impl.execution_stream();
      int index = (es->virtual_process->vp_id * es->virtual_process->nb_cores + es->th_id);
      return &mempools.thread_mempools[index];
    }

    template <size_t i, typename valueT>
    void set_arg_from_msg_keylist(ttg::span<keyT> &&keylist, ttg_data_copy_t *copy) {
      /* create a dummy task that holds the copy, which can be reused by others */
      task_t *dummy;
      parsec_execution_stream_s *es = world.impl().execution_stream();
      parsec_thread_mempool_t *mempool = get_task_mempool();
      dummy = new (parsec_thread_mempool_allocate(mempool)) task_t(mempool, &this->self);
      // TODO: do we need to copy static_stream_goal in dummy?

      /* set the received value as the dummy's only data */
      dummy->parsec_task.data[0].data_in = copy;

      /* save the current task and set the dummy task */
      auto parsec_ttg_caller_save = parsec_ttg_caller;
      parsec_ttg_caller = dummy;

      /* iterate over the keys and have them use the copy we made */
      parsec_task_t *task_ring = nullptr;
      for (auto &&key : keylist) {
        set_arg_local_impl<i>(key, *reinterpret_cast<valueT *>(copy->device_private), copy, &task_ring);
      }

      if (nullptr != task_ring) {
        auto &world_impl = world.impl();
        __parsec_schedule(world_impl.execution_stream(), task_ring, 0);
      }

      /* restore the previous task */
      parsec_ttg_caller = parsec_ttg_caller_save;

      /* release the dummy task */
      complete_task_and_release(es, &dummy->parsec_task);
      parsec_thread_mempool_free(mempool, &dummy->parsec_task);
    }

    // there are 6 types of set_arg:
    // - case 1: nonvoid Key, complete Value type
    // - case 2: nonvoid Key, void Value, mixed (data+control) inputs
    // - case 3: nonvoid Key, void Value, no inputs
    // - case 4:    void Key, complete Value type
    // - case 5:    void Key, void Value, mixed (data+control) inputs
    // - case 6:    void Key, void Value, no inputs
    // implementation of these will be further split into "local-only" and global+local

    template <std::size_t i>
    void set_arg_from_msg(void *data, std::size_t size) {
      using valueT = std::tuple_element_t<i, actual_input_tuple_type>;
      using msg_t = detail::msg_t;
      msg_t *msg = static_cast<msg_t *>(data);
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        /* unpack the keys */
        uint64_t pos = 0;
        std::vector<keyT> keylist;
        int num_keys = msg->tt_id.num_keys;
        keylist.reserve(num_keys);
        auto rank = world.rank();
        for (int k = 0; k < num_keys; ++k) {
          keyT key;
          pos = unpack(key, msg->bytes, pos);
          assert(keymap(key) == rank);
          keylist.push_back(std::move(key));
        }
        // case 1
        if constexpr (!ttg::meta::is_void_v<valueT>) {
          using decvalueT = std::decay_t<valueT>;
          if constexpr (!ttg::has_split_metadata<decvalueT>::value) {
            ttg_data_copy_t *copy = detail::create_new_datacopy(decvalueT{});
            unpack(*static_cast<decvalueT *>(copy->device_private), msg->bytes, pos);

            set_arg_from_msg_keylist<i, decvalueT>(ttg::span<keyT>(&keylist[0], num_keys), copy);
          } else {
            /* unpack the header and start the RMA transfers */
            ttg::SplitMetadataDescriptor<decvalueT> descr;
            using metadata_t = decltype(descr.get_metadata(std::declval<decvalueT>()));
            size_t metadata_size = sizeof(metadata_t);

            /* unpack the metadata */
            metadata_t metadata;
            std::memcpy(&metadata, msg->bytes + pos, metadata_size);
            pos += metadata_size;

            /* unpack the remote rank */
            int remote;
            std::memcpy(&remote, msg->bytes + pos, sizeof(remote));
            pos += sizeof(remote);

            assert(remote < world.size());

            /* extract the number of chunks */
            int32_t num_iovecs;
            std::memcpy(&num_iovecs, msg->bytes + pos, sizeof(num_iovecs));
            pos += sizeof(num_iovecs);

            ttg_data_copy_t *copy = detail::create_new_datacopy(descr.create_from_metadata(metadata));
            /* nothing else to do if the object is empty */
            if (0 == num_iovecs) {
              set_arg_from_msg_keylist<i, decvalueT>(keylist, copy);
            } else {
              /* extract the callback tag */
              parsec_ce_tag_t cbtag;
              std::memcpy(&cbtag, msg->bytes + pos, sizeof(cbtag));
              pos += sizeof(cbtag);

              /* create the value from the metadata */
              auto activation = new detail::rma_delayed_activate(
                  std::move(keylist), copy, num_iovecs, [this](std::vector<keyT> &&keylist, ttg_data_copy_t *copy) {
                    set_arg_from_msg_keylist<i, decvalueT>(keylist, copy);
                    this->world.impl().decrement_inflight_msg();
                  });
              auto &val = *static_cast<decvalueT *>(copy->device_private);

              using ActivationT = std::decay_t<decltype(*activation)>;

              int nv = 0;
              /* process payload iovecs */
              auto iovecs = descr.get_data(val);
              /* start the RMA transfers */
              for (auto &&iov : iovecs) {
                ++nv;
                parsec_ce_mem_reg_handle_t rreg;
                int32_t rreg_size_i;
                std::memcpy(&rreg_size_i, msg->bytes + pos, sizeof(rreg_size_i));
                pos += sizeof(rreg_size_i);
                rreg = static_cast<parsec_ce_mem_reg_handle_t>(msg->bytes + pos);
                pos += rreg_size_i;
                // std::intptr_t *fn_ptr = reinterpret_cast<std::intptr_t *>(msg->bytes + pos);
                // pos += sizeof(*fn_ptr);
                std::intptr_t fn_ptr;
                std::memcpy(&fn_ptr, msg->bytes + pos, sizeof(fn_ptr));
                pos += sizeof(fn_ptr);

                /* register the local memory */
                parsec_ce_mem_reg_handle_t lreg;
                size_t lreg_size;
                parsec_ce.mem_register(iov.data, PARSEC_MEM_TYPE_NONCONTIGUOUS, iov.num_bytes, parsec_datatype_int8_t,
                                       iov.num_bytes, &lreg, &lreg_size);
                world.impl().increment_inflight_msg();
                /* TODO: PaRSEC should treat the remote callback as a tag, not a function pointer! */
                parsec_ce.get(&parsec_ce, lreg, 0, rreg, 0, iov.num_bytes, remote,
                              &detail::get_complete_cb<ActivationT>, activation,
                              /*world.impl().parsec_ttg_rma_tag()*/
                              cbtag, &fn_ptr, sizeof(std::intptr_t));
              }

              assert(num_iovecs == nv);
              assert(size == (pos + sizeof(msg_header_t)));
            }
          }
          // case 2 and 3
        } else if constexpr (!ttg::meta::is_void_v<keyT> && std::is_void_v<valueT>) {
          for (auto &&key : keylist) {
            set_arg<i, keyT, ttg::Void>(key, ttg::Void{});
          }
        }
        // case 4
      } else if constexpr (ttg::meta::is_void_v<keyT> && !std::is_void_v<valueT>) {
        using decvalueT = std::decay_t<valueT>;
        decvalueT val;
        /* TODO: handle split-metadata case as with non-void keys */
        unpack(val, msg->bytes, 0);
        set_arg<i, keyT, valueT>(std::move(val));
        // case 5 and 6
      } else if constexpr (ttg::meta::is_void_v<keyT> && std::is_void_v<valueT>) {
        set_arg<i, keyT, ttg::Void>(ttg::Void{});
      } else {
        abort();
      }
    }

    template <std::size_t i>
    void finalize_argstream_from_msg(void *data, std::size_t size) {
      using msg_t = detail::msg_t;
      msg_t *msg = static_cast<msg_t *>(data);
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        /* unpack the key */
        uint64_t pos = 0;
        auto rank = world.rank();
        keyT key;
        pos = unpack(key, msg->bytes, pos);
        assert(keymap(key) == rank);
        finalize_argstream<i>(key);
      } else {
        auto rank = world.rank();
        assert(keymap() == rank);
        finalize_argstream<i>();
      }
    }

    template <std::size_t i>
    void argstream_set_size_from_msg(void *data, std::size_t size) {
      using msg_t = detail::msg_t;
      auto msg = static_cast<msg_t *>(data);
      uint64_t pos = 0;
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        /* unpack the key */
        auto rank = world.rank();
        keyT key;
        pos = unpack(key, msg->bytes, pos);
        assert(keymap(key) == rank);
        std::size_t argstream_size;
        pos = unpack(argstream_size, msg->bytes, pos);
        set_argstream_size<i>(key, argstream_size);
      } else {
        auto rank = world.rank();
        assert(keymap() == rank);
        std::size_t argstream_size;
        pos = unpack(argstream_size, msg->bytes, pos);
        set_argstream_size<i>(argstream_size);
      }
    }

    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg_local(
        const Key &key, Value &&value) {
      set_arg_local_impl<i>(key, std::forward<Value>(value));
    }

    template <std::size_t i, typename Key = keyT, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg_local(
        Value &&value) {
      set_arg_local_impl<i>(ttg::Void{}, std::forward<Value>(value));
    }

    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg_local(
        const Key &key, const Value &value) {
      set_arg_local_impl<i>(key, value);
    }

    template <std::size_t i, typename Key = keyT, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg_local(
        const Value &value) {
      set_arg_local_impl<i>(ttg::Void{}, value);
    }

    template <std::size_t i, typename Key = keyT, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg_local(
        std::shared_ptr<const Value> &valueptr) {
      set_arg_local_impl<i>(ttg::Void{}, *valueptr);
    }

    template <typename Key>
    task_t *create_new_task(const Key &key) {
      constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
      auto &world_impl = world.impl();
      task_t *newtask;
      parsec_thread_mempool_t *mempool = get_task_mempool();
      char *taskobj = (char *)parsec_thread_mempool_allocate(mempool);
      int32_t priority;
      if constexpr (!keyT_is_Void) {
        priority = priomap(key);
        /* placement-new the task */
        newtask = new (taskobj) task_t(key, mempool, &this->self, world_impl.taskpool(), this, priority);
      } else {
        priority = priomap();
        /* placement-new the task */
        newtask = new (taskobj) task_t(mempool, &this->self, world_impl.taskpool(), this, priority);
      }

      newtask->function_template_class_ptr[static_cast<std::size_t>(ttg::ExecutionSpace::Host)] =
          reinterpret_cast<detail::parsec_static_op_t>(&TT::static_op<ttg::ExecutionSpace::Host>);
      if constexpr (derived_has_cuda_op())
        newtask->function_template_class_ptr[static_cast<std::size_t>(ttg::ExecutionSpace::CUDA)] =
            reinterpret_cast<detail::parsec_static_op_t>(&TT::static_op<ttg::ExecutionSpace::CUDA>);

      for (int i = 0; i < static_stream_goal.size(); ++i) {
        newtask->stream[i].goal = static_stream_goal[i];
      }

      ttg::trace(world.rank(), ":", get_name(), " : ", key, ": creating task");
      return newtask;
    }

    // Used to set the i'th argument
    template <std::size_t i, typename Key, typename Value>
    void set_arg_local_impl(const Key &key, Value &&value, ttg_data_copy_t *copy_in = nullptr,
                            parsec_task_t **task_ring = nullptr) {
      using valueT = std::tuple_element_t<i, input_values_full_tuple_type>;
      constexpr const bool input_is_const = std::is_const_v<std::tuple_element_t<i, input_args_type>>;
      constexpr const bool valueT_is_Void = ttg::meta::is_void_v<valueT>;
      constexpr const bool keyT_is_Void = ttg::meta::is_void_v<Key>;

      if constexpr (!valueT_is_Void) {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": received value for argument : ", i,
                   " : value = ", value);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": received value for argument : ", i);
      }

      parsec_key_t hk = 0;
      if constexpr (!keyT_is_Void) {
        hk = reinterpret_cast<parsec_key_t>(&key);
        assert(keymap(key) == world.rank());
      }

      task_t *task;
      auto &world_impl = world.impl();
      auto &reducer = std::get<i>(input_reducers);
      bool release = false;
      bool remove_from_hash = true;
      /* If we have only one input and no reducer on that input we can skip the hash table */
      if (numins > 1 || reducer) {
        parsec_hash_table_lock_bucket(&tasks_table, hk);
        if (nullptr == (task = (task_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
          task = create_new_task(key);
          world_impl.increment_created();
          parsec_hash_table_nolock_insert(&tasks_table, &task->tt_ht_item);
        } else if (!reducer && numins == (task->in_data_count + 1)) {
          /* remove while we have the lock */
          parsec_hash_table_nolock_remove(&tasks_table, hk);
          remove_from_hash = false;
          release = true;
        }
        parsec_hash_table_unlock_bucket(&tasks_table, hk);
      } else {
        task = create_new_task(key);
        world_impl.increment_created();
        remove_from_hash = false;
      }

      if (reducer) {  // is this a streaming input? reduce the received value
        // N.B. Right now reductions are done eagerly, without spawning tasks
        //      this means we must lock
        parsec_hash_table_lock_bucket(&tasks_table, hk);

        if constexpr (!ttg::meta::is_void_v<valueT>) {  // for data values
          // have a value already? if not, set, otherwise reduce
          ttg_data_copy_t *copy = nullptr;
          if (nullptr == (copy = static_cast<ttg_data_copy_t *>(task->parsec_task.data[i].data_in))) {
            using decay_valueT = std::decay_t<valueT>;
            if (nullptr == copy_in) {
              copy = detail::create_new_datacopy(std::forward<Value>(value));
            } else {
              copy = detail::register_data_copy<valueT>(copy_in, task, input_is_const);
            }
            task->parsec_task.data[i].data_in = copy;
          } else {
            reducer(*reinterpret_cast<std::decay_t<valueT> *>(copy->device_private), value);
          }
        } else {
          reducer();  // even if this was a control input, must execute the reducer for possible side effects
        }
        task->stream[i].size++;
        release = (task->stream[i].size == task->stream[i].goal);
        if (release) {
          parsec_hash_table_nolock_remove(&tasks_table, hk);
          remove_from_hash = false;
        }
        parsec_hash_table_unlock_bucket(&tasks_table, hk);
      } else {
        /* whether the task needs to be deferred or not */
        bool needs_deferring = false;
        if constexpr (!valueT_is_Void) {
          if (nullptr != task->parsec_task.data[i].data_in) {
            ttg::print_error(get_name(), " : ", key, ": error argument is already set : ", i);
            throw std::logic_error("bad set arg");
          }

          ttg_data_copy_t *copy = copy_in;
          if (nullptr == copy_in && nullptr != parsec_ttg_caller) {
            copy = detail::find_copy_in_task(parsec_ttg_caller, &value);
          }

          if (nullptr != copy) {
            /* register_data_copy might provide us with a different copy if !input_is_const */
            copy = detail::register_data_copy<valueT>(copy, task, input_is_const);
          } else {
            copy = detail::create_new_datacopy(std::forward<Value>(value));
          }
          /* if we registered as a writer and were the first to register with this copy
           * we need to defer the release of this task to give other tasks a chance to
           * make a copy of the original data */
          needs_deferring = (copy->push_task != nullptr);
          task->parsec_task.data[i].data_in = copy;
        }
        if (needs_deferring) {
          if (nullptr == task->deferred_release) {
            if (remove_from_hash) {
              task->deferred_release = &release_task_to_scheduler<true>;
            } else {
              task->deferred_release = &release_task_to_scheduler<false>;
            }
            task->tt_ptr = this;
          }
        }
        release = !needs_deferring;
      }
      if (release) {
        if (remove_from_hash) {
          release_task<true>(this, task, task_ring);
        } else {
          release_task<false>(this, task, task_ring);
        }
      }
    }

    template <bool RemoveFromHash>
    static void release_task_to_scheduler(void *tt_ptr, detail::parsec_ttg_task_base_t *base_task) {
      release_task<RemoveFromHash>(tt_ptr, base_task, nullptr);
    }

    template <bool RemoveFromHash>
    static void release_task(void *tt_ptr, detail::parsec_ttg_task_base_t *base_task,
                             parsec_task_t **task_ring = nullptr) {
      constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
      task_t *task = static_cast<task_t *>(base_task);
      ttT &tt = *reinterpret_cast<ttT *>(tt_ptr);

      /* if RemoveFromHash == false, someone has already removed the task from the hash table
       * so we know that the task is ready, no need to do atomic increments here */
      constexpr const bool is_ready = !RemoveFromHash;
      int32_t count;
      if constexpr (is_ready) {
        count = numins;
      } else {
        count = parsec_atomic_fetch_inc_int32(&task->in_data_count) + 1;
        assert(count <= tt.self.dependencies_goal);
      }
      auto &world_impl = tt.world.impl();

      if (count == numins) {
        parsec_execution_stream_t *es = world_impl.execution_stream();
        parsec_key_t hk = task->pkey();
        if (tt.tracing()) {
          if constexpr (!keyT_is_Void) {
            ttg::trace(tt.world.rank(), ":", tt.get_name(), " : ", task->key, ": submitting task for op ");
          } else {
            ttg::trace(tt.world.rank(), ":", tt.get_name(), ": submitting task for op ");
          }
        }
        if (RemoveFromHash) parsec_hash_table_remove(&tt.tasks_table, hk);
        if (nullptr == task_ring) {
          __parsec_schedule(es, &task->parsec_task, 0);
        } else if (*task_ring == nullptr) {
          /* the first task is set directly */
          *task_ring = &task->parsec_task;
        } else {
          /* push into the ring */
          parsec_list_item_ring_push_sorted(&(*task_ring)->super, &task->parsec_task.super,
                                            offsetof(parsec_task_t, priority));
        }
      }
    }

    // cases 1+2
    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg(const Key &key,
                                                                                                       Value &&value) {
      set_arg_impl<i>(key, std::forward<Value>(value));
    }

    // cases 4+5+6
    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg(Value &&value) {
      set_arg_impl<i>(ttg::Void{}, std::forward<Value>(value));
    }

    template <std::size_t i, typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key>, void> set_arg() {
      set_arg_impl<i>(ttg::Void{}, ttg::Void{});
    }

    // case 3
    template <std::size_t i, typename Key>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, void> set_arg(const Key &key) {
      set_arg_impl<i>(key, ttg::Void{});
    }

    // Used to set the i'th argument
    template <std::size_t i, typename Key, typename Value>
    void set_arg_impl(const Key &key, Value &&value) {
      int owner;

      if constexpr (!ttg::meta::is_void_v<Key>)
        owner = keymap(key);
      else
        owner = keymap();
      if (owner == world.rank()) {
        if constexpr (!ttg::meta::is_void_v<keyT>)
          set_arg_local<i, keyT, Value>(key, std::forward<Value>(value));
        else
          set_arg_local<i, keyT, Value>(std::forward<Value>(value));
        return;
      }
      // the target task is remote. Pack the information and send it to
      // the corresponding peer.
      // TODO do we need to copy value?
      using msg_t = detail::msg_t;
      auto &world_impl = world.impl();
      uint64_t pos = 0;
      std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                           msg_header_t::MSG_SET_ARG, i, 1);
      using decvalueT = std::decay_t<Value>;
      /* pack the key */
      msg->tt_id.num_keys = 0;
      if constexpr (!ttg::meta::is_void_v<Key>) {
        pos = pack(key, msg->bytes, pos);
        msg->tt_id.num_keys = 1;
      }
      if constexpr (!ttg::meta::is_void_v<decvalueT>) {
        if constexpr (!ttg::has_split_metadata<decvalueT>::value) {
          // std::cout << "set_arg_from_msg unpacking from offset " << sizeof(keyT) << std::endl;
          pos = pack(value, msg->bytes, pos);
        } else {
          ttg_data_copy_t *copy;
          copy = detail::find_copy_in_task(parsec_ttg_caller, &value);
          if (nullptr == copy) {
            // We need to create a copy for this data, as it does not exist yet.
            copy = detail::create_new_datacopy(std::forward<Value>(value));
          }
          copy = detail::register_data_copy<decvalueT>(copy, nullptr, true);

          ttg::SplitMetadataDescriptor<decvalueT> descr;
          auto metadata = descr.get_metadata(value);
          size_t metadata_size = sizeof(metadata);
          /* pack the metadata */
          std::memcpy(msg->bytes + pos, &metadata, metadata_size);
          pos += metadata_size;
          /* pack the local rank */
          int rank = world.rank();
          std::memcpy(msg->bytes + pos, &rank, sizeof(rank));
          pos += sizeof(rank);

          auto iovecs = descr.get_data(*static_cast<decvalueT *>(copy->device_private));

          int32_t num_iovs = std::distance(std::begin(iovecs), std::end(iovecs));
          std::memcpy(msg->bytes + pos, &num_iovs, sizeof(num_iovs));
          pos += sizeof(num_iovs);

          /* TODO: at the moment, the tag argument to parsec_ce.get() is treated as a
           * raw function pointer instead of a preregistered AM tag, so play that game.
           * Once this is fixed in PaRSEC we need to use parsec_ttg_rma_tag instead! */
          parsec_ce_tag_t cbtag = reinterpret_cast<parsec_ce_tag_t>(&detail::get_remote_complete_cb);
          std::memcpy(msg->bytes + pos, &cbtag, sizeof(cbtag));
          pos += sizeof(cbtag);

          /**
           * register the generic iovecs and pack the registration handles
           * memory layout: [<lreg_size, lreg, release_cb_ptr>, ...]
           */
          for (auto &&iov : iovecs) {
            parsec_ce_mem_reg_handle_t lreg;
            size_t lreg_size;
            /* TODO: only register once when we can broadcast the data! */
            parsec_ce.mem_register(iov.data, PARSEC_MEM_TYPE_NONCONTIGUOUS, iov.num_bytes, parsec_datatype_int8_t,
                                   iov.num_bytes, &lreg, &lreg_size);
            auto lreg_ptr = std::shared_ptr<void>{lreg, [](void *ptr) {
                                                    parsec_ce_mem_reg_handle_t memreg = (parsec_ce_mem_reg_handle_t)ptr;
                                                    parsec_ce.mem_unregister(&memreg);
                                                  }};
            int32_t lreg_size_i = lreg_size;
            std::memcpy(msg->bytes + pos, &lreg_size_i, sizeof(lreg_size_i));
            pos += sizeof(lreg_size_i);
            std::memcpy(msg->bytes + pos, lreg, lreg_size_i);
            pos += lreg_size_i;
            /* TODO: can we avoid the extra indirection of going through std::function? */
            std::function<void(void)> *fn = new std::function<void(void)>([=]() mutable {
              /* shared_ptr of value and registration captured by value so resetting
               * them here will eventually release the memory/registration */
              detail::release_data_copy(copy);
              lreg_ptr.reset();
            });
            std::intptr_t fn_ptr{reinterpret_cast<std::intptr_t>(fn)};
            std::memcpy(msg->bytes + pos, &fn_ptr, sizeof(fn_ptr));
            pos += sizeof(fn_ptr);
          }
        }
      }
      parsec_taskpool_t *tp = world_impl.taskpool();
      tp->tdm.module->outgoing_message_start(tp, owner, NULL);
      tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
      // std::cout << "Sending AM with " << msg->op_id.num_keys << " keys " << std::endl;
      parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                        sizeof(msg_header_t) + pos);
    }

    template <int i, typename Iterator, typename Value>
    void broadcast_arg_local(Iterator &&begin, Iterator &&end, const Value &value) {
      parsec_task_t *task_ring = nullptr;
      ttg_data_copy_t *copy = nullptr;
      if (nullptr != parsec_ttg_caller) {
        copy = detail::find_copy_in_task(parsec_ttg_caller, &value);
      }

      for (auto it = begin; it != end; ++it) {
        set_arg_local_impl<i>(*it, value, copy, &task_ring);
      }
      /* submit all ready tasks at once */
      if (nullptr != task_ring) {
        __parsec_schedule(world.impl().execution_stream(), task_ring, 0);
      }
    }

    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>> &&
                         !ttg::has_split_metadata<std::decay_t<Value>>::value,
                     void>
    broadcast_arg(const ttg::span<const Key> &keylist, const Value &value) {
      auto world = ttg_default_execution_context();
      int rank = world.rank();

      bool have_remote = keylist.end() != std::find_if(keylist.begin(), keylist.end(),
                                                       [&](const Key &key) { return keymap(key) != rank; });

      if (have_remote) {
        std::vector<Key> keylist_sorted(keylist.begin(), keylist.end());

        /* Assuming there are no local keys, will be updated while processing remote keys */
        auto local_begin = keylist_sorted.end();
        auto local_end = keylist_sorted.end();

        /* sort the input key list by owner and check whether there are remote keys */
        std::sort(keylist_sorted.begin(), keylist_sorted.end(), [&](const Key &a, const Key &b) mutable {
          int rank_a = keymap(a);
          int rank_b = keymap(b);
          return rank_a < rank_b;
        });

        using msg_t = detail::msg_t;
        local_begin = keylist_sorted.end();
        auto &world_impl = world.impl();
        std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                             msg_header_t::MSG_SET_ARG, i);

        parsec_taskpool_t *tp = world_impl.taskpool();

        for (auto it = keylist_sorted.begin(); it < keylist_sorted.end(); /* increment inline */) {
          auto owner = keymap(*it);
          if (owner == rank) {
            /* make sure we don't lose local keys */
            local_begin = it;
            local_end =
                std::find_if_not(++it, keylist_sorted.end(), [&](const Key &key) { return keymap(key) == rank; });
            it = local_end;
            continue;
          }

          /* pack all keys for this owner */
          int num_keys = 0;
          uint64_t pos = 0;
          do {
            ++num_keys;
            pos = pack(*it, msg->bytes, pos);
            ++it;
          } while (it < keylist_sorted.end() && keymap(*it) == owner);
          msg->tt_id.num_keys = num_keys;

          /* TODO: use RMA to transfer the value */
          pos = pack(value, msg->bytes, pos);

          /* Send the message */
          tp->tdm.module->outgoing_message_start(tp, owner, NULL);
          tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
          parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                            sizeof(msg_header_t) + pos);
        }
        /* handle local keys */
        broadcast_arg_local<i>(local_begin, local_end, value);
      } else {
        /* only local keys */
        broadcast_arg_local<i>(keylist.begin(), keylist.end(), value);
      }
    }

    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>> &&
                         ttg::has_split_metadata<std::decay_t<Value>>::value,
                     void>
    splitmd_broadcast_arg(const ttg::span<const Key> &keylist, const Value &value) {
      using valueT = std::tuple_element_t<i, input_values_full_tuple_type>;
      auto world = ttg_default_execution_context();
      int rank = world.rank();
      bool have_remote = keylist.end() != std::find_if(keylist.begin(), keylist.end(),
                                                       [&](const Key &key) { return keymap(key) != rank; });

      if (have_remote) {
        using decvalueT = std::decay_t<Value>;

        /* sort the input key list by owner and check whether there are remote keys */
        std::vector<Key> keylist_sorted(keylist.begin(), keylist.end());
        std::sort(keylist_sorted.begin(), keylist_sorted.end(), [&](const Key &a, const Key &b) mutable {
          int rank_a = keymap(a);
          int rank_b = keymap(b);
          return rank_a < rank_b;
        });

        /* Assuming there are no local keys, will be updated while iterating over the keys */
        auto local_begin = keylist_sorted.end();
        auto local_end = keylist_sorted.end();

        ttg::SplitMetadataDescriptor<decvalueT> descr;
        auto iovs = descr.get_data(*const_cast<decvalueT *>(&value));
        int32_t num_iovs = std::distance(std::begin(iovs), std::end(iovs));
        std::vector<std::pair<int32_t, std::shared_ptr<void>>> memregs;
        memregs.reserve(num_iovs);

        /* register all iovs so the registration can be reused */
        for (auto &&iov : iovs) {
          parsec_ce_mem_reg_handle_t lreg;
          size_t lreg_size;
          parsec_ce.mem_register(iov.data, PARSEC_MEM_TYPE_NONCONTIGUOUS, iov.num_bytes, parsec_datatype_int8_t,
                                 iov.num_bytes, &lreg, &lreg_size);
          /* TODO: use a static function for deregistration here? */
          memregs.push_back(std::make_pair(static_cast<int32_t>(lreg_size),
                                           /* TODO: this assumes that parsec_ce_mem_reg_handle_t is void* */
                                           std::shared_ptr<void>{lreg, [](void *ptr) {
                                                                   parsec_ce_mem_reg_handle_t memreg =
                                                                       (parsec_ce_mem_reg_handle_t)ptr;
                                                                   parsec_ce.mem_unregister(&memreg);
                                                                 }}));
        }

        using msg_t = detail::msg_t;
        auto &world_impl = world.impl();
        std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                             msg_header_t::MSG_SET_ARG, i);
        auto metadata = descr.get_metadata(value);
        size_t metadata_size = sizeof(metadata);

        ttg_data_copy_t *copy;
        copy = detail::find_copy_in_task(parsec_ttg_caller, &value);
        assert(nullptr != copy);

        parsec_taskpool_t *tp = world_impl.taskpool();
        for (auto it = keylist_sorted.begin(); it < keylist_sorted.end(); /* increment done inline */) {
          auto owner = keymap(*it);
          if (owner == rank) {
            local_begin = it;
            /* find first non-local key */
            local_end =
                std::find_if_not(++it, keylist_sorted.end(), [&](const Key &key) { return keymap(key) == rank; });
            it = local_end;
            continue;
          }

          /* count keys and set it afterwards */
          uint64_t pos = 0;
          /* pack all keys for this owner */
          int num_keys = 0;
          do {
            ++num_keys;
            pos = pack(*it, msg->bytes, pos);
            ++it;
          } while (it < keylist_sorted.end() && keymap(*it) == owner);
          msg->tt_id.num_keys = num_keys;

          /* pack the metadata */
          std::memcpy(msg->bytes + pos, &metadata, metadata_size);
          pos += metadata_size;
          /* pack the local rank */
          int rank = world.rank();
          std::memcpy(msg->bytes + pos, &rank, sizeof(rank));
          pos += sizeof(rank);
          /* pack the number of iovecs */
          std::memcpy(msg->bytes + pos, &num_iovs, sizeof(num_iovs));
          pos += sizeof(num_iovs);

          /* TODO: at the moment, the tag argument to parsec_ce.get() is treated as a
           * raw function pointer instead of a preregistered AM tag, so play that game.
           * Once this is fixed in PaRSEC we need to use parsec_ttg_rma_tag instead! */
          parsec_ce_tag_t cbtag = reinterpret_cast<parsec_ce_tag_t>(&detail::get_remote_complete_cb);
          std::memcpy(msg->bytes + pos, &cbtag, sizeof(cbtag));
          pos += sizeof(cbtag);

          /**
           * pack the registration handles
           * memory layout: [<lreg_size, lreg, lreg_fn>, ...]
           */
          int idx = 0;
          for (auto &&iov : iovs) {
            // auto [lreg_size, lreg_ptr] = memregs[idx];
            int32_t lreg_size;
            std::shared_ptr<void> lreg_ptr;
            std::tie(lreg_size, lreg_ptr) = memregs[idx];
            std::memcpy(msg->bytes + pos, &lreg_size, sizeof(lreg_size));
            pos += sizeof(lreg_size);
            std::memcpy(msg->bytes + pos, lreg_ptr.get(), lreg_size);
            pos += lreg_size;
            /* create a function that will be invoked upon RMA completion at the target */
            std::shared_ptr<void> lreg_ptr_v = lreg_ptr;
            /* mark another reader on the copy */
            copy = detail::register_data_copy<valueT>(copy, nullptr, true);
            std::function<void(void)> *fn = new std::function<void(void)>([=]() mutable {
              /* shared_ptr of value and registration captured by value so resetting
               * them here will eventually release the memory/registration */
              detail::release_data_copy(copy);
              lreg_ptr_v.reset();
            });
            std::intptr_t fn_ptr{reinterpret_cast<std::intptr_t>(fn)};
            std::memcpy(msg->bytes + pos, &fn_ptr, sizeof(fn_ptr));
            pos += sizeof(fn_ptr);
            ++idx;
          }
          tp->tdm.module->outgoing_message_start(tp, owner, NULL);
          tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
          parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                            sizeof(msg_header_t) + pos);
        }
        /* handle local keys */
        broadcast_arg_local<i>(local_begin, local_end, value);
      } else {
        /* handle local keys */
        broadcast_arg_local<i>(keylist.begin(), keylist.end(), value);
      }
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of elements in args
    // Js: index sequence of input terminals to set
    template <typename Key, typename... Ts, size_t... Is, size_t... Js>
    std::enable_if_t<ttg::meta::is_none_void_v<Key>, void> set_args(std::index_sequence<Is...>,
                                                                    std::index_sequence<Js...>, const Key &key,
                                                                    const std::tuple<Ts...> &args) {
      static_assert(sizeof...(Js) == sizeof...(Is));
      constexpr size_t js[] = {Js...};
      int junk[] = {0, (set_arg<js[Is]>(key, TT::get<Is>(args)), 0)...};
      junk[0]++;
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of input terminals to set
    template <typename Key, typename... Ts, size_t... Is>
    std::enable_if_t<ttg::meta::is_none_void_v<Key>, void> set_args(std::index_sequence<Is...> is, const Key &key,
                                                                    const std::tuple<Ts...> &args) {
      set_args(std::index_sequence_for<Ts...>{}, is, key, args);
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of elements in args
    // Js: index sequence of input terminals to set
    template <typename Key = keyT, typename... Ts, size_t... Is, size_t... Js>
    std::enable_if_t<ttg::meta::is_void_v<Key>, void> set_args(std::index_sequence<Is...>, std::index_sequence<Js...>,
                                                               const std::tuple<Ts...> &args) {
      static_assert(sizeof...(Js) == sizeof...(Is));
      constexpr size_t js[] = {Js...};
      int junk[] = {0, (set_arg<js[Is], void>(TT::get<Is>(args)), 0)...};
      junk[0]++;
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of input terminals to set
    template <typename Key = keyT, typename... Ts, size_t... Is>
    std::enable_if_t<ttg::meta::is_void_v<Key>, void> set_args(std::index_sequence<Is...> is,
                                                               const std::tuple<Ts...> &args) {
      set_args(std::index_sequence_for<Ts...>{}, is, args);
    }

   public:
    // sets the default stream size for input \c i
    // \param size positive integer that specifies the default stream size
    template <std::size_t i>
    void set_static_argstream_size(std::size_t size) {
      assert(std::get<i>(input_reducers) && "TT::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "TT::set_static_argstream_size(key,size) called with size=0");

      this->trace(world.rank(), ":", get_name(), ": setting global stream size for terminal ", i);

      // Check if stream is already bounded
      if (static_stream_goal[i] > 0) {
        ttg::print_error(world.rank(), ":", get_name(), " : error stream is already bounded : ", i);
        throw std::runtime_error("TT::set_static_argstream_size called for a bounded stream");
      }

      static_stream_goal[i] = size;
    }

    /// sets stream size for input \c i
    /// \param size positive integer that specifies the stream size
    template <std::size_t i, typename Key>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, void> set_argstream_size(const Key &key, std::size_t size) {
      // preconditions
      assert(std::get<i>(input_reducers) && "TT::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "TT::set_argstream_size(key,size) called with size=0");

      // body
      const auto owner = keymap(key);
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), ":", key, " : forwarding stream size for terminal ", i);
        using msg_t = detail::msg_t;
        auto &world_impl = world.impl();
        uint64_t pos = 0;
        std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                             msg_header_t::MSG_SET_ARGSTREAM_SIZE, i, 1);
        /* pack the key */
        pos = pack(key, msg->bytes, pos);
        msg->tt_id.num_keys = 1;
        pos = pack(size, msg->bytes, pos);
        parsec_taskpool_t *tp = world_impl.taskpool();
        tp->tdm.module->outgoing_message_start(tp, owner, NULL);
        tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
        parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                          sizeof(msg_header_t) + pos);
      } else {
        ttg::trace(world.rank(), ":", get_name(), ":", key, " : setting stream size to ", size, " for terminal ", i);

        auto hk = reinterpret_cast<parsec_key_t>(&key);
        task_t *task;
        parsec_hash_table_lock_bucket(&tasks_table, hk);
        if (nullptr == (task = (task_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
          task = create_new_task(key);
          world.impl().increment_created();
          parsec_hash_table_nolock_insert(&tasks_table, &task->tt_ht_item);
        }

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        task->stream[i].goal = size;
        bool release = (task->stream[i].size == task->stream[i].goal);
        parsec_hash_table_unlock_bucket(&tasks_table, hk);

        if (release) release_task<true>(this, task);
      }
    }

    /// sets stream size for input \c i
    /// \param size positive integer that specifies the stream size
    template <std::size_t i, typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key>, void> set_argstream_size(std::size_t size) {
      // preconditions
      assert(std::get<i>(input_reducers) && "TT::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "TT::set_argstream_size(key,size) called with size=0");

      // body
      const auto owner = keymap();
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), " : forwarding stream size for terminal ", i);
        using msg_t = detail::msg_t;
        auto &world_impl = world.impl();
        uint64_t pos = 0;
        std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                             msg_header_t::MSG_SET_ARGSTREAM_SIZE, i, 1);
        /* pack the key */
        msg->tt_id.num_keys = 0;
        pos = pack(size, msg->bytes, pos);
        parsec_taskpool_t *tp = world_impl.taskpool();
        tp->tdm.module->outgoing_message_start(tp, owner, NULL);
        tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
        parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                          sizeof(msg_header_t) + pos);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : setting stream size to ", size, " for terminal ", i);

        parsec_key_t hk = 0;
        task_t *task;
        parsec_hash_table_lock_bucket(&tasks_table, hk);
        if (nullptr == (task = (task_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
          task = create_new_task(ttg::Void{});
          world.impl().increment_created();
          parsec_hash_table_nolock_insert(&tasks_table, &task->tt_ht_item);
        }

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        task->stream[i].goal = size;
        bool release = (task->stream[i].size == task->stream[i].goal);
        parsec_hash_table_unlock_bucket(&tasks_table, hk);

        if (release) release_task<true>(this, task);
      }
    }

    /// finalizes stream for input \c i
    template <std::size_t i, typename Key>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, void> finalize_argstream(const Key &key) {
      // preconditions
      assert(std::get<i>(input_reducers) && "TT::finalize_argstream called on nonstreaming input terminal");

      // body
      const auto owner = keymap(key);
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": forwarding stream finalize for terminal ", i);
        using msg_t = detail::msg_t;
        auto &world_impl = world.impl();
        uint64_t pos = 0;
        std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                             msg_header_t::MSG_FINALIZE_ARGSTREAM_SIZE, i, 1);
        /* pack the key */
        pos = pack(key, msg->bytes, pos);
        msg->tt_id.num_keys = 1;
        parsec_taskpool_t *tp = world_impl.taskpool();
        tp->tdm.module->outgoing_message_start(tp, owner, NULL);
        tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
        parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                          sizeof(msg_header_t) + pos);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": finalizing stream for terminal ", i);

        auto hk = reinterpret_cast<parsec_key_t>(&key);
        task_t *task = nullptr;
        parsec_hash_table_lock_bucket(&tasks_table, hk);
        if (nullptr == (task = (task_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
          ttg::print_error(world.rank(), ":", get_name(), ":", key,
                           " : error finalize called on stream that never received an input data: ", i);
          throw std::runtime_error("TT::finalize called on stream that never received an input data");
        }

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        task->stream[i].size = 1;
        parsec_hash_table_unlock_bucket(&tasks_table, hk);

        release_task<true>(this, task);
      }
    }

    /// finalizes stream for input \c i
    template <std::size_t i, bool key_is_void = ttg::meta::is_void_v<keyT>>
    std::enable_if_t<key_is_void, void> finalize_argstream() {
      // preconditions
      assert(std::get<i>(input_reducers) && "TT::finalize_argstream called on nonstreaming input terminal");

      // body
      const auto owner = keymap();
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), ": forwarding stream finalize for terminal ", i);
        using msg_t = detail::msg_t;
        auto &world_impl = world.impl();
        uint64_t pos = 0;
        std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                             msg_header_t::MSG_FINALIZE_ARGSTREAM_SIZE, i, 1);
        msg->tt_id.num_keys = 0;
        parsec_taskpool_t *tp = world_impl.taskpool();
        tp->tdm.module->outgoing_message_start(tp, owner, NULL);
        tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
        parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                          sizeof(msg_header_t) + pos);
      } else {
        ttg::trace(world.rank(), ":", get_name(), ": finalizing stream for terminal ", i);

        auto hk = static_cast<parsec_key_t>(0);
        task_t *task = nullptr;
        parsec_hash_table_lock_bucket(&tasks_table, hk);
        if (nullptr == (task = (task_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
          ttg::print_error(world.rank(), ":", get_name(),
                           " : error finalize called on stream that never received an input data: ", i);
          throw std::runtime_error("TT::finalize called on stream that never received an input data");
        }

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        task->stream[i].size = 1;
        parsec_hash_table_unlock_bucket(&tasks_table, hk);

        release_task<true>(this, task);
      }
    }

   private:
    // Copy/assign/move forbidden ... we could make it work using
    // PIMPL for this base class.  However, this instance of the base
    // class is tied to a specific instance of a derived class a
    // pointer to which is captured for invoking derived class
    // functions.  Thus, not only does the derived class has to be
    // involved but we would have to do it in a thread safe way
    // including for possibly already running tasks and remote
    // references.  This is not worth the effort ... wherever you are
    // wanting to move/assign an TT you should be using a pointer.
    TT(const TT &other) = delete;
    TT &operator=(const TT &other) = delete;
    TT(TT &&other) = delete;
    TT &operator=(TT &&other) = delete;

    // Registers the callback for the i'th input terminal
    template <typename terminalT, std::size_t i>
    void register_input_callback(terminalT &input) {
      using valueT = typename terminalT::value_type;
      //////////////////////////////////////////////////////////////////
      // case 1: nonvoid key, nonvoid value
      //////////////////////////////////////////////////////////////////
      if constexpr (!ttg::meta::is_void_v<keyT> && !std::is_void_v<valueT>) {
        auto move_callback = [this](const keyT &key, valueT &&value) {
          set_arg<i, keyT, valueT>(key, std::forward<valueT>(value));
        };
        auto send_callback = [this](const keyT &key, const valueT &value) {
          set_arg<i, keyT, const valueT &>(key, value);
        };
        auto broadcast_callback = [this](const ttg::span<const keyT> &keylist, const valueT &value) {
          if constexpr (ttg::has_split_metadata<std::decay_t<valueT>>::value) {
            splitmd_broadcast_arg<i, keyT, valueT>(keylist, value);
          } else {
            broadcast_arg<i, keyT, valueT>(keylist, value);
          }
        };
        auto setsize_callback = [this](const keyT &key, std::size_t size) { set_argstream_size<i>(key, size); };
        auto finalize_callback = [this](const keyT &key) { finalize_argstream<i>(key); };
        input.set_callback(send_callback, move_callback, broadcast_callback, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 2: nonvoid key, void value, mixed inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (!ttg::meta::is_void_v<keyT> && std::is_void_v<valueT>) {
        auto send_callback = [this](const keyT &key) { set_arg<i, keyT, ttg::Void>(key, ttg::Void{}); };
        auto setsize_callback = [this](const keyT &key, std::size_t size) { set_argstream_size<i>(key, size); };
        auto finalize_callback = [this](const keyT &key) { finalize_argstream<i>(key); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 3: nonvoid key, void value, no inputs
      // NOTE: subsumed in case 2 above, kept for historical reasons
      //////////////////////////////////////////////////////////////////
      //////////////////////////////////////////////////////////////////
      // case 4: void key, nonvoid value
      //////////////////////////////////////////////////////////////////
      else if constexpr (ttg::meta::is_void_v<keyT> && !std::is_void_v<valueT>) {
        auto move_callback = [this](valueT &&value) { set_arg<i, keyT, valueT>(std::forward<valueT>(value)); };
        auto send_callback = [this](const valueT &value) { set_arg<i, keyT, const valueT &>(value); };
        auto setsize_callback = [this](std::size_t size) { set_argstream_size<i>(size); };
        auto finalize_callback = [this]() { finalize_argstream<i>(); };
        input.set_callback(send_callback, move_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 5: void key, void value, mixed inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (ttg::meta::is_void_v<keyT> && std::is_void_v<valueT>) {
        auto send_callback = [this]() { set_arg<i, keyT, ttg::Void>(ttg::Void{}); };
        auto setsize_callback = [this](std::size_t size) { set_argstream_size<i>(size); };
        auto finalize_callback = [this]() { finalize_argstream<i>(); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 6: void key, void value, no inputs
      // NOTE: subsumed in case 5 above, kept for historical reasons
      //////////////////////////////////////////////////////////////////
      else
        abort();
    }

    template <std::size_t... IS>
    void register_input_callbacks(std::index_sequence<IS...>) {
      int junk[] = {
          0,
          (register_input_callback<std::tuple_element_t<IS, input_terminals_type>, IS>(std::get<IS>(input_terminals)),
           0)...};
      junk[0]++;
    }

    template <std::size_t... IS, typename inedgesT>
    void connect_my_inputs_to_incoming_edge_outputs(std::index_sequence<IS...>, inedgesT &inedges) {
      int junk[] = {0, (std::get<IS>(inedges).set_out(&std::get<IS>(input_terminals)), 0)...};
      junk[0]++;
    }

    template <std::size_t... IS, typename outedgesT>
    void connect_my_outputs_to_outgoing_edge_inputs(std::index_sequence<IS...>, outedgesT &outedges) {
      int junk[] = {0, (std::get<IS>(outedges).set_in(&std::get<IS>(output_terminals)), 0)...};
      junk[0]++;
    }

    template <typename input_terminals_tupleT, std::size_t... IS, typename flowsT>
    void _initialize_flows(std::index_sequence<IS...>, flowsT &&flows) {
      int junk[] = {0,
                    (*(const_cast<std::remove_const_t<decltype(flows[IS]->flow_flags)> *>(&(flows[IS]->flow_flags))) =
                         (std::is_const_v<std::tuple_element_t<IS, input_terminals_tupleT>> ? PARSEC_FLOW_ACCESS_READ
                                                                                            : PARSEC_FLOW_ACCESS_RW),
                     0)...};
      junk[0]++;
    }

    template <typename input_terminals_tupleT, typename flowsT>
    void initialize_flows(flowsT &&flows) {
      _initialize_flows<input_terminals_tupleT>(
          std::make_index_sequence<std::tuple_size<input_terminals_tupleT>::value>{}, flows);
    }

    void fence() override { ttg::default_execution_context().impl().fence(); }

    static int key_equal(parsec_key_t a, parsec_key_t b, void *user_data) {
      if constexpr (std::is_same_v<keyT, void>) {
        return 1;
      } else {
        keyT &ka = *(reinterpret_cast<keyT *>(a));
        keyT &kb = *(reinterpret_cast<keyT *>(b));
        return ka == kb;
      }
    }

    static uint64_t key_hash(parsec_key_t k, void *user_data) {
      constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
      if constexpr (keyT_is_Void || std::is_same_v<keyT, void>) {
        return 0;
      } else {
        keyT &kk = *(reinterpret_cast<keyT *>(k));
        using ttg::hash;
        uint64_t hv = hash<decltype(kk)>{}(kk);
        return hv;
      }
    }

    static char *key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data) {
      if constexpr (std::is_same_v<keyT, void>) {
        buffer[0] = '\0';
        return buffer;
      } else {
        keyT kk = *(reinterpret_cast<keyT *>(k));
        std::stringstream iss;
        iss << kk;
        memset(buffer, 0, buffer_size);
        iss.get(buffer, buffer_size);
        return buffer;
      }
    }

    parsec_key_fn_t tasks_hash_fcts = {key_equal, key_print, key_hash};

    static parsec_hook_return_t complete_task_and_release(parsec_execution_stream_t *es, parsec_task_t *t) {
      parsec_execution_stream_t *safe_es = parsec_ttg_es;
      parsec_ttg_es = es;
      auto *task = (detail::parsec_ttg_task_base_t *)t;
      for (int i = 0; i < task->data_count; i++) {
        ttg_data_copy_t *copy = static_cast<ttg_data_copy_t *>(task->parsec_task.data[i].data_in);
        if (nullptr == copy) continue;
        detail::release_data_copy(copy);
        task->parsec_task.data[i].data_in = nullptr;
      }
      task_t *tt = (task_t *)task;
      if (tt->deferred_release) {
        tt->deferred_release = nullptr;
        tt->tt_ptr = nullptr;
      }
      parsec_ttg_es = safe_es;
      return PARSEC_HOOK_RETURN_DONE;
    }

   public:
    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       ttg::World world, keymapT &&keymap_ = keymapT(), priomapT &&priomap_ = priomapT())
        : ttg::TTBase(name, numinedges, numouts)
        , world(world)
        // if using default keymap, rebind to the given world
        , keymap(std::is_same<keymapT, ttg::detail::default_keymap<keyT>>::value
                     ? decltype(keymap)(ttg::detail::default_keymap<keyT>(world))
                     : decltype(keymap)(std::forward<keymapT>(keymap_)))
        , priomap(decltype(keymap)(std::forward<priomapT>(priomap_)))
        , static_stream_goal() {
      // Cannot call these in base constructor since terminals not yet constructed
      if (innames.size() != numinedges) throw std::logic_error("ttg_parsec::TT: #input names != #input terminals");
      if (outnames.size() != numouts) throw std::logic_error("ttg_parsec::TT: #output names != #output terminals");

      auto &world_impl = world.impl();
      world_impl.register_op(this);

      if constexpr (numinedges == numins) {
        register_input_terminals(input_terminals, innames);
      } else {
        // create a name for the virtual control input
        register_input_terminals(input_terminals, std::array<std::string, 1>{std::string("Virtual Control")});
      }
      register_output_terminals(output_terminals, outnames);

      register_input_callbacks(std::make_index_sequence<numinedges>{});

      int i;

      memset(&self, 0, sizeof(parsec_task_class_t));

      self.name = get_name().c_str();
      self.task_class_id = get_instance_id();
      self.nb_parameters = 0;
      self.nb_locals = 0;
      self.nb_flows = numflows;

      //    function_id_to_instance[self.task_class_id] = this;

      if constexpr (derived_has_cuda_op()) {
        self.incarnations = (__parsec_chore_t *)malloc(3 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t *)self.incarnations)[0].type = PARSEC_DEV_CUDA;
        ((__parsec_chore_t *)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[0].hook = detail::hook_cuda;
        ((__parsec_chore_t *)self.incarnations)[1].type = PARSEC_DEV_CPU;
        ((__parsec_chore_t *)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[1].hook = detail::hook;
        ((__parsec_chore_t *)self.incarnations)[2].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t *)self.incarnations)[2].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[2].hook = NULL;
      } else {
        self.incarnations = (__parsec_chore_t *)malloc(2 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t *)self.incarnations)[0].type = PARSEC_DEV_CPU;
        ((__parsec_chore_t *)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[0].hook = detail::hook;
        ((__parsec_chore_t *)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t *)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[1].hook = NULL;
      }

      self.release_task = &parsec_release_task_to_mempool_update_nbtasks;
      self.complete_execution = complete_task_and_release;

      for (i = 0; i < numins; i++) {
        parsec_flow_t *flow = new parsec_flow_t;
        flow->name = strdup((std::string("flow in") + std::to_string(i)).c_str());
        flow->sym_type = PARSEC_SYM_INOUT;
        // see initialize_flows below
        // flow->flow_flags = PARSEC_FLOW_ACCESS_RW;
        flow->dep_in[0] = NULL;
        flow->dep_out[0] = NULL;
        flow->flow_index = i;
        flow->flow_datatype_mask = (1 << i);
        *((parsec_flow_t **)&(self.in[i])) = flow;
      }
      *((parsec_flow_t **)&(self.in[i])) = NULL;
      initialize_flows<input_terminals_type>(self.in);

      for (i = 0; i < numouts; i++) {
        parsec_flow_t *flow = new parsec_flow_t;
        flow->name = strdup((std::string("flow out") + std::to_string(i)).c_str());
        flow->sym_type = PARSEC_SYM_INOUT;
        flow->flow_flags = PARSEC_FLOW_ACCESS_READ;  // does PaRSEC use this???
        flow->dep_in[0] = NULL;
        flow->dep_out[0] = NULL;
        flow->flow_index = i;
        flow->flow_datatype_mask = (1 << i);
        *((parsec_flow_t **)&(self.out[i])) = flow;
      }
      *((parsec_flow_t **)&(self.out[i])) = NULL;

      self.flags = 0;
      self.dependencies_goal = numins; /* (~(uint32_t)0) >> (32 - numins); */

      int nbthreads = 0;
      auto *context = world_impl.context();
      for (int i = 0; i < context->nb_vp; i++) {
        nbthreads += context->virtual_processes[i]->nb_cores;
      }

      parsec_mempool_construct(&mempools, PARSEC_OBJ_CLASS(parsec_task_t), sizeof(task_t),
                               offsetof(parsec_task_t, mempool_owner), nbthreads);

      parsec_hash_table_init(&tasks_table, offsetof(detail::parsec_ttg_task_base_t, tt_ht_item), 8, tasks_hash_fcts,
                             NULL);
    }

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       keymapT &&keymap = keymapT(ttg::default_execution_context()), priomapT &&priomap = priomapT())
        : TT(name, innames, outnames, ttg::default_execution_context(), std::forward<keymapT>(keymap),
             std::forward<priomapT>(priomap)) {}

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
       const std::vector<std::string> &innames, const std::vector<std::string> &outnames, ttg::World world,
       keymapT &&keymap_ = keymapT(), priomapT &&priomap = priomapT())
        : TT(name, innames, outnames, world, std::forward<keymapT>(keymap_), std::forward<priomapT>(priomap)) {
      connect_my_inputs_to_incoming_edge_outputs(std::make_index_sequence<numinedges>{}, inedges);
      connect_my_outputs_to_outgoing_edge_inputs(std::make_index_sequence<numouts>{}, outedges);
    }
    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
       const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       keymapT &&keymap = keymapT(ttg::default_execution_context()), priomapT &&priomap = priomapT())
        : TT(inedges, outedges, name, innames, outnames, ttg::default_execution_context(),
             std::forward<keymapT>(keymap), std::forward<priomapT>(priomap)) {}

    // Destructor checks for unexecuted tasks
    virtual ~TT() { release(); }

    static void ht_iter_cb(void *item, void *cb_data) {
      task_t *task = (task_t *)item;
      ttT *op = (ttT *)cb_data;
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        std::cout << "Left over task " << op->get_name() << " " << task->key << std::endl;
      } else {
        std::cout << "Left over task " << op->get_name() << std::endl;
      }
    }

    virtual void release() override { do_release(); }

    void do_release() {
      if (!alive) {
        return;
      }
      alive = false;
      /* print all outstanding tasks */
      parsec_hash_table_for_all(&tasks_table, ht_iter_cb, this);
      parsec_hash_table_fini(&tasks_table);
      parsec_mempool_destruct(&mempools);
      // uintptr_t addr = (uintptr_t)self.incarnations;
      // free((void *)addr);
      free((__parsec_chore_t *)self.incarnations);
      for (int i = 0; i < numflows; i++) {
        if (NULL != self.in[i]) {
          free(self.in[i]->name);
          delete self.in[i];
        }
        if (NULL != self.out[i]) {
          free(self.out[i]->name);
          delete self.out[i];
        }
      }
      world.impl().deregister_op(this);
    }

    static constexpr const ttg::Runtime runtime = ttg::Runtime::PaRSEC;

    template <std::size_t i, typename Reducer>
    void set_input_reducer(Reducer &&reducer) {
      ttg::trace(world.rank(), ":", get_name(), " : setting reducer for terminal ", i);
      std::get<i>(input_reducers) = reducer;
    }

    template <std::size_t i, typename Reducer>
    void set_input_reducer(Reducer &&reducer, std::size_t size) {
      set_input_reducer<i>(std::forward<Reducer>(reducer));
      set_static_argstream_size<i>(size);
    }

    // Returns reference to input terminal i to facilitate connection --- terminal
    // cannot be copied, moved or assigned
    template <std::size_t i>
    std::tuple_element_t<i, input_terminals_type> *in() {
      return &std::get<i>(input_terminals);
    }

    // Returns reference to output terminal for purpose of connection --- terminal
    // cannot be copied, moved or assigned
    template <std::size_t i>
    std::tuple_element_t<i, output_terminalsT> *out() {
      return &std::get<i>(output_terminals);
    }

    // Manual injection of a task with all input arguments specified as a tuple
    template <typename Key = keyT>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const Key &key, const input_values_tuple_type &args) {
      TTG_OP_ASSERT_EXECUTABLE();
      /* trigger non-void inputs */
      set_args(ttg::meta::nonvoid_index_seq<actual_input_tuple_type>{}, key, args);
      /* trigger void inputs */
      using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
      set_args(void_index_seq{}, key, ttg::detail::make_void_tuple<void_index_seq::size()>());
    }

    // Manual injection of a key-free task and all input arguments specified as a tuple
    template <typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const input_values_tuple_type &args) {
      TTG_OP_ASSERT_EXECUTABLE();
      /* trigger non-void inputs */
      set_args(ttg::meta::nonvoid_index_seq<actual_input_tuple_type>{}, args);
      /* trigger void inputs */
      using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
      set_args(void_index_seq{}, ttg::detail::make_void_tuple<void_index_seq::size()>());
    }

    // Manual injection of a task that has no arguments
    template <typename Key = keyT>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const Key &key) {
      TTG_OP_ASSERT_EXECUTABLE();
      /* trigger void inputs */
      using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
      set_args(void_index_seq{}, key, ttg::detail::make_void_tuple<void_index_seq::size()>());
    }

    // Manual injection of a task that has no key or arguments
    template <typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke() {
      TTG_OP_ASSERT_EXECUTABLE();
      /* trigger void inputs */
      using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
      set_args(void_index_seq{}, ttg::detail::make_void_tuple<void_index_seq::size()>());
    }

    // overrides TTBase::invoke()
    void invoke() override {
      if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>)
        invoke<keyT>();
      else
        TTBase::invoke();
    }

   public:
    void make_executable() override {
      register_static_op_function();
      ttg::TTBase::make_executable();
    }

    /// keymap accessor
    /// @return the keymap
    const decltype(keymap) &get_keymap() const { return keymap; }

    /// keymap setter
    template <typename Keymap>
    void set_keymap(Keymap &&km) {
      keymap = km;
    }

    /// priority map accessor
    /// @return the priority map
    const decltype(priomap) &get_priomap() const { return priomap; }

    /// priomap setter
    /// @arg pm a function that maps a key to an integral priority value.
    template <typename Priomap>
    void set_priomap(Priomap &&pm) {
      priomap = pm;
    }

    // Register the static_op function to associate it to instance_id
    void register_static_op_function(void) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      ttg::trace("ttg_parsec(", rank, ") Inserting into static_id_to_op_map at ", get_instance_id());
      static_set_arg_fct_call_t call = std::make_pair(&TT::static_set_arg, this);
      auto &world_impl = world.impl();
      static_map_mutex.lock();
      static_id_to_op_map.insert(std::make_pair(get_instance_id(), call));
      if (delayed_unpack_actions.count(get_instance_id()) > 0) {
        auto tp = world_impl.taskpool();

        ttg::trace("ttg_parsec(", rank, ") There are ", delayed_unpack_actions.count(get_instance_id()),
                   " messages delayed with op_id ", get_instance_id());

        auto se = delayed_unpack_actions.equal_range(get_instance_id());
        std::vector<static_set_arg_fct_arg_t> tmp;
        for (auto it = se.first; it != se.second;) {
          assert(it->first == get_instance_id());
          tmp.push_back(it->second);
          it = delayed_unpack_actions.erase(it);
        }
        static_map_mutex.unlock();

        for (auto it : tmp) {
          ttg::print("ttg_parsec(", rank, ") Unpacking delayed message (", ", ", get_instance_id(), ", ",
                     std::get<1>(it), ", ", std::get<2>(it), ")");
          int rc = detail::static_unpack_msg(&parsec_ce, world_impl.parsec_ttg_tag(), std::get<1>(it), std::get<2>(it),
                                             std::get<0>(it), NULL);
          assert(rc == 0);
          free(std::get<1>(it));
        }

        tmp.clear();
      } else {
        static_map_mutex.unlock();
      }
    }
  };

#include "ttg/make_tt.h"

}  // namespace ttg_parsec

/**
 * The PaRSEC backend tracks data copies so we make a copy of the data
 * if the data is not being tracked yet or if the data is not const, i.e.,
 * the user may mutate the data after it was passed to send/broadcast.
 */
template <>
struct ttg::detail::value_copy_handler<ttg::Runtime::PaRSEC> {
 private:
  ttg_data_copy_t *copy_to_remove = nullptr;

 public:
  ~value_copy_handler() {
    if (nullptr != copy_to_remove) {
      ttg_parsec::detail::remove_data_copy(copy_to_remove, parsec_ttg_caller);
      ttg_parsec::detail::release_data_copy(copy_to_remove);
    }
  }

  template <typename Value>
  inline Value &&operator()(Value &&value) {
    if (nullptr == parsec_ttg_caller) {
      ttg::print("ERROR: ttg_send or ttg_broadcast called outside of a task!\n");
    }
    ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::find_copy_in_task(parsec_ttg_caller, &value);
    Value *value_ptr = &value;
    if (nullptr == copy) {
      /**
       * the value is not known, create a copy that we can track
       * depending on Value, this uses either the copy or move constructor
       */
      copy = ttg_parsec::detail::create_new_datacopy(std::forward<Value>(value));
      bool inserted = ttg_parsec::detail::add_copy_to_task(copy, parsec_ttg_caller);
      assert(inserted);
      value_ptr = reinterpret_cast<Value *>(copy->device_private);
      copy_to_remove = copy;
    } else {
      /* this copy won't be modified anymore so mark it as read-only */
      copy->readers = 1;
    }
    return std::move(*value_ptr);
  }

  template <typename Value>
  inline const Value &operator()(const Value &value) {
    if (nullptr == parsec_ttg_caller) {
      ttg::print("ERROR: ttg_send or ttg_broadcast called outside of a task!\n");
    }
    ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::find_copy_in_task(parsec_ttg_caller, &value);
    const Value *value_ptr = &value;
    if (nullptr == copy) {
      /**
       * the value is not known, create a copy that we can track
       * depending on Value, this uses either the copy or move constructor
       */
      copy = ttg_parsec::detail::create_new_datacopy(value);
      bool inserted = ttg_parsec::detail::add_copy_to_task(copy, parsec_ttg_caller);
      assert(inserted);
      value_ptr = reinterpret_cast<Value *>(copy->device_private);
      copy_to_remove = copy;
    }
    return *value_ptr;
  }

  /* we have to make a copy of non-const data as the user may modify it after
   * send/broadcast */
  template <typename Value, typename Enabler = std::enable_if_t<!std::is_const_v<Value>>>
  inline Value &operator()(Value &value) {
    if (nullptr == parsec_ttg_caller) {
      ttg::print("ERROR: ttg_send or ttg_broadcast called outside of a task!\n");
    }
    /* the value is not known, create a copy that we can track */
    ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::create_new_datacopy(value);
    bool inserted = ttg_parsec::detail::add_copy_to_task(copy, parsec_ttg_caller);
    assert(inserted);
    Value *value_ptr = reinterpret_cast<Value *>(copy->device_private);
    copy_to_remove = copy;
    return *value_ptr;
  }
};

#endif  // PARSEC_TTG_H_INCLUDED
