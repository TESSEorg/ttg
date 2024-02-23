// clang-format off
#ifndef PARSEC_TTG_H_INCLUDED
#define PARSEC_TTG_H_INCLUDED

/* set up env if this header was included directly */
#if !defined(TTG_IMPL_NAME)
#define TTG_USE_PARSEC 1
#endif  // !defined(TTG_IMPL_NAME)

/* Whether to defer a potential writer if there are readers.
 * This may avoid extra copies in exchange for concurrency.
 * This may cause deadlocks, so use with caution. */
#define TTG_PARSEC_DEFER_WRITER false

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
#ifdef TTG_HAVE_DEVICE
#include "ttg/device/task.h"
#endif  // TTG_HAVE_DEVICE

#include "ttg/serialization/data_descriptor.h"

#include "ttg/parsec/fwd.h"

#include "ttg/parsec/buffer.h"
#include "ttg/parsec/devicescratch.h"
#include "ttg/parsec/thread_local.h"
#include "ttg/parsec/devicefunc.h"
#include "ttg/parsec/ttvalue.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <experimental/type_traits>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// needed for MPIX_CUDA_AWARE_SUPPORT
#if defined(TTG_HAVE_MPI)
#include <mpi.h>
#if defined(TTG_HAVE_MPIEXT)
#include <mpi-ext.h>
#endif // TTG_HAVE_MPIEXT
#endif // TTG_HAVE_MPI


#include <parsec.h>
#include <parsec/class/parsec_hash_table.h>
#include <parsec/data_internal.h>
#include <parsec/execution_stream.h>
#include <parsec/interfaces/interface.h>
#include <parsec/mca/device/device.h>
#include <parsec/parsec_comm_engine.h>
#include <parsec/parsec_internal.h>
#include <parsec/scheduling.h>
#include <parsec/remote_dep.h>

#ifdef PARSEC_HAVE_DEV_CUDA_SUPPORT
#include <parsec/mca/device/cuda/device_cuda.h>
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT
#ifdef PARSEC_HAVE_DEV_HIP_SUPPORT
#include <parsec/mca/device/hip/device_hip.h>
#endif // PARSEC_HAVE_DEV_HIP_SUPPORT
#ifdef PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT
#include <parsec/mca/device/level_zero/device_level_zero.h>
#endif //PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT

#include <parsec/mca/device/device_gpu.h>
#if defined(PARSEC_PROF_TRACE)
#include <parsec/profiling.h>
#undef PARSEC_TTG_PROFILE_BACKEND
#if defined(PARSEC_PROF_GRAPHER)
#include <parsec/parsec_prof_grapher.h>
#endif
#endif
#include <cstdlib>
#include <cstring>

#if defined(TTG_PARSEC_DEBUG_TRACK_DATA_COPIES)
#include <unordered_set>
#endif

#include "ttg/parsec/ttg_data_copy.h"
#include "ttg/parsec/thread_local.h"
#include "ttg/parsec/ptr.h"
#include "ttg/parsec/task.h"
#include "ttg/parsec/parsec-ext.h"

#include "ttg/device/device.h"

#undef TTG_PARSEC_DEBUG_TRACK_DATA_COPIES

/* PaRSEC function declarations */
extern "C" {
void parsec_taskpool_termination_detected(parsec_taskpool_t *tp);
int parsec_add_fetch_runtime_task(parsec_taskpool_t *tp, int tasks);
}

namespace ttg_parsec {
  typedef void (*static_set_arg_fct_type)(void *, size_t, ttg::TTBase *);
  typedef std::pair<static_set_arg_fct_type, ttg::TTBase *> static_set_arg_fct_call_t;
  inline std::map<uint64_t, static_set_arg_fct_call_t> static_id_to_op_map;
  inline std::mutex static_map_mutex;
  typedef std::tuple<int, void *, size_t> static_set_arg_fct_arg_t;
  inline std::multimap<uint64_t, static_set_arg_fct_arg_t> delayed_unpack_actions;

  struct msg_header_t {
    typedef enum fn_id : std::int8_t {
      MSG_INVALID = -1,
      MSG_SET_ARG = 0,
      MSG_SET_ARGSTREAM_SIZE = 1,
      MSG_FINALIZE_ARGSTREAM_SIZE = 2,
      MSG_GET_FROM_PULL = 3 } fn_id_t;
    uint32_t taskpool_id = -1;
    uint64_t op_id = -1;
    std::size_t key_offset = 0;
    fn_id_t fn_id = MSG_INVALID;
    std::int8_t num_iovecs = 0;
    bool inline_data = false;
    int32_t param_id = -1;
    int num_keys = 0;
    int sender = -1;

    msg_header_t() = default;

    msg_header_t(fn_id_t fid, uint32_t tid, uint64_t oid, int32_t pid, int sender, int nk)
    : fn_id(fid)
    , taskpool_id(tid)
    , op_id(oid)
    , param_id(pid)
    , num_keys(nk)
    , sender(sender)
    { }
  };

  static void unregister_parsec_tags(void *_);

  namespace detail {

    constexpr const int PARSEC_TTG_MAX_AM_SIZE = 1 * 1024*1024;

    struct msg_t {
      msg_header_t tt_id;
      static constexpr std::size_t max_payload_size = PARSEC_TTG_MAX_AM_SIZE - sizeof(msg_header_t);
      unsigned char bytes[max_payload_size];

      msg_t() = default;
      msg_t(uint64_t tt_id,
            uint32_t taskpool_id,
            msg_header_t::fn_id_t fn_id,
            int32_t param_id,
            int sender,
            int num_keys = 1)
      : tt_id(fn_id, taskpool_id, tt_id, param_id, sender, num_keys)
      {}
    };

    inline std::size_t max_inline_size = msg_t::max_payload_size;

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

    inline bool &initialized_mpi() {
      static bool im = false;
      return im;
    }

  }  // namespace detail

  class WorldImpl : public ttg::base::WorldImplBase {
    ttg::Edge<> m_ctl_edge;
    bool _dag_profiling;
    bool _task_profiling;
    std::array<bool, static_cast<std::size_t>(ttg::ExecutionSpace::Invalid)>
               mpi_space_support = {true, false, false};

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

    static void ttg_parsec_ce_up(parsec_comm_engine_t *comm_engine, void *user_data)
    {
      parsec_ce.tag_register(WorldImpl::parsec_ttg_tag(), &detail::static_unpack_msg, user_data, detail::PARSEC_TTG_MAX_AM_SIZE);
      parsec_ce.tag_register(WorldImpl::parsec_ttg_rma_tag(), &detail::get_remote_complete_cb, user_data, 128);
    }

    static void ttg_parsec_ce_down(parsec_comm_engine_t *comm_engine, void *user_data)
    {
      parsec_ce.tag_unregister(WorldImpl::parsec_ttg_tag());
      parsec_ce.tag_unregister(WorldImpl::parsec_ttg_rma_tag());
    }

   public:
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
    int parsec_ttg_profile_backend_set_arg_start, parsec_ttg_profile_backend_set_arg_end;
    int parsec_ttg_profile_backend_bcast_arg_start, parsec_ttg_profile_backend_bcast_arg_end;
    int parsec_ttg_profile_backend_allocate_datacopy, parsec_ttg_profile_backend_free_datacopy;
#endif

    WorldImpl(int *argc, char **argv[], int ncores, parsec_context_t *c = nullptr)
        : WorldImplBase(query_comm_size(), query_comm_rank())
        , ctx(c)
        , own_ctx(c == nullptr)
#if defined(PARSEC_PROF_TRACE)
        , profiling_array(nullptr)
        , profiling_array_size(0)
#endif
       , _dag_profiling(false)
       , _task_profiling(false)
    {
      ttg::detail::register_world(*this);
      if (own_ctx) ctx = parsec_init(ncores, argc, argv);

      /* query MPI device support */
      if (ttg::detail::force_device_comm()
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
          || MPIX_Query_cuda_support()
#endif // MPIX_CUDA_AWARE_SUPPORT
         ) {
        mpi_space_support[static_cast<std::size_t>(ttg::ExecutionSpace::CUDA)] = true;
      }

      if (ttg::detail::force_device_comm()
#if defined(MPIX_HIP_AWARE_SUPPORT) && MPIX_HIP_AWARE_SUPPORT
          || MPIX_Query_hip_support()
#endif // MPIX_HIP_AWARE_SUPPORT
         ) {
        mpi_space_support[static_cast<std::size_t>(ttg::ExecutionSpace::HIP)] = true;
      }

#if defined(PARSEC_PROF_TRACE)
      if(parsec_profile_enabled) {
        profile_on();
#if defined(PARSEC_TTG_PROFILE_BACKEND)
        parsec_profiling_add_dictionary_keyword("PARSEC_TTG_SET_ARG_IMPL", "fill:000000", 0, NULL,
                                                (int*)&parsec_ttg_profile_backend_set_arg_start,
                                                (int*)&parsec_ttg_profile_backend_set_arg_end);
        parsec_profiling_add_dictionary_keyword("PARSEC_TTG_BCAST_ARG_IMPL", "fill:000000", 0, NULL,
                                                (int*)&parsec_ttg_profile_backend_bcast_arg_start,
                                                (int*)&parsec_ttg_profile_backend_bcast_arg_end);
        parsec_profiling_add_dictionary_keyword("PARSEC_TTG_DATACOPY", "fill:000000",
                                                sizeof(size_t), "size{int64_t}",
                                                (int*)&parsec_ttg_profile_backend_allocate_datacopy,
                                                (int*)&parsec_ttg_profile_backend_free_datacopy);
#endif
      }
#endif

      if( NULL != parsec_ce.tag_register) {
        parsec_ce.tag_register(WorldImpl::parsec_ttg_tag(), &detail::static_unpack_msg, this, detail::PARSEC_TTG_MAX_AM_SIZE);
        parsec_ce.tag_register(WorldImpl::parsec_ttg_rma_tag(), &detail::get_remote_complete_cb, this, 128);
      }

      create_tpool();
    }


    auto *context() { return ctx; }
    auto *execution_stream() { return parsec_my_execution_stream(); }
    auto *taskpool() { return tpool; }

    void create_tpool() {
      assert(nullptr == tpool);
      tpool = PARSEC_OBJ_NEW(parsec_taskpool_t);
      tpool->taskpool_id = -1;
      tpool->update_nb_runtime_task = parsec_add_fetch_runtime_task;
      tpool->taskpool_type = PARSEC_TASKPOOL_TYPE_TTG;
      tpool->taskpool_name = strdup("TTG Taskpool");
      parsec_taskpool_reserve_id(tpool);

      tpool->devices_index_mask = 0;
      for(int i = 0; i < (int)parsec_nb_devices; i++) {
          parsec_device_module_t *device = parsec_mca_device_get(i);
          if( NULL == device ) continue;
          tpool->devices_index_mask |= (1 << device->device_index);
      }

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
      tpool->tdm.module->taskpool_set_runtime_actions(tpool, 0);
      parsec_taskpool_enable(tpool, NULL, NULL, execution_stream(), size() > 1);

#if defined(PARSEC_PROF_TRACE)
      tpool->profiling_array = profiling_array;
#endif

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

    static constexpr int parsec_ttg_tag() { return PARSEC_DSL_TTG_TAG; }
    static constexpr int parsec_ttg_rma_tag() { return PARSEC_DSL_TTG_RMA_TAG; }

    MPI_Comm comm() const { return MPI_COMM_WORLD; }

    virtual void execute() override {
      if (!parsec_taskpool_started) {
        parsec_enqueue(ctx, tpool);
        tpool->tdm.module->taskpool_addto_runtime_actions(tpool, 1);
        tpool->tdm.module->taskpool_ready(tpool);
        [[maybe_unused]] auto ret = parsec_context_start(ctx);
        // ignore ret since all of its nonzero values are OK (e.g. -1 due to ctx already being active)
        parsec_taskpool_started = true;
      }
    }

    void destroy_tpool() {
#if defined(PARSEC_PROF_TRACE)
      // We don't want to release the profiling array, as it should be persistent
      // between fences() to allow defining a TT/TTG before a fence() and schedule
      // it / complete it after a fence()
      tpool->profiling_array = nullptr;
#endif
      assert(NULL != tpool->tdm.monitor);
      tpool->tdm.module->unmonitor_taskpool(tpool);
      parsec_taskpool_free(tpool);
      tpool = nullptr;
    }

    virtual void destroy() override {
      if (is_valid()) {
        if (parsec_taskpool_started) {
          // We are locally ready (i.e. we won't add new tasks)
          tpool->tdm.module->taskpool_addto_runtime_actions(tpool, -1);
          ttg::trace("ttg_parsec(", this->rank(), "): final waiting for completion");
          if (own_ctx)
            parsec_context_wait(ctx);
          else
            parsec_taskpool_wait(tpool);
        }
        release_ops();
        ttg::detail::deregister_world(*this);
        destroy_tpool();
        if (own_ctx) {
          unregister_parsec_tags(nullptr);
        } else {
          parsec_context_at_fini(unregister_parsec_tags, nullptr);
        }
#if defined(PARSEC_PROF_TRACE)
        if(nullptr != profiling_array) {
          free(profiling_array);
          profiling_array = nullptr;
          profiling_array_size = 0;
        }
#endif
        if (own_ctx) parsec_fini(&ctx);
        mark_invalid();
      }
    }

    ttg::Edge<> &ctl_edge() { return m_ctl_edge; }

    const ttg::Edge<> &ctl_edge() const { return m_ctl_edge; }

    void increment_created() { taskpool()->tdm.module->taskpool_addto_nb_tasks(taskpool(), 1); }

    void increment_inflight_msg() { taskpool()->tdm.module->taskpool_addto_runtime_actions(taskpool(), 1); }
    void decrement_inflight_msg() { taskpool()->tdm.module->taskpool_addto_runtime_actions(taskpool(), -1); }

    bool dag_profiling() override { return _dag_profiling; }

    virtual void dag_on(const std::string &filename) override {
#if defined(PARSEC_PROF_GRAPHER)
      if(!_dag_profiling) {
        profile_on();
        size_t len = strlen(filename.c_str())+32;
        char ext_filename[len];
        snprintf(ext_filename, len, "%s-%d.dot", filename.c_str(), rank());
        parsec_prof_grapher_init(ctx, ext_filename);
        _dag_profiling = true;
      }
#else
      ttg::print("Error: requested to create '", filename, "' to create a DAG of tasks,\n"
                 "but PaRSEC does not support graphing options. Reconfigure with PARSEC_PROF_GRAPHER=ON\n");
#endif
    }

    virtual void dag_off() override {
#if defined(PARSEC_PROF_GRAPHER)
      if(_dag_profiling) {
        parsec_prof_grapher_fini();
        _dag_profiling = false;
      }
#endif
    }

    virtual void profile_off() override {
#if defined(PARSEC_PROF_TRACE)
      _task_profiling = false;
#endif
    }

    virtual void profile_on() override {
#if defined(PARSEC_PROF_TRACE)
      _task_profiling = true;
#endif
    }

    virtual bool profiling() override { return _task_profiling; }

    bool mpi_support(ttg::ExecutionSpace space) {
      return mpi_space_support[static_cast<std::size_t>(space)];
    }

    virtual void final_task() override {
#ifdef TTG_USE_USER_TERMDET
      if(parsec_taskpool_started) {
        taskpool()->tdm.module->taskpool_set_nb_tasks(taskpool(), 0);
        parsec_taskpool_started = false;
      }
#endif  // TTG_USE_USER_TERMDET
    }

    template <typename keyT, typename output_terminalsT, typename derivedT, typename input_valueTs = ttg::typelist<>>
    void register_tt_profiling(const TT<keyT, output_terminalsT, derivedT, input_valueTs> *t) {
#if defined(PARSEC_PROF_TRACE)
      std::stringstream ss;
      build_composite_name_rec(t->ttg_ptr(), ss);
      ss << t->get_name();
      register_new_profiling_event(ss.str().c_str(), t->get_instance_id());
#endif
    }

   protected:
#if defined(PARSEC_PROF_TRACE)
    void build_composite_name_rec(const ttg::TTBase *t, std::stringstream &ss) {
      if(nullptr == t)
        return;
      build_composite_name_rec(t->ttg_ptr(), ss);
      ss << t->get_name() << "::";
    }

    void register_new_profiling_event(const char *name, int position) {
      if(2*position >= profiling_array_size) {
        size_t new_profiling_array_size = 64 * ((2*position + 63)/64 + 1);
        profiling_array = (int*)realloc((void*)profiling_array,
                                         new_profiling_array_size * sizeof(int));
        memset((void*)&profiling_array[profiling_array_size], 0, sizeof(int)*(new_profiling_array_size - profiling_array_size));
        profiling_array_size = new_profiling_array_size;
        tpool->profiling_array = profiling_array;
      }

      assert(0 == tpool->profiling_array[2*position]);
      assert(0 == tpool->profiling_array[2*position+1]);
      // TODO PROFILING: 0 and NULL should be replaced with something that depends on the key human-readable serialization...
      // Typically, we would put something like 3*sizeof(int32_t), "m{int32_t};n{int32_t};k{int32_t}" to say
      // there are three fields, named m, n and k, stored in this order, and each of size int32_t
      parsec_profiling_add_dictionary_keyword(name, "fill:000000", 64, "key{char[64]}",
                                              (int*)&tpool->profiling_array[2*position],
                                              (int*)&tpool->profiling_array[2*position+1]);
    }
#endif

    virtual void fence_impl(void) override {
      int rank = this->rank();
      if (!parsec_taskpool_started) {
        ttg::trace("ttg_parsec::(", rank, "): parsec taskpool has not been started, fence is a simple MPI_Barrier");
        MPI_Barrier(comm());
        return;
      }
      ttg::trace("ttg_parsec::(", rank, "): parsec taskpool is ready for completion");
      // We are locally ready (i.e. we won't add new tasks)
      tpool->tdm.module->taskpool_addto_runtime_actions(tpool, -1);
      ttg::trace("ttg_parsec(", rank, "): waiting for completion");
      parsec_taskpool_wait(tpool);

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
    bool own_ctx = false;  //< whether I own the context
    parsec_taskpool_t *tpool = nullptr;
    bool parsec_taskpool_started = false;
#if defined(PARSEC_PROF_TRACE)
    int        *profiling_array;
    std::size_t profiling_array_size;
#endif
  };

  static void unregister_parsec_tags(void *_pidx)
  {
    if(NULL != parsec_ce.tag_unregister) {
      parsec_ce.tag_unregister(WorldImpl::parsec_ttg_tag());
      parsec_ce.tag_unregister(WorldImpl::parsec_ttg_rma_tag());
    }
  }

  namespace detail {

    const parsec_symbol_t parsec_taskclass_param0 = {
      .flags = PARSEC_SYMBOL_IS_STANDALONE|PARSEC_SYMBOL_IS_GLOBAL,
      .name = "HASH0",
      .context_index = 0,
      .min = nullptr,
      .max = nullptr,
      .expr_inc = nullptr,
      .cst_inc = 0 };
    const parsec_symbol_t parsec_taskclass_param1 = {
      .flags = PARSEC_SYMBOL_IS_STANDALONE|PARSEC_SYMBOL_IS_GLOBAL,
      .name = "HASH1",
      .context_index = 1,
      .min = nullptr,
      .max = nullptr,
      .expr_inc = nullptr,
      .cst_inc = 0 };
    const parsec_symbol_t parsec_taskclass_param2 = {
      .flags = PARSEC_SYMBOL_IS_STANDALONE|PARSEC_SYMBOL_IS_GLOBAL,
      .name = "KEY0",
      .context_index = 2,
      .min = nullptr,
      .max = nullptr,
      .expr_inc = nullptr,
      .cst_inc = 0 };
    const parsec_symbol_t parsec_taskclass_param3 = {
      .flags = PARSEC_SYMBOL_IS_STANDALONE|PARSEC_SYMBOL_IS_GLOBAL,
      .name = "KEY1",
      .context_index = 3,
      .min = nullptr,
      .max = nullptr,
      .expr_inc = nullptr,
      .cst_inc = 0 };

    inline ttg_data_copy_t *find_copy_in_task(parsec_ttg_task_base_t *task, const void *ptr) {
      ttg_data_copy_t *res = nullptr;
      if (task == nullptr || ptr == nullptr) {
        return res;
      }
      for (int i = 0; i < task->data_count; ++i) {
        auto copy = static_cast<ttg_data_copy_t *>(task->copies[i]);
        if (NULL != copy && copy->get_ptr() == ptr) {
          res = copy;
          break;
        }
      }
      return res;
    }

    inline int find_index_of_copy_in_task(parsec_ttg_task_base_t *task, const void *ptr) {
      int i = -1;
      if (task == nullptr || ptr == nullptr) {
        return i;
      }
      for (i = 0; i < task->data_count; ++i) {
        auto copy = static_cast<ttg_data_copy_t *>(task->copies[i]);
        if (NULL != copy && copy->get_ptr() == ptr) {
          return i;
        }
      }
      return -1;
    }

    inline bool add_copy_to_task(ttg_data_copy_t *copy, parsec_ttg_task_base_t *task) {
      if (task == nullptr || copy == nullptr) {
        return false;
      }

      if (MAX_PARAM_COUNT < task->data_count) {
        throw std::logic_error("Too many data copies, check MAX_PARAM_COUNT!");
      }

      task->copies[task->data_count] = copy;
      task->data_count++;
      return true;
    }

    inline void remove_data_copy(ttg_data_copy_t *copy, parsec_ttg_task_base_t *task) {
      int i;
      /* find and remove entry; copies are usually appended and removed, so start from back */
      for (i = task->data_count-1; i >= 0; --i) {
        if (copy == task->copies[i]) {
          break;
        }
      }
      if (i < 0) return;
      /* move all following elements one up */
      for (; i < task->data_count - 1; ++i) {
        task->copies[i] = task->copies[i + 1];
      }
      /* null last element */
      task->copies[i] = nullptr;
      task->data_count--;
    }

#if defined(TTG_PARSEC_DEBUG_TRACK_DATA_COPIES)
#warning "ttg::PaRSEC enables data copy tracking"
    static std::unordered_set<ttg_data_copy_t *> pending_copies;
    static std::mutex pending_copies_mutex;
#endif
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
    static int64_t parsec_ttg_data_copy_uid = 0;
#endif

    template <typename Value>
    inline ttg_data_copy_t *create_new_datacopy(Value &&value) {
      using value_type = std::decay_t<Value>;
      ttg_data_copy_t *copy;
      if constexpr (std::is_base_of_v<ttg::TTValue<value_type>, value_type>) {
        copy = new value_type(std::forward<Value>(value));
      } else if constexpr (std::is_rvalue_reference_v<decltype(value)> ||
                           std::is_copy_constructible_v<std::decay_t<Value>>) {
        copy = new ttg_data_value_copy_t<value_type>(std::forward<Value>(value));
      } else {
        throw std::logic_error("Trying to copy-construct data that is not copy-constructible!");
      }
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      // Keep track of additional memory usage
      if(ttg::default_execution_context().impl().profiling()) {
        copy->size = sizeof(Value);
        copy->uid = parsec_atomic_fetch_inc_int64(&parsec_ttg_data_copy_uid);
        parsec_profiling_ts_trace_flags(ttg::default_execution_context().impl().parsec_ttg_profile_backend_allocate_datacopy,
                                        static_cast<uint64_t>(copy->uid),
                                        PROFILE_OBJECT_ID_NULL, &copy->size,
                                        PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
      }
#endif
#if defined(TTG_PARSEC_DEBUG_TRACK_DATA_COPIES)
      {
        const std::lock_guard<std::mutex> lock(pending_copies_mutex);
        auto rc = pending_copies.insert(copy);
        assert(std::get<1>(rc));
      }
#endif
      return copy;
    }

#if 0
    template <std::size_t... IS, typename Key = keyT>
    void invoke_pull_terminals(std::index_sequence<IS...>, const Key &key, detail::parsec_ttg_task_base_t *task) {
      int junk[] = {0, (invoke_pull_terminal<IS>(
                            std::get<IS>(input_terminals), key, task),
                        0)...};
      junk[0]++;
    }
#endif // 0

    template<typename TT, std::size_t I>
    inline void transfer_ownership_impl(ttg_data_copy_t *copy, int device) {
      if constexpr(!std::is_const_v<std::tuple_element_t<I, typename TT::input_values_tuple_type>>) {
        copy->transfer_ownership(PARSEC_FLOW_ACCESS_RW, device);
      }
    }

    template<typename TT, std::size_t... Is>
    inline void transfer_ownership(parsec_ttg_task_t<TT> *me, int device, std::index_sequence<Is...>) {
      /* transfer ownership of each data */
      int junk[] = {0, (transfer_ownership_impl<TT, Is>(me->copies[Is], device), 0)...};
      junk[0]++;
    }

    template<typename TT>
    inline parsec_hook_return_t hook(struct parsec_execution_stream_s *es, parsec_task_t *parsec_task) {
      parsec_ttg_task_t<TT> *me = (parsec_ttg_task_t<TT> *)parsec_task;
      if constexpr(std::tuple_size_v<typename TT::input_values_tuple_type> > 0) {
        transfer_ownership<TT>(me, 0, std::make_index_sequence<std::tuple_size_v<typename TT::input_values_tuple_type>>{});
      }
      return me->template invoke_op<ttg::ExecutionSpace::Host>();
    }

    template<typename TT>
    inline parsec_hook_return_t hook_cuda(struct parsec_execution_stream_s *es, parsec_task_t *parsec_task) {
      if constexpr(TT::derived_has_cuda_op()) {
        parsec_ttg_task_t<TT> *me = (parsec_ttg_task_t<TT> *)parsec_task;
        return me->template invoke_op<ttg::ExecutionSpace::CUDA>();
      } else {
        throw std::runtime_error("PaRSEC CUDA hook invoked on a TT that does not support CUDA operations!");
      }
    }

    template<typename TT>
    inline parsec_hook_return_t hook_hip(struct parsec_execution_stream_s *es, parsec_task_t *parsec_task) {
      if constexpr(TT::derived_has_hip_op()) {
        parsec_ttg_task_t<TT> *me = (parsec_ttg_task_t<TT> *)parsec_task;
        return me->template invoke_op<ttg::ExecutionSpace::HIP>();
      } else {
        throw std::runtime_error("PaRSEC HIP hook invoked on a TT that does not support HIP operations!");
      }
    }

    template<typename TT>
    inline parsec_hook_return_t hook_level_zero(struct parsec_execution_stream_s *es, parsec_task_t *parsec_task) {
      if constexpr(TT::derived_has_level_zero_op()) {
        parsec_ttg_task_t<TT> *me = (parsec_ttg_task_t<TT> *)parsec_task;
        return me->template invoke_op<ttg::ExecutionSpace::L0>();
      } else {
        throw std::runtime_error("PaRSEC HIP hook invoked on a TT that does not support HIP operations!");
      }
    }

    template <typename KeyT, typename ActivationCallbackT>
    class rma_delayed_activate {
      std::vector<KeyT> _keylist;
      std::atomic<int> _outstanding_transfers;
      ActivationCallbackT _cb;
      detail::ttg_data_copy_t *_copy;

     public:
      rma_delayed_activate(std::vector<KeyT> &&key, detail::ttg_data_copy_t *copy, int num_transfers, ActivationCallbackT cb)
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
      if (copy->is_mutable() && nullptr == copy->get_next_task()) {
        /* current task mutated the data but there are no consumers so prepare
        * the copy to be freed below */
        copy->reset_readers();
      }

      int32_t readers = copy->num_readers();
      if (readers > 1) {
        /* potentially more than one reader, decrement atomically */
        readers = copy->decrement_readers();
      } else if (readers == 1) {
        /* make sure readers drop to zero */
        readers = copy->decrement_readers<false>();
      }
      /* if there was only one reader (the current task) or
       * a mutable copy and a successor, we release the copy */
      if (1 == readers || copy->is_mutable()) {
        if (nullptr != copy->get_next_task()) {
          /* Release the deferred task.
           * The copy was mutable and will be mutated by the released task,
           * so simply transfer ownership.
           */
          parsec_task_t *next_task = copy->get_next_task();
          copy->set_next_task(nullptr);
          parsec_ttg_task_base_t *deferred_op = (parsec_ttg_task_base_t *)next_task;
          copy->mark_mutable();
          deferred_op->release_task();
        } else if ((1 == copy->num_ref()) || (1 == copy->drop_ref())) {
          /* we are the last reference, delete the copy */
#if defined(TTG_PARSEC_DEBUG_TRACK_DATA_COPIES)
          {
            const std::lock_guard<std::mutex> lock(pending_copies_mutex);
            size_t rc = pending_copies.erase(copy);
            assert(1 == rc);
          }
#endif
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
          // Keep track of additional memory usage
          if(ttg::default_execution_context().impl().profiling()) {
            parsec_profiling_ts_trace_flags(ttg::default_execution_context().impl().parsec_ttg_profile_backend_free_datacopy,
                                            static_cast<uint64_t>(copy->uid),
                                            PROFILE_OBJECT_ID_NULL, &copy->size,
                                            PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO);
          }
#endif
          delete copy;
        }
      }
    }

    template <typename Value>
    inline ttg_data_copy_t *register_data_copy(ttg_data_copy_t *copy_in, parsec_ttg_task_base_t *task, bool readonly) {
      ttg_data_copy_t *copy_res = copy_in;
      bool replace = false;
      int32_t readers = copy_in->num_readers();

      assert(readers != 0);

      if (readonly && !copy_in->is_mutable()) {
        /* simply increment the number of readers */
        readers = copy_in->increment_readers();
      }

      if (readers == copy_in->mutable_tag) {
        if (copy_res->get_next_task() != nullptr) {
          if (readonly) {
            parsec_ttg_task_base_t *next_task = reinterpret_cast<parsec_ttg_task_base_t *>(copy_res->get_next_task());
            if (next_task->defer_writer) {
              /* there is a writer but it signalled that it wants to wait for readers to complete */
              return copy_res;
            }
          }
        }
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
        if (1 == copy_in->num_readers() && !task->defer_writer) {
          /**
           * no other readers, mark copy as mutable and defer the release
           * of the task
           */
          copy_in->mark_mutable();
          assert(nullptr == copy_in->get_next_task());
          copy_in->set_next_task(&task->parsec_task);
        } else {
          if (task->defer_writer && nullptr == copy_in->get_next_task()) {
            /* we're the first writer and want to wait for all readers to complete */
            copy_res->set_next_task(&task->parsec_task);
          } else {
            /* there are writers and/or waiting already of this copy already, make a copy that we can mutate */
            copy_res = NULL;
          }
        }
      }

      if (NULL == copy_res) {
        ttg_data_copy_t *new_copy = detail::create_new_datacopy(*static_cast<Value *>(copy_in->get_ptr()));
        if (replace && nullptr != copy_in->get_next_task()) {
          /* replace the task that was deferred */
          parsec_ttg_task_base_t *deferred_op = (parsec_ttg_task_base_t *)copy_in->get_next_task();
          new_copy->mark_mutable();
          /* replace the copy in the deferred task */
          for (int i = 0; i < deferred_op->data_count; ++i) {
            if (deferred_op->copies[i] == copy_in) {
              deferred_op->copies[i] = new_copy;
              break;
            }
          }
          copy_in->set_next_task(nullptr);
          deferred_op->release_task();
          copy_in->reset_readers();            // set the copy back to being read-only
          copy_in->increment_readers<false>(); // register as reader
          copy_res = copy_in;                  // return the copy we were passed
        } else {
          if (!readonly) {
            new_copy->mark_mutable();
          }
          copy_res = new_copy;  // return the new copy
        }
      }
      return copy_res;
    }

  }  // namespace detail

  inline void ttg_initialize(int argc, char **argv, int num_threads, parsec_context_t *ctx) {
    if (detail::initialized_mpi()) throw std::runtime_error("ttg_parsec::ttg_initialize: can only be called once");

    // make sure it's not already initialized
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {  // MPI not initialized? do it, remember that we did it
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
      if (!provided)
        throw std::runtime_error("ttg_parsec::ttg_initialize: MPI_Init_thread did not provide MPI_THREAD_MULTIPLE");
      detail::initialized_mpi() = true;
    } else {  // no way to test that MPI was initialized with MPI_THREAD_MULTIPLE, cross fingers and proceed
    }

    if (num_threads < 1) num_threads = ttg::detail::num_threads();
    auto world_ptr = new ttg_parsec::WorldImpl{&argc, &argv, num_threads, ctx};
    std::shared_ptr<ttg::base::WorldImplBase> world_sptr{static_cast<ttg::base::WorldImplBase *>(world_ptr)};
    ttg::World world{std::move(world_sptr)};
    ttg::detail::set_default_world(std::move(world));

    // query the first device ID
    detail::first_device_id = -1;
    for (int i = 0; i < parsec_nb_devices; ++i) {
      bool is_gpu = parsec_mca_device_is_gpu(i);
      if (detail::first_device_id == -1 && is_gpu) {
        detail::first_device_id = i;
      } else if (detail::first_device_id > -1 && !is_gpu) {
        throw std::runtime_error("PaRSEC: Found non-GPU device in GPU ID range!");
      }
    }

    /* parse the maximum inline size */
    const char* ttg_max_inline_cstr = std::getenv("TTG_MAX_INLINE");
    if (nullptr != ttg_max_inline_cstr) {
      std::size_t inline_size = std::atol(ttg_max_inline_cstr);
      if (inline_size < detail::max_inline_size) {
        detail::max_inline_size = inline_size;
      }
    }
  }
  inline void ttg_finalize() {
    // We need to notify the current taskpool of termination if we are in user termination detection mode
    // or the parsec_context_wait() in destroy_worlds() will never complete
    if(0 == ttg::default_execution_context().rank())
      ttg::default_execution_context().impl().final_task();
    ttg::detail::set_default_world(ttg::World{});  // reset the default world
    detail::ptr_impl::drop_all_ptr();
    ttg::detail::destroy_worlds<ttg_parsec::WorldImpl>();
    if (detail::initialized_mpi()) MPI_Finalize();
  }
  inline ttg::World ttg_default_execution_context() { return ttg::get_default_world(); }
  [[noreturn]]
  inline void ttg_abort() { MPI_Abort(ttg_default_execution_context().impl().comm(), 1); std::abort(); }
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

  inline void make_executable_hook(ttg::World& world) {
    MPI_Barrier(world.impl().comm());
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
    using have_cuda_op_non_type_t = decltype(T::have_cuda_op);

    template <typename T>
    using have_hip_op_non_type_t = decltype(T::have_hip_op);

    template <typename T>
    using have_level_zero_op_non_type_t = decltype(T::have_level_zero_op);

    bool alive = true;

    static constexpr int numinedges = std::tuple_size_v<input_tuple_type>;     // number of input edges
    static constexpr int numins = std::tuple_size_v<actual_input_tuple_type>;  // number of input arguments
    static constexpr int numouts = std::tuple_size_v<output_terminalsT>;       // number of outputs
    static constexpr int numflows = std::max(numins, numouts);                 // max number of flows

   public:
    /// @return true if derivedT::have_cuda_op exists and is defined to true
    static constexpr bool derived_has_cuda_op() {
      if constexpr (ttg::meta::is_detected_v<have_cuda_op_non_type_t, derivedT>) {
        return derivedT::have_cuda_op;
      } else {
        return false;
      }
    }

    /// @return true if derivedT::have_hip_op exists and is defined to true
    static constexpr bool derived_has_hip_op() {
      if constexpr (ttg::meta::is_detected_v<have_hip_op_non_type_t, derivedT>) {
        return derivedT::have_hip_op;
      } else {
        return false;
      }
    }

    /// @return true if derivedT::have_hip_op exists and is defined to true
    static constexpr bool derived_has_level_zero_op() {
      if constexpr (ttg::meta::is_detected_v<have_level_zero_op_non_type_t, derivedT>) {
        return derivedT::have_level_zero_op;
      } else {
        return false;
      }
    }

    /// @return true if the TT supports device execution
    static constexpr bool derived_has_device_op() {
      return (derived_has_cuda_op() || derived_has_hip_op() || derived_has_level_zero_op());
    }

    using ttT = TT;
    using key_type = keyT;
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
    using task_t = detail::parsec_ttg_task_t<ttT>;

    friend detail::parsec_ttg_task_base_t;
    friend task_t;

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

    template <std::size_t... IS>
    static constexpr auto make_get_from_pull_fcts(std::index_sequence<IS...>) {
      using resultT = decltype(get_from_pull_msg_fcts);
      return resultT{{&TT::get_from_pull_msg<IS>...}};
    }
    constexpr static std::array<void (TT::*)(void *, std::size_t), numinedges> get_from_pull_msg_fcts =
        make_get_from_pull_fcts(std::make_index_sequence<numinedges>{});

    template<std::size_t... IS>
    constexpr static auto make_input_is_const(std::index_sequence<IS...>) {
      using resultT = decltype(input_is_const);
      return resultT{{std::is_const_v<std::tuple_element_t<IS, input_args_type>>...}};
    }
    constexpr static std::array<bool, numins> input_is_const = make_input_is_const(std::make_index_sequence<numins>{});

    ttg::World world;
    ttg::meta::detail::keymap_t<keyT> keymap;
    ttg::meta::detail::keymap_t<keyT> priomap;
    // For now use same type for unary/streaming input terminals, and stream reducers assigned at runtime
    ttg::meta::detail::input_reducers_t<actual_input_tuple_type>
        input_reducers;  //!< Reducers for the input terminals (empty = expect single value)
    std::array<parsec_task_class_t*, numins> inpute_reducers_taskclass = { nullptr };
    std::array<std::size_t, numins> static_stream_goal = { std::numeric_limits<std::size_t>::max() };
    int num_pullins = 0;

    bool m_defer_writer = TTG_PARSEC_DEFER_WRITER;

   public:
    ttg::World get_world() const { return world; }

   private:
    /// dispatches a call to derivedT::op if Space == Host, otherwise to derivedT::op_cuda if Space == CUDA
    /// @return void if called a synchronous function, or ttg::coroutine_handle<> if called a coroutine (if non-null,
    ///    points to the suspended coroutine)
    template <ttg::ExecutionSpace Space, typename... Args>
    auto op(Args &&...args) {
      derivedT *derived = static_cast<derivedT *>(this);
      // TODO: do we still distinguish op and op_cuda? How do we handle support for multiple devices?
      //if constexpr (Space == ttg::ExecutionSpace::Host) {
        using return_type = decltype(derived->op(std::forward<Args>(args)...));
        if constexpr (std::is_same_v<return_type,void>) {
          derived->op(std::forward<Args>(args)...);
          return;
        }
        else {
          return derived->op(std::forward<Args>(args)...);
        }
#if 0
      }
      else if constexpr (Space == ttg::ExecutionSpace::CUDA) {
        using return_type = decltype(derived->op_cuda(std::forward<Args>(args)...));
        if constexpr (std::is_same_v<return_type,void>) {
          derived->op_cuda(std::forward<Args>(args)...);
          return;
        }
        else {
          return derived->op_cuda(std::forward<Args>(args)...);
        }
      }
      else
        ttg::abort();
#endif // 0
    }

    template <std::size_t i, typename terminalT, typename Key>
    void invoke_pull_terminal(terminalT &in, const Key &key, detail::parsec_ttg_task_base_t *task) {
      if (in.is_pull_terminal) {
        auto owner = in.container.owner(key);
        if (owner != world.rank()) {
          get_pull_terminal_data_from<i>(owner, key);
        } else {
          // push the data to the task
          set_arg<i>(key, (in.container).get(key));
        }
      }
    }

    template <std::size_t i, typename Key>
    void get_pull_terminal_data_from(const int owner,
                                     const Key &key) {
      using msg_t = detail::msg_t;
      auto &world_impl = world.impl();
      parsec_taskpool_t *tp = world_impl.taskpool();
      std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), tp->taskpool_id,
                                                            msg_header_t::MSG_GET_FROM_PULL, i,
                                                            world.rank(), 1);
      /* pack the key */
      size_t pos = 0;
      pos = pack(key, msg->bytes, pos);
      tp->tdm.module->outgoing_message_start(tp, owner, NULL);
      tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
      parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                        sizeof(msg_header_t) + pos);
    }

    template <std::size_t... IS, typename Key = keyT>
    void invoke_pull_terminals(std::index_sequence<IS...>, const Key &key, detail::parsec_ttg_task_base_t *task) {
      int junk[] = {0, (invoke_pull_terminal<IS>(
                            std::get<IS>(input_terminals), key, task),
                        0)...};
      junk[0]++;
    }

    template <std::size_t... IS>
    static input_refs_tuple_type make_tuple_of_ref_from_array(task_t *task, std::index_sequence<IS...>) {
      return input_refs_tuple_type{static_cast<std::tuple_element_t<IS, input_refs_tuple_type>>(
          *reinterpret_cast<std::remove_reference_t<std::tuple_element_t<IS, input_refs_tuple_type>> *>(
              task->copies[IS]->get_ptr()))...};
    }

#ifdef TTG_HAVE_DEVICE
    /**
     * Submit callback called by PaRSEC once all input transfers have completed.
     */
    template <ttg::ExecutionSpace Space>
    static int device_static_submit(parsec_device_gpu_module_t  *gpu_device,
                                    parsec_gpu_task_t           *gpu_task,
                                    parsec_gpu_exec_stream_t    *gpu_stream) {

      task_t *task = (task_t*)gpu_task->ec;
      // get the device task from the coroutine handle
      ttg::device::Task dev_task = ttg::device::detail::device_task_handle_type::from_address(task->suspended_task_address);

      task->dev_ptr->stream = gpu_stream;

      //std::cout << "device_static_submit task " << task << std::endl;

      // get the promise which contains the views
      auto dev_data = dev_task.promise();

      /* we should still be waiting for the transfer to complete */
      assert(dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_WAIT_TRANSFER ||
             dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_WAIT_KERNEL);

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) && defined(TTG_HAVE_CUDA)
      {
        parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t *)gpu_stream;
        int device = detail::parsec_device_to_ttg_device(gpu_device->super.device_index);
        ttg::device::detail::set_current(device, cuda_stream->cuda_stream);
      }
#endif // defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) && defined(TTG_HAVE_CUDA)

#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT) && defined(TTG_HAVE_HIP)
      {
        parsec_hip_exec_stream_t *hip_stream = (parsec_hip_exec_stream_t *)gpu_stream;
        int device = detail::parsec_device_to_ttg_device(gpu_device->super.device_index);
        ttg::device::detail::set_current(device, hip_stream->hip_stream);
      }
#endif // defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) && defined(TTG_HAVE_CUDA)

#if defined(PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT) && defined(TTG_HAVE_LEVEL_ZERO)
      {
        parsec_level_zero_exec_stream_t *stream;
        stream = (parsec_level_zero_exec_stream_t *)gpu_stream;
        int device = detail::parsec_device_to_ttg_device(gpu_device->super.device_index);
        ttg::device::detail::set_current(device, stream->swq->queue);
      }
#endif // defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) && defined(TTG_HAVE_CUDA)

      /* Here we call back into the coroutine again after the transfers have completed */
      static_op<Space>(&task->parsec_task);

      ttg::device::detail::reset_current();

      /* we will come back into this function once the kernel and transfers are done */
      int rc = PARSEC_HOOK_RETURN_DONE;
      if (nullptr != task->suspended_task_address) {
        /* Get a new handle for the promise*/
        dev_task = ttg::device::detail::device_task_handle_type::from_address(task->suspended_task_address);
        dev_data = dev_task.promise();

        assert(dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_WAIT_KERNEL ||
               dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_SENDOUT     ||
               dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_COMPLETE);

        if (ttg::device::detail::TTG_DEVICE_CORO_SENDOUT == dev_data.state() ||
            ttg::device::detail::TTG_DEVICE_CORO_COMPLETE == dev_data.state()) {
          /* the task started sending so we won't come back here */
          //std::cout << "device_static_submit task " << task << " complete" << std::endl;
        } else {
          //std::cout << "device_static_submit task " << task << " return-again" << std::endl;
          rc = PARSEC_HOOK_RETURN_AGAIN;
        }
      } else {
        /* the task is done so we won't come back here */
        //std::cout << "device_static_submit task " << task << " complete" << std::endl;
      }
      return rc;
    }

    static void
    static_device_stage_in(parsec_gpu_task_t        *gtask,
                         uint32_t                  flow_mask,
                         parsec_gpu_exec_stream_t *gpu_stream) {
      /* register any memory that hasn't been registered yet */
      for (int i = 0; i < MAX_PARAM_COUNT; ++i) {
        if (flow_mask & (1<<i)) {
          task_t *task = (task_t*)gtask->ec;
          parsec_data_copy_t *copy = task->parsec_task.data[i].data_in;
          if (0 == (copy->flags & TTG_PARSEC_DATA_FLAG_REGISTERED)) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
            // register host memory for faster device access
            cudaError_t status;
            //status = cudaHostRegister(copy->device_private, gtask->flow_nb_elts[i], cudaHostRegisterPortable);
            //assert(cudaSuccess == status);
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT
            //copy->flags |= TTG_PARSEC_DATA_FLAG_REGISTERED;
          }
        }
      }
    }

    static int
    static_device_stage_in_hook(parsec_gpu_task_t        *gtask,
                                uint32_t                  flow_mask,
                                parsec_gpu_exec_stream_t *gpu_stream) {
      static_device_stage_in(gtask, flow_mask, gpu_stream);
      return parsec_default_gpu_stage_in(gtask, flow_mask, gpu_stream);
    }

    template <ttg::ExecutionSpace Space>
    static parsec_hook_return_t device_static_op(parsec_task_t* parsec_task) {
      static_assert(derived_has_device_op());

      int dev_index;
      double ratio = 1.0;

      task_t *task = (task_t*)parsec_task;
      parsec_execution_stream_s *es = task->tt->world.impl().execution_stream();

      //std::cout << "device_static_op: task " << parsec_task << std::endl;

      /* set up a device task */
      parsec_gpu_task_t *gpu_task;
      /* PaRSEC wants to free the gpu_task, because F***K ownerships */
      gpu_task = static_cast<parsec_gpu_task_t*>(std::calloc(1, sizeof(*gpu_task)));
      PARSEC_OBJ_CONSTRUCT(gpu_task, parsec_list_item_t);
      gpu_task->ec = parsec_task;
      gpu_task->task_type = 0; // user task
      gpu_task->load = 1;    // TODO: can we do better?
      gpu_task->last_data_check_epoch = -1; // used internally
      gpu_task->pushout = 0;
      gpu_task->submit = &TT::device_static_submit<Space>;

      /* set the gpu_task so it's available in register_device_memory */
      task->dev_ptr->gpu_task = gpu_task;

      // first invocation of the coroutine to get the coroutine handle
      static_op<Space>(parsec_task);

      /* when we come back here, the flows in gpu_task are set (see register_device_memory) */

      if (nullptr == task->suspended_task_address) {
        /* short-cut in case the task returned immediately */
        return PARSEC_HOOK_RETURN_DONE;
      }

      // get the device task from the coroutine handle
      auto dev_task = ttg::device::detail::device_task_handle_type::from_address(task->suspended_task_address);

      // get the promise which contains the views
      ttg::device::detail::device_task_promise_type& dev_data = dev_task.promise();

      /* for now make sure we're waiting for transfers and the coro hasn't skipped this step */
      assert(dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_WAIT_TRANSFER);

      /* set up a temporary task-class to correctly specify the flows */
      parsec_task_class_t tc = *task->parsec_task.task_class;

      tc.name = task->parsec_task.task_class->name;
      // input flows are set up during register_device_memory as part of the first invocation above
      for (int i = 0; i < MAX_PARAM_COUNT; ++i) {
        tc.in[i]  = gpu_task->flow[i];
        tc.out[i] = gpu_task->flow[i];
      }
      tc.nb_flows = MAX_PARAM_COUNT;

      /* swap in the new task class */
      const parsec_task_class_t* tmp = task->parsec_task.task_class;
      *const_cast<parsec_task_class_t**>(&task->parsec_task.task_class) = &tc;

      /* TODO: is this the right place to set the mask? */
      task->parsec_task.chore_mask = PARSEC_DEV_ALL;
      /* get a device and come back if we need another one */
      int64_t task_load = 1;
      dev_index = parsec_get_best_device(parsec_task, &task_load);

      /* swap back the original task class */
      task->parsec_task.task_class = tmp;

      gpu_task->load = task_load;
      assert(dev_index >= 0);
      if (!parsec_mca_device_is_gpu(dev_index)) {
          return PARSEC_HOOK_RETURN_NEXT; /* Fall back */
      }

      parsec_device_gpu_module_t *device = (parsec_device_gpu_module_t*)parsec_mca_device_get(dev_index);
      assert(NULL != device);

      task->dev_ptr->device = device;

      switch(device->super.type) {

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        case PARSEC_DEV_CUDA:
          if constexpr (Space == ttg::ExecutionSpace::CUDA) {
            /* TODO: we need custom staging functions because PaRSEC looks at the
             *       task-class to determine the number of flows. */
            gpu_task->stage_in  = static_device_stage_in_hook;
            gpu_task->stage_out = parsec_default_gpu_stage_out;
            return parsec_device_kernel_scheduler(&device->super, es, gpu_task);
          }
          break;
#endif
#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
        case PARSEC_DEV_HIP:
          if constexpr (Space == ttg::ExecutionSpace::HIP) {
            gpu_task->stage_in  = static_device_stage_in_hook;
            gpu_task->stage_out = parsec_default_gpu_stage_out;
            return parsec_device_kernel_scheduler(&device->super, es, gpu_task);
          }
          break;
#endif // PARSEC_HAVE_DEV_HIP_SUPPORT
#if defined(PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT)
        case PARSEC_DEV_LEVEL_ZERO:
          if constexpr (Space == ttg::ExecutionSpace::L0) {
            gpu_task->stage_in  = static_device_stage_in_hook;
            gpu_task->stage_out = parsec_default_gpu_stage_out;
            return parsec_device_kernel_scheduler(&device->super, es, gpu_task);
          }
          break;
#endif // PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT
        default:
          break;
      }
      ttg::print_error(task->tt->get_name(), " : received mismatching device type ", (int)device->super.type, " from PaRSEC");
      ttg::abort();
      return PARSEC_HOOK_RETURN_DONE; // will not be reacehed
    }
#endif  // TTG_HAVE_DEVICE

    template <ttg::ExecutionSpace Space>
    static parsec_hook_return_t static_op(parsec_task_t *parsec_task) {

      task_t *task = (task_t*)parsec_task;
      void* suspended_task_address =
#ifdef TTG_HAS_COROUTINE
        task->suspended_task_address;  // non-null = need to resume the task
#else
        nullptr;
#endif
      //std::cout << "static_op: suspended_task_address " << suspended_task_address << std::endl;
      if (suspended_task_address == nullptr) {  // task is a coroutine that has not started or an ordinary function

        ttT *baseobj = task->tt;
        derivedT *obj = static_cast<derivedT *>(baseobj);
        assert(detail::parsec_ttg_caller == nullptr);
        detail::parsec_ttg_caller = static_cast<detail::parsec_ttg_task_base_t*>(task);
        if (obj->tracing()) {
          if constexpr (!ttg::meta::is_void_v<keyT>)
            ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : ", task->key, ": executing");
          else
            ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : executing");
        }

        if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          auto input = make_tuple_of_ref_from_array(task, std::make_index_sequence<numinvals>{});
          TTG_PROCESS_TT_OP_RETURN(suspended_task_address, baseobj->template op<Space>(task->key, std::move(input), obj->output_terminals));
        } else if constexpr (!ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          TTG_PROCESS_TT_OP_RETURN(suspended_task_address, baseobj->template op<Space>(task->key, obj->output_terminals));
        } else if constexpr (ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          auto input = make_tuple_of_ref_from_array(task, std::make_index_sequence<numinvals>{});
          TTG_PROCESS_TT_OP_RETURN(suspended_task_address, baseobj->template op<Space>(std::move(input), obj->output_terminals));
        } else if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          TTG_PROCESS_TT_OP_RETURN(suspended_task_address, baseobj->template op<Space>(obj->output_terminals));
        } else {
          ttg::abort();
        }
        detail::parsec_ttg_caller = nullptr;
      }
      else {  // resume the suspended coroutine
#ifdef TTG_HAVE_DEVICE
        ttg::device::Task coro = ttg::device::detail::device_task_handle_type::from_address(suspended_task_address);
        assert(detail::parsec_ttg_caller == nullptr);
        detail::parsec_ttg_caller = static_cast<detail::parsec_ttg_task_base_t*>(task);
        // TODO: unify the outputs tls handling
        auto old_output_tls_ptr = task->tt->outputs_tls_ptr_accessor();
        task->tt->set_outputs_tls_ptr();
        coro.resume();
        if (coro.completed()) {
          coro.destroy();
          suspended_task_address = nullptr;
        }
        task->tt->set_outputs_tls_ptr(old_output_tls_ptr);
        detail::parsec_ttg_caller = nullptr;
#if 0
#ifdef TTG_HAS_COROUTINE
        auto ret = static_cast<ttg::resumable_task>(ttg::coroutine_handle<>::from_address(suspended_task_address));
        assert(ret.ready());
        ret.resume();
        if (ret.completed()) {
          ret.destroy();
          suspended_task_address = nullptr;
        }
        else { // not yet completed
          // leave suspended_task_address as is
        }
        task->suspended_task_address = suspended_task_address;
#else
#endif  // TTG_HAS_COROUTINE
        ttg::abort();  // should not happen
#endif  // 0
#else  // TTG_HAVE_DEVICE
ttg::abort();  // should not happen
#endif  // TTG_HAVE_DEVICE
      }
      task->suspended_task_address = suspended_task_address;

      if (suspended_task_address == nullptr) {
        ttT *baseobj = task->tt;
        derivedT *obj = static_cast<derivedT *>(baseobj);
        if (obj->tracing()) {
          if constexpr (!ttg::meta::is_void_v<keyT>)
            ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : ", task->key, ": done executing");
          else
            ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : done executing");
        }
      }

// XXX the below code is not needed, should be removed once the fib test has been changed
#if 0
#ifdef TTG_HAS_COROUTINE
      if (suspended_task_address) {
        // right now can events are not properly implemented, we are only testing the workflow with dummy events
        // so mark the events finished manually, parsec will rerun this task again and it should complete the second time
        auto events = static_cast<ttg::resumable_task>(ttg::coroutine_handle<>::from_address(suspended_task_address)).events();
        for (auto &event_ptr : events) {
          event_ptr->finish();
        }
        assert(ttg::coroutine_handle<>::from_address(suspended_task_address).promise().ready());

        // TODO: shove {ptr to parsec_task, ptr to this function} to the list of tasks suspended by this thread (hence stored in TLS)
        // thread will loop over its list (after running every task? periodically? need a dedicated queue of ready tasks?)
        // and resume the suspended tasks whose events are ready (N.B. ptr to parsec_task is enough to get the list of pending events)
        // event clearance will in device case handled by host callbacks run by the dedicated device runtime thread

        // TODO PARSEC_HOOK_RETURN_AGAIN -> PARSEC_HOOK_RETURN_ASYNC when event tracking and task resumption (by this thread) is ready
        return PARSEC_HOOK_RETURN_AGAIN;
      }
      else
#endif  // TTG_HAS_COROUTINE
#endif // 0
        return PARSEC_HOOK_RETURN_DONE;
    }

    template <ttg::ExecutionSpace Space>
    static parsec_hook_return_t static_op_noarg(parsec_task_t *parsec_task) {
      task_t *task = static_cast<task_t*>(parsec_task);

      void* suspended_task_address =
#ifdef TTG_HAS_COROUTINE
        task->suspended_task_address;  // non-null = need to resume the task
#else
        nullptr;
#endif
      if (suspended_task_address == nullptr) {  // task is a coroutine that has not started or an ordinary function
        ttT *baseobj = (ttT *)task->object_ptr;
        derivedT *obj = (derivedT *)task->object_ptr;
        assert(detail::parsec_ttg_caller == NULL);
        detail::parsec_ttg_caller = task;
        if constexpr (!ttg::meta::is_void_v<keyT>) {
          TTG_PROCESS_TT_OP_RETURN(suspended_task_address, baseobj->template op<Space>(task->key, obj->output_terminals));
        } else if constexpr (ttg::meta::is_void_v<keyT>) {
          TTG_PROCESS_TT_OP_RETURN(suspended_task_address, baseobj->template op<Space>(obj->output_terminals));
        } else  // unreachable
          ttg:: abort();
        detail::parsec_ttg_caller = NULL;
      }
      else {
#ifdef TTG_HAS_COROUTINE
        auto ret = static_cast<ttg::resumable_task>(ttg::coroutine_handle<>::from_address(suspended_task_address));
        assert(ret.ready());
        ret.resume();
        if (ret.completed()) {
          ret.destroy();
          suspended_task_address = nullptr;
        }
        else { // not yet completed
          // leave suspended_task_address as is
        }
#else
        ttg::abort();  // should not happen
#endif
      }
      task->suspended_task_address = suspended_task_address;

      if (suspended_task_address) {
        ttg::abort();  // not yet implemented
        // see comments in static_op()
        return PARSEC_HOOK_RETURN_AGAIN;
      }
      else
        return PARSEC_HOOK_RETURN_DONE;
    }

    template <std::size_t i>
    static parsec_hook_return_t static_reducer_op(parsec_execution_stream_s *es, parsec_task_t *parsec_task) {
      using rtask_t = detail::reducer_task_t;
      using value_t = std::tuple_element_t<i, actual_input_tuple_type>;
      constexpr const bool val_is_void = ttg::meta::is_void_v<value_t>;
      constexpr const bool input_is_const = std::is_const_v<value_t>;
      rtask_t *rtask = (rtask_t*)parsec_task;
      task_t *parent_task = static_cast<task_t*>(rtask->parent_task);
      ttT *baseobj = parent_task->tt;
      derivedT *obj = static_cast<derivedT *>(baseobj);

      auto& reducer = std::get<i>(baseobj->input_reducers);

      //std::cout << "static_reducer_op " << parent_task->key << std::endl;

      if (obj->tracing()) {
        if constexpr (!ttg::meta::is_void_v<keyT>)
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : ", parent_task->key, ": reducer executing");
        else
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : reducer executing");
      }

      /* the copy to reduce into */
      detail::ttg_data_copy_t *target_copy;
      target_copy = parent_task->copies[i];
      assert(val_is_void || nullptr != target_copy);
      /* once we hit 0 we have to stop since another thread might enqueue a new reduction task */
      std::size_t c = 0;
      std::size_t size = 0;
      assert(parent_task->streams[i].reduce_count > 0);
      if (rtask->is_first) {
        if (0 == (parent_task->streams[i].reduce_count.fetch_sub(1, std::memory_order_acq_rel)-1)) {
          /* we were the first and there is nothing to be done */
          if (obj->tracing()) {
            if constexpr (!ttg::meta::is_void_v<keyT>)
              ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : ", parent_task->key, ": first reducer empty");
            else
              ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : first reducer empty");
          }

          return PARSEC_HOOK_RETURN_DONE;
        }
      }

      assert(detail::parsec_ttg_caller == NULL);
      detail::parsec_ttg_caller = rtask->parent_task;

      do {
        if constexpr(!val_is_void) {
          /* the copies to reduce out of */
          detail::ttg_data_copy_t *source_copy;
          parsec_list_item_t *item;
          item = parsec_lifo_pop(&parent_task->streams[i].reduce_copies);
          if (nullptr == item) {
            // maybe someone is changing the goal right now
            break;
          }
          source_copy = ((detail::ttg_data_copy_self_t *)(item))->self;
          assert(target_copy->num_readers() == target_copy->mutable_tag);
          assert(source_copy->num_readers() > 0);
          reducer(*reinterpret_cast<std::decay_t<value_t> *>(target_copy->get_ptr()),
                  *reinterpret_cast<std::decay_t<value_t> *>(source_copy->get_ptr()));
          detail::release_data_copy(source_copy);
        } else if constexpr(val_is_void) {
          reducer(); // invoke control reducer
        }
        // there is only one task working on this stream, so no need to be atomic here
        size = ++parent_task->streams[i].size;
        //std::cout << "static_reducer_op size " << size << " of " << parent_task->streams[i].goal << std::endl;
      } while ((c = (parent_task->streams[i].reduce_count.fetch_sub(1, std::memory_order_acq_rel)-1)) > 0);
      //} while ((c = (--task->streams[i].reduce_count)) > 0);

      /* finalize_argstream sets goal to 1, so size may be larger than goal */
      bool complete = (size >= parent_task->streams[i].goal);

      //std::cout << "static_reducer_op size " << size
      //          << " of " << parent_task->streams[i].goal << " complete " << complete
      //          << " c " << c << std::endl;
      if (complete && c == 0) {
        if constexpr(input_is_const) {
          /* make the consumer task a reader if its input is const */
          target_copy->reset_readers();
        }
        /* task may not be runnable yet because other inputs are missing, have release_task decide */
        parent_task->remove_from_hash = true;
        parent_task->release_task(parent_task);
      }

      detail::parsec_ttg_caller = NULL;

      if (obj->tracing()) {
        if constexpr (!ttg::meta::is_void_v<keyT>)
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : ", parent_task->key, ": done executing");
        else
          ttg::trace(obj->get_world().rank(), ":", obj->get_name(), " : done executing");
      }

      return PARSEC_HOOK_RETURN_DONE;
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
    uint64_t pack(T &obj, void *bytes, uint64_t pos, detail::ttg_data_copy_t *copy = nullptr) {
      const ttg_data_descriptor *dObj = ttg::get_data_descriptor<ttg::meta::remove_cvr_t<T>>();
      uint64_t payload_size = dObj->payload_size(&obj);
      if (copy) {
        /* reset any tracked data, we don't care about the packing from the payload size */
        copy->iovec_reset();
      }

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
            ttg::abort();
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
        case msg_header_t::MSG_GET_FROM_PULL: {
          assert(hd->param_id >= 0);
          assert(hd->param_id < obj->get_from_pull_msg_fcts.size());
          auto member = obj->get_from_pull_msg_fcts[hd->param_id];
          (obj->*member)(data, size);
          break;
        }
        default:
          ttg::abort();
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
    void set_arg_from_msg_keylist(ttg::span<keyT> &&keylist, detail::ttg_data_copy_t *copy) {
      /* create a dummy task that holds the copy, which can be reused by others */
      task_t *dummy;
      parsec_execution_stream_s *es = world.impl().execution_stream();
      parsec_thread_mempool_t *mempool = get_task_mempool();
      dummy = new (parsec_thread_mempool_allocate(mempool)) task_t(mempool, &this->self);
      dummy->set_dummy(true);
      // TODO: do we need to copy static_stream_goal in dummy?

      /* set the received value as the dummy's only data */
      dummy->copies[0] = copy;

      /* We received the task on this world, so it's using the same taskpool */
      dummy->parsec_task.taskpool = world.impl().taskpool();

      /* save the current task and set the dummy task */
      auto parsec_ttg_caller_save = detail::parsec_ttg_caller;
      detail::parsec_ttg_caller = dummy;

      /* iterate over the keys and have them use the copy we made */
      parsec_task_t *task_ring = nullptr;
      for (auto &&key : keylist) {
        set_arg_local_impl<i>(key, *reinterpret_cast<valueT *>(copy->get_ptr()), copy, &task_ring);
      }

      if (nullptr != task_ring) {
        auto &world_impl = world.impl();
        parsec_task_t *vp_task_ring[1] = { task_ring };
        __parsec_schedule_vp(world_impl.execution_stream(), vp_task_ring, 0);
      }

      /* restore the previous task */
      detail::parsec_ttg_caller = parsec_ttg_caller_save;

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
        /* TODO: can we avoid copying all the keys?! */
        uint64_t pos = msg->tt_id.key_offset;
        uint64_t key_end_pos;
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
        key_end_pos = pos;
        /* jump back to the beginning of the message to get the value */
        pos = 0;
        // case 1
        if constexpr (!ttg::meta::is_void_v<valueT>) {
          using decvalueT = std::decay_t<valueT>;
          int32_t num_iovecs = msg->tt_id.num_iovecs;
          //bool inline_data = msg->inline_data;
          detail::ttg_data_copy_t *copy;
          if constexpr (ttg::has_split_metadata<decvalueT>::value) {
            ttg::SplitMetadataDescriptor<decvalueT> descr;
            using metadata_t = decltype(descr.get_metadata(std::declval<decvalueT>()));

            /* unpack the metadata */
            metadata_t metadata;
            pos = unpack(metadata, msg->bytes, pos);

            //std::cout << "set_arg_from_msg splitmd num_iovecs " << num_iovecs << std::endl;

            copy = detail::create_new_datacopy(descr.create_from_metadata(metadata));
          } else if constexpr (!ttg::has_split_metadata<decvalueT>::value) {
            copy = detail::create_new_datacopy(decvalueT{});
#if 0
            // TODO: first attempt at sending directly to the device
            parsec_gpu_data_copy_t* gpu_elem;
            gpu_elem = PARSEC_DATA_GET_COPY(master, gpu_device->super.device_index);
            int i = detail::first_device_id;
            int devid = detail::first_device_id;
            while (i < parsec_nb_devices) {
              if (nullptr == gpu_elem) {
                gpu_elem = PARSEC_OBJ_NEW(parsec_data_copy_t);
                gpu_elem->flags = PARSEC_DATA_FLAG_PARSEC_OWNED | PARSEC_DATA_FLAG_PARSEC_MANAGED;
                gpu_elem->coherency_state = PARSEC_DATA_COHERENCY_INVALID;
                gpu_elem->version = 0;
                gpu_elem->coherency_state = PARSEC_DATA_COHERENCY_OWNED;
              }
              if (nullptr == gpu_elem->device_private) {
                gpu_elem->device_private = zone_malloc(gpu_device->memory, gpu_task->flow_nb_elts[i]);
                if (nullptr == gpu_elem->device_private) {
                  devid++;
                  continue;
                }
                break;
              }
            }
#endif // 0
            /* unpack the object, potentially discovering iovecs */
            pos = unpack(*static_cast<decvalueT *>(copy->get_ptr()), msg->bytes, pos);
            //std::cout << "set_arg_from_msg iovec_begin num_iovecs " << num_iovecs << " distance " << std::distance(copy->iovec_begin(), copy->iovec_end()) << std::endl;
            assert(std::distance(copy->iovec_begin(), copy->iovec_end()) == num_iovecs);
          }

          if (num_iovecs == 0) {
            set_arg_from_msg_keylist<i, decvalueT>(ttg::span<keyT>(&keylist[0], num_keys), copy);
          } else {
            /* unpack the header and start the RMA transfers */

            /* get the remote rank */
            int remote = msg->tt_id.sender;
            assert(remote < world.size());

            auto &val = *static_cast<decvalueT *>(copy->get_ptr());

            bool inline_data = msg->tt_id.inline_data;

            int nv = 0;
            /* start the RMA transfers */
            auto handle_iovecs_fn =
              [&](auto&& iovecs) {

                if (inline_data) {
                  /* unpack the data from the message */
                  for (auto &&iov : iovecs) {
                    ++nv;
                    std::memcpy(iov.data, msg->bytes + pos, iov.num_bytes);
                    pos += iov.num_bytes;
                  }
                } else {
                  /* extract the callback tag */
                  parsec_ce_tag_t cbtag;
                  std::memcpy(&cbtag, msg->bytes + pos, sizeof(cbtag));
                  pos += sizeof(cbtag);

                  /* create the value from the metadata */
                  auto activation = new detail::rma_delayed_activate(
                      std::move(keylist), copy, num_iovecs, [this](std::vector<keyT> &&keylist, detail::ttg_data_copy_t *copy) {
                        set_arg_from_msg_keylist<i, decvalueT>(keylist, copy);
                        this->world.impl().decrement_inflight_msg();
                      });

                  using ActivationT = std::decay_t<decltype(*activation)>;

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
                    //std::cout << "set_arg_from_msg: get rreg " << rreg << " remote " << remote << std::endl;
                    parsec_ce.get(&parsec_ce, lreg, 0, rreg, 0, iov.num_bytes, remote,
                                  &detail::get_complete_cb<ActivationT>, activation,
                                  /*world.impl().parsec_ttg_rma_tag()*/
                                  cbtag, &fn_ptr, sizeof(std::intptr_t));
                  }
                }
            };
            if constexpr (ttg::has_split_metadata<decvalueT>::value) {
              ttg::SplitMetadataDescriptor<decvalueT> descr;
              handle_iovecs_fn(descr.get_data(val));
            } else if constexpr (!ttg::has_split_metadata<decvalueT>::value) {
              handle_iovecs_fn(copy->iovec_span());
              copy->iovec_reset();
            }

            assert(num_iovecs == nv);
            assert(size == (key_end_pos + sizeof(msg_header_t)));

            if (inline_data) {
              set_arg_from_msg_keylist<i, decvalueT>(ttg::span<keyT>(&keylist[0], num_keys), copy);
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
      } else {  // unreachable
        ttg::abort();
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

    template <std::size_t i>
    void get_from_pull_msg(void *data, std::size_t size) {
      using msg_t = detail::msg_t;
      msg_t *msg = static_cast<msg_t *>(data);
      auto &in = std::get<i>(input_terminals);
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        /* unpack the key */
        uint64_t pos = 0;
        keyT key;
        pos = unpack(key, msg->bytes, pos);
        set_arg<i>(key, (in.container).get(key));
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
      int32_t priority = 0;
      if constexpr (!keyT_is_Void) {
        priority = priomap(key);
        /* placement-new the task */
        newtask = new (taskobj) task_t(key, mempool, &this->self, world_impl.taskpool(), this, priority);
      } else {
        priority = priomap();
        /* placement-new the task */
        newtask = new (taskobj) task_t(mempool, &this->self, world_impl.taskpool(), this, priority);
      }

      for (int i = 0; i < static_stream_goal.size(); ++i) {
        newtask->streams[i].goal = static_stream_goal[i];
      }

      ttg::trace(world.rank(), ":", get_name(), " : ", key, ": creating task");
      return newtask;
    }


    template <std::size_t i>
    detail::reducer_task_t *create_new_reducer_task(task_t *task, bool is_first) {
      /* make sure we can reuse the existing memory pool and don't have to create a new one */
      static_assert(sizeof(task_t) >= sizeof(detail::reducer_task_t));
      constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;
      auto &world_impl = world.impl();
      detail::reducer_task_t *newtask;
      parsec_thread_mempool_t *mempool = get_task_mempool();
      char *taskobj = (char *)parsec_thread_mempool_allocate(mempool);
      // use the priority of the task we stream into
      int32_t priority = 0;
      if constexpr (!keyT_is_Void) {
        priority = priomap(task->key);
        ttg::trace(world.rank(), ":", get_name(), " : ", task->key, ": creating reducer task");
      } else {
        priority = priomap();
        ttg::trace(world.rank(), ":", get_name(), ": creating reducer task");
      }
      /* placement-new the task */
      newtask = new (taskobj) detail::reducer_task_t(task, mempool, inpute_reducers_taskclass[i],
                                                     world_impl.taskpool(), priority, is_first);

      return newtask;
    }


    // Used to set the i'th argument
    template <std::size_t i, typename Key, typename Value>
    void set_arg_local_impl(const Key &key, Value &&value, detail::ttg_data_copy_t *copy_in = nullptr,
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
      bool discover_task = true;
      bool get_pull_data = false;
      bool has_lock = false;
      /* If we have only one input and no reducer on that input we can skip the hash table */
      if (numins > 1 || reducer) {
        has_lock = true;
        parsec_hash_table_lock_bucket(&tasks_table, hk);
        if (nullptr == (task = (task_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
          task = create_new_task(key);
          world_impl.increment_created();
          parsec_hash_table_nolock_insert(&tasks_table, &task->tt_ht_item);
          get_pull_data = !is_lazy_pull();
          if( world_impl.dag_profiling() ) {
#if defined(PARSEC_PROF_GRAPHER)
            parsec_prof_grapher_task(&task->parsec_task, world_impl.execution_stream()->th_id, 0,
                                     key_hash(make_key(task->parsec_task.taskpool, task->parsec_task.locals), nullptr));
#endif
          }
        } else if (!reducer && numins == (task->in_data_count + 1)) {
          /* remove while we have the lock */
          parsec_hash_table_nolock_remove(&tasks_table, hk);
          remove_from_hash = false;
        }
        /* if we have a reducer, we need to hold on to the lock for just a little longer */
        if (!reducer) {
          parsec_hash_table_unlock_bucket(&tasks_table, hk);
          has_lock = false;
        }
      } else {
        task = create_new_task(key);
        world_impl.increment_created();
        remove_from_hash = false;
        if( world_impl.dag_profiling() ) {
#if defined(PARSEC_PROF_GRAPHER)
          parsec_prof_grapher_task(&task->parsec_task, world_impl.execution_stream()->th_id, 0,
                                   key_hash(make_key(task->parsec_task.taskpool, task->parsec_task.locals), nullptr));
#endif
        }
      }

      if( world_impl.dag_profiling() ) {
#if defined(PARSEC_PROF_GRAPHER)
        if(NULL != detail::parsec_ttg_caller && !detail::parsec_ttg_caller->is_dummy()) {
          int orig_index = detail::find_index_of_copy_in_task(detail::parsec_ttg_caller, &value);
          char orig_str[32];
          char dest_str[32];
          if(orig_index >= 0) {
            snprintf(orig_str, 32, "%d", orig_index);
          } else {
            strncpy(orig_str, "_", 32);
          }
          snprintf(dest_str, 32, "%lu", i);
          parsec_flow_t orig{ .name = orig_str, .sym_type = PARSEC_SYM_INOUT, .flow_flags = PARSEC_FLOW_ACCESS_RW,
                              .flow_index = 0, .flow_datatype_mask = ~0 };
          parsec_flow_t dest{ .name = dest_str, .sym_type = PARSEC_SYM_INOUT, .flow_flags = PARSEC_FLOW_ACCESS_RW,
                              .flow_index = 0, .flow_datatype_mask = ~0 };
          parsec_prof_grapher_dep(&detail::parsec_ttg_caller->parsec_task, &task->parsec_task, discover_task ? 1 : 0, &orig, &dest);
        }
#endif
      }

      auto get_copy_fn = [&](detail::parsec_ttg_task_base_t *task, auto&& value, bool is_const){
        detail::ttg_data_copy_t *copy = copy_in;
        if (nullptr == copy && nullptr != detail::parsec_ttg_caller) {
          copy = detail::find_copy_in_task(detail::parsec_ttg_caller, &value);
        }
        if (nullptr != copy) {
          /* retain the data copy */
          copy = detail::register_data_copy<valueT>(copy, task, is_const);
        } else {
          /* create a new copy */
          copy = detail::create_new_datacopy(std::forward<Value>(value));
          if (!is_const) {
            copy->mark_mutable();
          }
        }
        return copy;
      };

      if (reducer && 1 != task->streams[i].goal) {  // is this a streaming input? reduce the received value
        auto submit_reducer_task = [&](auto *parent_task){
          /* check if we need to create a task */
          std::size_t c = parent_task->streams[i].reduce_count.fetch_add(1, std::memory_order_release);
          //std::cout << "submit_reducer_task " << key << " c " << c << std::endl;
          if (0 == c) {
            /* we are responsible for creating the reduction task */
            detail::reducer_task_t *reduce_task;
            reduce_task = create_new_reducer_task<i>(parent_task, false);
            reduce_task->release_task(reduce_task); // release immediately
          }
        };

        if constexpr (!ttg::meta::is_void_v<valueT>) {  // for data values
          // have a value already? if not, set, otherwise reduce
          detail::ttg_data_copy_t *copy = nullptr;
          if (nullptr == (copy = task->copies[i])) {
            using decay_valueT = std::decay_t<valueT>;

            /* first input value, create a task and bind it to the copy */
            //std::cout << "Creating new reducer task for " << key << std::endl;
            detail::reducer_task_t *reduce_task;
            reduce_task = create_new_reducer_task<i>(task, true);

            /* protected by the bucket lock */
            task->streams[i].size = 1;
            task->streams[i].reduce_count.store(1, std::memory_order_relaxed);

            /* get the copy to use as input for this task */
            detail::ttg_data_copy_t *copy = get_copy_fn(reduce_task, std::forward<Value>(value), false);

            /* put the copy into the task */
            task->copies[i] = copy;

            /* release the task if we're not deferred
             * TODO: can we delay that until we get the second value?
             */
            if (copy->get_next_task() != &reduce_task->parsec_task) {
              reduce_task->release_task(reduce_task);
            }

            /* now we can unlock the bucket */
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
          } else {
            /* unlock the bucket, the lock is not needed anymore */
            parsec_hash_table_unlock_bucket(&tasks_table, hk);

            /* get the copy to use as input for this task */
            detail::ttg_data_copy_t *copy = get_copy_fn(task, std::forward<Value>(value), true);

            /* enqueue the data copy to be reduced */
            parsec_lifo_push(&task->streams[i].reduce_copies, &copy->super);
            submit_reducer_task(task);
          }
        } else {
          /* unlock the bucket, the lock is not needed anymore */
          parsec_hash_table_unlock_bucket(&tasks_table, hk);
          /* submit reducer for void values to handle side effects */
          submit_reducer_task(task);
        }
        //if (release) {
        //  parsec_hash_table_nolock_remove(&tasks_table, hk);
        //  remove_from_hash = false;
        //}
        //parsec_hash_table_unlock_bucket(&tasks_table, hk);
      } else {
        /* unlock the bucket, the lock is not needed anymore */
        if (has_lock) {
          parsec_hash_table_unlock_bucket(&tasks_table, hk);
        }
        /* whether the task needs to be deferred or not */
        if constexpr (!valueT_is_Void) {
          if (nullptr != task->copies[i]) {
            ttg::print_error(get_name(), " : ", key, ": error argument is already set : ", i);
            throw std::logic_error("bad set arg");
          }

          /* get the copy to use as input for this task */
          detail::ttg_data_copy_t *copy = get_copy_fn(task, std::forward<Value>(value), input_is_const);

          /* if we registered as a writer and were the first to register with this copy
           * we need to defer the release of this task to give other tasks a chance to
           * make a copy of the original data */
          release = (copy->get_next_task() != &task->parsec_task);
          task->copies[i] = copy;
        } else {
          release = true;
        }
      }
      task->remove_from_hash = remove_from_hash;
      if (release) {
        release_task(task, task_ring);
      }
      /* if not pulling lazily, pull the data here */
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        if (get_pull_data) {
          invoke_pull_terminals(std::make_index_sequence<std::tuple_size_v<input_values_tuple_type>>{}, task->key, task);
        }
      }
    }

    void release_task(task_t *task,
                      parsec_task_t **task_ring = nullptr) {
      constexpr const bool keyT_is_Void = ttg::meta::is_void_v<keyT>;

      /* if remove_from_hash == false, someone has already removed the task from the hash table
       * so we know that the task is ready, no need to do atomic increments here */
      bool is_ready = !task->remove_from_hash;
      int32_t count;
      if (is_ready) {
        count = numins;
      } else {
        count = parsec_atomic_fetch_inc_int32(&task->in_data_count) + 1;
        assert(count <= self.dependencies_goal);
      }

      auto &world_impl = world.impl();
      ttT *baseobj = task->tt;

      if (count == numins) {
        parsec_execution_stream_t *es = world_impl.execution_stream();
        parsec_key_t hk = task->pkey();
        if (tracing()) {
          if constexpr (!keyT_is_Void) {
            ttg::trace(world.rank(), ":", get_name(), " : ", task->key, ": submitting task for op ");
          } else {
            ttg::trace(world.rank(), ":", get_name(), ": submitting task for op ");
          }
        }
        if (task->remove_from_hash) parsec_hash_table_remove(&tasks_table, hk);
        if (nullptr == task_ring) {
          parsec_task_t *vp_task_rings[1] = { &task->parsec_task };
          __parsec_schedule_vp(es, vp_task_rings, 0);
        } else if (*task_ring == nullptr) {
          /* the first task is set directly */
          *task_ring = &task->parsec_task;
        } else {
          /* push into the ring */
          parsec_list_item_ring_push_sorted(&(*task_ring)->super, &task->parsec_task.super,
                                            offsetof(parsec_task_t, priority));
        }
      } else if constexpr (!ttg::meta::is_void_v<keyT>) {
        if ((baseobj->num_pullins + count == numins) && baseobj->is_lazy_pull()) {
          /* lazily pull the pull terminal data */
          baseobj->invoke_pull_terminals(std::make_index_sequence<std::tuple_size_v<input_values_tuple_type>>{}, task->key, task);
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

    template<typename Value, typename Key>
    bool can_inline_data(Value* value_ptr, detail::ttg_data_copy_t *copy, const Key& key, std::size_t num_keys) {
      using decvalueT = std::decay_t<Value>;
      bool inline_data = false;
      /* check whether to send data in inline */
      std::size_t iov_size = 0;
      std::size_t metadata_size = 0;
      if constexpr (ttg::has_split_metadata<std::decay_t<Value>>::value) {
        ttg::SplitMetadataDescriptor<decvalueT> descr;
        auto iovs = descr.get_data(*const_cast<decvalueT *>(value_ptr));
        iov_size = std::accumulate(iovs.begin(), iovs.end(), 0,
                                    [](std::size_t s, auto& iov){ return s + iov.num_bytes; });
        auto metadata = descr.get_metadata(*const_cast<decvalueT *>(value_ptr));
        metadata_size = ttg::default_data_descriptor<decltype(metadata)>::payload_size(&metadata);
      } else {
        /* TODO: how can we query the iovecs of the buffers here without actually packing the data? */
        metadata_size = ttg::default_data_descriptor<ttg::meta::remove_cvr_t<Value>>::payload_size(value_ptr);
        iov_size = std::accumulate(copy->iovec_begin(), copy->iovec_end(), 0,
                                    [](std::size_t s, auto& iov){ return s + iov.num_bytes; });
      }
      /* key is packed at the end */
      std::size_t key_pack_size = ttg::default_data_descriptor<Key>::payload_size(&key);
      std::size_t pack_size = key_pack_size + metadata_size + iov_size;
      if (pack_size < detail::max_inline_size) {
        inline_data = true;
      }
      return inline_data;
    }

    // Used to set the i'th argument
    template <std::size_t i, typename Key, typename Value>
    void set_arg_impl(const Key &key, Value &&value, detail::ttg_data_copy_t *copy_in = nullptr) {
      int owner;
      using decvalueT = std::decay_t<Value>;
      using norefvalueT = std::remove_reference_t<Value>;
      norefvalueT *value_ptr = &value;

#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      if(world.impl().profiling()) {
        parsec_profiling_ts_trace(world.impl().parsec_ttg_profile_backend_set_arg_start, 0, 0, NULL);
      }
#endif

      if constexpr (!ttg::meta::is_void_v<Key>)
        owner = keymap(key);
      else
        owner = keymap();
      if (owner == world.rank()) {
        if constexpr (!ttg::meta::is_void_v<keyT>)
          set_arg_local_impl<i>(key, std::forward<Value>(value), copy_in);
        else
          set_arg_local_impl<i>(ttg::Void{}, std::forward<Value>(value), copy_in);
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
          if(world.impl().profiling()) {
            parsec_profiling_ts_trace(world.impl().parsec_ttg_profile_backend_set_arg_end, 0, 0, NULL);
          }
#endif
        return;
      }
      // the target task is remote. Pack the information and send it to
      // the corresponding peer.
      // TODO do we need to copy value?
      using msg_t = detail::msg_t;
      auto &world_impl = world.impl();
      uint64_t pos = 0;
      int num_iovecs = 0;
      std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                           msg_header_t::MSG_SET_ARG, i, world_impl.rank(), 1);

      if constexpr (!ttg::meta::is_void_v<decvalueT>) {

        detail::ttg_data_copy_t *copy = copy_in;
        /* make sure we have a data copy to register with */
        if (nullptr == copy) {
          copy = detail::find_copy_in_task(detail::parsec_ttg_caller, value_ptr);
          if (nullptr == copy) {
            // We need to create a copy for this data, as it does not exist yet.
            copy = detail::create_new_datacopy(std::forward<Value>(value));
            // use the new value from here on out
            value_ptr = static_cast<norefvalueT*>(copy->get_ptr());
          }
        }

        bool inline_data = can_inline_data(value_ptr, copy, key, 1);
        msg->tt_id.inline_data = inline_data;

        auto handle_iovec_fn = [&](auto&& iovecs){

          if (inline_data) {
            /* inline data is packed right after the tt_id in the message */
            for (auto &&iov : iovecs) {
              std::memcpy(msg->bytes + pos, iov.data, iov.num_bytes);
              pos += iov.num_bytes;
            }
          } else {

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
              copy = detail::register_data_copy<decvalueT>(copy, nullptr, true);
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
              std::memcpy(msg->bytes + pos, lreg, lreg_size);
              pos += lreg_size;
              //std::cout << "set_arg_impl lreg " << lreg << std::endl;
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
        };

        if constexpr (ttg::has_split_metadata<std::decay_t<Value>>::value) {
          ttg::SplitMetadataDescriptor<decvalueT> descr;
          auto iovs = descr.get_data(*const_cast<decvalueT *>(value_ptr));
          num_iovecs = std::distance(std::begin(iovs), std::end(iovs));
          /* pack the metadata */
          auto metadata = descr.get_metadata(*const_cast<decvalueT *>(value_ptr));
          size_t metadata_size = sizeof(metadata);
          pos = pack(metadata, msg->bytes, pos);
          //std::cout << "set_arg_impl splitmd num_iovecs " << num_iovecs << std::endl;
          handle_iovec_fn(iovs);
        } else if constexpr (!ttg::has_split_metadata<std::decay_t<Value>>::value) {
          /* serialize the object */
          //std::cout << "PRE pack num_iovecs " << std::distance(copy->iovec_begin(), copy->iovec_end()) << std::endl;
          pos = pack(*value_ptr, msg->bytes, pos, copy);
          num_iovecs = std::distance(copy->iovec_begin(), copy->iovec_end());
          //std::cout << "POST pack num_iovecs " << num_iovecs << std::endl;
          /* handle any iovecs contained in it */
          handle_iovec_fn(copy->iovec_span());
          copy->iovec_reset();
        }

        msg->tt_id.num_iovecs = num_iovecs;
      }

      /* pack the key */
      msg->tt_id.num_keys = 0;
      msg->tt_id.key_offset = pos;
      if constexpr (!ttg::meta::is_void_v<Key>) {
        size_t tmppos = pack(key, msg->bytes, pos);
        pos = tmppos;
        msg->tt_id.num_keys = 1;
      }

      parsec_taskpool_t *tp = world_impl.taskpool();
      tp->tdm.module->outgoing_message_start(tp, owner, NULL);
      tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
      //std::cout << "set_arg_impl send_am owner " << owner << " sender " << msg->tt_id.sender << std::endl;
      parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                        sizeof(msg_header_t) + pos);
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      if(world.impl().profiling()) {
        parsec_profiling_ts_trace(world.impl().parsec_ttg_profile_backend_set_arg_end, 0, 0, NULL);
      }
#endif
#if defined(PARSEC_PROF_GRAPHER)
      if(NULL != detail::parsec_ttg_caller && !detail::parsec_ttg_caller->is_dummy()) {
        int orig_index = detail::find_index_of_copy_in_task(detail::parsec_ttg_caller, value_ptr);
        char orig_str[32];
        char dest_str[32];
        if(orig_index >= 0) {
          snprintf(orig_str, 32, "%d", orig_index);
        } else {
          strncpy(orig_str, "_", 32);
        }
        snprintf(dest_str, 32, "%lu", i);
        parsec_flow_t orig{ .name = orig_str, .sym_type = PARSEC_SYM_INOUT, .flow_flags = PARSEC_FLOW_ACCESS_RW,
                            .flow_index = 0, .flow_datatype_mask = ~0 };
        parsec_flow_t dest{ .name = dest_str, .sym_type = PARSEC_SYM_INOUT, .flow_flags = PARSEC_FLOW_ACCESS_RW,
                            .flow_index = 0, .flow_datatype_mask = ~0 };
        task_t *task = create_new_task(key);
        parsec_prof_grapher_dep(&detail::parsec_ttg_caller->parsec_task, &task->parsec_task, 0, &orig, &dest);
        delete task;
      }
#endif
    }

    template <int i, typename Iterator, typename Value>
    void broadcast_arg_local(Iterator &&begin, Iterator &&end, const Value &value) {
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      if(world.impl().profiling()) {
        parsec_profiling_ts_trace(world.impl().parsec_ttg_profile_backend_bcast_arg_start, 0, 0, NULL);
      }
#endif
      parsec_task_t *task_ring = nullptr;
      detail::ttg_data_copy_t *copy = nullptr;
      if (nullptr != detail::parsec_ttg_caller) {
        copy = detail::find_copy_in_task(detail::parsec_ttg_caller, &value);
      }

      for (auto it = begin; it != end; ++it) {
        set_arg_local_impl<i>(*it, value, copy, &task_ring);
      }
      /* submit all ready tasks at once */
      if (nullptr != task_ring) {
        parsec_task_t *vp_task_ring[1] = { task_ring };
        __parsec_schedule_vp(world.impl().execution_stream(), vp_task_ring, 0);
      }
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      if(world.impl().profiling()) {
        parsec_profiling_ts_trace(world.impl().parsec_ttg_profile_backend_set_arg_end, 0, 0, NULL);
      }
#endif
    }

    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>,
                     void>
    broadcast_arg(const ttg::span<const Key> &keylist, const Value &value) {
      using valueT = std::tuple_element_t<i, input_values_full_tuple_type>;
      auto world = ttg_default_execution_context();
      auto np = world.size();
      int rank = world.rank();
      uint64_t pos = 0;
      bool have_remote = keylist.end() != std::find_if(keylist.begin(), keylist.end(),
                                                       [&](const Key &key) { return keymap(key) != rank; });

      if (have_remote) {
        using decvalueT = std::decay_t<Value>;

        /* sort the input key list by owner and check whether there are remote keys */
        std::vector<Key> keylist_sorted(keylist.begin(), keylist.end());
        std::sort(keylist_sorted.begin(), keylist_sorted.end(), [&](const Key &a, const Key &b) mutable {
          int rank_a = keymap(a);
          int rank_b = keymap(b);
          // sort so that the keys for my rank are first, rank+1 next, ..., wrapping around to 0
          int pos_a = (rank_a + np - rank) % np;
          int pos_b = (rank_b + np - rank) % np;
          return pos_a < pos_b;
        });

        /* Assuming there are no local keys, will be updated while iterating over the keys */
        auto local_begin = keylist_sorted.end();
        auto local_end = keylist_sorted.end();

        int32_t num_iovs = 0;

        detail::ttg_data_copy_t *copy;
        copy = detail::find_copy_in_task(detail::parsec_ttg_caller, &value);
        assert(nullptr != copy);

        using msg_t = detail::msg_t;
        auto &world_impl = world.impl();
        std::unique_ptr<msg_t> msg = std::make_unique<msg_t>(get_instance_id(), world_impl.taskpool()->taskpool_id,
                                                             msg_header_t::MSG_SET_ARG, i, world_impl.rank());

        /* check if we inline the data */
        /* TODO: this assumes the worst case: that all keys are packed at once (i.e., go to the same remote). Can we do better?*/
        bool inline_data = can_inline_data(&value, copy, keylist_sorted[0], keylist_sorted.size());
        msg->tt_id.inline_data = inline_data;

        std::vector<std::pair<int32_t, std::shared_ptr<void>>> memregs;
        auto handle_iovs_fn = [&](auto&& iovs){

          if (inline_data) {
            /* inline data is packed right after the tt_id in the message */
            for (auto &&iov : iovs) {
              std::memcpy(msg->bytes + pos, iov.data, iov.num_bytes);
              pos += iov.num_bytes;
            }
          } else {

            /* TODO: at the moment, the tag argument to parsec_ce.get() is treated as a
              * raw function pointer instead of a preregistered AM tag, so play that game.
              * Once this is fixed in PaRSEC we need to use parsec_ttg_rma_tag instead! */
            parsec_ce_tag_t cbtag = reinterpret_cast<parsec_ce_tag_t>(&detail::get_remote_complete_cb);
            std::memcpy(msg->bytes + pos, &cbtag, sizeof(cbtag));
            pos += sizeof(cbtag);

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
                                                                      //std::cout << "broadcast_arg memunreg lreg " << memreg << std::endl;
                                                                      parsec_ce.mem_unregister(&memreg);
                                                                    }}));
              //std::cout << "broadcast_arg memreg lreg " << lreg << std::endl;
            }
          }
        };

        if constexpr (ttg::has_split_metadata<std::decay_t<Value>>::value) {
          ttg::SplitMetadataDescriptor<decvalueT> descr;
          /* pack the metadata */
          auto metadata = descr.get_metadata(value);
          size_t metadata_size = sizeof(metadata);
          pos = pack(metadata, msg->bytes, pos);
          auto iovs = descr.get_data(*const_cast<decvalueT *>(&value));
          num_iovs = std::distance(std::begin(iovs), std::end(iovs));
          memregs.reserve(num_iovs);
          handle_iovs_fn(iovs);
          //std::cout << "broadcast_arg splitmd num_iovecs " << num_iovs << std::endl;
        } else if constexpr (!ttg::has_split_metadata<std::decay_t<Value>>::value) {
          /* serialize the object once */
          pos = pack(value, msg->bytes, pos, copy);
          num_iovs = std::distance(copy->iovec_begin(), copy->iovec_end());
          handle_iovs_fn(copy->iovec_span());
          copy->iovec_reset();
        }

        msg->tt_id.num_iovecs = num_iovs;

        std::size_t save_pos = pos;

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

          /* rewind the buffer and start packing a new set of memregs and keys */
          pos = save_pos;
          /**
           * pack the registration handles
           * memory layout: [<lreg_size, lreg, lreg_fn>, ...]
           * NOTE: we need to pack these for every receiver to ensure correct ref-counting of the registration
           */
          if (!inline_data) {
            for (int idx = 0; idx < num_iovs; ++idx) {
              // auto [lreg_size, lreg_ptr] = memregs[idx];
              int32_t lreg_size;
              std::shared_ptr<void> lreg_ptr;
              std::tie(lreg_size, lreg_ptr) = memregs[idx];
              std::memcpy(msg->bytes + pos, &lreg_size, sizeof(lreg_size));
              pos += sizeof(lreg_size);
              std::memcpy(msg->bytes + pos, lreg_ptr.get(), lreg_size);
              pos += lreg_size;
              //std::cout << "broadcast_arg lreg_ptr " << lreg_ptr.get() << std::endl;
              /* mark another reader on the copy */
              copy = detail::register_data_copy<valueT>(copy, nullptr, true);
              /* create a function that will be invoked upon RMA completion at the target */
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

          /* mark the beginning of the keys */
          msg->tt_id.key_offset = pos;

          /* pack all keys for this owner */
          int num_keys = 0;
          do {
            ++num_keys;
            pos = pack(*it, msg->bytes, pos);
            ++it;
          } while (it < keylist_sorted.end() && keymap(*it) == owner);
          msg->tt_id.num_keys = num_keys;

          tp->tdm.module->outgoing_message_start(tp, owner, NULL);
          tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
          //std::cout << "broadcast_arg send_am owner " << owner << std::endl;
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
    /// sets the default stream size for input \c i
    /// \param size positive integer that specifies the default stream size
    template <std::size_t i>
    void set_static_argstream_size(std::size_t size) {
      assert(std::get<i>(input_reducers) && "TT::set_static_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "TT::set_static_argstream_size(key,size) called with size=0");

      this->trace(world.rank(), ":", get_name(), ": setting global stream size for terminal ", i);

      // Check if stream is already bounded
      if (static_stream_goal[i] < std::numeric_limits<std::size_t>::max()) {
        ttg::print_error(world.rank(), ":", get_name(), " : error stream is already bounded : ", i);
        throw std::runtime_error("TT::set_static_argstream_size called for a bounded stream");
      }

      static_stream_goal[i] = size;
    }

    /// sets stream size for input \c i
    /// \param size positive integer that specifies the stream size
    /// \param key the task identifier that expects this number of inputs in the streaming terminal
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
                                                             msg_header_t::MSG_SET_ARGSTREAM_SIZE, i,
                                                             world_impl.rank(), 1);
        /* pack the key */
        pos = pack(key, msg->bytes, pos);
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
          if( world.impl().dag_profiling() ) {
#if defined(PARSEC_PROF_GRAPHER)
            parsec_prof_grapher_task(&task->parsec_task, world.impl().execution_stream()->th_id, 0, *(uintptr_t*)&(task->parsec_task.locals[0]));
#endif
          }
        }
        parsec_hash_table_unlock_bucket(&tasks_table, hk);

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        // 1) "lock" the stream by incrementing the reduce_count
        // 2) set the goal
        // 3) "unlock" the stream
        // only one thread will see the reduce_count be zero and the goal match the size
        task->streams[i].reduce_count.fetch_add(1, std::memory_order_acquire);
        task->streams[i].goal = size;
        auto c = task->streams[i].reduce_count.fetch_sub(1, std::memory_order_release);
        if (1 == c && (task->streams[i].size >= size)) {
          release_task(task);
        }
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
                                                             msg_header_t::MSG_SET_ARGSTREAM_SIZE, i,
                                                             world_impl.rank(), 0);
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
          if( world.impl().dag_profiling() ) {
#if defined(PARSEC_PROF_GRAPHER)
            parsec_prof_grapher_task(&task->parsec_task, world.impl().execution_stream()->th_id, 0, *(uintptr_t*)&(task->parsec_task.locals[0]));
#endif
          }
        }
        parsec_hash_table_unlock_bucket(&tasks_table, hk);

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        // 1) "lock" the stream by incrementing the reduce_count
        // 2) set the goal
        // 3) "unlock" the stream
        // only one thread will see the reduce_count be zero and the goal match the size
        task->streams[i].reduce_count.fetch_add(1, std::memory_order_acquire);
        task->streams[i].goal = size;
        auto c = task->streams[i].reduce_count.fetch_sub(1, std::memory_order_release);
        if (1 == c && (task->streams[i].size >= size)) {
          release_task(task);
        }
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
                                                             msg_header_t::MSG_FINALIZE_ARGSTREAM_SIZE, i,
                                                             world_impl.rank(), 1);
        /* pack the key */
        pos = pack(key, msg->bytes, pos);
        parsec_taskpool_t *tp = world_impl.taskpool();
        tp->tdm.module->outgoing_message_start(tp, owner, NULL);
        tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
        parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                          sizeof(msg_header_t) + pos);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": finalizing stream for terminal ", i);

        auto hk = reinterpret_cast<parsec_key_t>(&key);
        task_t *task = nullptr;
        //parsec_hash_table_lock_bucket(&tasks_table, hk);
        if (nullptr == (task = (task_t *)parsec_hash_table_find(&tasks_table, hk))) {
          ttg::print_error(world.rank(), ":", get_name(), ":", key,
                           " : error finalize called on stream that never received an input data: ", i);
          throw std::runtime_error("TT::finalize called on stream that never received an input data");
        }

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        // 1) "lock" the stream by incrementing the reduce_count
        // 2) set the goal
        // 3) "unlock" the stream
        // only one thread will see the reduce_count be zero and the goal match the size
        task->streams[i].reduce_count.fetch_add(1, std::memory_order_acquire);
        task->streams[i].goal = 1;
        auto c = task->streams[i].reduce_count.fetch_sub(1, std::memory_order_release);
        if (1 == c && (task->streams[i].size >= 1)) {
          release_task(task);
        }
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
                                                             msg_header_t::MSG_FINALIZE_ARGSTREAM_SIZE, i,
                                                             world_impl.rank(), 0);
        parsec_taskpool_t *tp = world_impl.taskpool();
        tp->tdm.module->outgoing_message_start(tp, owner, NULL);
        tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
        parsec_ce.send_am(&parsec_ce, world_impl.parsec_ttg_tag(), owner, static_cast<void *>(msg.get()),
                          sizeof(msg_header_t) + pos);
      } else {
        ttg::trace(world.rank(), ":", get_name(), ": finalizing stream for terminal ", i);

        auto hk = static_cast<parsec_key_t>(0);
        task_t *task = nullptr;
        if (nullptr == (task = (task_t *)parsec_hash_table_find(&tasks_table, hk))) {
          ttg::print_error(world.rank(), ":", get_name(),
                           " : error finalize called on stream that never received an input data: ", i);
          throw std::runtime_error("TT::finalize called on stream that never received an input data");
        }

        // TODO: Unfriendly implementation, cannot check if stream is already bounded
        // TODO: Unfriendly implementation, cannot check if stream has been finalized already

        // commit changes
        // 1) "lock" the stream by incrementing the reduce_count
        // 2) set the goal
        // 3) "unlock" the stream
        // only one thread will see the reduce_count be zero and the goal match the size
        task->streams[i].reduce_count.fetch_add(1, std::memory_order_acquire);
        task->streams[i].goal = 1;
        auto c = task->streams[i].reduce_count.fetch_sub(1, std::memory_order_release);
        if (1 == c && (task->streams[i].size >= 1)) {
          release_task(task);
        }
      }
    }

    void copy_mark_pushout(detail::ttg_data_copy_t *copy) {

      assert(detail::parsec_ttg_caller->dev_ptr && detail::parsec_ttg_caller->dev_ptr->gpu_task);
      parsec_gpu_task_t *gpu_task = detail::parsec_ttg_caller->dev_ptr->gpu_task;
      auto check_parsec_data = [&](parsec_data_t* data) {
        if (data->owner_device != 0) {
          /* find the flow */
          int flowidx = 0;
          while (flowidx < MAX_PARAM_COUNT &&
                gpu_task->flow[flowidx]->flow_flags != PARSEC_FLOW_ACCESS_NONE) {
            if (detail::parsec_ttg_caller->parsec_task.data[flowidx].data_in->original == data) {
              /* found the right data, set the corresponding flow as pushout */
              break;
            }
            ++flowidx;
          }
          if (flowidx == MAX_PARAM_COUNT) {
            throw std::runtime_error("Cannot add more than MAX_PARAM_COUNT flows to a task!");
          }
          if (gpu_task->flow[flowidx]->flow_flags == PARSEC_FLOW_ACCESS_NONE) {
            /* no flow found, add one and mark it pushout */
            detail::parsec_ttg_caller->parsec_task.data[flowidx].data_in = data->device_copies[0];
            gpu_task->flow_nb_elts[flowidx] = data->nb_elts;
          }
          /* need to mark the flow RW to make PaRSEC happy */
          ((parsec_flow_t *)gpu_task->flow[flowidx])->flow_flags |= PARSEC_FLOW_ACCESS_RW;
          gpu_task->pushout |= 1<<flowidx;
        }
      };
      copy->foreach_parsec_data(check_parsec_data);
    }


    /* check whether a data needs to be pushed out */
    template <std::size_t i, typename Value, typename RemoteCheckFn>
    std::enable_if_t<!std::is_void_v<std::decay_t<Value>>,
                     void>
    do_prepare_send(const Value &value, RemoteCheckFn&& remote_check) {
      using valueT = std::tuple_element_t<i, input_values_full_tuple_type>;
      static constexpr const bool value_is_const = std::is_const_v<valueT>;

      /* get the copy */
      detail::ttg_data_copy_t *copy;
      copy = detail::find_copy_in_task(detail::parsec_ttg_caller, &value);

      /* if there is no copy we don't need to prepare anything */
      if (nullptr == copy) {
        return;
      }

      detail::parsec_ttg_task_base_t *caller = detail::parsec_ttg_caller;
      bool need_pushout = false;

      if (caller->data_flags & detail::ttg_parsec_data_flags::MARKED_PUSHOUT) {
        /* already marked pushout, skip the rest */
        return;
      }

      /* TODO: remove this once we support reductions on the GPU */
      auto &reducer = std::get<i>(input_reducers);
      if (reducer) {
        /* reductions are currently done only on the host so push out */
        copy_mark_pushout(copy);
        caller->data_flags |= detail::ttg_parsec_data_flags::MARKED_PUSHOUT;
        return;
      }

      if constexpr (value_is_const) {
        if (caller->data_flags & detail::ttg_parsec_data_flags::IS_MODIFIED) {
          /* The data has been modified previously. PaRSEC requires us to pushout
           * data if we transition from a writer to one or more readers. */
          need_pushout = true;
        }

        /* check for multiple readers */
        if (caller->data_flags & detail::ttg_parsec_data_flags::SINGLE_READER) {
          caller->data_flags |= detail::ttg_parsec_data_flags::MULTIPLE_READER;
        }

        if (caller->data_flags & detail::ttg_parsec_data_flags::SINGLE_WRITER) {
          /* there is a writer already, we will need to create a copy */
          need_pushout = true;
        }

        caller->data_flags |= detail::ttg_parsec_data_flags::SINGLE_READER;
      } else {
        if (caller->data_flags & detail::ttg_parsec_data_flags::SINGLE_WRITER) {
          caller->data_flags |= detail::ttg_parsec_data_flags::MULTIPLE_WRITER;
          need_pushout = true;
        } else {
          if (caller->data_flags & detail::ttg_parsec_data_flags::SINGLE_READER) {
            /* there are readers, we will need to create a copy */
            need_pushout = true;
          }
          caller->data_flags |= detail::ttg_parsec_data_flags::SINGLE_WRITER;
        }
      }

      if constexpr (!derived_has_device_op()) {
        need_pushout = true;
      }

      /* check if there are non-local successors if it's a device task */
      if (!need_pushout) {
        bool device_supported = false;
        if constexpr (derived_has_cuda_op()) {
          device_supported = !world.impl().mpi_support(ttg::ExecutionSpace::CUDA);
        } else if constexpr (derived_has_hip_op()) {
          device_supported = !world.impl().mpi_support(ttg::ExecutionSpace::HIP);
        } else if constexpr (derived_has_level_zero_op()) {
          device_supported = !world.impl().mpi_support(ttg::ExecutionSpace::L0);
        }
        /* if MPI supports the device we don't care whether we have remote peers
         * because we can send from the device directly */
        if (!device_supported) {
          need_pushout = remote_check();
        }
      }

      if (need_pushout) {
        copy_mark_pushout(copy);
        caller->data_flags |= detail::ttg_parsec_data_flags::MARKED_PUSHOUT;
      }
    }

    /* check whether a data needs to be pushed out */
    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>,
                     void>
    prepare_send(const ttg::span<const Key> &keylist, const Value &value) {
      auto remote_check = [&](){
          auto world = ttg_default_execution_context();
          int rank = world.rank();
          uint64_t pos = 0;
          bool remote = keylist.end() != std::find_if(keylist.begin(), keylist.end(),
                                                      [&](const Key &key) { return keymap(key) != rank; });
          return remote;
        };
      do_prepare_send<i>(value, remote_check);
    }

    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>,
                     void>
    prepare_send(const Value &value) {
      auto remote_check = [&](){
          auto world = ttg_default_execution_context();
          int rank = world.rank();
          return (keymap() != rank);
        };
      do_prepare_send<i>(value, remote_check);
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
      if (input.is_pull_terminal) {
        num_pullins++;
      }
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
            broadcast_arg<i, keyT, valueT>(keylist, value);
        };
        auto prepare_send_callback = [this](const ttg::span<const keyT> &keylist, const valueT &value) {
            prepare_send<i, keyT, valueT>(keylist, value);
        };
        auto setsize_callback = [this](const keyT &key, std::size_t size) { set_argstream_size<i>(key, size); };
        auto finalize_callback = [this](const keyT &key) { finalize_argstream<i>(key); };
        input.set_callback(send_callback, move_callback, broadcast_callback,
                           setsize_callback, finalize_callback, prepare_send_callback);
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
        auto prepare_send_callback = [this](const valueT &value) {
            prepare_send<i, void>(value);
        };
        input.set_callback(send_callback, move_callback, {}, setsize_callback, finalize_callback, prepare_send_callback);
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
        ttg::abort();
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

#if 0
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
#endif // 0

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
        uint64_t hv = hash<std::decay_t<decltype(kk)>>{}(kk);
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

    static parsec_key_t make_key(const parsec_taskpool_t *tp, const parsec_assignment_t *as) {
        // we use the parsec_assignment_t array as a scratchpad to store the hash and address of the key
        keyT *key = *(keyT**)&(as[2]);
        return reinterpret_cast<parsec_key_t>(key);
    }

    static char *parsec_ttg_task_snprintf(char *buffer, size_t buffer_size, const parsec_task_t *parsec_task) {
      if(buffer_size == 0)
        return buffer;

      if constexpr (ttg::meta::is_void_v<keyT>) {
        snprintf(buffer, buffer_size, "%s()[]<%d>", parsec_task->task_class->name, parsec_task->priority);
      }  else {
        const task_t *task = reinterpret_cast<const task_t*>(parsec_task);
        std::stringstream ss;
        ss << task->key;

        std::string keystr = ss.str();
        std::replace(keystr.begin(), keystr.end(), '(', ':');
        std::replace(keystr.begin(), keystr.end(), ')', ':');

        snprintf(buffer, buffer_size, "%s(%s)[]<%d>", parsec_task->task_class->name, keystr.c_str(), parsec_task->priority);
      }
      return buffer;
    }

#if defined(PARSEC_PROF_TRACE)
    static void *parsec_ttg_task_info(void *dst, const void *data, size_t size)
    {
      const task_t *task = reinterpret_cast<const task_t *>(data);

      if constexpr (ttg::meta::is_void_v<keyT>) {
        snprintf(reinterpret_cast<char*>(dst), size, "()");
      } else {
        std::stringstream ss;
        ss << task->key;
        snprintf(reinterpret_cast<char*>(dst), size, "%s", ss.str().c_str());
      }
      return dst;
    }
#endif

    parsec_key_fn_t tasks_hash_fcts = {key_equal, key_print, key_hash};

    template<std::size_t I>
    inline static void increment_data_version_impl(task_t *task) {
      if constexpr (!std::is_const_v<std::tuple_element_t<I, typename TT::input_values_tuple_type>>) {
        if (task->copies[I] != nullptr){
          task->copies[I]->inc_current_version();
        }
      }
    }

    template<std::size_t... Is>
    inline static void increment_data_versions(task_t *task, std::index_sequence<Is...>) {
      /* increment version of each mutable data */
      int junk[] = {0, (increment_data_version_impl<Is>(task), 0)...};
      junk[0]++;
    }

    static parsec_hook_return_t complete_task_and_release(parsec_execution_stream_t *es, parsec_task_t *parsec_task) {

      //std::cout << "complete_task_and_release: task " << parsec_task << std::endl;

      task_t *task = (task_t*)parsec_task;

      /* if we still have a coroutine handle we invoke it one more time to get the sends/broadcasts */
      if (task->suspended_task_address) {
#ifdef TTG_HAVE_DEVICE
        /* increment versions of all data we might have modified
         * this must happen before we issue the sends */
        //increment_data_versions(task, std::make_index_sequence<std::tuple_size_v<typename TT::input_values_tuple_type>>{});

        // get the device task from the coroutine handle
        auto dev_task = ttg::device::detail::device_task_handle_type::from_address(task->suspended_task_address);

        // get the promise which contains the views
        auto dev_data = dev_task.promise();

        /* for now make sure we're waiting for the kernel to complete and the coro hasn't skipped this step */
        assert(dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_SENDOUT);

        /* execute the sends we stored */
        if (dev_data.state() == ttg::device::detail::TTG_DEVICE_CORO_SENDOUT) {
          /* set the current task, needed inside the sends */
          detail::parsec_ttg_caller = task;
          dev_data.do_sends();
          detail::parsec_ttg_caller = nullptr;
        }
#else // TTG_HAVE_DEVICE
        ttg::abort();  // should not happen
#endif // TTG_HAVE_DEVICE
        /* the coroutine should have completed and we cannot access the promise anymore */
        task->suspended_task_address = nullptr;
      }

      /* release our data copies */
      for (int i = 0; i < task->data_count; i++) {
        detail::ttg_data_copy_t *copy = task->copies[i];
        if (nullptr == copy) continue;
        detail::release_data_copy(copy);
        task->copies[i] = nullptr;
      }
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
        , priomap(decltype(keymap)(std::forward<priomapT>(priomap_))) {
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

      self.name = strdup(get_name().c_str());
      self.task_class_id = get_instance_id();
      self.nb_parameters = 0;
      self.nb_locals = 0;
      //self.nb_flows = numflows;
      self.nb_flows = MAX_PARAM_COUNT; // we're not using all flows but have to
                                       // trick the device handler into looking at all of them

      if( world_impl.profiling() ) {
        // first two ints are used to store the hash of the key.
        self.nb_parameters = (sizeof(void*)+sizeof(int)-1)/sizeof(int);
        // seconds two ints are used to store a pointer to the key of the task.
        self.nb_locals     = self.nb_parameters + (sizeof(void*)+sizeof(int)-1)/sizeof(int);

        // If we have parameters and locals, we need to define the corresponding dereference arrays
        self.params[0] = &detail::parsec_taskclass_param0;
        self.params[1] = &detail::parsec_taskclass_param1;

        self.locals[0] = &detail::parsec_taskclass_param0;
        self.locals[1] = &detail::parsec_taskclass_param1;
        self.locals[2] = &detail::parsec_taskclass_param2;
        self.locals[3] = &detail::parsec_taskclass_param3;
      }
      self.make_key = make_key;
      self.key_functions = &tasks_hash_fcts;
      self.task_snprintf = parsec_ttg_task_snprintf;

#if defined(PARSEC_PROF_TRACE)
      self.profile_info = &parsec_ttg_task_info;
#endif

      world_impl.taskpool()->nb_task_classes = std::max(world_impl.taskpool()->nb_task_classes, static_cast<decltype(world_impl.taskpool()->nb_task_classes)>(self.task_class_id+1));
      //    function_id_to_instance[self.task_class_id] = this;
      //self.incarnations = incarnations_array.data();
//#if 0
      if constexpr (derived_has_cuda_op()) {
        self.incarnations = (__parsec_chore_t *)malloc(3 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t *)self.incarnations)[0].type = PARSEC_DEV_CUDA;
        ((__parsec_chore_t *)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[0].hook = &detail::hook_cuda<TT>;
        ((__parsec_chore_t *)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t *)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[1].hook = NULL;
      } else if (derived_has_hip_op()) {
        self.incarnations = (__parsec_chore_t *)malloc(3 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t *)self.incarnations)[0].type = PARSEC_DEV_HIP;
        ((__parsec_chore_t *)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[0].hook = &detail::hook_hip<TT>;

        ((__parsec_chore_t *)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t *)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[1].hook = NULL;
      } else if (derived_has_level_zero_op()) {
        self.incarnations = (__parsec_chore_t *)malloc(3 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t *)self.incarnations)[0].type = PARSEC_DEV_LEVEL_ZERO;
        ((__parsec_chore_t *)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[0].hook = &detail::hook_level_zero<TT>;

        ((__parsec_chore_t *)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t *)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[1].hook = NULL;
      } else {
        self.incarnations = (__parsec_chore_t *)malloc(2 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t *)self.incarnations)[0].type = PARSEC_DEV_CPU;
        ((__parsec_chore_t *)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[0].hook = &detail::hook<TT>;
        ((__parsec_chore_t *)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t *)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[1].hook = NULL;
      }
//#endif // 0

      self.release_task = &parsec_release_task_to_mempool_update_nbtasks;
      self.complete_execution = complete_task_and_release;

      for (i = 0; i < MAX_PARAM_COUNT; i++) {
        parsec_flow_t *flow = new parsec_flow_t;
        flow->name = strdup((std::string("flow in") + std::to_string(i)).c_str());
        flow->sym_type = PARSEC_SYM_INOUT;
        // see initialize_flows below
        // flow->flow_flags = PARSEC_FLOW_ACCESS_RW;
        flow->dep_in[0] = NULL;
        flow->dep_out[0] = NULL;
        flow->flow_index = i;
        flow->flow_datatype_mask = ~0;
        *((parsec_flow_t **)&(self.in[i])) = flow;
      }
      //*((parsec_flow_t **)&(self.in[i])) = NULL;
      //initialize_flows<input_terminals_type>(self.in);

      for (i = 0; i < MAX_PARAM_COUNT; i++) {
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
      //*((parsec_flow_t **)&(self.out[i])) = NULL;

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
      //DO NOT MOVE THIS - information about the number of pull terminals is only available after connecting the edges.
      if constexpr (numinedges > 0) {
        register_input_callbacks(std::make_index_sequence<numinedges>{});
      }
    }
    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
       const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       keymapT &&keymap = keymapT(ttg::default_execution_context()), priomapT &&priomap = priomapT())
        : TT(inedges, outedges, name, innames, outnames, ttg::default_execution_context(),
             std::forward<keymapT>(keymap), std::forward<priomapT>(priomap)) {}

    // Destructor checks for unexecuted tasks
    virtual ~TT() {
      if(nullptr != self.name ) {
        free((void*)self.name);
        self.name = nullptr;
      }

      for (std::size_t i = 0; i < numins; ++i) {
        if (inpute_reducers_taskclass[i] != nullptr) {
          std::free(inpute_reducers_taskclass[i]);
          inpute_reducers_taskclass[i] = nullptr;
        }
      }
      release();
    }

    static void ht_iter_cb(void *item, void *cb_data) {
      task_t *task = (task_t *)item;
      ttT *op = (ttT *)cb_data;
      if constexpr (!ttg::meta::is_void_v<keyT>) {
        std::cout << "Left over task " << op->get_name() << " " << task->key << std::endl;
      } else {
        std::cout << "Left over task " << op->get_name() << std::endl;
      }
    }

    void print_incomplete_tasks() {
      parsec_hash_table_for_all(&tasks_table, ht_iter_cb, this);
    }

    virtual void release() override { do_release(); }

    void do_release() {
      if (!alive) {
        return;
      }
      alive = false;
      /* print all outstanding tasks */
      print_incomplete_tasks();
      parsec_hash_table_fini(&tasks_table);
      parsec_mempool_destruct(&mempools);
      // uintptr_t addr = (uintptr_t)self.incarnations;
      // free((void *)addr);
      free((__parsec_chore_t *)self.incarnations);
      for (int i = 0; i < MAX_PARAM_COUNT; i++) {
        if (NULL != self.in[i]) {
          free(self.in[i]->name);
          delete self.in[i];
          self.in[i] = nullptr;
        }
        if (NULL != self.out[i]) {
          free(self.out[i]->name);
          delete self.out[i];
          self.out[i] = nullptr;
        }
      }
      world.impl().deregister_op(this);
    }

    static constexpr const ttg::Runtime runtime = ttg::Runtime::PaRSEC;

    /// define the reducer function to be called when additional inputs are
    /// received on a streaming terminal
    ///   @tparam <i> the index of the input terminal that is used as a streaming terminal
    ///   @param[in] reducer: a function of prototype (input_type<i> &a, const input_type<i> &b)
    ///                       that function should aggregate b into a
    template <std::size_t i, typename Reducer>
    void set_input_reducer(Reducer &&reducer) {
      ttg::trace(world.rank(), ":", get_name(), " : setting reducer for terminal ", i);
      std::get<i>(input_reducers) = reducer;

      parsec_task_class_t *tc = inpute_reducers_taskclass[i];
      if (nullptr == tc) {
        tc = (parsec_task_class_t *)std::calloc(1, sizeof(*tc));
        inpute_reducers_taskclass[i] = tc;

        tc->name = strdup((get_name() + std::string(" reducer ") + std::to_string(i)).c_str());
        tc->task_class_id = get_instance_id();
        tc->nb_parameters = 0;
        tc->nb_locals = 0;
        tc->nb_flows = numflows;

        auto &world_impl = world.impl();

        if( world_impl.profiling() ) {
          // first two ints are used to store the hash of the key.
          tc->nb_parameters = (sizeof(void*)+sizeof(int)-1)/sizeof(int);
          // seconds two ints are used to store a pointer to the key of the task.
          tc->nb_locals     = self.nb_parameters + (sizeof(void*)+sizeof(int)-1)/sizeof(int);

          // If we have parameters and locals, we need to define the corresponding dereference arrays
          tc->params[0] = &detail::parsec_taskclass_param0;
          tc->params[1] = &detail::parsec_taskclass_param1;

          tc->locals[0] = &detail::parsec_taskclass_param0;
          tc->locals[1] = &detail::parsec_taskclass_param1;
          tc->locals[2] = &detail::parsec_taskclass_param2;
          tc->locals[3] = &detail::parsec_taskclass_param3;
        }
        tc->make_key = make_key;
        tc->key_functions = &tasks_hash_fcts;
        tc->task_snprintf = parsec_ttg_task_snprintf;

#if defined(PARSEC_PROF_TRACE)
        tc->profile_info = &parsec_ttg_task_info;
#endif

        world_impl.taskpool()->nb_task_classes = std::max(world_impl.taskpool()->nb_task_classes, static_cast<decltype(world_impl.taskpool()->nb_task_classes)>(self.task_class_id+1));

#if 0
        // FIXME: currently only support reduction on the host
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
        } else
#endif // 0
        {
          tc->incarnations = (__parsec_chore_t *)malloc(2 * sizeof(__parsec_chore_t));
          ((__parsec_chore_t *)tc->incarnations)[0].type = PARSEC_DEV_CPU;
          ((__parsec_chore_t *)tc->incarnations)[0].evaluate = NULL;
          ((__parsec_chore_t *)tc->incarnations)[0].hook = &static_reducer_op<i>;
          ((__parsec_chore_t *)tc->incarnations)[1].type = PARSEC_DEV_NONE;
          ((__parsec_chore_t *)tc->incarnations)[1].evaluate = NULL;
          ((__parsec_chore_t *)tc->incarnations)[1].hook = NULL;
        }

        /* the reduction task does not alter the termination detection because the target task will execute */
        tc->release_task = &parsec_release_task_to_mempool;
        tc->complete_execution = NULL;
      }
    }

    /// define the reducer function to be called when additional inputs are
    /// received on a streaming terminal
    ///   @tparam <i> the index of the input terminal that is used as a streaming terminal
    ///   @param[in] reducer: a function of prototype (input_type<i> &a, const input_type<i> &b)
    ///                       that function should aggregate b into a
    ///   @param[in] size: the default number of inputs that are received in this streaming terminal,
    ///                    for each task
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
      if constexpr(!std::is_same_v<Key, key_type>) {
        key_type k = key; /* cast that type into the key type we know */
        invoke(k, args);
      } else {
        /* trigger non-void inputs */
        set_args(ttg::meta::nonvoid_index_seq<actual_input_tuple_type>{}, key, args);
        /* trigger void inputs */
        using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
        set_args(void_index_seq{}, key, ttg::detail::make_void_tuple<void_index_seq::size()>());
      }
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

      if constexpr(!std::is_same_v<Key, key_type>) {
        key_type k = key; /* cast that type into the key type we know */
        invoke(k);
      } else {
        /* trigger void inputs */
        using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
        set_args(void_index_seq{}, key, ttg::detail::make_void_tuple<void_index_seq::size()>());
      }
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

  private:
    template<typename Key, typename Arg, typename... Args, std::size_t I, std::size_t... Is>
    void invoke_arglist(std::index_sequence<I, Is...>, const Key& key, Arg&& arg, Args&&... args) {
      using arg_type = std::decay_t<Arg>;
      if constexpr (ttg::meta::is_ptr_v<arg_type>) {
        /* add a reference to the object */
        auto copy = ttg_parsec::detail::get_copy(arg);
        copy->add_ref();
        /* reset readers so that the value can flow without copying */
        copy->reset_readers();
        auto& val = *arg;
        set_arg_impl<I>(key, val, copy);
        ttg_parsec::detail::release_data_copy(copy);
        if constexpr (std::is_rvalue_reference_v<Arg>) {
          /* if the ptr was moved in we reset it */
          arg.reset();
        }
      } else if constexpr (!ttg::meta::is_ptr_v<arg_type>) {
        set_arg<I>(key, std::forward<Arg>(arg));
      }
      if constexpr (sizeof...(Is) > 0) {
        /* recursive next argument */
        invoke_arglist(std::index_sequence<Is...>{}, key, std::forward<Args>(args)...);
      }
    }

  public:
    // Manual injection of a task with all input arguments specified as variadic arguments
    template <typename Key = keyT, typename Arg, typename... Args>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const Key &key, Arg&& arg, Args&&... args) {
      static_assert(sizeof...(Args)+1 == std::tuple_size_v<actual_input_tuple_type>,
                    "Number of arguments to invoke must match the number of task inputs.");
      TTG_OP_ASSERT_EXECUTABLE();
      /* trigger non-void inputs */
      invoke_arglist(ttg::meta::nonvoid_index_seq<actual_input_tuple_type>{}, key,
                     std::forward<Arg>(arg), std::forward<Args>(args)...);
      //set_args(ttg::meta::nonvoid_index_seq<actual_input_tuple_type>{}, key, args);
      /* trigger void inputs */
      using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
      set_args(void_index_seq{}, key, ttg::detail::make_void_tuple<void_index_seq::size()>());
    }

    void set_defer_writer(bool value) {
      m_defer_writer = value;
    }

    bool get_defer_writer(bool value) {
      return m_defer_writer;
    }

   public:
    void make_executable() override {
      world.impl().register_tt_profiling(this);
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
      priomap = std::forward<Priomap>(pm);
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
          if(ttg::tracing())
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
  ttg_parsec::detail::ttg_data_copy_t *copy_to_remove = nullptr;
  bool do_release = true;

 public:
  value_copy_handler() = default;
  value_copy_handler(const value_copy_handler& h) = delete;
  value_copy_handler(value_copy_handler&& h)
  : copy_to_remove(h.copy_to_remove)
  {
    h.copy_to_remove = nullptr;
  }

  value_copy_handler& operator=(const value_copy_handler& h) = delete;
  value_copy_handler& operator=(value_copy_handler&& h)
  {
    std::swap(copy_to_remove, h.copy_to_remove);
    return *this;
  }

  ~value_copy_handler() {
    if (nullptr != copy_to_remove) {
      ttg_parsec::detail::remove_data_copy(copy_to_remove, ttg_parsec::detail::parsec_ttg_caller);
      if (do_release) {
        ttg_parsec::detail::release_data_copy(copy_to_remove);
      }
    }
  }

  template <typename Value>
  inline std::add_lvalue_reference_t<Value> operator()(Value &&value) {
    static_assert(std::is_rvalue_reference_v<decltype(value)> ||
                  std::is_copy_constructible_v<std::decay_t<Value>>,
                  "Data sent without being moved must be copy-constructible!");

    auto caller = ttg_parsec::detail::parsec_ttg_caller;
    if (nullptr == caller) {
      ttg::print("ERROR: ttg::send or ttg::broadcast called outside of a task!\n");
    }
    using value_type = std::remove_reference_t<Value>;
    ttg_parsec::detail::ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::find_copy_in_task(caller, &value);
    value_type *value_ptr = &value;
    if (nullptr == copy) {
      /**
       * the value is not known, create a copy that we can track
       * depending on Value, this uses either the copy or move constructor
       */
      copy = ttg_parsec::detail::create_new_datacopy(std::forward<Value>(value));
      bool inserted = ttg_parsec::detail::add_copy_to_task(copy, caller);
      assert(inserted);
      value_ptr = reinterpret_cast<value_type *>(copy->get_ptr());
      copy_to_remove = copy;
    } else {
      if constexpr (std::is_rvalue_reference_v<decltype(value)>) {
        /* this copy won't be modified anymore so mark it as read-only */
        copy->reset_readers();
      }
      /* the value was potentially changed, so increment version */
      copy->inc_current_version();
    }
    /* We're coming from a writer so mark the data as modified.
     * That way we can force a pushout in prepare_send if we move to read-only tasks (needed by PaRSEC). */
    caller->data_flags = ttg_parsec::detail::ttg_parsec_data_flags::IS_MODIFIED;
    return *value_ptr;
  }

  template<typename Value>
  inline std::add_lvalue_reference_t<Value> operator()(ttg_parsec::detail::persistent_value_ref<Value> vref) {
    auto caller = ttg_parsec::detail::parsec_ttg_caller;
    if (nullptr == caller) {
      ttg::print("ERROR: ttg::send or ttg::broadcast called outside of a task!\n");
    }
    ttg_parsec::detail::ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::find_copy_in_task(caller, &vref.value_ref);
    if (nullptr == copy) {
      // no need to create a new copy since it's derived from the copy already
      copy = const_cast<ttg_parsec::detail::ttg_data_copy_t *>(static_cast<const ttg_parsec::detail::ttg_data_copy_t *>(&vref.value_ref));
      bool inserted = ttg_parsec::detail::add_copy_to_task(copy, caller);
      assert(inserted);
      copy_to_remove = copy; // we want to remove the copy from the task once done sending
      do_release = false; // we don't release the copy since we didn't allocate it
      copy->add_ref(); // add a reference so that TTG does not attempt to delete this object
    }
    return vref.value_ref;
  }

  template <typename Value>
  inline const Value &operator()(const Value &value) {
    static_assert(std::is_copy_constructible_v<std::decay_t<Value>>,
                  "Data sent without being moved must be copy-constructible!");
    auto caller = ttg_parsec::detail::parsec_ttg_caller;
    if (nullptr == caller) {
      ttg::print("ERROR: ttg::send or ttg::broadcast called outside of a task!\n");
    }
    ttg_parsec::detail::ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::find_copy_in_task(caller, &value);
    const Value *value_ptr = &value;
    if (nullptr == copy) {
      /**
       * the value is not known, create a copy that we can track
       * depending on Value, this uses either the copy or move constructor
       */
      copy = ttg_parsec::detail::create_new_datacopy(value);
      bool inserted = ttg_parsec::detail::add_copy_to_task(copy, caller);
      assert(inserted);
      value_ptr = reinterpret_cast<Value *>(copy->get_ptr());
      copy_to_remove = copy;
    }
    caller->data_flags = ttg_parsec::detail::ttg_parsec_data_flags::NONE;
    return *value_ptr;
  }

};

#endif  // PARSEC_TTG_H_INCLUDED
// clang-format on
