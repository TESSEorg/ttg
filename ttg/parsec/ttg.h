#ifndef PARSEC_TTG_H_INCLUDED
#define PARSEC_TTG_H_INCLUDED

#include "../ttg.h"
#include "../util/meta.h"
#include "../util/serialization.h"
#include "../ttg/util/hash.h"

#include <array>
#include <cassert>
#include <experimental/type_traits>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include <list>

#include <parsec.h>
#include <parsec/class/parsec_hash_table.h>
#include <parsec/data_internal.h>
#include <parsec/mca/device/device.h>
#include <parsec/execution_stream.h>
#include <parsec/interfaces/interface.h>
#include <parsec/parsec_internal.h>
#include <parsec/scheduling.h>
#include <cstdlib>
#include <cstring>

extern "C" {
    void parsec_taskpool_termination_detected(parsec_taskpool_t *tp);
    int parsec_add_fetch_runtime_task(parsec_taskpool_t *tp, int tasks);
}

namespace parsec {
  namespace ttg {
      typedef void (*static_set_arg_fct_type)(void *, size_t, ::ttg::OpBase*);
      typedef std::pair<static_set_arg_fct_type, ::ttg::OpBase*> static_set_arg_fct_call_t;
      static std::map<uint64_t, static_set_arg_fct_call_t> static_id_to_op_map;
      static std::mutex static_map_mutex;
      typedef std::tuple<int, void *, size_t>static_set_arg_fct_arg_t;
      static std::multimap<uint64_t, static_set_arg_fct_arg_t> delayed_unpack_actions;

      struct msg_header_t {
          uint32_t taskpool_id;
          uint64_t op_id;
          std::size_t param_id;
      };

      static int static_unpack_msg(parsec_comm_engine_t *ce, uint64_t tag,  void *data, long unsigned int size, int src_rank,  void *obj) {
          static_set_arg_fct_type static_set_arg_fct;
          int rank;
          parsec_taskpool_t *tp = NULL;
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
          msg_header_t *msg = static_cast<msg_header_t*>(data);
          uint64_t op_id = msg->op_id;
          tp = parsec_taskpool_lookup( msg->taskpool_id );
          assert(NULL != tp);
          static_map_mutex.lock();
          try {
              auto op_pair = static_id_to_op_map.at( op_id );
              static_map_mutex.unlock();
              tp->tdm.module->incoming_message_start(tp, src_rank, NULL, NULL, 0, NULL);
              static_set_arg_fct = op_pair.first;
              static_set_arg_fct(data, size, op_pair.second);
              tp->tdm.module->incoming_message_end(tp, NULL);
              return 0;
          } catch (const std::out_of_range & e) {
              void *data_cpy = malloc(size);
              assert(data_cpy != 0);
              memcpy(data_cpy, data, size);
              if(::ttg::tracing()) {
                  ::ttg::print("parsec::ttg(", rank, ") Delaying delivery of message (", src_rank, ", ", op_id, ", ", data_cpy, ", ", size, ")");
              }
              delayed_unpack_actions.insert(std::make_pair(op_id, std::make_tuple( src_rank, data_cpy, size )));
              static_map_mutex.unlock();
              return 1;
          }
    }

    class World {
        const int _PARSEC_TTG_TAG = 10; // This TAG should be 'allocated' at the PaRSEC level
     public:
        static const int PARSEC_TTG_MAX_AM_SIZE = 1024*1024;
      World(int *argc, char **argv[], int ncores) {
          ctx = parsec_init(ncores, argc, argv);
          es = ctx->virtual_processes[0]->execution_streams[0];

          parsec_ce.tag_register(_PARSEC_TTG_TAG, parsec::ttg::static_unpack_msg, this,
                                 PARSEC_TTG_MAX_AM_SIZE);
        
          tpool = (parsec_taskpool_t *)calloc(1, sizeof(parsec_taskpool_t));
          tpool->taskpool_id = -1;
          tpool->update_nb_runtime_task = parsec_add_fetch_runtime_task;
          tpool->taskpool_type = PARSEC_TASKPOOL_TYPE_TTG;
          parsec_taskpool_reserve_id(tpool);

          parsec_termdet_open_dyn_module(tpool);
          tpool->tdm.module->monitor_taskpool(tpool, parsec_taskpool_termination_detected);
          // In TTG, we use the pending actions to denote that the
          // taskpool is not ready, i.e. some local tasks could still
          // be added by the main thread. It should then be initialized
          // to 0, execute will set it to 1 and mark the tpool as ready,
          // and the fence() will decrease it back to 0. 
          tpool->tdm.module->taskpool_set_nb_pa(tpool, 0); 
          parsec_taskpool_enable(tpool, NULL, NULL, es, size() > 1);
      }

      ~World() {
          while (!op_register.empty()) {
              std::cout << "Destroying OpBase " << (*op_register.begin()) << std::endl;
              (*op_register.begin())->release();
          }
          parsec_ce.tag_unregister(_PARSEC_TTG_TAG);
          parsec_fini(&ctx);
          free(tpool);
      }

      const int &parsec_ttg_tag() { return _PARSEC_TTG_TAG; }
        
      MPI_Comm comm() const { return MPI_COMM_WORLD; }

      int size() const {
        int size;
        MPI_Comm_size(comm(), &size);
        return size;
      }

      int rank() const {
        int rank;
        MPI_Comm_rank(comm(), &rank);
        return rank;
      }

      void execute() {
        parsec_enqueue(ctx, tpool);
        tpool->tdm.module->taskpool_addto_nb_pa(tpool, 1);
        tpool->tdm.module->taskpool_ready(tpool);
        int ret = parsec_context_start(ctx);
        if(ret != 0) throw std::runtime_error("TTG: parsec_context_start failed");
      }

      auto *context() { return ctx; }
      auto *execution_stream() { return es; }
      auto *taskpool() { return tpool; }

      void fence() {
          int rank;
          parsec_taskpool_t *tp = taskpool();
          if( ::ttg::tracing() ) {
              MPI_Comm_rank(MPI_COMM_WORLD, &rank);
              ::ttg::print("parsec::ttg(", rank,  "): parsec taskpool is ready for completion");
          }
          // We are locally ready (i.e. we won't add new tasks)
          tpool->tdm.module->taskpool_addto_nb_pa(tpool, -1);
          if( ::ttg::tracing() ) {
              ::ttg::print("parsec::ttg(", rank, "): waiting for completion");
          }
          parsec_context_wait(ctx);
      }

      void increment_created() { taskpool()->tdm.module->taskpool_addto_nb_tasks(taskpool(), 1); }
      void increment_sent_to_sched() { parsec_atomic_fetch_inc_int32(&sent_to_sched_counter()); }

      int32_t sent_to_sched() const { return this->sent_to_sched_counter(); }

      void register_op(::ttg::OpBase* op) {
          // TODO: do we need locking here?
          op_register.push_back(op);
      }

      void deregister_op(::ttg::OpBase* op) {
          // TODO: do we need locking here?
          op_register.remove(op);
      }

     private:
      parsec_context_t *ctx = nullptr;
      parsec_execution_stream_t *es = nullptr;
      parsec_taskpool_t *tpool = nullptr;
      std::list<::ttg::OpBase*> op_register;

      volatile int32_t &sent_to_sched_counter() const {
        static volatile int32_t sent_to_sched = 0;
        return sent_to_sched;
      }
    };

    namespace detail {
      World *&default_world_accessor() {
        static World *world_ptr = nullptr;
        return world_ptr;
      }
    }  // namespace detail

    inline World &get_default_world() {
      if (detail::default_world_accessor() != nullptr) {
        return *detail::default_world_accessor();
      } else {
        throw std::logic_error("parsec::ttg::set_default_world() must be called before use");
      }
    }
    inline void set_default_world(World &world) { detail::default_world_accessor() = &world; }
    inline void set_default_world(World *world) { detail::default_world_accessor() = world; }

    /// the default keymap implementation maps key to ttg::hash{}(key) % nproc
    template <typename keyT>
    struct default_keymap : ::ttg::detail::default_keymap_impl<keyT> {
     public:
      default_keymap() = default;
      default_keymap(World &world) : ::ttg::detail::default_keymap_impl<keyT>(world.size()) {}
    };

  }  // namespace ttg
}  // namespace parsec

extern "C" {
  typedef void (*parsec_static_op_t)(void *);  // static_op will be cast to this type

  typedef struct my_op_s {
    parsec_task_t parsec_task;
    int32_t in_data_count;
    // TODO need to augment PaRSEC backend's my_op_s by stream size info, etc.  ... in_data_count will need to be replaced by something like this
    //  int counter;                            // Tracks the number of arguments set
    //  std::array<std::size_t, numins> nargs;  // Tracks the number of expected values (0 = finalized)
    //  std::array<std::size_t, numins>
    //      stream_size;                        // Expected number of values to receive, only used for streaming inputs
    //  // (0 = unbounded stream)
    parsec_hash_table_item_t op_ht_item;
    parsec_static_op_t function_template_class_ptr[::ttg::runtime_traits<::ttg::Runtime::PaRSEC>::num_execution_spaces];
    void *object_ptr;
    void (*static_set_arg)(int, int);
    parsec_key_t key;
  } my_op_t;
}

extern "C" {
static inline parsec_hook_return_t hook(struct parsec_execution_stream_s *es, parsec_task_t *task) {
  my_op_t *me = (my_op_t *) task;
  me->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::Host)](task);
  (void) es;
  return PARSEC_HOOK_RETURN_DONE;
}
static inline parsec_hook_return_t hook_cuda(struct parsec_execution_stream_s *es, parsec_task_t *task) {
  my_op_t *me = (my_op_t *) task;
  me->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::CUDA)](task);
  (void) es;
  return PARSEC_HOOK_RETURN_DONE;
}
}

static inline uint64_t parsec_tasks_hash_fct(parsec_key_t key, int nb_bits, void *data) {
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

static parsec_key_fn_t parsec_tasks_hash_fcts = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

namespace parsec {
  namespace ttg {
    namespace detail {
    static inline std::map<World*, std::set<std::shared_ptr<void>>>& ptr_registry_accessor() {
      static std::map<World*, std::set<std::shared_ptr<void>>> registry;
      return registry;
    };
    static inline std::map<World*, ::ttg::Edge<>>& clt_edge_registry_accessor() {
      static std::map<World*, ::ttg::Edge<>> registry;
      return registry;
    };
    static inline std::map<World*, std::set<std::shared_ptr<std::promise<void>>>>& status_registry_accessor() {
      static std::map<World*, std::set<std::shared_ptr<std::promise<void>>>> registry;
      return registry;
    };
    }

    template <typename... RestOfArgs>
    inline void ttg_initialize(int argc, char **argv, int taskpool_size, RestOfArgs &&...) {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
      set_default_world(new World(&argc, &argv, taskpool_size));
    }
    inline void ttg_finalize() {
      auto &status = ::parsec::ttg::detail::status_registry_accessor();
      status.clear();
      auto &edges = ::parsec::ttg::detail::clt_edge_registry_accessor();
      edges.clear();
      auto &ptrs = ::parsec::ttg::detail::ptr_registry_accessor();
      ptrs.clear();
      World *default_world = &get_default_world();
      delete default_world;
      set_default_world(nullptr);
      MPI_Finalize();
    }
    inline void ttg_abort() { MPI_Abort(get_default_world().comm(), 1); }
    inline World &ttg_default_execution_context() { return get_default_world(); }
    inline void ttg_execute(World &world) { world.execute(); }
    inline void ttg_fence(World &world) {
      world.fence();

      // flag registered statuses
      {
        auto& registry = detail::status_registry_accessor();
        auto iter = registry.find(&world);
        if (iter != registry.end()) {
          auto &statuses = iter->second;
          for (auto &status: statuses) {
            status->set_value();
          }
          statuses.clear();  // clear out the statuses
        }
      }

    }

    template <typename T>
    inline void ttg_register_ptr(World& world, const std::shared_ptr<T>& ptr) {
      auto& registry = detail::ptr_registry_accessor();
      auto iter = registry.find(&world);
      if (iter != registry.end()) {
        auto& ptr_set = iter->second;
        assert(ptr_set.find(ptr) == ptr_set.end());  // prevent duplicate registration
        ptr_set.insert(ptr);
      } else {
        registry.insert(std::make_pair(&world, std::set<std::shared_ptr<void>>({ptr})));
      }
    }

    inline void ttg_register_status(World& world, const std::shared_ptr<std::promise<void>>& status_ptr) {
      auto& registry = detail::status_registry_accessor();
      auto iter = registry.find(&world);
      if (iter != registry.end()) {
        auto& ptr_set = iter->second;
        assert(ptr_set.find(status_ptr) == ptr_set.end());  // prevent duplicate registration
        ptr_set.insert(status_ptr);
      } else {
        registry.insert(std::make_pair(&world, std::set<std::shared_ptr<std::promise<void>>>({status_ptr})));
      }
    }

    inline ::ttg::Edge<>& ttg_ctl_edge(World& world) {
      auto& registry = detail::clt_edge_registry_accessor();
      auto iter = registry.find(&world);
      if (iter != registry.end()) {
        return iter->second;
      } else {
        registry.insert(std::make_pair(&world, ::ttg::Edge<>{}));
        return registry[&world];
      }
    }

    inline void ttg_sum(World &world, double &value) {
      double result = 0.0;
      MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, world.comm());
      value = result;
    }
    /// broadcast
    /// @tparam T a serializable type
    template <typename T>
    void ttg_broadcast(World &world, T &data, int source_rank) {
      assert(world.size() == 1);
    }

    struct ParsecBaseOp {
     protected:
      //  static std::map<int, ParsecBaseOp*> function_id_to_instance;
      parsec_hash_table_t tasks_table;
      parsec_task_class_t self;
    };
    // std::map<int, ParsecBaseOp*> ParsecBaseOp::function_id_to_instance = {};

    namespace detail {
        struct msg_t {
            msg_header_t op_id;
            unsigned char bytes[World::PARSEC_TTG_MAX_AM_SIZE - sizeof(msg_header_t)];

            msg_t() = default;
            msg_t(uint64_t op_id, uint32_t taskpool_id, std::size_t param_id) : op_id{taskpool_id, op_id, param_id} {}
        };
    }  // namespace detail

    template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
    class Op : public ::ttg::OpBase, ParsecBaseOp {
     private:
      using opT = Op<keyT, output_terminalsT, derivedT, input_valueTs...>;
      parsec_mempool_t mempools;
      std::map<std::pair<int, int>, int> mempools_index;

      //check for a non-type member named have_cuda_op
      template <typename T>
      using have_cuda_op_non_type_t = decltype(&T::have_cuda_op);

      bool alive = true;

     public:
      static constexpr int numins = sizeof...(input_valueTs);                    // number of input arguments
      static constexpr int numouts = std::tuple_size<output_terminalsT>::value;  // number of outputs

      /// @return true if derivedT::have_cuda_op exists and is defined to true
      static constexpr bool derived_has_cuda_op() {
        if constexpr (std::experimental::is_detected_v<have_cuda_op_non_type_t, derivedT>) {
          return derivedT::have_cuda_op;
        } else {
          return false;
        }
      }

      using input_terminals_type = std::tuple<::ttg::In<keyT, input_valueTs>...>;
      using input_edges_type = std::tuple<::ttg::Edge<keyT, std::decay_t<input_valueTs>>...>;
      static_assert(::ttg::meta::is_none_Void_v<input_valueTs...>, "::ttg::Void is for internal use only, do not use it");
      // if have data inputs and (always last) control input, convert last input to Void to make logic easier
      using input_values_full_tuple_type = std::tuple<::ttg::meta::void_to_Void_t<std::decay_t<input_valueTs>>...>;
      using input_refs_full_tuple_type = std::tuple<std::add_lvalue_reference_t<::ttg::meta::void_to_Void_t<input_valueTs>>...>;
      using input_values_tuple_type = std::conditional_t<::ttg::meta::is_none_void_v<input_valueTs...>,input_values_full_tuple_type,typename ::ttg::meta::drop_last_n<input_values_full_tuple_type,std::size_t{1}>::type>;
      using input_refs_tuple_type = std::conditional_t<::ttg::meta::is_none_void_v<input_valueTs...>,input_refs_full_tuple_type,typename ::ttg::meta::drop_last_n<input_refs_full_tuple_type,std::size_t{1}>::type>;
      using input_unwrapped_values_tuple_type = input_values_tuple_type;
      static constexpr int numinvals = std::tuple_size_v<input_refs_tuple_type>; // number of input arguments with values (i.e. omitting the control input, if any)

      using output_terminals_type = output_terminalsT;
      using output_edges_type = typename ::ttg::terminals_to_edges<output_terminalsT>::type;

      template <std::size_t i, typename resultT, typename InTuple>
      static resultT get(InTuple &&intuple) {
        return static_cast<resultT>(std::get<i>(std::forward<InTuple>(intuple)));
      };
      template <std::size_t i, typename InTuple>
      static auto &get(InTuple &&intuple) {
        return std::get<i>(std::forward<InTuple>(intuple));
      };

     private:
      input_terminals_type input_terminals;
      output_terminalsT output_terminals;
      std::array<void (Op::*)(void*, std::size_t), numins> set_arg_from_msg_fcts;

      World &world;
      ::ttg::meta::detail::keymap_t<keyT> keymap;
      // For now use same type for unary/streaming input terminals, and stream reducers assigned at runtime
      ::ttg::meta::detail::input_reducers_t<input_valueTs...>
          input_reducers;  //!< Reducers for the input terminals (empty = expect single value)

     public:
      World &get_world() const { return world; }

     private:


      template <std::size_t...IS>
      static auto make_set_args_fcts(std::index_sequence<IS...>) {
          using resultT = decltype(set_arg_from_msg_fcts);
          return resultT{{&Op::set_arg_from_msg<IS>...}};
      }

      /// dispatches a call to derivedT::op if Space == Host, otherwise to derivedT::op_cuda if Space == CUDA
      template <::ttg::ExecutionSpace Space, typename ... Args>
      void op(Args&& ... args) {
        derivedT *derived = static_cast<derivedT*>(this);
        if constexpr (Space == ::ttg::ExecutionSpace::Host)
          derived->op(std::forward<Args>(args)...);
        else if constexpr (Space == ::ttg::ExecutionSpace::CUDA)
          derived->op_cuda(std::forward<Args>(args)...);
        else abort();
      }

      template <std::size_t...IS>
      static input_refs_tuple_type make_tuple_of_ref_from_array(my_op_t *task, std::index_sequence<IS...>) {
        return input_refs_tuple_type{
          static_cast<typename std::tuple_element<IS, input_refs_tuple_type>::type>(
            *reinterpret_cast<std::remove_reference_t<typename std::tuple_element<IS, input_refs_tuple_type>::type>*>(
              task->parsec_task.data[IS].data_in->device_private)) ... };
      }

      template <::ttg::ExecutionSpace Space>
      static void static_op(parsec_task_t *my_task) {

        my_op_t *task = (my_op_t *)my_task;
        opT *baseobj = (opT *)task->object_ptr;
        derivedT *obj = (derivedT *)task->object_ptr;
        if (obj->tracing()) {
          if constexpr (!::ttg::meta::is_void_v<keyT>)
            ::ttg::print(obj->get_world().rank(), ":", obj->get_name(), " : ", *(keyT*) task->key, ": executing");
          else
            ::ttg::print(obj->get_world().rank(), ":", obj->get_name(), " : executing");
        }

        if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          input_refs_tuple_type input = make_tuple_of_ref_from_array(task, std::make_index_sequence<numinvals>{});
          baseobj->template op<Space>(*(keyT*)task->key, std::move(input), obj->output_terminals);
        } else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          baseobj->template op<Space>(*(keyT*)task->key, obj->output_terminals);
        } else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          input_refs_tuple_type input = make_tuple_of_ref_from_array(task, std::make_index_sequence<numinvals>{});
          baseobj->template op<Space>(std::move(input), obj->output_terminals);
        } else if constexpr (::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          baseobj->template op<Space>(obj->output_terminals);
        } else abort();

        if (obj->tracing()) {
          if constexpr (!::ttg::meta::is_void_v<keyT>)
                           ::ttg::print(obj->get_world().rank(), ":", obj->get_name(), " : ", *(keyT*)task->key, ": done executing");
          else
            ::ttg::print(obj->get_world().rank(), ":", obj->get_name(), " : done executing");
        }
      }

      template <::ttg::ExecutionSpace Space>
      static void static_op_noarg(parsec_task_t *my_task) {
        my_op_t *task = (my_op_t *)my_task;
        opT *baseobj = (opT *)task->object_ptr;
        derivedT *obj = (derivedT *)task->object_ptr;
        if constexpr(!::ttg::meta::is_void_v<keyT>) {
            baseobj->template op<Space>(*(keyT*)task->key, obj->output_terminals);
        } else if constexpr(::ttg::meta::is_void_v<keyT>) {
            baseobj->template op<Space>(obj->output_terminals);
        }
        else abort();
      }

     protected:
      template <typename T> uint64_t unpack(T &obj, void *_bytes, uint64_t pos) {
        const ttg_data_descriptor *dObj = ::ttg::get_data_descriptor<T>();
        uint64_t header_size, payload_size;
        void *buffer;
        char *bytes = static_cast<char *>(_bytes);
        int contiguous;
        dObj->get_info(static_cast<void*>(&bytes[pos]), &header_size, &payload_size, &contiguous, &buffer);
        assert(0 == header_size);
        if (NULL != buffer) {
          dObj->unpack_payload(&obj, payload_size, 0, buffer);
        } else {
          dObj->unpack_payload(&obj, payload_size, pos, _bytes);
        }
        return pos + payload_size;
      }

      template <typename T> uint64_t pack(T &obj, void *bytes, uint64_t pos) {
        const ttg_data_descriptor *dObj = ::ttg::get_data_descriptor<T>();
        uint64_t header_size, payload_size;
        void *buffer;
        int contiguous;
        dObj->get_info(&obj, &header_size, &payload_size, &contiguous, &buffer);
        assert(0 == header_size);
        if (NULL != buffer) {
          dObj->pack_payload(buffer, payload_size, pos, bytes);
        } else {
          dObj->pack_payload(&obj, payload_size, pos, bytes);
        }
        return pos + payload_size;
      }

      static void static_set_arg(void *data, std::size_t size, ::ttg::OpBase *bop) {
          assert(size >= sizeof(msg_header_t) &&
                 "Trying to unpack as message that does not hold enough bytes to represent a single header");
          msg_header_t *hd = static_cast<msg_header_t*>(data);
          derivedT *obj = reinterpret_cast<derivedT*>(bop);
          if(-1 != hd->param_id) {
            auto member = obj->set_arg_from_msg_fcts[hd->param_id];
            (obj->*member)(data, size);
          } else {
            if constexpr (::ttg::meta::is_empty_tuple_v<input_refs_tuple_type>) {
              if constexpr (::ttg::meta::is_void_v<keyT>) {
                obj->template set_arg<keyT>();
              } else {
                using msg_t = detail::msg_t;
                msg_t *msg = static_cast<msg_t*>(data);
                keyT key;
                obj->unpack(key, static_cast<void*>(msg->bytes), 0);
                obj->template set_arg<keyT>(key);
              }
            } else {
              abort();
            }
          }
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
          using valueT = typename std::tuple_element<i, input_terminals_type>::type::value_type;
          using msg_t = detail::msg_t;
          msg_t *msg = static_cast<msg_t*>(data);
          // case 1
          if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && !std::is_void_v<valueT>) {
                  keyT key;
                  using decvalueT = std::decay_t<valueT>;
                  decvalueT val;
                  uint64_t pos = unpack(key, msg->bytes, 0);
                  pos = unpack(val, msg->bytes, pos);
                  set_arg<i, keyT, valueT>(key, std::move(val));
                  // case 2
              } else if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
                  keyT key;
                  unpack(key, msg->bytes, 0);
                  set_arg<i, keyT, ::ttg::Void>(key, ::ttg::Void{});
                  // case 3
              } else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
                  keyT key;
                  unpack(key, msg->bytes, 0);
                  set_arg<keyT>(key);
                  // case 4
              } else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && !std::is_void_v<valueT>) {
                  using decvalueT = std::decay_t<valueT>;
                  decvalueT val;
                  unpack(val, msg->bytes, 0);
                  set_arg<i, keyT, valueT>(std::move(val));
                  // case 5
              } else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
                  set_arg<i, keyT, ::ttg::Void>(::ttg::Void{});
                  // case 6
              } else if constexpr (::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
                  set_arg<keyT>();
              } else {
              abort();
          }
      }

      template <std::size_t i, typename Key, typename Value>
      std::enable_if_t<!::ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void>
      set_arg_local(const Key &key, Value && value) {
        set_arg_local_impl<i>(key,std::forward<Value>(value));
      }

      template <std::size_t i, typename Key = keyT, typename Value>
      std::enable_if_t<::ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void>
      set_arg_local(Value && value) {
        set_arg_local_impl<i>(::ttg::Void{},std::forward<Value>(value));
      }

      template <typename Key = keyT> void release_op_task(my_op_t *task) {
          if constexpr (!::ttg::meta::is_void_v<Key>) {
              Key *key = (Key*)task->key;
              delete(key);
          }
          parsec_mempool_free(&mempools, task);
      }

      // Used to set the i'th argument
      template <std::size_t i, typename Key, typename Value>
      void set_arg_local_impl(const Key &key, Value && value) {
        using valueT = typename std::tuple_element<i, input_values_full_tuple_type>::type;
        constexpr const bool valueT_is_Void = ::ttg::meta::is_void_v<valueT>;

        if (tracing()) {
          if constexpr (!valueT_is_Void) {
            ::ttg::print(world.rank(),
                         ":",
                         get_name(),
                         " : ",
                         key,
                         ": received value for argument : ",
                         i,
                         " : value = ",
                         value);
          } else {
            ::ttg::print(world.rank(),
                         ":",
                         get_name(),
                         " : ",
                         key,
                         ": received value for argument : ",
                         i);
          }
        }

        parsec_key_t hk = reinterpret_cast<parsec_key_t>(&key);
        my_op_t *task = NULL;
        if (NULL == (task = (my_op_t *)parsec_hash_table_find(&tasks_table, hk))) {
          my_op_t *newtask;
          parsec_execution_stream_s *es = world.execution_stream();
          parsec_thread_mempool_t *mempool =
              &mempools.thread_mempools[mempools_index[std::pair<int, int>(es->virtual_process->vp_id, es->th_id)]];
          newtask = (my_op_t *)parsec_thread_mempool_allocate(mempool);
          memset((void *)newtask, 0, sizeof(my_op_t));
          newtask->parsec_task.mempool_owner = mempool;

          PARSEC_OBJ_CONSTRUCT(&newtask->parsec_task, parsec_list_item_t);
          newtask->parsec_task.task_class = &this->self;
          newtask->parsec_task.taskpool = world.taskpool();
          newtask->parsec_task.status = PARSEC_TASK_STATUS_HOOK;
          newtask->in_data_count = 0;

          newtask->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::Host)] = reinterpret_cast<parsec_static_op_t>(&Op::static_op<::ttg::ExecutionSpace::Host>);
          if constexpr (derived_has_cuda_op())
            newtask->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::CUDA)] = reinterpret_cast<parsec_static_op_t>(&Op::static_op<::ttg::ExecutionSpace::CUDA>);
          newtask->object_ptr = static_cast<derivedT *>(this);
          if constexpr (::ttg::meta::is_void_v<keyT>) {
              newtask->key = 0;
          } else {
              keyT *new_key = new keyT(key);
              newtask->key = reinterpret_cast<parsec_key_t>(new_key);
          }
          
          parsec_mfence();
          parsec_hash_table_lock_bucket(&tasks_table, hk);
          if (NULL != (task = (my_op_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
            release_op_task(newtask);
          } else {
            if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": creating task");
            newtask->op_ht_item.key = newtask->key;
            world.increment_created();
            parsec_hash_table_nolock_insert(&tasks_table, &newtask->op_ht_item);
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
            task = newtask;
          }
        }

        if constexpr (!valueT_is_Void) {
          if (NULL != task->parsec_task.data[i].data_in) {
            ::ttg::print_error(get_name(), " : ", key, ": error argument is already set : ", i);
            throw std::logic_error("bad set arg");
          }
          // TODO:
          //    - find if value (which is a ref) exists in data[?].data_in
          //    - if it does, drop the reference, and check if it was a const type or not
          //    - if it is a const type, then the source task cannot modify it, and
          //    - if the target task uses the data as read-only, it is not necessary to
          //    - create a new data copy and we should reuse it
          parsec_data_copy_t *copy =  PARSEC_OBJ_NEW(parsec_data_copy_t);
          copy->device_private = (void*)( new valueT(value) );
          task->parsec_task.data[i].data_in = copy;

          // uncomment this if you want to test deserialization ... also comment out the placement new above
          //    auto* ddesc = ::ttg::get_data_descriptor<valueT>();
          //    void* value_ptr = (void*)&value;
          //    uint64_t hs, ps;
          //    int is_contiguous;
          //    void* buf;
          //    (*(ddesc->get_info))(value_ptr, &hs, &ps, &is_contiguous, &buf);
          //    assert(is_contiguous);
          //    (*(ddesc->unpack_header))(copy->device_private, hs, value_ptr);
          //    (*(ddesc->unpack_payload))(copy->device_private, ps, 0, value_ptr);
        }

        int32_t count = parsec_atomic_fetch_inc_int32(&task->in_data_count)+1;
        assert(count <= self.dependencies_goal);

        if (count == self.dependencies_goal) {
          world.increment_sent_to_sched();
          parsec_execution_stream_t *es = world.execution_stream();
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
          parsec_hash_table_remove(&tasks_table, hk);
          __parsec_schedule(es, &task->parsec_task, 0);
        }
      }

      // cases 1+2
      template <std::size_t i, typename Key, typename Value>
      std::enable_if_t<!::ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void>
      set_arg(const Key &key, Value &&value) {
        set_arg_impl<i>(key,std::forward<Value>(value));
      }

      // cases 4+5
      template <std::size_t i, typename Key, typename Value>
      std::enable_if_t<::ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void>
      set_arg(Value &&value) {
          set_arg_impl<i>(::ttg::Void{},std::forward<Value>(value));
      }

      // Used to set the i'th argument
      template <std::size_t i, typename Key, typename Value>
      void set_arg_impl(const Key &key, Value &&value) {
        using valueT = typename std::tuple_element<i, input_values_full_tuple_type>::type;
        int owner;
        if constexpr (!::ttg::meta::is_void_v<Key>)
          owner = keymap(key);
        else
          owner = keymap();
        if( owner == ttg_default_execution_context().rank() ) {
          if constexpr (!::ttg::meta::is_void_v<keyT>)
            set_arg_local<i, keyT, Value>(key, std::forward<Value>(value));
          else
            set_arg_local<i, keyT, Value>(std::forward<Value>(value));
          return;
        }
        // the target task is remote. Pack the information and send it to
        // the corresponding peer.
        // TODO do we need to copy value?
        using msg_t = detail::msg_t;
        msg_t* msg = new msg_t(get_instance_id(), world.taskpool()->taskpool_id, i);

        uint64_t pos = 0;
        pos = pack(key, msg->bytes, pos);
        pos = pack(value, msg->bytes, pos);
        parsec_taskpool_t *tp = world.taskpool();
        tp->tdm.module->outgoing_message_start(tp, owner, NULL);       
        tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
        parsec_ce.send_am(&parsec_ce, world.parsec_ttg_tag(), owner, static_cast<void*>(msg), sizeof(msg_header_t)+pos);
        delete msg;
      }

      // case 3
      template <typename Key = keyT>
      std::enable_if_t<!::ttg::meta::is_void_v<Key>,void>
      set_arg(const Key &key) {
        static_assert(::ttg::meta::is_empty_tuple_v<input_refs_tuple_type>, "logic error: set_arg (case 3) called but input_refs_tuple_type is nonempty");

        const auto owner = keymap(key);
        if( owner == ttg_default_execution_context().rank() ) {
          // create PaRSEC task
          // and give it to the scheduler
          my_op_t *task;
          parsec_execution_stream_s *es = world.execution_stream();
          parsec_thread_mempool_t *mempool =
              &mempools.thread_mempools[mempools_index[std::pair<int, int>(es->virtual_process->vp_id, es->th_id)]];
          task = (my_op_t *) parsec_thread_mempool_allocate(mempool);
          memset((void *) task, 0, sizeof(my_op_t));
          task->parsec_task.mempool_owner = mempool;

          PARSEC_OBJ_CONSTRUCT(task, parsec_list_item_t);
          task->parsec_task.task_class = &this->self;
          task->parsec_task.taskpool = world.taskpool();
          task->parsec_task.status = PARSEC_TASK_STATUS_HOOK;

          task->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::Host)] = reinterpret_cast<parsec_static_op_t>(&Op::static_op_noarg<::ttg::ExecutionSpace::Host>);
          if constexpr (derived_has_cuda_op())
            task->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::CUDA)] = reinterpret_cast<parsec_static_op_t>(&Op::static_op_noarg<::ttg::ExecutionSpace::CUDA>);
          task->object_ptr = static_cast<derivedT *>(this);
          keyT *kp = new keyT(key);
          task->key = reinterpret_cast<parsec_key_t>(kp);
          task->parsec_task.data[0].data_in = static_cast<parsec_data_copy_t *>(NULL);
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": creating task");
          world.increment_created();
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
          world.increment_sent_to_sched();
          __parsec_schedule(es, &task->parsec_task, 0);
        } else {
          using msg_t = detail::msg_t;
          // We pass -1 to signal that we just need to call set_arg(key) on the other end
          msg_t* msg = new msg_t(get_instance_id(), world.taskpool()->taskpool_id, -1);

          uint64_t pos = 0;
          const ttg_data_descriptor *dKey = ::ttg::get_data_descriptor<Key>();
          pos = dKey->pack_payload(&key, sizeof(Key), pos, msg->bytes);
          parsec_taskpool_t *tp = world.taskpool();
          tp->tdm.module->outgoing_message_start(tp, owner, NULL);       
          tp->tdm.module->outgoing_message_pack(tp, owner, NULL, NULL, 0);
          parsec_ce.send_am(&parsec_ce, world.parsec_ttg_tag(), owner, static_cast<void*>(msg), sizeof(msg_header_t)+pos);
          delete msg;
        }
      }

      // case 6
      template <typename Key = keyT>
      std::enable_if_t<::ttg::meta::is_void_v<Key>,void>
      set_arg() {
        static_assert(::ttg::meta::is_empty_tuple_v<input_refs_tuple_type>, "logic error: set_arg (case 3) called but input_refs_tuple_type is nonempty");

        const auto owner = keymap();
        if( owner == ttg_default_execution_context().rank() ) {

          // create PaRSEC task
          // and give it to the scheduler
          my_op_t *task;
          parsec_execution_stream_s *es = world.execution_stream();
          parsec_thread_mempool_t *mempool =
              &mempools.thread_mempools[mempools_index[std::pair<int, int>(es->virtual_process->vp_id, es->th_id)]];
          task = (my_op_t *) parsec_thread_mempool_allocate(mempool);
          memset((void *) task, 0, sizeof(my_op_t));
          task->parsec_task.mempool_owner = mempool;

          PARSEC_OBJ_CONSTRUCT(task, parsec_list_item_t);
          task->parsec_task.task_class = &this->self;
          task->parsec_task.taskpool = world.taskpool();
          task->parsec_task.status = PARSEC_TASK_STATUS_HOOK;

          task->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::Host)] = reinterpret_cast<parsec_static_op_t>(&Op::static_op_noarg<::ttg::ExecutionSpace::Host>);
          if constexpr (derived_has_cuda_op())
            task->function_template_class_ptr[static_cast<std::size_t>(::ttg::ExecutionSpace::CUDA)] = reinterpret_cast<parsec_static_op_t>(&Op::static_op_noarg<::ttg::ExecutionSpace::CUDA>);
          task->object_ptr = static_cast<derivedT *>(this);
          task->key = 0;
          task->parsec_task.data[0].data_in = static_cast<parsec_data_copy_t *>(NULL);
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : creating task");
          world.increment_created();
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : submitting task for op ");
          world.increment_sent_to_sched();
          __parsec_schedule(es, &task->parsec_task, 0);
        }
      }

      // Used by invoke to set all arguments associated with a task
      template <typename Key, size_t... IS>
      std::enable_if_t<::ttg::meta::is_none_void_v<Key>,void>
      set_args(std::index_sequence<IS...>, const Key &key, const input_refs_tuple_type &args) {
        int junk[] = {0, (set_arg<IS>(key, Op::get<IS>(args)), 0)...};
        junk[0]++;
      }

      // Used by invoke to set all arguments associated with a task
      template <typename Key = keyT, size_t... IS>
      std::enable_if_t<::ttg::meta::is_void_v<Key>,void>
      set_args(std::index_sequence<IS...>, const input_refs_tuple_type &args) {
        int junk[] = {0, (set_arg<IS>(Op::get<IS>(args)), 0)...};
        junk[0]++;
      }

     public:

      /// sets stream size for input \c i
      /// \param size positive integer that specifies the stream size
      template <std::size_t i, typename Key>
      std::enable_if_t<!::ttg::meta::is_void_v<Key>, void>
      set_argstream_size(const Key &key, std::size_t size) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
        assert(size > 0 && "Op::set_argstream_size(key,size) called with size=0");

        // body
        const auto owner = keymap(key);
        abort();  // TODO implement set_argstream_size
      }

      /// sets stream size for input \c i
      /// \param size positive integer that specifies the stream size
      template <std::size_t i, typename Key = keyT>
      std::enable_if_t<::ttg::meta::is_void_v<Key>, void>
      set_argstream_size(std::size_t size) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
        assert(size > 0 && "Op::set_argstream_size(key,size) called with size=0");

        // body
        const auto owner = keymap();
        abort();  // TODO implement set_argstream_size
      }

      /// finalizes stream for input \c i
      template <std::size_t i, typename Key>
      std::enable_if_t<!::ttg::meta::is_void_v<Key>, void>
      finalize_argstream(const Key &key) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::finalize_argstream called on nonstreaming input terminal");

        // body
        const auto owner = keymap(key);
        abort();  // TODO implement set_argstream_size
      }

      /// finalizes stream for input \c i
      template <std::size_t i, typename Key>
      std::enable_if_t<::ttg::meta::is_void_v<Key>, void>
      finalize_argstream() {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::finalize_argstream called on nonstreaming input terminal");

        // body
        const auto owner = keymap();
        abort();  // TODO implement set_argstream_size
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
      // wanting to move/assign an Op you should be using a pointer.
      Op(const Op &other) = delete;
      Op &operator=(const Op &other) = delete;
      Op(Op &&other) = delete;
      Op &operator=(Op &&other) = delete;

      // Registers the callback for the i'th input terminal
      template <typename terminalT, std::size_t i>
      void register_input_callback(terminalT &input) {
        using valueT = typename terminalT::value_type;
        // case 1
        if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && !std::is_void_v<valueT>) {
          auto move_callback = [this](const keyT &key, valueT &&value) {
            set_arg<i, keyT, valueT>(key, std::forward<valueT>(value));
          };
          auto send_callback = [this](const keyT &key, const valueT &value) {
            set_arg<i, keyT, const valueT &>(key, value);
          };
          input.set_callback(send_callback, move_callback);
        }
        // case 2
        else if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
          auto send_callback = [this](const keyT &key) {
            set_arg<i, keyT, ::ttg::Void>(key, ::ttg::Void{});
          };
          input.set_callback(send_callback, send_callback);
        }
        // case 3
        else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
          auto send_callback = [this](const keyT &key) {
            set_arg<keyT>(key);
          };
          input.set_callback(send_callback, send_callback);
        }
        // case 4
        else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && !std::is_void_v<valueT>) {
          auto move_callback = [this](valueT &&value) {
            set_arg<i, keyT, valueT>(std::forward<valueT>(value));
          };
          auto send_callback = [this](const valueT &value) {
            set_arg<i, keyT, const valueT &>(value);
          };
          input.set_callback(send_callback, move_callback);
        }
        // case 5
        else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
          auto send_callback = [this]() {
            set_arg<i, keyT, ::ttg::Void>(::ttg::Void{});
          };
          input.set_callback(send_callback, send_callback);
        }
        // case 6
        else if constexpr (::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_refs_tuple_type> && std::is_void_v<valueT>) {
          auto send_callback = [this]() {
            set_arg<keyT>();
          };
          input.set_callback(send_callback, send_callback);
        } else abort();
      }

      template <std::size_t... IS>
      void register_input_callbacks(std::index_sequence<IS...>) {
        int junk[] = {0, (register_input_callback<typename std::tuple_element<IS, input_terminals_type>::type, IS>(
                              std::get<IS>(input_terminals)),
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
        int junk[] = {
            0,
            (*(const_cast<std::remove_const_t<decltype(flows[IS]->flow_flags)> *>(&(flows[IS]->flow_flags))) =
                 (std::is_const<typename std::tuple_element<IS, input_terminals_tupleT>::type>::value ? PARSEC_FLOW_ACCESS_READ
                                                                                                      : PARSEC_FLOW_ACCESS_RW),
             0)...};
        junk[0]++;
      }

      template <typename input_terminals_tupleT, typename flowsT>
      void initialize_flows(flowsT &&flows) {
        _initialize_flows<input_terminals_tupleT>(
            std::make_index_sequence<std::tuple_size<input_terminals_tupleT>::value>{}, flows);
      }

      void fence() override { get_default_world().fence(); }

      static int key_equal(parsec_key_t a, parsec_key_t b, void *user_data) {
          if constexpr (std::is_same_v<keyT, void>) {
                  return 1;
              } else {
                  keyT ka = *( reinterpret_cast<keyT*>(a) );
                  keyT kb = *( reinterpret_cast<keyT*>(b) );
                  return ka == kb;
          }
      }

      static uint64_t key_hash(parsec_key_t k, void *user_data) {
          if constexpr (std::is_same_v<keyT, void>) {
                  return 0;
              } else {
              keyT kk = *( reinterpret_cast<keyT*>(k) );
              using ::ttg::hash;
              return hash<decltype(kk)>{}(kk);
          }
      }

      static char *key_print(char *buffer, size_t buffer_size, parsec_key_t k, void *user_data) {
          if constexpr (std::is_same_v<keyT, void>) {
                  buffer[0] = '\0';
                  return buffer;
              } else {
              keyT kk = *( reinterpret_cast<keyT*>(k) );
              // use streambuf here?
              snprintf(buffer, buffer_size, "%lu", k);
              return buffer;
          }
      }

      parsec_key_fn_t tasks_hash_fcts = {
          key_equal,
          key_print,
          key_hash
      };
      
     static parsec_hook_return_t complete_task_and_release(parsec_execution_stream_t *es, parsec_task_t *task) {
         if constexpr (!::ttg::meta::is_void_v<keyT>) {
                 my_op_t *op = (my_op_t*)task;
                 keyT *key = (keyT*)op->key;
                 delete(key);
         }
         for(int i = 0; i < MAX_PARAM_COUNT; i++) {
             if(NULL != task->data[i].data_in) {
                 PARSEC_OBJ_RELEASE(task->data[i].data_in);
             }
         }
         return PARSEC_HOOK_RETURN_DONE;
     }
        
     public:
      template <typename keymapT = default_keymap<keyT>>
      Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         World &world, keymapT &&keymap_ = keymapT())
          : ::ttg::OpBase(name, numins, numouts)
          , set_arg_from_msg_fcts(make_set_args_fcts(std::make_index_sequence<numins>{}))
          , world(world)
          // if using default keymap, rebind to the given world
          , keymap(std::is_same<keymapT, default_keymap<keyT>>::value
                       ? decltype(keymap)(default_keymap<keyT>(world))
                       : decltype(keymap)(std::forward<keymapT>(keymap_))) {
        // Cannot call these in base constructor since terminals not yet constructed
        if (innames.size() != std::tuple_size<input_terminals_type>::value)
          throw std::logic_error("parsec::ttg::OP: #input names != #input terminals");
        if (outnames.size() != std::tuple_size<output_terminalsT>::value)
          throw std::logic_error("parsec::ttg::OP: #output names != #output terminals");

        world.register_op(this);

        register_input_terminals(input_terminals, innames);
        register_output_terminals(output_terminals, outnames);

        register_input_callbacks(std::make_index_sequence<numins>{});

        int i;

        memset(&self, 0, sizeof(parsec_task_class_t));

        self.name = get_name().c_str();
        self.task_class_id = get_instance_id();
        self.nb_parameters = 0;
        self.nb_locals = 0;
        self.nb_flows = std::max((int)numins, (int)numouts);

        //    function_id_to_instance[self.task_class_id] = this;

        if constexpr (derived_has_cuda_op()) {
          self.incarnations = (__parsec_chore_t *) malloc(3 * sizeof(__parsec_chore_t));
          ((__parsec_chore_t *) self.incarnations)[0].type = PARSEC_DEV_CUDA;
          ((__parsec_chore_t *) self.incarnations)[0].evaluate = NULL;
          ((__parsec_chore_t *) self.incarnations)[0].hook = hook_cuda;
          ((__parsec_chore_t *) self.incarnations)[1].type = PARSEC_DEV_CPU;
          ((__parsec_chore_t *) self.incarnations)[1].evaluate = NULL;
          ((__parsec_chore_t *) self.incarnations)[1].hook = hook;
          ((__parsec_chore_t *) self.incarnations)[2].type = PARSEC_DEV_NONE;
          ((__parsec_chore_t *) self.incarnations)[2].evaluate = NULL;
          ((__parsec_chore_t *) self.incarnations)[2].hook = NULL;
        } else {
          self.incarnations = (__parsec_chore_t *) malloc(2 * sizeof(__parsec_chore_t));
          ((__parsec_chore_t *) self.incarnations)[0].type = PARSEC_DEV_CPU;
          ((__parsec_chore_t *) self.incarnations)[0].evaluate = NULL;
          ((__parsec_chore_t *) self.incarnations)[0].hook = hook;
          ((__parsec_chore_t *) self.incarnations)[1].type = PARSEC_DEV_NONE;
          ((__parsec_chore_t *) self.incarnations)[1].evaluate = NULL;
          ((__parsec_chore_t *) self.incarnations)[1].hook = NULL;
        }

        self.release_task = parsec_release_task_to_mempool_update_nbtasks;
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

        int k = 0;
        auto *context = world.context();
        for (int i = 0; i < context->nb_vp; i++) {
          for (int j = 0; j < context->virtual_processes[i]->nb_cores; j++) {
            mempools_index[std::pair<int, int>(i, j)] = k++;
          }
        }
        parsec_mempool_construct(&mempools, PARSEC_OBJ_CLASS(parsec_task_t),
                                 sizeof(my_op_t),
                                 offsetof(parsec_task_t, mempool_owner), k);

        parsec_hash_table_init(&tasks_table, offsetof(my_op_t, op_ht_item), 8, tasks_hash_fcts, NULL);
      }

      template <typename keymapT = default_keymap<keyT>>
      Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         keymapT &&keymap = keymapT(get_default_world()))
          : Op(name, innames, outnames, get_default_world(), std::forward<keymapT>(keymap)) {}

      template <typename keymapT = default_keymap<keyT>>
      Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames, World &world,
         keymapT &&keymap_ = keymapT())
          : Op(name, innames, outnames, world, std::forward<keymapT>(keymap_)) {
        connect_my_inputs_to_incoming_edge_outputs(std::make_index_sequence<numins>{}, inedges);
        connect_my_outputs_to_outgoing_edge_inputs(std::make_index_sequence<numouts>{}, outedges);
      }
      template <typename keymapT = default_keymap<keyT>>
      Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         keymapT &&keymap = keymapT(get_default_world()))
          : Op(inedges, outedges, name, innames, outnames, get_default_world(), std::forward<keymapT>(keymap)) {}

      // Destructor checks for unexecuted tasks
      ~Op() {
         release();
      }

      virtual void release() override {
        std::cout << "Op dtor " << this << std::endl;
        if (!alive) { return; }
        alive = false;
        parsec_hash_table_fini(&tasks_table);
        parsec_mempool_destruct(&mempools);
        uintptr_t addr = (uintptr_t)self.incarnations;
        free((void*)addr);
        for(int i = 0; i < self.nb_flows; i++) {
            if( NULL != self.in[i] ) {
                free(self.in[i]->name);
                delete self.in[i];
            }
            if( NULL != self.out[i]) {
                free(self.out[i]->name);
                delete self.out[i];
            }
        }
        world.deregister_op(this);
      }

      static constexpr const ::ttg::Runtime runtime = ::ttg::Runtime::PaRSEC;

      template <std::size_t i, typename Reducer>
      void set_input_reducer(Reducer &&reducer) {
        std::get<i>(input_reducers) = reducer;
      }

      // Returns reference to input terminal i to facilitate connection --- terminal
      // cannot be copied, moved or assigned
      template <std::size_t i>
      typename std::tuple_element<i, input_terminals_type>::type *in() {
        return &std::get<i>(input_terminals);
      }

      // Returns reference to output terminal for purpose of connection --- terminal
      // cannot be copied, moved or assigned
      template <std::size_t i>
      typename std::tuple_element<i, output_terminalsT>::type *out() {
        return &std::get<i>(output_terminals);
      }

      // Manual injection of a task with all input arguments specified as a tuple
      template <typename Key = keyT> std::enable_if_t<!::ttg::meta::is_void_v<Key>,void>
      invoke(const Key &key, const input_refs_tuple_type &args)
      {
        TTG_OP_ASSERT_EXECUTABLE();
        set_args(std::make_index_sequence<std::tuple_size<input_refs_tuple_type>::value>{}, key, args);
      }

      // Manual injection of a key-free task and all input arguments specified as a tuple
      template <typename Key = keyT> std::enable_if_t<::ttg::meta::is_void_v<Key>,void>
      invoke(const input_refs_tuple_type &args) {
        TTG_OP_ASSERT_EXECUTABLE();
        set_args(std::make_index_sequence<std::tuple_size<input_refs_tuple_type>::value>{}, args);
      }

      // Manual injection of a task that has no arguments
      template <typename Key = keyT> std::enable_if_t<!::ttg::meta::is_void_v<Key>,void> invoke(const Key &key) {
        TTG_OP_ASSERT_EXECUTABLE();
        set_arg<keyT>(key);
      }

      // Manual injection of a task that has no key or arguments
      template <typename Key = keyT> std::enable_if_t<::ttg::meta::is_void_v<Key>,void> invoke() {
        TTG_OP_ASSERT_EXECUTABLE();
        set_arg<keyT>();
      }

      void make_executable() override {
        register_static_op_function();
        OpBase::make_executable();
      }

      /// keymap accessor
      /// @return the keymap
      const decltype(keymap) &get_keymap() const { return keymap; }

      // Register the static_op function to associate it to instance_id
      void register_static_op_function(void) {
          int rank;
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
          if (tracing()) {
              ::ttg::print("parsec::ttg(", rank,  ") Inserting into static_id_to_op_map at ", get_instance_id());
          }
          static_set_arg_fct_call_t call = std::make_pair(&Op::static_set_arg, this);

          static_map_mutex.lock();
          static_id_to_op_map.insert(std::make_pair(get_instance_id(), call));
          if( delayed_unpack_actions.count(get_instance_id()) > 0 ) {
              auto tp = world.taskpool();

              if (tracing()) {
                  ::ttg::print("parsec::ttg(", rank, ") There are ", delayed_unpack_actions.count(get_instance_id()), " messages delayed with op_id ", get_instance_id());
              }

              auto se = delayed_unpack_actions.equal_range( get_instance_id() );
              std::vector<static_set_arg_fct_arg_t> tmp;
              for(auto it = se.first; it != se.second; ) {
                  assert(it->first == get_instance_id());
                  tmp.push_back( it->second );
                  it = delayed_unpack_actions.erase(it);
              }
              static_map_mutex.unlock();

              for(auto it : tmp) {
                  if(tracing()) {
                      ::ttg::print("parsec::ttg(", rank, ") Unpacking delayed message (", ", ",
                                   get_instance_id(), ", ", std::get<1>(it), ", ", std::get<2>(it), ")");
                  }
                  int rc = static_unpack_msg(&parsec_ce, world.parsec_ttg_tag(),  std::get<1>(it), std::get<2>(it), std::get<0>(it), NULL);
                  assert(rc == 0);
                  free(std::get<1>(it));
              }

              tmp.clear();
          } else {
              static_map_mutex.unlock();
          }
      }
    };

#include "../wrap.h"

  }  // namespace ttg
}  // namespace parsec

#endif  // PARSEC_TTG_H_INCLUDED
