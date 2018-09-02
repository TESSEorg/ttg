#ifndef PARSEC_TTG_H_INCLUDED
#define PARSEC_TTG_H_INCLUDED

#include "../ttg.h"
#include "../util/meta.h"

#include <array>
#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include <parsec.h>
#include <parsec/class/parsec_hash_table.h>
#include <parsec/data_internal.h>
#include <parsec/devices/device.h>
#include <parsec/execution_stream.h>
#include <parsec/interfaces/interface.h>
#include <parsec/parsec_internal.h>
#include <parsec/scheduling.h>
#include <parsec/remote_dep.h>
#include <cstdlib>
#include <cstring>

extern "C" int parsec_ptg_update_runtime_task(parsec_taskpool_t *tp, int tasks);
extern "C" int remote_dep_dequeue_send(int rank, parsec_remote_deps_t* deps);

namespace parsec {
  namespace ttg {

    class World {
     public:
      World(int *argc, char **argv[], int ncores) {
        ctx = parsec_init(ncores, argc, argv);
        tpool = (parsec_taskpool_t *)calloc(1, sizeof(parsec_taskpool_t));
        tpool->taskpool_id = 1;
        tpool->nb_tasks = 0;
        tpool->nb_pending_actions = 1;
        tpool->update_nb_runtime_task = parsec_ptg_update_runtime_task;
        es = ctx->virtual_processes[0]->execution_streams[0];
      }

      ~World() { parsec_fini(&ctx); }

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
        int ret = parsec_context_start(ctx);
        if(ret != 0) throw std::runtime_error("TTG: parsec_context_start failed");
      }

      void fence() {
          int ws, mr;
          MPI_Comm_size(MPI_COMM_WORLD, &ws);
          if(ws > 1) {
              MPI_Comm_rank(MPI_COMM_WORLD, &mr);
              fprintf(stderr, "On rank %d: (very) poor man's fence: giving 10s to complete before entering the wait\n", mr);
              sleep(10);
          }
          parsec_context_wait(ctx);
      }

      auto *context() { return ctx; }
      auto *execution_stream() { return es; }
      auto *taskpool() { return tpool; }

      void increment_created() { parsec_atomic_fetch_inc_int32(&created_counter()); }
      void increment_sent_to_sched() { parsec_atomic_fetch_inc_int32(&sent_to_sched_counter()); }

      int32_t created() const { return this->created_counter(); }
      int32_t sent_to_sched() const { return this->sent_to_sched_counter(); }

     private:
      parsec_context_t *ctx = nullptr;
      parsec_execution_stream_t *es = nullptr;
      parsec_taskpool_t *tpool = nullptr;

      volatile int32_t &created_counter() const {
        static volatile int32_t created = 0;
        return created;
      }
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

    /// the default keymap implementation maps key to std::hash{}(key) % nproc
    template <typename keyT>
    struct default_keymap : ::ttg::detail::default_keymap_impl<keyT> {
     public:
      default_keymap() = default;
      default_keymap(World &world) : ::ttg::detail::default_keymap_impl<keyT>(world.size()) {}
    };

  }  // namespace ttg
}  // namespace parsec

extern "C" {
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
  void (*function_template_class_ptr)(void *);
  void *object_ptr;
  void (*static_set_arg)(int, int);
  parsec_key_t key;
  void *user_tuple; /* user_tuple will past the end of my_op_s (to allow for proper alignment)
                     * This points to the beginning of the tuple. */
} my_op_t;

static parsec_hook_return_t hook(struct parsec_execution_stream_s *es, parsec_task_t *task);
}

static parsec_hook_return_t do_nothing(parsec_execution_stream_t *es, parsec_task_t *task) {
  (void)es;
  (void)task;
  return PARSEC_HOOK_RETURN_DONE;
}

static parsec_hook_return_t hook(struct parsec_execution_stream_s *es, parsec_task_t *task) {
  my_op_t *me = (my_op_t *)task;
  me->function_template_class_ptr(task);
  (void)es;
  return PARSEC_HOOK_RETURN_DONE;
}

namespace ttg {
  namespace overload {
    template <>
    inline parsec_key_t unique_hash<parsec_key_t, uint64_t>(const uint64_t &t) {
      return t;
    }
    template <>
    inline parsec_key_t unique_hash<parsec_key_t, int>(const int &t) {
      return t;
    }
    template <>
    inline parsec_key_t unique_hash<parsec_key_t, ::ttg::Void>(const ::ttg::Void &t) {
      return 0;
    }
  }  // namespace overload
}  // namespace ttg

static uint64_t parsec_tasks_hash_fct(parsec_key_t key, int nb_bits, void *data) {
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

class PrintThread : public std::ostringstream {
 public:
  PrintThread() = default;

  ~PrintThread() {
    std::lock_guard<std::mutex> guard(_mutexPrint);
    std::cout << this->str();
    std::cout.flush();
  }

 private:
  static std::mutex _mutexPrint;
};

std::mutex PrintThread::_mutexPrint{};
namespace parsec {
  namespace ttg {
    typedef void (*static_set_arg_fct_type)(void *, size_t, ::ttg::OpBase*);
    static std::map<uint64_t, 
                    std::pair<static_set_arg_fct_type,
                              ::ttg::OpBase* > > static_id_to_op_map;

    static void static_unpack_msg(void *data, size_t size) {
        void (*static_set_arg_fct)(void *, size_t, ::ttg::OpBase *);
        typedef struct {
            uint64_t op_id;
        } msg_header_t;
        msg_header_t *msg = static_cast<msg_header_t*>(data);
        auto op_pair = static_id_to_op_map.at( msg->op_id );
        static_set_arg_fct = op_pair.first;
        static_set_arg_fct(data, size, op_pair.second);
    }

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
      World *default_world = &get_default_world();
      delete default_world;
      set_default_world(nullptr);
      MPI_Finalize();
    }
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
    template <typename Key, typename Value, typename Enabler = void>
    struct msg_t;

    template <typename Key, typename Value>
    struct msg_t<Key,Value,std::enable_if_t<::ttg::meta::is_none_void_v<Key,Value>>> {
      uint64_t op_id;
      std::size_t param_id;
      Key key;
      Value val;
      msg_t() = default;
      msg_t(uint64_t op_id, std::size_t param_id, const Key& key, const Value& val) : op_id(op_id), param_id(param_id), key(key), val(val) {}
    };

    template <typename Key, typename Value>
    struct msg_t<Key,Value,std::enable_if_t<::ttg::meta::is_void_v<Key> && !::ttg::meta::is_void_v<Value>>> {
      uint64_t op_id;
      std::size_t param_id;
      Value val;
      msg_t() = default;
      msg_t(uint64_t op_id, std::size_t param_id, const ::ttg::Void&, const Value& val) : op_id(op_id), param_id(param_id), val(val) {}
    };

    template <typename Key, typename Value>
    struct msg_t<Key,Value,std::enable_if_t<!::ttg::meta::is_void_v<Key> && ::ttg::meta::is_void_v<Value>>> {
      uint64_t op_id;
      std::size_t param_id;
      keyT key;
      msg_t() = default;
      msg_t(uint64_t op_id, std::size_t param_id, const Key& key, const ::ttg::Void&) : op_id(op_id), param_id(param_id), key(key) {}
    };

    template <typename Key, typename Value>
    struct msg_t<Key,Value,std::enable_if_t<::ttg::meta::is_all_void_v<Key,Value>>> {
      uint64_t op_id;
      std::size_t param_id;
      msg_t() = default;
      msg_t(uint64_t op_id, std::size_t param_id, const ::ttg::Void&, const ::ttg::Void&) : op_id(op_id), param_id(param_id) {}
    };

    }  // namespace detail

    template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
    class Op : public ::ttg::OpBase, ParsecBaseOp {
     private:
      using opT = Op<keyT, output_terminalsT, derivedT, input_valueTs...>;
      parsec_mempool_t mempools;
      std::map<std::pair<int, int>, int> mempools_index;

     public:
      static constexpr int numins = sizeof...(input_valueTs);                    // number of input arguments
      static constexpr int numouts = std::tuple_size<output_terminalsT>::value;  // number of outputs or
      // results

      // PaRSEC for now passes data as tuple of ptrs (datacopies have these pointers also)
      // N.B. perhaps make data_wrapper_t<T> = parsec_data_copy_t (rather, a T-dependent wrapper around it to automate
      // the casts that will be inevitably needed)
      template <typename T>
      using data_wrapper_t = T *;
      template <typename T>
      struct data_wrapper_traits {
        using type = T;
      };
      template <typename T>
      struct data_wrapper_traits<data_wrapper_t<T>> {
        using type = T;
      };
      template <typename T>
      struct data_wrapper_traits<data_wrapper_t<T> &> {
        using type = T &;
      };
      template <typename T>
      struct data_wrapper_traits<const data_wrapper_t<T> &> {
        using type = const T &;
      };
      template <typename T>
      struct data_wrapper_traits<data_wrapper_t<T> &&> {
        using type = T &&;
      };
      template <typename T>
      struct data_wrapper_traits<const data_wrapper_t<T> &&> {
        using type = const T &&;
      };
      template <typename T>
      using data_unwrapped_t = typename data_wrapper_traits<T>::type;

      template <typename Wrapper>
      static auto &&unwrap(Wrapper &&wrapper) {
        // what will be the return type? Either const or non-const lvalue ref
        // * (T*&) => T&
        // * (const T*&) => const T&
        // * (T*&&) => T&
        // * (const T*&&) => const T&
        // so the input type alone is not enough, will have to cast to the desired type in unwrap_to
        // or have to use a struct instead of a pointer and overload indirection operator (operator*)
        // probably not a good way forward since it appears that operator* always return an lvalue ref.
        // not produce an rvalue ref need an explicit cast.

        return *wrapper;
      }
      // extend this to tell PaRSEC how the data is being used (even is this is to increment the counter only)
      template <typename Result, typename Wrapper>
      static Result unwrap_to(Wrapper &&wrapper) {
        return static_cast<Result>(unwrap(std::forward<Wrapper>(wrapper)));
      }
      // this wraps a (moved) copy T must be copyable and/or movable (depends on the use)
      template <typename T>
      static data_wrapper_t<std::remove_reference_t<T>> wrap(T &&data) {
        return new std::remove_reference_t<T>(std::forward<T>(data));
      }
      template <typename T>
      static data_wrapper_t<T> wrap(data_wrapper_t<T> &&data) {
        return std::move(data);
      }
      template <typename T>
      static const data_wrapper_t<T> &wrap(const data_wrapper_t<T> &data) {
        return data;
      }

      using input_terminals_type = std::tuple<::ttg::In<keyT, input_valueTs>...>;
      using input_edges_type = std::tuple<::ttg::Edge<keyT, std::decay_t<input_valueTs>>...>;
      static_assert(::ttg::meta::is_none_void_v<input_valueTs...> || std::tuple_size_v<input_terminals_type> == 1, "only single void input can be handled (i.e. can't mix void and nonvoid inputs)");
      using input_values_tuple_type = std::conditional_t<::ttg::meta::is_none_void_v<input_valueTs...>,std::tuple<data_wrapper_t<std::decay_t<input_valueTs>>...>,std::tuple<>>;
      using input_unwrapped_values_tuple_type = std::tuple<std::decay_t<input_valueTs>...>;

      using output_terminals_type = output_terminalsT;
      using output_edges_type = typename ::ttg::terminals_to_edges<output_terminalsT>::type;

      // these are aware of result type, can communicate this info back to PaRSEC
      template <std::size_t i, typename resultT>
      static resultT get(input_values_tuple_type &intuple) {
        return unwrap_to<resultT>(std::get<i>(intuple));
      };
      template <std::size_t i, typename resultT>
      static resultT get(const input_values_tuple_type &intuple) {
        return unwrap_to<resultT>(std::get<i>(intuple));
      };
      template <std::size_t i, typename resultT>
      static resultT get(input_values_tuple_type &&intuple) {
        return unwrap_to<resultT>(std::get<i>(intuple));
      };
      template <std::size_t i>
      static auto &get(input_values_tuple_type &intuple) {
        return unwrap(std::get<i>(intuple));
      };
      template <std::size_t i>
      static const auto &get(const input_values_tuple_type &intuple) {
        return unwrap(std::get<i>(intuple));
      };
      template <std::size_t i>
      static auto &&get(input_values_tuple_type &&intuple) {
        return unwrap(std::get<i>(intuple));
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

      static void static_op(parsec_task_t *my_task) {
        my_op_t *task = (my_op_t *)my_task;
        derivedT *obj = (derivedT *)task->object_ptr;
        if (obj->tracing()) {
          PrintThread{} << obj->get_name() << " : " << keyT((uintptr_t)task->key) << ": executing" << std::endl;
        }

        if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_unwrapped_values_tuple_type>) {
          obj->op(keyT((uintptr_t) task->key), std::move(*static_cast<input_values_tuple_type *>(task->user_tuple)),
                  obj->output_terminals);
        } else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_unwrapped_values_tuple_type>) {
          obj->op(keyT((uintptr_t) task->key), obj->output_terminals);
        } else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_unwrapped_values_tuple_type>) {
          obj->op(std::move(*static_cast<input_values_tuple_type *>(task->user_tuple)),
                  obj->output_terminals);
        } else {
          obj->op(obj->output_terminals);
        }

        if (obj->tracing())
          PrintThread{} << obj->get_name() << " : " << keyT((uintptr_t)task->key) << ": done executing" << std::endl;
      }

      static void static_op_noarg(parsec_task_t *my_task) {
        my_op_t *task = (my_op_t *)my_task;
        derivedT *obj = (derivedT *)task->object_ptr;
        if constexpr(!::ttg::meta::is_void_v<keyT>) {
          obj->op(keyT((uintptr_t) task->key), obj->output_terminals);
        } else {
          obj->op(obj->output_terminals);
        }
      }

     protected:
      static void static_set_arg(void *data, std::size_t size, ::ttg::OpBase *bop) {
          typedef struct {
              uint64_t op_id;
              std::size_t param_id;
          } header_t;
          assert(size >= sizeof(header_t) &&
                 "Trying to unpack as message that does not hold enough bytes to represent a single header");
          header_t *hd = static_cast<header_t*>(data);
          derivedT *obj = reinterpret_cast<derivedT*>(bop);
          auto member = obj->set_arg_from_msg_fcts[hd->param_id];
          (obj->*member)(data, size);
      }

      template <std::size_t i>
      void set_arg_from_msg(void *data, std::size_t size) {
          using valueT = typename std::tuple_element<i, input_terminals_type>::type::value_type;
          using msg_t = detail::msg_t<keyT,valueT>;
          assert(size == sizeof(msg_t) &&
                 "Trying to unpack as message that does not hold the right number of bytes for this type");
          msg_t *msg = static_cast<msg_t*>(data);
          if constexpr (::ttg::meta::is_none_void_v<keyT,valueT>)
            set_arg<i, valueT>(msg->key, std::forward<valueT>(msg->val));
          else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_void_v<valueT>)
            set_arg<i, valueT>(msg->key);
          else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_void_v<valueT>)
            set_arg<i, valueT>(std::forward<valueT>(msg->val));
          else if constexpr (::ttg::meta::is_all_void_v<keyT,valueT>)
            set_arg<i, valueT>();
      }
        
      // Used to set the i'th argument
      template <std::size_t i, typename T>
      void set_arg_local(const keyT &key, T &&value) {
        using valueT = data_unwrapped_t<typename std::tuple_element<i, input_values_tuple_type>::type>;

        if (tracing()) PrintThread{} << get_name() << " : " << key << ": setting argument : " << i << std::endl;

        using ::ttg::unique_hash;
        parsec_key_t hk = unique_hash<parsec_key_t>(key);
        my_op_t *task = NULL;
        constexpr const std::size_t alignment_of_input_tuple = std::alignment_of<input_values_tuple_type>::value;
        if (NULL == (task = (my_op_t *)parsec_hash_table_find(&tasks_table, hk))) {
          my_op_t *newtask;
          parsec_execution_stream_s *es = world.execution_stream();
          parsec_thread_mempool_t *mempool =
              &mempools.thread_mempools[mempools_index[std::pair<int, int>(es->virtual_process->vp_id, es->th_id)]];
          newtask = (my_op_t *)parsec_thread_mempool_allocate(mempool);
          memset((void *)newtask, 0, sizeof(my_op_t) + sizeof(input_values_tuple_type) + alignment_of_input_tuple);
          newtask->parsec_task.mempool_owner = mempool;

          OBJ_CONSTRUCT(&newtask->parsec_task, parsec_list_item_t);
          newtask->parsec_task.task_class = &this->self;
          newtask->parsec_task.taskpool = world.taskpool();
          newtask->parsec_task.status = PARSEC_TASK_STATUS_HOOK;
          newtask->in_data_count = 0;

          newtask->function_template_class_ptr = reinterpret_cast<void (*)(void *)>(&Op::static_op);
          newtask->object_ptr = static_cast<derivedT *>(this);
          newtask->key = hk;

          parsec_mfence();
          parsec_hash_table_lock_bucket(&tasks_table, hk);
          if (NULL != (task = (my_op_t *)parsec_hash_table_nolock_find(&tasks_table, hk))) {
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
            free(newtask);
          } else {
            newtask->op_ht_item.key = hk;
            parsec_atomic_fetch_inc_int32(&world.taskpool()->nb_tasks);
            world.increment_created();
            parsec_hash_table_nolock_insert(&tasks_table, &newtask->op_ht_item);
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
            task = newtask;
            if (tracing()) PrintThread{} << get_name() << " : " << key << ": creating task" << std::endl;
          }
        }

        assert(task->key == hk);

        if (NULL != task->parsec_task.data[i].data_in) {
          std::cerr << get_name() << " : " << key << ": error argument is already set : " << i << std::endl;
          throw std::logic_error("bad set arg");
        }

        void *task_body_tail_ptr =
            reinterpret_cast<void *>(reinterpret_cast<intptr_t>(task) + static_cast<intptr_t>(sizeof(my_op_t)));
        std::size_t task_body_tail_size = sizeof(input_values_tuple_type) + alignment_of_input_tuple;
        task->user_tuple = std::align(alignment_of_input_tuple, sizeof(input_values_tuple_type), task_body_tail_ptr,
                                      task_body_tail_size);
        assert(task->user_tuple != nullptr);
        input_values_tuple_type *tuple = static_cast<input_values_tuple_type *>(task->user_tuple);

        // Q. do we need to worry about someone calling set_arg "directly", i.e. through BaseOp::send and passing their
        // own (non-PaRSEC) value rather than value owned by PaRSEC?
        // N.B. wrap(data_wrapper_t) does nothing, so no double wrapping here
        std::get<i>(*tuple) = data_wrapper_t<valueT>(wrap(std::forward<T>(value)));
        parsec_data_copy_t *copy = OBJ_NEW(parsec_data_copy_t);
        task->parsec_task.data[i].data_in = copy;
        copy->device_private = (void *)(std::get<i>(*tuple));  // tuple holds pointers already
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

        int32_t count = parsec_atomic_fetch_inc_int32(&task->in_data_count)+1;
        assert(count <= self.dependencies_goal);

        if (count == self.dependencies_goal) {
          world.increment_sent_to_sched();
          parsec_execution_stream_t *es = world.execution_stream();
          if (tracing()) PrintThread{} << get_name() << " : " << key << ": invoking op" << std::endl;
          parsec_hash_table_remove(&tasks_table, hk);
          __parsec_schedule(es, &task->parsec_task, 0);
        }
      }

      // Used to set the i'th argument
      template <std::size_t i, typename T>
      void set_arg(const keyT &key, T &&value) {
        using valueT = data_unwrapped_t<typename std::tuple_element<i, input_values_tuple_type>::type>;
        auto owner = keymap(key);
        if( owner == ttg_default_execution_context().rank() ) {
            set_arg_local<i>(key, std::forward<T>(value));
            return;
        }
        // the target task is remote. Pack the information and send it to
        // the corresponding peer.
        // TODO do we need to copy value?
        using msg_t = detail::msg_t<keyT,valueT>;
        msg_t msg(get_instance_id(), i, key, value);
        parsec_remote_deps_t* deps = (parsec_remote_deps_t*)remote_deps_allocate(&parsec_remote_dep_context.freelist);
        deps->root = get_default_world().rank();
        deps->outgoing_mask = (1 << i);
        deps->max_priority = 0;
        deps->taskpool = world.taskpool();
        struct remote_dep_output_param_s* output = &deps->output[0];
        int _array_pos = owner / (8 * sizeof(uint32_t));
        int _array_mask = 1 << (owner % (8 * sizeof(uint32_t)));
        output->rank_bits[_array_pos] |= _array_mask;
        output->deps_mask |= (1 << i);
        output->count_bits = 1;
        output->priority = 0;

        deps->msg.deps = (remote_dep_datakey_t)deps;
        deps->msg.output_mask = (1 << i);
        deps->msg.tag = 10;  // TODO: change me
        deps->msg.taskpool_id = deps->taskpool->taskpool_id;
        deps->msg.task_class_id = this->self.task_class_id;
        deps->msg.length = (sizeof(msg_t) + sizeof(int) - 1) / sizeof(int);
        memcpy(deps->msg.locals, &msg, sizeof(msg_t));

        remote_dep_dequeue_send(owner, deps);
      }

      // Used to generate tasks with no input arguments
      template <typename Key = keyT> std::enable_if_t<!::ttg::meta::is_void_v<Key>,void> set_arg_empty(const keyT &key) {
        if (tracing()) std::cout << get_name() << " : " << key << ": invoking op " << std::endl;
        // create PaRSEC task
        // and give it to the scheduler
        my_op_t *task;
        parsec_execution_stream_s *es = world.execution_stream();
        parsec_thread_mempool_t *mempool =
            &mempools.thread_mempools[mempools_index[std::pair<int, int>(es->virtual_process->vp_id, es->th_id)]];
        task = (my_op_t *)parsec_thread_mempool_allocate(mempool);
        memset((void *)task, 0, sizeof(my_op_t));
        task->parsec_task.mempool_owner = mempool;

        OBJ_CONSTRUCT(task, parsec_list_item_t);
        task->parsec_task.task_class = &this->self;
        task->parsec_task.taskpool = world.taskpool();
        task->parsec_task.status = PARSEC_TASK_STATUS_HOOK;

        task->function_template_class_ptr = reinterpret_cast<void (*)(void *)>(&Op::static_op_noarg);
        task->object_ptr = static_cast<derivedT *>(this);
        using ::ttg::unique_hash;
        task->key = unique_hash<parsec_key_t>(key);
        task->parsec_task.data[0].data_in = static_cast<parsec_data_copy_t *>(NULL);
        __parsec_schedule(es, &task->parsec_task, 0);
      }

      // Used to generate tasks with no input arguments
      template <typename Key = keyT> std::enable_if_t<::ttg::meta::is_void_v<Key>,void> set_arg_empty() {
        if (tracing()) std::cout << get_name() << " : invoking op " << std::endl;
        // create PaRSEC task
        // and give it to the scheduler
        my_op_t *task;
        parsec_execution_stream_s *es = world.execution_stream();
        parsec_thread_mempool_t *mempool =
            &mempools.thread_mempools[mempools_index[std::pair<int, int>(es->virtual_process->vp_id, es->th_id)]];
        task = (my_op_t *)parsec_thread_mempool_allocate(mempool);
        memset((void *)task, 0, sizeof(my_op_t));
        task->parsec_task.mempool_owner = mempool;

        OBJ_CONSTRUCT(task, parsec_list_item_t);
        task->parsec_task.task_class = &this->self;
        task->parsec_task.taskpool = world.taskpool();
        task->parsec_task.status = PARSEC_TASK_STATUS_HOOK;

        task->function_template_class_ptr = reinterpret_cast<void (*)(void *)>(&Op::static_op_noarg);
        task->object_ptr = static_cast<derivedT *>(this);
        using ::ttg::unique_hash;
        task->key = unique_hash<parsec_key_t>(0);
        task->parsec_task.data[0].data_in = static_cast<parsec_data_copy_t *>(NULL);
        __parsec_schedule(es, &task->parsec_task, 0);
      }

      // Used by invoke to set all arguments associated with a task
      template <size_t... IS>
      void set_args(std::index_sequence<IS...>, const keyT &key, const input_values_tuple_type &args) {
        int junk[] = {0, (set_arg<IS>(key, Op::get<IS>(args)), 0)...};
        junk[0]++;
      }

     public:

      /// sets stream size for input \c i
      /// \param size positive integer that specifies the stream size
      template <std::size_t i>
      void set_argstream_size(const keyT &key, std::size_t size) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
        assert(size > 0 && "Op::set_argstream_size(key,size) called with size=0");

        // body
        const auto owner = keymap(key);
        abort();  // TODO implement set_argstream_size
      }

      /// finalizes stream for input \c i
      template <std::size_t i>
      void finalize_argstream(const keyT &key) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::finalize_argstream called on nonstreaming input terminal");

        // body
        const auto owner = keymap(key);
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
        using move_callbackT = std::function<void(const keyT &, valueT &&)>;
        using send_callbackT = std::function<void(const keyT &, const valueT &)>;
        auto move_callback = [this](const keyT &key, valueT &&value) {
          // std::cout << "move_callback\n";
          set_arg<i, valueT>(key, std::forward<valueT>(value));
        };
        auto send_callback = [this](const keyT &key, const valueT &value) {
          // std::cout << "send_callback\n";
          set_arg<i, const valueT &>(key, value);
        };

        input.set_callback(send_callbackT(send_callback), move_callbackT(move_callback));
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
                 (std::is_const<typename std::tuple_element<IS, input_terminals_tupleT>::type>::value ? FLOW_ACCESS_READ
                                                                                                      : FLOW_ACCESS_RW),
             0)...};
        junk[0]++;
      }

      template <typename input_terminals_tupleT, typename flowsT>
      void initialize_flows(flowsT &&flows) {
        _initialize_flows<input_terminals_tupleT>(
            std::make_index_sequence<std::tuple_size<input_terminals_tupleT>::value>{}, flows);
      }

      void fence() override { get_default_world().fence(); }

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

        register_input_terminals(input_terminals, innames);
        register_output_terminals(output_terminals, outnames);

        register_input_callbacks(std::make_index_sequence<numins>{});

        register_static_op_function();

        int i;

        memset(&self, 0, sizeof(parsec_task_class_t));

        self.name = get_name().c_str();
        self.task_class_id = get_instance_id();
        self.nb_parameters = 0;
        self.nb_locals = 0;
        self.nb_flows = std::max((int)numins, (int)numouts);

        //    function_id_to_instance[self.task_class_id] = this;

        self.incarnations = (__parsec_chore_t *)malloc(2 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t *)self.incarnations)[0].type = PARSEC_DEV_CPU;
        ((__parsec_chore_t *)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[0].hook = hook;
        ((__parsec_chore_t *)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t *)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t *)self.incarnations)[1].hook = NULL;

        self.release_task = parsec_release_task_to_mempool_update_nbtasks;
        self.complete_execution = do_nothing;

        for (i = 0; i < numins; i++) {
          parsec_flow_t *flow = new parsec_flow_t;
          flow->name = strdup((std::string("flow in") + std::to_string(i)).c_str());
          flow->sym_type = SYM_INOUT;
          // see initialize_flows below
          // flow->flow_flags = FLOW_ACCESS_RW;
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
          flow->sym_type = SYM_INOUT;
          flow->flow_flags = FLOW_ACCESS_READ;  // does PaRSEC use this???
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
        // + alignment_of_input_tuple to allow alignment of input_values_tuple_type
        parsec_mempool_construct(&mempools, OBJ_CLASS(parsec_task_t),
                                 sizeof(my_op_t) + sizeof(input_values_tuple_type) + alignof(input_values_tuple_type),
                                 offsetof(parsec_task_t, mempool_owner), k);

        parsec_hash_table_init(&tasks_table, offsetof(my_op_t, op_ht_item), 10, parsec_tasks_hash_fcts, NULL);
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
        parsec_hash_table_fini(&tasks_table);
        parsec_mempool_destruct(&mempools);
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
      void invoke(const keyT &key, const input_values_tuple_type &args) {
        // That task is going to complete, so count it as to execute
        parsec_atomic_fetch_inc_int32(&world.taskpool()->nb_tasks);
        set_args(std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{}, key, args);
      }

      // Manual injection of a task that has no arguments
      template <typename Key = keyT> std::enable_if_t<!::ttg::meta::is_void_v<Key>,void> invoke(const keyT &key) {
        // That task is going to complete, so count it as to execute
        parsec_atomic_fetch_inc_int32(&world.taskpool()->nb_tasks);
        set_arg_empty(key);
      }

      // Manual injection of a task that has no key or arguments
      template <typename Key = keyT> std::enable_if_t<::ttg::meta::is_void_v<Key>,void> invoke() {
        // That task is going to complete, so count it as to execute
        parsec_atomic_fetch_inc_int32(&world.taskpool()->nb_tasks);
        set_arg_empty();
      }

      void make_executable() override { OpBase::make_executable(); }

      /// keymap accessor
      /// @return the keymap
      const decltype(keymap) &get_keymap() const { return keymap; }

      // Register the static_op function to associate it to instance_id
      void register_static_op_function(void) {
          static_id_to_op_map.insert( std::pair<uint64_t, 
                                                std::pair<void (*)(void *, size_t, 
                                                                   ::ttg::OpBase *), 
                                                ::ttg::OpBase*>>
                                      (get_instance_id(), 
                                       std::pair<void (*)(void *, size_t, ::ttg::OpBase *),
                                                 ::ttg::OpBase*>(&Op::static_set_arg, this) ));
      }
    };

#include "../wrap.h"

  }  // namespace ttg
}  // namespace parsec

#endif  // PARSEC_TTG_H_INCLUDED
