#ifndef PARSEC_TTG_H_INCLUDED
#define PARSEC_TTG_H_INCLUDED

#include "../ttg.h"

#include <array>
#include <cassert>
#include <functional>
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
#include <cstdlib>
#include <cstring>

extern "C" int parsec_ptg_update_runtime_task(parsec_taskpool_t *tp, int tasks);

namespace parsec {
  namespace ttg {

    class World {
     public:
      World(int *argc, char **argv[], int ncores) {
        ctx = parsec_init(ncores, argc, argv);
        tpool = (parsec_taskpool_t *)calloc(1, sizeof(parsec_taskpool_t));
        tpool->taskpool_id = 1;
        tpool->nb_tasks = 1;
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
      }

      void fence() {
        parsec_atomic_dec_32b((volatile uint32_t *)&tpool->nb_tasks);
        parsec_context_wait(ctx);
      }

      auto *context() { return ctx; }
      auto *execution_stream() { return es; }
      auto *taskpool() { return tpool; }

      void increment_created() { parsec_atomic_inc_32b(&created_counter()); }
      void increment_sent_to_sched() { parsec_atomic_inc_32b(&sent_to_sched_counter()); }

      uint32_t created() const { return this->created_counter(); }
      uint32_t sent_to_sched() const { return this->sent_to_sched(); }

     private:
      parsec_context_t *ctx = nullptr;
      parsec_execution_stream_t *es = nullptr;
      parsec_taskpool_t *tpool = nullptr;

      volatile uint32_t &created_counter() const {
        static volatile uint32_t created = 0;
        return created;
      }
      volatile uint32_t &sent_to_sched_counter() const {
        static volatile uint32_t sent_to_sched = 0;
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
        throw "parsec::ttg::set_default_world() must be called before use";
      }
    }
    inline void set_default_world(World &world) { detail::default_world_accessor() = &world; }
    inline void set_default_world(World *world) { detail::default_world_accessor() = world; }

    namespace detail {
      /// the default keymap implementation maps key to std::hash{}(key) % nproc
      template <typename keyT>
      class default_keymap {
       public:
        default_keymap(World &world = get_default_world()) : nproc(world.size()) {}
        template <typename... Args>
        auto operator()(const keyT &key) const {
          return std::hash<keyT>{}(key) % nproc;
        }

       private:
        int nproc;
      };
    }  // namespace detail

  }  // namespace ttg
}  // namespace parsec

extern "C" {
typedef struct my_op_s {
  parsec_task_t parsec_task;
  uint32_t in_data_count;
  parsec_hash_table_item_t op_ht_item;
  void (*function_template_class_ptr)(void *);
  void *object_ptr;
  void (*static_set_arg)(int, int);
  uint64_t key;
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
    uint64_t unique_hash<uint64_t, uint64_t>(const uint64_t &t) {
      return t;
    }
  }  // namespace overload
}  // namespace ttg

static uint32_t parsec_tasks_hash_fct(uintptr_t key, uint32_t hash_size, void *data) {
  /* Use all the bits of the 64 bits key, project on the lowest base bits (0 <= hash < 1024) */
  (void)data;
  int b = 0, base = 10;     /* We start at 10, as size is 1024 at least */
  uint32_t mask = 0x3FFULL; /* same thing: assume size is 1024 at least */
  uint32_t h = key;
  while (hash_size != (1u << base)) {
    assert(base < 32);
    base++;
    mask = (mask << 1) | 1;
  }
  while (b < 64) {
    b += base;
    h ^= key >> b;
  }
  return (uint32_t)(key & mask);
}

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
    inline void ttg_fence(World &world) { world.fence(); }
    inline void ttg_sum(World &world, double &value) {
      double result = 0.0;
      MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, world.comm());
      value = result;
    }

    struct ParsecBaseOp {
     protected:
      //  static std::map<int, ParsecBaseOp*> function_id_to_instance;
      parsec_hash_table_t tasks_table;
      parsec_task_class_t self;
    };
    // std::map<int, ParsecBaseOp*> ParsecBaseOp::function_id_to_instance = {};

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
        // probably not a godo way forward since it appears that operator* always return an lvalue ref.
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

      using input_values_tuple_type = std::tuple<data_wrapper_t<input_valueTs>...>;
      using input_terminals_type = std::tuple<::ttg::In<keyT, input_valueTs>...>;
      using input_edges_type = std::tuple<::ttg::Edge<keyT, std::decay_t<input_valueTs>>...>;

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

      World &world;
      std::function<std::int64_t(const keyT &)> keymap;

     protected:
      World &get_world() { return world; }

     private:
      struct OpArgs {
        int counter;                      // Tracks the number of arguments set
        std::array<bool, numins> argset;  // Tracks if a given arg is already set;
        input_values_tuple_type t;        // The input values
        derivedT *derived;                // Pointer to derived class instance
        keyT key;                         // Task key

        OpArgs() : counter(numins), argset(), t() { std::fill(argset.begin(), argset.end(), false); }

        void run() { derived->op(key, t, derived->output_terminals); }

        virtual ~OpArgs() {}  // Will be deleted via TaskInterface*
      };

      static void static_op(parsec_task_t *my_task) {
        my_op_t *task = (my_op_t *)my_task;
        derivedT *obj = (derivedT *)task->object_ptr;
        if (obj->tracing()) {
          PrintThread{} << obj->get_name() << " : " << keyT(task->key) << ": executing" << std::endl;
        }
        obj->op(keyT(task->key), std::move(*static_cast<input_values_tuple_type *>(task->user_tuple)),
                obj->output_terminals);
        if (obj->tracing())
          PrintThread{} << obj->get_name() << " : " << keyT(task->key) << ": done executing" << std::endl;
      }

      static void static_op_noarg(parsec_task_t *my_task) {
        my_op_t *task = (my_op_t *)my_task;
        derivedT *obj = (derivedT *)task->object_ptr;
        obj->op(keyT(task->key), std::tuple<>(), obj->output_terminals);
      }

      using cacheT = std::map<keyT, OpArgs>;
      cacheT cache;

      // Used to set the i'th argument
      template <std::size_t i, typename T>
      void set_arg(const keyT &key, T &&value) {
        using valueT = data_unwrapped_t<typename std::tuple_element<i, input_values_tuple_type>::type>;

        if (tracing()) PrintThread{} << get_name() << " : " << key << ": setting argument : " << i << std::endl;

        using ::ttg::unique_hash;
        uint64_t hk = unique_hash<uint64_t>(key);
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
            parsec_hash_table_nolock_insert(&tasks_table, &newtask->op_ht_item);
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
            world.increment_created();
            parsec_atomic_inc_32b((volatile uint32_t *)&world.taskpool()->nb_tasks);
            task = newtask;
            if (tracing()) PrintThread{} << get_name() << " : " << key << ": creating task" << std::endl;
          }
        }

        assert(task->key == hk);

        if (NULL != task->parsec_task.data[i].data_in) {
          std::cerr << get_name() << " : " << key << ": error argument is already set : " << i << std::endl;
          throw "bad set arg";
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

        int count = parsec_atomic_inc_32b(&task->in_data_count);
        assert(count <= self.dependencies_goal);

        if (count == self.dependencies_goal) {
          world.increment_sent_to_sched();
          parsec_execution_stream_t *es = world.execution_stream();
          if (tracing()) PrintThread{} << get_name() << " : " << key << ": invoking op" << std::endl;
          __parsec_schedule(es, &task->parsec_task, 0);
          parsec_hash_table_remove(&tasks_table, hk);
        }
      }

      // Used to generate tasks with no input arguments
      void set_arg_empty(const keyT &key) {
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
        task->key = unique_hash<uint64_t>(key);
        task->parsec_task.data[0].data_in = static_cast<parsec_data_copy_t *>(NULL);
        __parsec_schedule(es, &task->parsec_task, 0);
      }

      // Used by invoke to set all arguments associated with a task
      template <size_t... IS>
      void set_args(std::index_sequence<IS...>, const keyT &key, const input_values_tuple_type &args) {
        int junk[] = {0, (set_arg<IS>(key, Op::get<IS>(args)), 0)...};
        junk[0]++;
      }

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

      void fence() override { get_default_world().fence(); }

     public:
      template <typename keymapT = detail::default_keymap<keyT>>
      Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         World &world, keymapT &&keymap_ = keymapT())
          : ::ttg::OpBase(name, numins, numouts)
          , world(world)
          // if using default keymap, rebind to the given world
          , keymap(std::is_same<keymapT, detail::default_keymap<keyT>>::value
                       ? decltype(keymap)(detail::default_keymap<keyT>(world))
                       : decltype(keymap)(std::forward<keymapT>(keymap_))) {
        // Cannot call these in base constructor since terminals not yet constructed
        if (innames.size() != std::tuple_size<input_terminals_type>::value)
          throw "parsec::ttg::OP: #input names != #input terminals";
        if (outnames.size() != std::tuple_size<output_terminalsT>::value)
          throw "parsec::ttg::OP: #output names != #output terminals";

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
          flow->flow_flags = FLOW_ACCESS_RW;
          flow->dep_in[0] = NULL;
          flow->dep_out[0] = NULL;
          flow->flow_index = i;
          flow->flow_datatype_mask = (1 << i);
          *((parsec_flow_t **)&(self.in[i])) = flow;
        }
        *((parsec_flow_t **)&(self.in[i])) = NULL;

        for (i = 0; i < numouts; i++) {
          parsec_flow_t *flow = new parsec_flow_t;
          flow->name = strdup((std::string("flow out") + std::to_string(i)).c_str());
          flow->sym_type = SYM_INOUT;
          flow->flow_flags = FLOW_ACCESS_READ;
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

        parsec_hash_table_init(&tasks_table, offsetof(my_op_t, op_ht_item), 1024, parsec_tasks_hash_fct, NULL);
      }

      template <typename keymapT = detail::default_keymap<keyT>>
      Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         keymapT &&keymap = keymapT(get_default_world()))
          : Op(name, innames, outnames, get_default_world(), std::forward<keymapT>(keymap)) {}

      template <typename keymapT = detail::default_keymap<keyT>>
      Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames, World &world,
         keymapT &&keymap_ = keymapT())
          : Op(name, innames, outnames, world, std::forward<keymapT>(keymap_)) {
        connect_my_inputs_to_incoming_edge_outputs(std::make_index_sequence<numins>{}, inedges);
        connect_my_outputs_to_outgoing_edge_inputs(std::make_index_sequence<numouts>{}, outedges);
      }
      template <typename keymapT = detail::default_keymap<keyT>>
      Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         keymapT &&keymap = keymapT(get_default_world()))
          : Op(inedges, outedges, name, innames, outnames, get_default_world(), std::forward<keymapT>(keymap)) {}

      // Destructor checks for unexecuted tasks
      ~Op() {
        if (cache.size() != 0) {
          int rank = 0;
          std::cerr << rank << ":"
                    << "warning: unprocessed tasks in destructor of operation '" << get_name() << "'" << std::endl;
          std::cerr << rank << ":"
                    << "   T => argument assigned     F => argument unassigned" << std::endl;
          int nprint = 0;
          for (auto item : cache) {
            if (nprint++ > 10) {
              std::cerr << "   etc." << std::endl;
              break;
            }
            std::cerr << rank << ":"
                      << "   unused: " << item.first << " : ( ";
            for (std::size_t i = 0; i < numins; i++) std::cerr << (item.second.argset[i] ? "T" : "F") << " ";
            std::cerr << ")" << std::endl;
          }
        }
        parsec_hash_table_fini(&tasks_table);
        parsec_mempool_destruct(&mempools);
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
        parsec_atomic_inc_32b((volatile uint32_t *)&world.taskpool()->nb_tasks);
        set_args(std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{}, key, args);
      }

      // Manual injection of a task that has no arguments
      void invoke(const keyT &key) {
        // That task is going to complete, so count it as to execute
        parsec_atomic_inc_32b((volatile uint32_t *)&world.taskpool()->nb_tasks);
        set_arg_empty(key);
      }
    };

#include "../wrap.h"

  }  // namespace ttg
}  // namespace parsec

#endif  // PARSEC_TTG_H_INCLUDED
