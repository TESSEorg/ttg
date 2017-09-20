#ifndef PARSEC_TTG_H_INCLUDED
#define PARSEC_TTG_H_INCLUDED

#include "../ttg.h"

#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <mutex>

#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/class/parsec_hash_table.h>
#include <parsec/devices/device.h>
#include <parsec/parsec_internal.h>
#include <parsec/scheduling.h>
#include <parsec/execution_stream.h>
#include <parsec/interfaces/interface.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
typedef struct my_op_s {
    parsec_task_t parsec_task;
    uint32_t in_data_count;
    parsec_hash_table_item_t op_ht_item;
    void (*function_template_class_ptr)(void*);
    void *object_ptr;
    void (*static_set_arg)(int, int);
    uint64_t key;
    void *user_tuple; /* user_tuple starts here, but overshoots the data structure
                       * It is declared as a void * so that the field is aligned with
                       * an addressable byte. */
} my_op_t;

namespace parsec {
namespace ttg {
    
extern parsec_taskpool_t* taskpool;
extern parsec_context_t *parsec;

}
}

static parsec_hook_return_t hook(struct parsec_execution_stream_s* es,
                                 parsec_task_t* task);
}

static parsec_hook_return_t do_nothing(parsec_execution_stream_t* es,
                                       parsec_task_t* task) {
  (void)es;
  (void)task;
  return PARSEC_HOOK_RETURN_DONE;
}

static parsec_hook_return_t hook(struct parsec_execution_stream_s* es,
                                 parsec_task_t* task) {
    my_op_t* me = (my_op_t*)task;
    me->function_template_class_ptr(task);
    (void)es;
    return PARSEC_HOOK_RETURN_DONE;
}

namespace ttg {
namespace overload {
template<> uint64_t unique_hash<uint64_t, uint64_t>(const uint64_t& t) {
  return t;
}
}
}

static uint32_t parsec_tasks_hash_fct(uintptr_t key, uint32_t hash_size, void *data)
{
    /* Use all the bits of the 64 bits key, project on the lowest base bits (0 <= hash < 1024) */
    (void)data;
    int b = 0, base = 10; /* We start at 10, as size is 1024 at least */
    uint32_t mask = 0x3FFULL; /* same thing: assume size is 1024 at least */
    uint32_t h = key;
    while( hash_size != (1u<<base) ) { assert(base < 32); base++; mask = (mask<<1)|1; }
    while( b < 64 ) {
        b += base;
        h ^= key >> b;
    }
    return (uint32_t)( key & mask);
}

class PrintThread: public std::ostringstream
{
public:
    PrintThread() = default;

    ~PrintThread()
    {
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
 
struct ParsecBaseOp {
 protected:
    //  static std::map<int, ParsecBaseOp*> function_id_to_instance;
    parsec_hash_table_t tasks_table;
    parsec_task_class_t self;
};
//std::map<int, ParsecBaseOp*> ParsecBaseOp::function_id_to_instance = {};

static void init(int cores, int *argc, char **argv[])
{
    parsec = parsec_init(cores, argc, argv);
    taskpool = (parsec_taskpool_t*)calloc(1, sizeof(parsec_taskpool_t));
    taskpool->taskpool_id = 1;
    taskpool->nb_tasks = 1;
    taskpool->nb_pending_actions = 1;
    taskpool->update_nb_runtime_task = parsec_ptg_update_runtime_task;
}

static void fini(void)
{
    parsec_fini(&parsec);
}

static void start(void)
{
    parsec_enqueue(parsec, taskpool);
    int ret = parsec_context_start(parsec);
}

extern volatile uint32_t created;
extern volatile uint32_t sent_to_sched;
 
template <typename keyT, typename output_terminalsT, typename derivedT,
          typename... input_valueTs>
class Op : public ::ttg::OpBase, ParsecBaseOp {
 private:
  using opT = Op<keyT, output_terminalsT, derivedT, input_valueTs...>;
  parsec_mempool_t mempools;
  std::map< std::pair<int,int>, int > mempools_index;

 public:
    void fence() {
        parsec_atomic_dec_32b((volatile uint32_t*)&taskpool->nb_tasks);
        parsec_context_wait(parsec);
    }
  
  static constexpr int numins =
      sizeof...(input_valueTs);  // number of input arguments
  static constexpr int numouts =
      std::tuple_size<output_terminalsT>::value;  // number of outputs or
                                                  // results

  using input_values_tuple_type = std::tuple<input_valueTs...>;
  using input_terminals_type = std::tuple<::ttg::In<keyT, input_valueTs>...>;
  using input_edges_type = std::tuple<::ttg::Edge<keyT, input_valueTs>...>;

  using output_terminals_type = output_terminalsT;
  using output_edges_type =
      typename ::ttg::terminals_to_edges<output_terminalsT>::type;

 private:
  input_terminals_type input_terminals;
  output_terminalsT output_terminals;

  struct OpArgs {
    int counter;                      // Tracks the number of arguments set
    std::array<bool, numins> argset;  // Tracks if a given arg is already set;
    input_values_tuple_type t;        // The input values
    derivedT* derived;                // Pointer to derived class instance
    keyT key;                         // Task key

    OpArgs() : counter(numins), argset(), t() { std::fill(argset.begin(), argset.end(), false); }

    void run() { derived->op(key, t, derived->output_terminals); }

    virtual ~OpArgs() {}  // Will be deleted via TaskInterface*
  };
  
  static void static_op(parsec_task_t* my_task) {
      my_op_t* task = (my_op_t*)my_task;
      derivedT* obj = (derivedT*)task->object_ptr;
      if (obj->tracing()) {
          PrintThread{} << obj->get_name() << " : " << keyT(task->key) << ": executing"
                        << std::endl;
      }
      obj->op(keyT(task->key),
              std::move(*static_cast<input_values_tuple_type*>((void*)&task->user_tuple)),
              obj->output_terminals);
      if (obj->tracing())
          PrintThread{} << obj->get_name() << " : " << keyT(task->key) << ": done executing"
                        << std::endl;
  }

  static void static_op_noarg(parsec_task_t* my_task) {
      my_op_t* task = (my_op_t*)my_task;
      derivedT* obj = (derivedT*)task->object_ptr;
      obj->op(keyT(task->key), std::tuple<>(), obj->output_terminals);
  }

  using cacheT = std::map<keyT, OpArgs>;
  cacheT cache;

  // Used to set the i'th argument
  template <std::size_t i, typename T>
  void set_arg(const keyT& key, T&& value) {
    using valueT =
        typename std::tuple_element<i, input_values_tuple_type>::type;

    if (tracing())
        PrintThread{} << get_name() << " : " << key << ": setting argument : " << i
                      << std::endl;

    using ::ttg::unique_hash;
    uint64_t hk = unique_hash<uint64_t>(key);
    my_op_t *task = NULL;
    if( NULL == (task = (my_op_t*)parsec_hash_table_find(&tasks_table, hk)) ) {
        my_op_t *newtask;
        parsec_execution_stream_s *es = parsec_my_execution_stream();
        parsec_thread_mempool_t *mempool = &mempools.thread_mempools[ mempools_index[std::pair<int,int>(es->virtual_process->vp_id, es->th_id)] ];
        newtask = (my_op_t *) parsec_thread_mempool_allocate(mempool);
        memset((void*)newtask, 0, sizeof(my_op_t));
        newtask->parsec_task.mempool_owner = mempool;
        
        OBJ_CONSTRUCT(&newtask->parsec_task, parsec_list_item_t);
        newtask->parsec_task.task_class = &this->self;
        newtask->parsec_task.taskpool = taskpool;
        newtask->parsec_task.status = PARSEC_TASK_STATUS_HOOK;
        newtask->in_data_count = 0;

        newtask->function_template_class_ptr =
            reinterpret_cast<void (*)(void*)>(&Op::static_op);
        newtask->object_ptr = static_cast<derivedT*>(this);
        newtask->key = hk;

        parsec_mfence();
        parsec_hash_table_lock_bucket(&tasks_table, hk);
        if( NULL != (task = (my_op_t*)parsec_hash_table_nolock_find(&tasks_table, hk)) ) {
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
            free(newtask);
        } else {
            newtask->op_ht_item.key = hk;
            parsec_hash_table_nolock_insert(&tasks_table, &newtask->op_ht_item);
            parsec_hash_table_unlock_bucket(&tasks_table, hk);
            parsec_atomic_inc_32b(&created);
            parsec_atomic_inc_32b((volatile uint32_t*)&taskpool->nb_tasks);
            task = newtask;
            if(tracing())
                PrintThread{} << get_name() << " : " << key << ": creating task"
                              << std::endl;
        }
    }

    assert(task->key == hk);

    if( NULL != task->parsec_task.data[i].data_in ) {
        std::cerr << get_name() << " : " << key
                << ": error argument is already set : " << i << std::endl;
        throw "bad set arg";
    }
    
    input_values_tuple_type *tuple = static_cast<input_values_tuple_type *>((void*)&task->user_tuple);
    new (&std::get<i>(*tuple)) valueT(std::forward<T>(value));
    parsec_data_copy_t* copy = OBJ_NEW(parsec_data_copy_t);
    task->parsec_task.data[i].data_in = copy;
    copy->device_private = (void*)(&std::get<i>(*tuple));
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
      parsec_atomic_inc_32b(&sent_to_sched);
      parsec_execution_stream_t *es = parsec_my_execution_stream();
      if (tracing())
          PrintThread{} << get_name() << " : " << key << ": invoking op"
                        << std::endl;
      __parsec_schedule(es, &task->parsec_task, 0);
      parsec_hash_table_remove(&tasks_table, hk);
    }
  }

  // Used to generate tasks with no input arguments
  void set_arg_empty(const keyT& key) {
    if (tracing())
      std::cout << get_name() << " : " << key << ": invoking op " << std::endl;
    // create PaRSEC task
    // and give it to the scheduler
    my_op_t *task;
    parsec_execution_stream_s *es = parsec_my_execution_stream();
    parsec_thread_mempool_t *mempool = &mempools.thread_mempools[ mempools_index[std::pair<int,int>(es->virtual_process->vp_id, es->th_id)] ];
    task = (my_op_t *) parsec_thread_mempool_allocate(mempool);
    memset((void*)task, 0, sizeof(my_op_t));
    task->parsec_task.mempool_owner = mempool;

    OBJ_CONSTRUCT(task, parsec_list_item_t);
    task->parsec_task.task_class = &this->self;
    task->parsec_task.taskpool = taskpool;
    task->parsec_task.status = PARSEC_TASK_STATUS_HOOK;

    task->function_template_class_ptr =
        reinterpret_cast<void (*)(void*)>(&Op::static_op_noarg);
    task->object_ptr = static_cast<derivedT*>(this);
    using ::ttg::unique_hash;
    task->key = unique_hash<uint64_t>(key);
    task->parsec_task.data[0].data_in = static_cast<parsec_data_copy_t*>(NULL);
    __parsec_schedule(es, &task->parsec_task, 0);
  }

  // Used by invoke to set all arguments associated with a task
  template <size_t... IS>
  void set_args(std::index_sequence<IS...>, const keyT& key,
                const input_values_tuple_type& args) {
    int junk[] = {0, (set_arg<IS>(key, std::get<IS>(args)), 0)...};
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
  Op(const Op& other) = delete;
  Op& operator=(const Op& other) = delete;
  Op(Op&& other) = delete;
  Op& operator=(Op&& other) = delete;

  // Registers the callback for the i'th input terminal
  template <typename terminalT, std::size_t i>
  void register_input_callback(terminalT& input) {
      using valueT = typename terminalT::value_type;
      using move_callbackT = std::function<void(const keyT&, valueT&&)>;
      using send_callbackT = std::function<void(const keyT&, const valueT&)>;
      auto move_callback = [this](const keyT& key, valueT&& value) {
          //std::cout << "move_callback\n";
          set_arg<i, valueT>(key, std::forward<valueT>(value));
      };
      auto send_callback = [this](const keyT& key, const valueT& value) {
          //std::cout << "send_callback\n";
          set_arg<i, const valueT&>(key, value);
      };
      
      input.set_callback(send_callbackT(send_callback),move_callbackT(move_callback));
  }

  template <std::size_t... IS>
  void register_input_callbacks(std::index_sequence<IS...>) {
    int junk[] = {
        0,
        (register_input_callback<
             typename std::tuple_element<IS, input_terminals_type>::type, IS>(
             std::get<IS>(input_terminals)),
         0)...};
    junk[0]++;
  }

  template <std::size_t... IS, typename inedgesT>
  void connect_my_inputs_to_incoming_edge_outputs(std::index_sequence<IS...>,
                                                  inedgesT& inedges) {
    int junk[] = {
        0,
        (std::get<IS>(inedges).set_out(&std::get<IS>(input_terminals)), 0)...};
    junk[0]++;
  }

  template <std::size_t... IS, typename outedgesT>
  void connect_my_outputs_to_outgoing_edge_inputs(std::index_sequence<IS...>,
                                                  outedgesT& outedges) {
    int junk[] = {
        0,
        (std::get<IS>(outedges).set_in(&std::get<IS>(output_terminals)), 0)...};
    junk[0]++;
  }

 public:
  Op(const std::string& name, const std::vector<std::string>& innames,
     const std::vector<std::string>& outnames)
      : ::ttg::OpBase(name, numins, numouts) {
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

    self.incarnations = (__parsec_chore_t*)malloc(2 * sizeof(__parsec_chore_t));
    ((__parsec_chore_t*)self.incarnations)[0].type = PARSEC_DEV_CPU;
    ((__parsec_chore_t*)self.incarnations)[0].evaluate = NULL;
    ((__parsec_chore_t*)self.incarnations)[0].hook = hook;
    ((__parsec_chore_t*)self.incarnations)[1].type = PARSEC_DEV_NONE;
    ((__parsec_chore_t*)self.incarnations)[1].evaluate = NULL;
    ((__parsec_chore_t*)self.incarnations)[1].hook = NULL;

    self.release_task = parsec_release_task_to_mempool_update_nbtasks;
    self.complete_execution = do_nothing;

    for (i = 0; i < numins; i++) {
      parsec_flow_t* flow = new parsec_flow_t;
      flow->name = strdup((std::string("flow in") + std::to_string(i)).c_str());
      flow->sym_type = SYM_INOUT;
      flow->flow_flags = FLOW_ACCESS_RW;
      flow->dep_in[0] = NULL;
      flow->dep_out[0] = NULL;
      flow->flow_index = i;
      flow->flow_datatype_mask = (1 << i);
      *((parsec_flow_t**)&(self.in[i])) = flow;
    }
    *((parsec_flow_t**)&(self.in[i])) = NULL;

    for (i = 0; i < numouts; i++) {
      parsec_flow_t* flow = new parsec_flow_t;
      flow->name =
          strdup((std::string("flow out") + std::to_string(i)).c_str());
      flow->sym_type = SYM_INOUT;
      flow->flow_flags = FLOW_ACCESS_RW;
      flow->dep_in[0] = NULL;
      flow->dep_out[0] = NULL;
      flow->flow_index = i;
      flow->flow_datatype_mask = (1 << i);
      *((parsec_flow_t**)&(self.out[i])) = flow;
    }
    *((parsec_flow_t**)&(self.out[i])) = NULL;

    self.flags = 0;
    self.dependencies_goal = numins; /* (~(uint32_t)0) >> (32 - numins); */

    int k = 0;
    for(int i = 0; i < parsec->nb_vp; i++) {
        for(int j = 0; j < parsec->virtual_processes[i]->nb_cores; j++) {
            mempools_index[ std::pair<int,int>(i,j) ] = k++;
        }
    }
    parsec_mempool_construct(&mempools, OBJ_CLASS(parsec_task_t), sizeof(my_op_t)+sizeof(input_values_tuple_type),
                             offsetof(parsec_task_t, mempool_owner), k);
    
    parsec_hash_table_init(&tasks_table, offsetof(my_op_t, op_ht_item), 1024, parsec_tasks_hash_fct, NULL);
  }

  Op(const input_edges_type& inedges, const output_edges_type& outedges,
     const std::string& name, const std::vector<std::string>& innames,
     const std::vector<std::string>& outnames)
      : Op(name, innames, outnames) {
    connect_my_inputs_to_incoming_edge_outputs(
        std::make_index_sequence<numins>{}, inedges);
    connect_my_outputs_to_outgoing_edge_inputs(
        std::make_index_sequence<numouts>{}, outedges);
  }

  // Destructor checks for unexecuted tasks
  ~Op() {
    if (cache.size() != 0) {
      int rank = 0;
      std::cerr << rank << ":"
                << "warning: unprocessed tasks in destructor of operation '"
                << get_name() << "'" << std::endl;
      std::cerr << rank << ":"
                << "   T => argument assigned     F => argument unassigned"
                << std::endl;
      int nprint = 0;
      for (auto item : cache) {
        if (nprint++ > 10) {
          std::cerr << "   etc." << std::endl;
          break;
        }
        std::cerr << rank << ":"
                  << "   unused: " << item.first << " : ( ";
        for (std::size_t i = 0; i < numins; i++)
          std::cerr << (item.second.argset[i] ? "T" : "F") << " ";
        std::cerr << ")" << std::endl;
      }
    }
    parsec_hash_table_fini(&tasks_table);
    parsec_mempool_destruct(&mempools);
  }

  // Returns reference to input terminal i to facilitate connection --- terminal
  // cannot be copied, moved or assigned
  template <std::size_t i>
  typename std::tuple_element<i, input_terminals_type>::type* in() {
    return &std::get<i>(input_terminals);
  }

  // Returns reference to output terminal for purpose of connection --- terminal
  // cannot be copied, moved or assigned
  template <std::size_t i>
  typename std::tuple_element<i, output_terminalsT>::type* out() {
    return &std::get<i>(output_terminals);
  }

  // Manual injection of a task with all input arguments specified as a tuple
  void invoke(const keyT& key, const input_values_tuple_type& args) {
      // That task is going to complete, so count it as to execute
      parsec_atomic_inc_32b((volatile uint32_t*)&taskpool->nb_tasks);
      set_args(std::make_index_sequence<
               std::tuple_size<input_values_tuple_type>::value>{},
               key, args);
  }

  // Manual injection of a task that has no arguments
  void invoke(const keyT& key) {
      // That task is going to complete, so count it as to execute
      parsec_atomic_inc_32b((volatile uint32_t*)&taskpool->nb_tasks);
      set_arg_empty(key);
  }
};

    // Class to wrap a callable with signature
    //
    // void op(const input_keyT&, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
    //
    template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
    class WrapOp : public Op<keyT, output_terminalsT, WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>,
                             input_valuesT...> {
      using baseT =
          Op<keyT, output_terminalsT, WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>, input_valuesT...>;
      funcT func;

     public:
      WrapOp(const funcT& func, const typename baseT::input_edges_type& inedges,
             const typename baseT::output_edges_type& outedges, const std::string& name,
             const std::vector<std::string>& innames, const std::vector<std::string>& outnames)
          : baseT(inedges, outedges, name, innames, outnames), func(func) {}

        void op(const keyT& key, typename baseT::input_values_tuple_type&& args, output_terminalsT& out) {
            func(key, std::forward<typename baseT::input_values_tuple_type>(args), out);
      }
    };

    // Class to wrap a callable with signature
    //
    // void op(const input_keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)
    //
    template <typename funcT, typename keyT, typename output_terminalsT, typename... input_valuesT>
    class WrapOpArgs : public Op<keyT, output_terminalsT, WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>,
                                 input_valuesT...> {
      using baseT =
          Op<keyT, output_terminalsT, WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>, input_valuesT...>;
      funcT func;

      template <std::size_t... S>
      void call_func_from_tuple(const keyT& key, typename baseT::input_values_tuple_type&& args,
                                output_terminalsT& out, std::index_sequence<S...>) {
          func(key,
               std::forward<typename std::tuple_element<S,typename baseT::input_values_tuple_type>::type>(std::get<S>(args))...,
               out);
      }

     public:
      WrapOpArgs(const funcT& func, const typename baseT::input_edges_type& inedges,
                 const typename baseT::output_edges_type& outedges, const std::string& name,
                 const std::vector<std::string>& innames, const std::vector<std::string>& outnames)
          : baseT(inedges, outedges, name, innames, outnames), func(func) {}

      void op(const keyT& key, typename baseT::input_values_tuple_type&& args, output_terminalsT& out) {
          call_func_from_tuple(
                               key, std::forward<typename baseT::input_values_tuple_type>(args), out,
                               std::make_index_sequence<std::tuple_size<typename baseT::input_values_tuple_type>::value>{});
      };
    };

    // Factory function to assist in wrapping a callable with signature
    //
    // void op(const input_keyT&, std::tuple<input_valuesT...>&&, std::tuple<output_terminalsT...>&)
    template <typename keyT, typename funcT, typename... input_valuesT, typename... output_edgesT>
    auto wrapt(const funcT& func, const std::tuple<::ttg::Edge<keyT, input_valuesT>...>& inedges,
               const std::tuple<output_edgesT...>& outedges, const std::string& name = "wrapper",
               const std::vector<std::string>& innames = std::vector<std::string>(
                   std::tuple_size<std::tuple<::ttg::Edge<keyT, input_valuesT>...>>::value, "input"),
               const std::vector<std::string>& outnames =
                   std::vector<std::string>(std::tuple_size<std::tuple<output_edgesT...>>::value, "output")) {
      using input_terminals_type = std::tuple<typename ::ttg::Edge<keyT, input_valuesT>::input_terminal_type...>;
      using output_terminals_type = typename ::ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;
      using callable_type =
          std::function<void(const keyT&, std::tuple<input_valuesT...>&&, output_terminals_type&)>;
      callable_type f(func);  // primarily to check types
      using wrapT = WrapOp<funcT, keyT, output_terminals_type, input_valuesT...>;

      return std::make_unique<wrapT>(func, inedges, outedges, name, innames, outnames);
    }

    // Factory function to assist in wrapping a callable with signature
    //
    // void op(const input_keyT&, input_valuesT&&..., std::tuple<output_terminalsT...>&)
    template <typename keyT, typename funcT, typename... input_valuesT, typename... output_edgesT>
    auto wrap(const funcT& func, const std::tuple<::ttg::Edge<keyT, input_valuesT>...>& inedges,
              const std::tuple<output_edgesT...>& outedges, const std::string& name = "wrapper",
              const std::vector<std::string>& innames = std::vector<std::string>(
                  std::tuple_size<std::tuple<::ttg::Edge<keyT, input_valuesT>...>>::value, "input"),
              const std::vector<std::string>& outnames =
                  std::vector<std::string>(std::tuple_size<std::tuple<output_edgesT...>>::value, "output")) {
      using input_terminals_type = std::tuple<typename ::ttg::Edge<keyT, input_valuesT>::input_terminal_type...>;
      using output_terminals_type = typename ::ttg::edges_to_output_terminals<std::tuple<output_edgesT...>>::type;
      using wrapT = WrapOpArgs<funcT, keyT, output_terminals_type, input_valuesT...>;

      return std::make_unique<wrapT>(func, inedges, outedges, name, innames, outnames);
    }

    
}  // namespace ttg
}  // namespace parsec

#endif  // PARSEC_TTG_H_INCLUDED
