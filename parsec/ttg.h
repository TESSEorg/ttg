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

#include <parsec.h>
#include <parsec/parsec_internal.h>
#include <parsec/devices/device.h>
#include <parsec/data_internal.h>
#include <parsec/scheduling.h>

extern parsec_execution_unit_t* eu;
extern parsec_handle_t* handle;

static int static_global_function_id = 0;

extern "C" {
    typedef struct my_op_s : public parsec_execution_context_t {
        void(*function_template_class_ptr)(void*) ;
        void* object_ptr;
        void(*static_set_arg)(int, int);
        uint64_t key;
    } my_op_t;

    static parsec_hook_return_t hook(struct parsec_execution_unit_s* eu,
                                     parsec_execution_context_t* task);
}

// how to improve:
// 1. terminals -> ports to match established terminology in ...
// 2. add namespace ttg, TTGName -> ttg::Name; e.g. TTGOpBase -> ttg::OpBase
// 3. to make runtimes interoperable: use inline namespaces?
//    namespace madness {
//      inline namespace ::ttg
//    }

template <typename keyT, typename valueT>
class Edge;  // Forward decl.

template <typename keyT, typename valueT>
class TTGIn : public TTGTerminalBase {
 public:
  typedef valueT value_type;
  typedef keyT key_type;
  typedef Edge<keyT, valueT> edge_type;
  using send_callback_type = std::function<void(const keyT&, const valueT&)>;
  static constexpr bool is_an_input_terminal = true;

 private:
  bool initialized;
  send_callback_type send_callback;

  // No moving, copying, assigning permitted
  TTGIn(TTGIn&& other) = delete;
  TTGIn(const TTGIn& other) = delete;
  TTGIn& operator=(const TTGIn& other) = delete;
  TTGIn& operator=(const TTGIn&& other) = delete;

 public:
  TTGIn() : initialized(false) {}

  TTGIn(const send_callback_type& send_callback)
      : initialized(true), send_callback(send_callback) {}

  // callback (std::function) is used to erase the operator type and argument
  // index
  void set_callback(const send_callback_type& send_callback) {
    initialized = true;
    this->send_callback = send_callback;
  }

  void send(const keyT& key, const valueT& value) {
    if (!initialized) throw "sending to uninitialzed callback";
    send_callback(key, value);
  }

  // An optimized implementation will need a separate callback for broadcast
  // with a specific value for rangeT
  template <typename rangeT>
  void broadcast(const rangeT& keylist, const valueT& value) {
    if (!initialized) throw "broadcasting to uninitialzed callback";
    for (auto key : keylist) send(key, value);
  }
};

// Output terminal
template <typename keyT, typename valueT>
class TTGOut : public TTGTerminalBase {
 public:
  typedef valueT value_type;
  typedef keyT key_type;
  typedef TTGIn<keyT, valueT> input_terminal_type;
  typedef Edge<keyT, valueT> edge_type;
  static constexpr bool is_an_output_terminal = true;

 private:
  std::vector<input_terminal_type*> successors;

  // No moving, copying, assigning permitted
  TTGOut(TTGOut&& other) = delete;
  TTGOut(const TTGOut& other) = delete;
  TTGOut& operator=(const TTGOut& other) = delete;
  TTGOut& operator=(const TTGOut&& other) = delete;

 public:
  TTGOut() {}

  void connect(input_terminal_type& successor) {
    successors.push_back(&successor);
    static_cast<TTGTerminalBase*>(this)->connect_base(&successor);
  }

  void send(const keyT& key, const valueT& value) {
    for (auto successor : successors) successor->send(key, value);
  }

  // An optimized implementation will need a separate callback for broadcast
  // with a specific value for rangeT
  template <typename rangeT>
  void broadcast(const rangeT& keylist, const valueT& value) {
    for (auto successor : successors) successor->broadcast(keylist, value);
  }
};

template <typename keyT, typename valueT>
class Edge {
 private:
  // An EdgeImpl represents a single edge that most usually will
  // connect a single output terminal with a single
  // input terminal.  However, we had to relax this constraint in
  // order to easily accommodate connecting an input/output edge to
  // an operation that to the outside looked like a single op but
  // internally was implemented as multiple operations.  Thus, the
  // input/output edge has to connect to multiple terminals.
  // Permitting multiple end points makes this much easier to
  // compose, easier to implement, and likely more efficient at
  // runtime.  This is why outs/ins are vectors rather than pointers
  // to a single terminal.
  struct EdgeImpl {
    std::string name;
    std::vector<TTGIn<keyT, valueT>*> outs;
    std::vector<TTGOut<keyT, valueT>*> ins;

    EdgeImpl() : name(""), outs(), ins() {}

    EdgeImpl(const std::string& name) : name(name), outs(), ins() {}

    void set_in(TTGOut<keyT, valueT>* in) {
      if (ins.size())
        std::cout << "Edge: " << name << " : has multiple inputs" << std::endl;
      ins.push_back(in);
      try_to_connect_new_in(in);
    }

    void set_out(TTGIn<keyT, valueT>* out) {
      if (outs.size())
        std::cout << "Edge: " << name << " : has multiple outputs" << std::endl;
      outs.push_back(out);
      try_to_connect_new_out(out);
    }

    void try_to_connect_new_in(TTGOut<keyT, valueT>* in) const {
      for (auto out : outs)
        if (in && out) in->connect(*out);
    }

    void try_to_connect_new_out(TTGIn<keyT, valueT>* out) const {
      for (auto in : ins)
        if (in && out) in->connect(*out);
    }

    ~EdgeImpl() {
      if (ins.size() == 0 || outs.size() == 0) {
        std::cerr << "Edge: destroying edge pimpl with either in or out not "
                     "assigned --- DAG may be incomplete"
                  << std::endl;
      }
    }
  };

  // We have a vector here to accomodate fusing multiple edges together
  // when connecting them all to a single terminal.
  mutable std::vector<std::shared_ptr<EdgeImpl>>
      p;  // Need shallow copy semantics

 public:
  typedef TTGIn<keyT, valueT> input_terminal_type;
  typedef TTGOut<keyT, valueT> output_terminal_type;
  typedef keyT key_type;
  typedef valueT value_type;
  static constexpr bool is_an_edge = true;

  Edge(const std::string name = "anonymous edge") : p(1) {
    p[0] = std::make_shared<EdgeImpl>(name);
  }

  template <typename... valuesT>
  Edge(const Edge<keyT, valuesT>&... edges) : p(0) {
    std::vector<Edge<keyT, valueT>> v = {edges...};
    for (auto& edge : v) {
      p.insert(p.end(), edge.p.begin(), edge.p.end());
    }
  }

  void set_in(TTGOut<keyT, valueT>* in) const {
    for (auto& edge : p) edge->set_in(in);
  }

  void set_out(TTGIn<keyT, valueT>* out) const {
    for (auto& edge : p) edge->set_out(out);
  }
};

// Fuse edges into one ... all the types have to be the same ... just using
// valuesT for variadic args
template <typename keyT, typename... valuesT>
auto fuse(const Edge<keyT, valuesT>&... args) {
  using valueT = typename std::tuple_element<
      0, std::tuple<valuesT...>>::type;  // grab first type
  return Edge<keyT, valueT>(
      args...);  // This will force all valuesT to be the same
}

// Make a tuple of Edges ... needs some type checking injected
template <typename... inedgesT>
auto edges(const inedgesT&... args) {
  return std::make_tuple(args...);
}

template <size_t i, typename keyT, typename valueT,
          typename... output_terminalsT>
void send(const keyT& key, const valueT& value,
          std::tuple<output_terminalsT...>& t) {
  std::get<i>(t).send(key, value);
}

template <size_t i, typename rangeT, typename valueT,
          typename... output_terminalsT>
void broadcast(const rangeT& keylist, const valueT& value,
               std::tuple<output_terminalsT...>& t) {
  std::get<i>(t).broadcast(keylist, value);
}

// Make type of tuple of edges from type of tuple of terminals
template <typename termsT>
struct terminals_to_edges;
template <typename... termsT>
struct terminals_to_edges<std::tuple<termsT...>> {
  typedef std::tuple<typename termsT::edge_type...> type;
};

// Make type of tuple of output terminals from type of tuple of edges
template <typename edgesT>
struct edges_to_output_terminals;
template <typename... edgesT>
struct edges_to_output_terminals<std::tuple<edgesT...>> {
  typedef std::tuple<typename edgesT::output_terminal_type...> type;
};

// Make type of tuple of input terminals from type of tuple of edges
template <typename edgesT>
struct edges_to_input_terminals;
template <typename... edgesT>
struct edges_to_input_terminals<std::tuple<edgesT...>> {
  typedef std::tuple<typename edgesT::input_terminal_type...> type;
};

static parsec_hook_return_t do_nothing(parsec_execution_unit_t *eu, parsec_execution_context_t *task)
{
    (void)eu;
    (void)task;
    return PARSEC_HOOK_RETURN_DONE;
}

static parsec_hook_return_t hook(struct parsec_execution_unit_s* eu,
                                 parsec_execution_context_t* task)
{
    my_op_t* me = static_cast<my_op_t*>(task);
    me->function_template_class_ptr(task);
    (void)eu;
    return PARSEC_HOOK_RETURN_DONE;
}

struct ParsecBaseOp {
protected:
    static std::map<int, ParsecBaseOp*> function_id_to_instance;
};
std::map<int, ParsecBaseOp*> ParsecBaseOp::function_id_to_instance = {};

template <typename keyT, typename output_terminalsT, typename derivedT,
          typename... input_valueTs>
class TTGOp : public TTGOpBase, ParsecBaseOp {
 protected:
    parsec_function_t self;
 private:
  using opT = TTGOp<keyT, output_terminalsT, derivedT, input_valueTs...>;

 public:
  static constexpr int numins =
      sizeof...(input_valueTs);  // number of input arguments
  static constexpr int numouts =
      std::tuple_size<output_terminalsT>::value;  // number of outputs or
                                                  // results

  using input_values_tuple_type = std::tuple<input_valueTs...>;
  using input_terminals_type = std::tuple<TTGIn<keyT, input_valueTs>...>;
  using input_edges_type = std::tuple<Edge<keyT, input_valueTs>...>;

  using output_terminals_type = output_terminalsT;
  using output_edges_type =
      typename terminals_to_edges<output_terminalsT>::type;

 private:
  input_terminals_type input_terminals;
  output_terminalsT output_terminals;

  struct OpArgs {
    int counter;                      // Tracks the number of arguments set
    std::array<bool, numins> argset;  // Tracks if a given arg is already set;
    input_values_tuple_type t;        // The input values
    derivedT* derived;                // Pointer to derived class instance
    keyT key;                         // Task key

    OpArgs() : counter(numins), argset(), t() {}

    void run() {
        derived->op(key, t, derived->output_terminals);
    }

    virtual ~OpArgs() {}  // Will be deleted via TaskInterface*
  };

  static void static_op(parsec_execution_context_t* my_task) {
        my_op_t* task = static_cast<my_op_t*>(my_task);
        /*        
                  struct data_repo_entry_s     *data_repo;
                  struct parsec_data_copy_s    *data_in;
                  struct parsec_data_copy_s    *data_out;
        */
        derivedT* obj = (derivedT*)task->object_ptr;
        obj->op((keyT)task->key,
                *reinterpret_cast<input_values_tuple_type*>(task->data[0].data_in),
                obj->output_terminals);
#ifdef OR_BETTER
            *static_cast<input_values_tuple_type*>(task->data[0].data_in->device_private));
#endif
  }

  using cacheT = std::map<keyT, OpArgs>;
  cacheT cache;

  // Used to set the i'th argument
  template <std::size_t i>
  void set_arg(const keyT& key, const typename std::tuple_element<
                                    i, input_values_tuple_type>::type& value) {
    using valueT = typename std::tuple_element<i, input_terminals_type>::type;

    if (tracing()) std::cout << get_name() << " : " << key << ": setting argument : " << i << std::endl;
        auto it = cache.find(key);
        if( it == cache.end() ) {
            cache[key] = OpArgs();
        }
        OpArgs& args = cache[key];
        if (args.argset[i]) {
            std::cerr << get_name() << " : " << key << ": error argument is already set : " << i << std::endl;
            throw "bad set arg";
        }
        args.argset[i] = true;        
        std::get<i>(args.t) = value;
        args.counter--;
        if (args.counter == 0) {
            if (tracing()) std::cout << get_name() << " : " << key << ": invoking op " << std::endl;
            //static_cast<derivedT*>(this)->op(key, args.t);
            cache.erase(key);
            //create PaRSEC task
            // and give it to the scheduler
            my_op_t* task = static_cast<my_op_t*>(calloc(1, sizeof(my_op_t)));

            OBJ_CONSTRUCT(task, parsec_list_item_t);
            task->function = &this->self;
            task->parsec_handle = handle;
            task->status = PARSEC_TASK_STATUS_HOOK;
            
            task->function_template_class_ptr =
                reinterpret_cast<void (*)(void*)>(&TTGOp::static_op);
            task->object_ptr = static_cast<derivedT*>(this);
            task->key = key;
            task->data[0].data_in = static_cast<parsec_data_copy_t*>(malloc(sizeof(value)));
            memcpy(task->data[0].data_in, &value, sizeof(value));
            __parsec_schedule(eu, task, 0);
        }
  }

  // Used to generate tasks with no input arguments
  void set_arg_empty(const keyT& key) {
        if (tracing()) std::cout << get_name() << " : " << key << ": invoking op " << std::endl;
        //create PaRSEC task
        // and give it to the scheduler
        my_op_t* task = static_cast<my_op_t*>(calloc(1, sizeof(my_op_t)));

        OBJ_CONSTRUCT(task, parsec_list_item_t);
        task->function = &this->self;
        task->parsec_handle = handle;
        task->status = PARSEC_TASK_STATUS_HOOK;
            
        task->function_template_class_ptr =
            reinterpret_cast<void (*)(void*)>(&TTGOp::static_op);
        task->object_ptr = static_cast<derivedT*>(this);
        task->key = key;
        task->data[0].data_in = static_cast<parsec_data_copy_t*>(NULL);
        __parsec_schedule(eu, task, 0);
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
  TTGOp(const TTGOp& other) = delete;
  TTGOp& operator=(const TTGOp& other) = delete;
  TTGOp(TTGOp&& other) = delete;
  TTGOp& operator=(TTGOp&& other) = delete;

  // Registers the callback for the i'th input terminal
  template <typename terminalT, std::size_t i>
  void register_input_callback(terminalT& input) {
    using callbackT =
        std::function<void(const typename terminalT::key_type&,
                           const typename terminalT::value_type&)>;
    auto callback = [this](const typename terminalT::key_type& key,
                           const typename terminalT::value_type& value) {
      set_arg<i>(key, value);
    };
    input.set_callback(callbackT(callback));
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
  TTGOp(const std::string& name, const std::vector<std::string>& innames,
        const std::vector<std::string>& outnames)
      : TTGOpBase(name, numins, numouts) {
    // Cannot call these in base constructor since terminals not yet constructed
    if (innames.size() != std::tuple_size<input_terminals_type>::value)
      throw "TTGOP: #input names != #input terminals";
    if (outnames.size() != std::tuple_size<output_terminalsT>::value)
      throw "TTGOP: #output names != #output terminals";

    register_input_terminals(input_terminals, innames);
    register_output_terminals(output_terminals, outnames);

    register_input_callbacks(std::make_index_sequence<numins>{});

    int i;

        memset(&self, 0, sizeof(parsec_function_t));
        
        self.name = name.c_str();
        self.function_id = static_global_function_id++;
        self.nb_parameters = 0;
        self.nb_locals = 0;
        self.nb_flows = std::max((int)numins, (int)numouts);

        function_id_to_instance[self.function_id] = this;
        
        self.incarnations = (__parsec_chore_t *) malloc(2 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t*)self.incarnations)[0].type = PARSEC_DEV_CPU;
        ((__parsec_chore_t*)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t*)self.incarnations)[0].hook = hook;
        ((__parsec_chore_t*)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t*)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t*)self.incarnations)[1].hook = NULL;

        self.release_task = do_nothing;

        for( i = 0; i < numins; i++ ) {
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

        for( i = 0; i < numouts; i++ ) {
            parsec_flow_t* flow = new parsec_flow_t;
            flow->name = strdup((std::string("flow out") + std::to_string(i)).c_str());
            flow->sym_type = SYM_INOUT;
            flow->flow_flags = FLOW_ACCESS_RW;
            flow->dep_in[0] = NULL;
            flow->dep_out[0] = NULL;
            flow->flow_index = i;
            flow->flow_datatype_mask = (1 << i);
            *((parsec_flow_t**)&(self.out[i]))  = flow;
        }
        *((parsec_flow_t**)&(self.out[i])) = NULL;
    
        self.flags = 0;
        self.dependencies_goal = 0;
  }
    
  TTGOp(const input_edges_type& inedges, const output_edges_type& outedges,
        const std::string& name, const std::vector<std::string>& innames,
        const std::vector<std::string>& outnames)
      : TTGOp(name, innames, outnames) {
      
    connect_my_inputs_to_incoming_edge_outputs(
        std::make_index_sequence<numins>{}, inedges);
    connect_my_outputs_to_outgoing_edge_inputs(
        std::make_index_sequence<numouts>{}, outedges);
  }

  // Destructor checks for unexecuted tasks
  ~TTGOp() {
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
  }

  // Returns reference to input terminal i to facilitate connection --- terminal
  // cannot be copied, moved or assigned
  template <std::size_t i>
  typename std::tuple_element<i, input_terminals_type>::type& in() {
    return std::get<i>(input_terminals);
  }

  // Returns reference to output terminal for purpose of connection --- terminal
  // cannot be copied, moved or assigned
  template <std::size_t i>
  typename std::tuple_element<i, output_terminalsT>::type& out() {
    return std::get<i>(output_terminals);
  }

  // Manual injection of a task with all input arguments specified as a tuple
  void invoke(const keyT& key, const input_values_tuple_type& args) {
    set_args(std::make_index_sequence<
                 std::tuple_size<input_values_tuple_type>::value>{},
             key, args);
  }

  // Manual injection of a task that has no arguments
  void invoke(const keyT& key) { set_arg_empty(key); }
};

// Class to wrap a callable with signature
//
// void op(const input_keyT&, const std::tuple<input_valuesT...>&,
// std::tuple<output_terminalsT...>&)
//
template <typename funcT, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
class WrapOp
    : public TTGOp<keyT, output_terminalsT,
                   WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>,
                   input_valuesT...> {
  using baseT = TTGOp<keyT, output_terminalsT,
                      WrapOp<funcT, keyT, output_terminalsT, input_valuesT...>,
                      input_valuesT...>;
  funcT func;

 public:
  WrapOp(const funcT& func, const typename baseT::input_edges_type& inedges,
         const typename baseT::output_edges_type& outedges,
         const std::string& name, const std::vector<std::string>& innames,
         const std::vector<std::string>& outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(func) {}

  void op(const keyT& key, const typename baseT::input_values_tuple_type& args,
          output_terminalsT& out) {
    func(key, args, out);
  }
};

// Class to wrap a callable with signature
//
// void op(const input_keyT&, const std::tuple<input_valuesT...>&,
// std::tuple<output_terminalsT...>&)
//
template <typename funcT, typename keyT, typename output_terminalsT,
          typename... input_valuesT>
class WrapOpArgs
    : public TTGOp<keyT, output_terminalsT,
                   WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>,
                   input_valuesT...> {
  using baseT =
      TTGOp<keyT, output_terminalsT,
            WrapOpArgs<funcT, keyT, output_terminalsT, input_valuesT...>,
            input_valuesT...>;
  funcT func;

  template <std::size_t... S>
  void call_func_from_tuple(const keyT& key,
                            const typename baseT::input_values_tuple_type& args,
                            output_terminalsT& out, std::index_sequence<S...>) {
    func(key, std::get<S>(args)..., out);
  }

 public:
  WrapOpArgs(const funcT& func, const typename baseT::input_edges_type& inedges,
             const typename baseT::output_edges_type& outedges,
             const std::string& name, const std::vector<std::string>& innames,
             const std::vector<std::string>& outnames)
      : baseT(inedges, outedges, name, innames, outnames), func(func) {}

  void op(const keyT& key, const typename baseT::input_values_tuple_type& args,
          output_terminalsT& out) {
    call_func_from_tuple(
        key, args, out,
        std::make_index_sequence<
            std::tuple_size<typename baseT::input_values_tuple_type>::value>{});
  };
};

// Factory function to assist in wrapping a callable with signature
//
// void op(const input_keyT&, const std::tuple<input_valuesT...>&,
// std::tuple<output_terminalsT...>&)
template <typename keyT, typename funcT, typename... input_valuesT,
          typename... output_edgesT>
auto wrapt(const funcT& func,
           const std::tuple<Edge<keyT, input_valuesT>...>& inedges,
           const std::tuple<output_edgesT...>& outedges,
           const std::string& name = "wrapper",
           const std::vector<std::string>& innames = std::vector<std::string>(
               std::tuple_size<std::tuple<Edge<keyT, input_valuesT>...>>::value,
               "input"),
           const std::vector<std::string>& outnames = std::vector<std::string>(
               std::tuple_size<std::tuple<output_edgesT...>>::value,
               "output")) {
  using input_terminals_type =
      std::tuple<typename Edge<keyT, input_valuesT>::input_terminal_type...>;
  using output_terminals_type =
      typename edges_to_output_terminals<std::tuple<output_edgesT...>>::type;
  using callable_type =
      std::function<void(const keyT&, const std::tuple<input_valuesT...>&,
                         output_terminals_type&)>;
  callable_type f(func);  // pimarily to check types
  using wrapT = WrapOp<funcT, keyT, output_terminals_type, input_valuesT...>;

  return std::make_unique<wrapT>(func, inedges, outedges, name, innames,
                                 outnames);
}

// Factory function to assist in wrapping a callable with signature
//
// void op(const input_keyT&, input_valuesT&...,
// std::tuple<output_terminalsT...>&)
template <typename keyT, typename funcT, typename... input_valuesT,
          typename... output_edgesT>
auto wrap(const funcT& func,
          const std::tuple<Edge<keyT, input_valuesT>...>& inedges,
          const std::tuple<output_edgesT...>& outedges,
          const std::string& name = "wrapper",
          const std::vector<std::string>& innames = std::vector<std::string>(
              std::tuple_size<std::tuple<Edge<keyT, input_valuesT>...>>::value,
              "input"),
          const std::vector<std::string>& outnames = std::vector<std::string>(
              std::tuple_size<std::tuple<output_edgesT...>>::value, "output")) {
  using input_terminals_type =
      std::tuple<typename Edge<keyT, input_valuesT>::input_terminal_type...>;
  using output_terminals_type =
      typename edges_to_output_terminals<std::tuple<output_edgesT...>>::type;
  using wrapT =
      WrapOpArgs<funcT, keyT, output_terminals_type, input_valuesT...>;

  return std::make_unique<wrapT>(func, inedges, outedges, name, innames,
                                 outnames);
}

#endif  // PARSEC_TTG_H_INCLUDED
