#ifndef MADNESS_FLOW_H_INCLUDED
#define MADNESS_FLOW_H_INCLUDED

#include <iostream>
#include <tuple>
#include <vector>
#include <functional>
#include <memory>
#include <map>

// We are forming an execution graph to represent data streaming
// through a series of operations or computational steps.  Each node
// in the graph represents an operation on the incoming data.  Each
// arc in the graph represents a flow of data from one operation to
// the next.  This operation graph need not be acylic, hence we can
// easily represent iteration and recursion.
//
// There can be multiple independent instances of an operation in the
// execution graph. E.g., consider evaluating a[i]*b[i]*c[i]*d[i]*e[i]
// for each element (i) in the vectors a, b, c, d, e --- we will need
// several (four) independent instances of the multiply operation.
//
// For each set of input arguments flowing to an operation, a task is
// (logically) created.  The task graph will be acyclic since a task
// can only send data to its successors and not to itself.
// 
// Tasks take arguments from input data flows, execute an operation,
// and inject results into output data flows to send data to
// successor tasks.
//
// Tasks (like PaRSEC) are identified by a key (or index) --- think of
// it as the index of a for-loop iteration with each iteration
// executed in a separate task. In the vector multiplication example
// above, the key would be the vector index (i). The place where
// computation occurs is determined by the key, and arguments (data)
// are sent to the task using flows labelled by the key.
//
// Each piece of data in a flow comprises a (key,value) pair.  The key
// labels the task instance to which the value is being sent, and the
// flow is connected (by position) with a specific argument of the
// operation that the task will execute.  Since the task and all of
// its arguments are assoicated with the same key, they clearly must
// all have the same key type.
//
// Ouputs (results) of a task are injected into flows that connect
// with arguments of sucessor (dependent) tasks.  These tasks may be
// associated with different key types.  Why?  Imagine an algorithm on
// a tree (in which data are labelled by the node names) feeding
// results into a matrix (in which data are labelled by the
// [row,column] indicies) feeding results into a matrix algorithm
// (with tasks labelled by triplets [i,j,k]), etc.
//
// The act of inserting data into any of the input flows associated
// with an operation (logically) creates the task that will execute
// it.  Once all input arguments are assigned, the task is (or can be)
// executed.
//
// For wrapping simple callables, essentially all of the boilerplate
// can be automated via templates.  Only more complex operations
// must manually define a class.
//
// The essential classes (soon to be abstracted further to permit
// multiple co-existing implementations) are
//
// Flow<keyT,valueT> --- plumbing connecting the output of one task
// with the input of one or more successors.  Provides
// send/broadcast/add_callback methods and defines key_type and
// value_type.  Copy/assignment semantics???
//
// Flows< Flow<key0T,value0T>, ... > --- essentially a tuple of flows
// that can have any type for their keys and values.  Can be used to
// provide the output flows for an operation.  It can be empty; i.e.,
// just Flows<> is permissible. An important case of Flows is where
// the constituent Flow types all share same key; then it defines
// value_tuple_type and key_plus_value_tuple_type.  Provides
// send<i>/broadcast<i>/get<i>/all()/size() methods.  It must have
// shallow copy semantics since copying is necessary but each copy
// must refer to the same underlying data structure.
//
// InFlows --- a bundle of Flow's that all share same key type but can
// have different value types. Just syntactic sugar.
//
// OpTuple --- a mixin class that provides the functionality necessary
// to turn a function (defined as derived class method) into an
// operation that can be connected via flows and have data in the
// flows trigger computation.
//
// Op --- provides some syntactic convenience compared to OpTuple
//
// BaseOp --- presently used to turn tracing log messages on/off for
// all operation instances.


// A flow is plumbing used to connect the output of an operation to
// the input of another.  In the present implementation it never holds
// data ... it is just keeps track of callbacks that are executed
// when something is inserted into the flow to pass that datum to the
// next operation.
//
// It necessarily has shallow copy semantics to ensure all copies
// refer to the same underlying object, to ensure correct lifetime of
// objects, and to make it easier to plug and play.
//
// Flow, as well as higher level constructs built out of it, like Flows (a tuple of Flow's)
// and FlowArray, must meet a FlowBundle concept requirements to be usable
// by Op facilities.
template <typename keyT, typename valueT>
class Flow {
    using callbackT = std::function<void(const keyT&, const valueT&)>;

    struct FlowImpl {
        mutable std::vector<callbackT> callbacks;
    };

    std::shared_ptr<FlowImpl> p; // For shallow copy

public:

    typedef valueT value_type;

    Flow() : p(std::make_shared<FlowImpl>()) {}

    // A callback must be convertible to a std::function<void(const keyT&, const valueT&)>

    // Method is const in sense that it does not inject anything into
    // the flow and is only used when constructing the DAG
    void add_callback(const callbackT& callback) const {
        p->callbacks.push_back(callback);
    }        

    ///////////////////////////////////////////////////////////
    // everything below is required by the FlowBundle concept

    typedef keyT key_type;
    typedef std::tuple<key_type, value_type> key_plus_output_tuple_type;
    typedef std::tuple<value_type> values_tuple_type;

    // Send the given (key,value) into the flow
    template <std::size_t i = 0>
    void send(const keyT& key, const valueT& value) {
        static_assert(i == 0, "Flow::send<i> expects i=0");
        for (auto callback : p->callbacks) callback(key,value);
    }

    // Broadcast the given value to all keys
    template <typename rangeT, std::size_t i = 0>
    void broadcast(const rangeT& keylist, const valueT& value) {
        static_assert(i == 0, "Flow::broadcast<i> expects i=0");
        for (auto key : keylist) send(key,value);
    }

    // Gets the i'th flow
    template <std::size_t i = 0>
    auto & get() {
      static_assert(i == 0, "Flow::get<i> expects i=0");
      return *this;
    }

    // Returns a tuple containing all flows.  Primarily used for unpacking using std::tie.
    auto all() {return std::make_tuple(*this);}

    // Returns a tuple containing all flows.  Primarily used for unpacking using std::tie.
    const auto all() const {return std::make_tuple(*this);}

    // Returns the number of flows
    static constexpr std::size_t size() { return 1; }
};


// Clone a flow --- connect an input flow to an output flow so that
// content injected into the input is propagated to the output, but
// not vice versa.  I.e., the output flow is downstream of the input.
//
// Relies on the input flow providing add_callback()
template <typename keyT, typename valueT>
Flow<keyT,valueT> clone(const Flow<keyT,valueT>& inflow) {
    Flow<keyT,valueT> outflow;
    inflow.add_callback([outflow](const keyT& key, const valueT& value){const_cast<Flow<keyT,valueT>*>(&outflow)->send(key,value);});
    return outflow;
}

// machinery to help define key_type in Flows when possible
namespace detail {
struct dont_define_key_typedefs {};

template <typename keyT, typename...valueTs>
struct do_define_key_typedefs {
  typedef keyT key_type;
  typedef std::tuple<keyT, valueTs...> key_plus_output_tuple_type;
};

struct invalid_key_type {};

template <typename...>
struct key_type {
  typedef invalid_key_type type;
};

template <typename flowT>
struct key_type<flowT> {
  typedef typename flowT::key_type type;
};

template <typename flowT, typename... flowTs>
struct key_type<flowT, flowTs...> : public key_type<flowTs...> {
  typedef key_type<flowTs...> baseT;

  typedef typename std::conditional<
      std::is_same<typename key_type<flowT>::type,
                   typename baseT::type>::value,
      typename baseT::type, invalid_key_type>::type type;
};

template <typename T>
struct value_type {
  typedef typename T::value_type type;
};

template <typename... flowTs>
struct flows_typedefs
    : public std::conditional<
          std::is_same<typename key_type<flowTs...>::type,
                       invalid_key_type>::value,
          dont_define_key_typedefs,
          do_define_key_typedefs<
              typename key_type<flowTs...>::type,
              typename detail::value_type<flowTs>::type...>>::type {
  typedef std::tuple<typename detail::value_type<flowTs>::type...>
      values_tuple_type;
};

/// trait to determine if T has key_type
template<class T>
class has_key_type {
    /// true case
    template<class U>
    static std::true_type __test(typename U::key_type*);
    /// false case
    template<class >
    static std::false_type __test(...);
  public:
    static constexpr const bool value = std::is_same<std::true_type,
        decltype(__test<T>(0))>::value;
};

}  // namespace detail

// An tuple of flows, each of which has arbitrary key and value types.  Can
// be empty.
template <typename...flowTs>
class Flows : public detail::flows_typedefs<flowTs...> {
    std::tuple<flowTs...> t;
public:
    Flows() : t() {}

    template <typename...argsT>
    Flows(const argsT... args) : t(args...) {}
    
    template <typename...argsT>
    Flows(const std::tuple<argsT...>& t) : t(t) {}
    
    // Gets the i'th flow
    template <std::size_t i>
    auto & get() {return std::get<i>(t);}

    // Sends to the i'th flow
    template <std::size_t i, typename keyT, typename valueT>
    void send(const keyT& key, const valueT& value) {
        get<i>().send(key,value);
    }

    // Broadcasts to the i'th flow
    template <std::size_t i, typename rangeT, typename valueT>
    void broadcast(const rangeT& keylist, const valueT& value) {
        get<i>().broadcast(keylist, value);
    }

    // Returns a tuple containing all flows.  Primarily used for unpacking using std::tie.
    auto & all() {return t;}

    // Returns a tuple containing all flows.  Primarily used for unpacking using std::tie.
    const auto & all() const {return t;}

    // Returns the number of flows
    static constexpr std::size_t size() {return std::tuple_size<std::tuple<flowTs...>>::value;}
};

// FlowArray is an alias for Flows with same key type
template <typename keyT, typename...valueTs>
using FlowArray = Flows<Flow<keyT,valueTs>...>;

// InFlows is the old name for FlowArray
template <typename keyT, typename...valueTs>
using InFlows = FlowArray<keyT,valueTs...>;

// Factory function occasionally needed where type deduction fails 
template <typename...flowsT>
auto make_flows(flowsT...args) {return Flows<flowsT...>(args...);}

// Data/functionality common to all Ops
class BaseOp {
    static bool trace; // If true prints trace of all assignments and all op invocations
public:
    
    // Sets trace to value and returns previous setting
    static bool set_trace(bool value) {std::swap(trace,value); return value;}

    static bool get_trace() {return trace;}

    static bool tracing() {return trace;}
};

// With more than one source file this will need to be moved
bool BaseOp::trace = false;

// Mix-in class used with CRTP to implement operations in a flow.  The
// derived class should implement an operation with this name and
// signature.
//
// void op(const input_keyT& key,
//         const std::tuple<input_valuesT...>& input_values,
//         output_flowsT& output_flows)
//
// [Aside: derive from Op not OpTuple to pass input values as arguments
// to op() rather than packed in a tuple.  Some template voodoo should
// be able to fuse the two classes if that is preferable.]
//
// where
//
// input_key is the key associated with the task and all of its input
// arguments,
//
// input_values is tuple that provides the value of each input flow as
// separate argument.

// output_flows is a copy of the object provided to take output --- it
// can be anything but most naturally would be an instance of
// Flows<...>.
//
// input_flowsT must presently be InFlows<keyT,value0T,...>

template <typename input_flowsT, typename output_flowsT, typename derivedT>
class OpTuple {
public:
    typedef typename input_flowsT::key_type input_key_type;
    typedef typename input_flowsT::values_tuple_type input_values_tuple_type;
    typedef output_flowsT output_type;
    
private:
    static constexpr int numargs = input_flowsT::size(); // Number of arguments in the input flows
    
    static_assert(input_flowsT::size() != 0,
                  "OpTuple<input_flowsT,...> expects non-empty input_flowsT");
    static_assert(detail::has_key_type<input_flowsT>::value,
                  "OpTuple<input_flowsT,...> expects a set of input flows with matching key types");
        
    struct OpTupleImpl : private BaseOp {
        
        struct OpArgs{
            int counter;            // Tracks the number of arguments set
            std::array<bool,numargs> argset; // Tracks if a given arg is already set;
            typename input_flowsT::values_tuple_type t; // The flow values
            
            OpArgs() : counter(numargs), argset(), t() {}
        };
        
        std::string name;  // mostly used for debugging output
        output_flowsT outputs;
        std::map<input_key_type,OpArgs> cache; // Contains tasks waiting for input to become complete
        OpTuple<input_flowsT, output_flowsT, derivedT>* owner; // So can forward calls to derived class
        
        // Used in the callback from a flow to set i'th argument
        template <typename valueT, std::size_t i>
        void set_arg(const input_key_type& key, const valueT& value) {
            // In parallel case we would be using a write accessor to a
            // local concurrent container to obtain exclusive access to
            // counter and to avoid race condition on first insert.
            
            // Lots of optimiatizations and scheduling strategies to
            // insert here ... for now always shove things into the cache
            // and execute when ready.
            
            if (tracing()) std::cout << name << " : " << key << ": setting argument : " << i << std::endl;
            
            OpArgs& args = cache[key];
            
            if (args.argset[i]) {
                std::cerr << name << " : " << key << ": error argument is already set : " << i << std::endl;
                throw "bad set arg";
            }
            
            args.argset[i] = true;        
            
            std::get<i>(args.t) = value;
            args.counter--;
            if (args.counter == 0) {
                if (tracing()) std::cout << name << " : " << key << ": invoking op " << std::endl;
                static_cast<derivedT*>(owner)->op(key, args.t, outputs);
                cache.erase(key);
            }
        }
        
        // Registers the callback for the i'th input flow
        template <typename flowT, std::size_t i>
        void register_input_callback(const flowT& input) {
            using callbackT = std::function<void(const input_key_type&, const typename flowT::value_type&)>;
            
            if (tracing()) std::cout << name << " : registering callback for argument : " << i << std::endl;
            
            auto callback = [this](const input_key_type& key, const typename flowT::value_type& value){set_arg<typename flowT::value_type,i>(key,value);};
            input.add_callback(callbackT(callback));
        }
        
        template<typename Tuple, std::size_t...IS>
        void register_input_callbacks(const Tuple& inputs, std::index_sequence<IS...>) {
            int junk[] = {0,(register_input_callback<typename std::tuple_element<IS,Tuple>::type, IS>(std::get<IS>(inputs)),0)...}; junk[0]++;
        }

        OpTupleImpl(const std::string& name) : name(name), outputs() {}

        template <typename arg_input_flowsT>
        OpTupleImpl(const arg_input_flowsT& inputs, const output_flowsT& outputs,  const std::string& name)
            : name(name), outputs(outputs)
        {
            register_input_callbacks(inputs.all(),
                                     std::make_index_sequence<std::tuple_size<typename arg_input_flowsT::values_tuple_type>::value>{});
        }

        // Destructor checks for unexecuted tasks
        ~OpTupleImpl() {
            if (cache.size() != 0) {
                std::cerr << "warning: unprocessed tasks in destructor of operation '" << name << "'" << std::endl;
                std::cerr << "   T => argument assigned     F => argument unassigned" << std::endl;
                int nprint=0;
                for (auto item : cache) {
                    if (nprint++ > 10) {
                        std::cerr << "   etc." << std::endl;
                        break;
                    }
                    std::cerr << "   unused: " << item.first << " : ( ";
                    for (std::size_t i=0; i<numargs; i++) std::cerr << (item.second.argset[i] ? "T" : "F") << " ";
                    std::cerr << ")" << std::endl;
                }
            }
        }
    };

    std::unique_ptr<OpTupleImpl> impl;
        
    OpTuple(const OpTuple& other) = delete;
        
    OpTuple& operator=(const OpTuple& other) = delete;

public:
    // Default constructor makes operation that still needs to be connected to input and output flows
    OpTuple(const std::string& name = std::string("unnamed op")) : impl(new OpTupleImpl(name)) {impl->owner= this;}

    // Full constructor makes operation connected to both input and output flows
    // This format for the constructor forces the type constraint and automates some conversions.
    template <typename arg_input_flowsT>
    OpTuple(const arg_input_flowsT& inputs, const output_flowsT& outputs,
            const std::string& name = std::string("unnamed tuple op"))
        : impl(new OpTupleImpl(inputs, outputs, name))
    {impl->owner = this;}

    OpTuple (OpTuple&& other) {
        if (this != &other) {
            impl.reset();
            std::swap(impl,other.impl);
            impl->owner= this;
        }
    }

    // Connects an incompletely constructed operation to its input and output flows
    void connect(const input_flowsT& inputs, const output_flowsT& outputs) {
        impl->outputs = outputs;
        impl->register_input_callbacks(inputs.all(),
                                       std::make_index_sequence<std::tuple_size<typename input_flowsT::values_tuple_type>::value>{});
    }        
   
    void set_name(const std::string& name) {impl->name = name;}

    const std::string& get_name() const {return impl->name;}
};


// Call derivedT::op() unpacking input values as arguments so the signature of op is
//
// void op(const keyT& key, const input_valuesT& args ..., output_type& output);
//
template <typename input_flowsT, typename output_flowsT, typename derivedT>
class Op : public OpTuple<input_flowsT, output_flowsT, Op<input_flowsT, output_flowsT, derivedT>>  {
public:
    using opT = OpTuple<input_flowsT, output_flowsT, Op<input_flowsT, output_flowsT, derivedT>>;
    typedef typename input_flowsT::key_type input_key_type;
    typedef typename input_flowsT::values_tuple_type input_values_tuple_type;
    typedef output_flowsT output_type;


    // Helper routine to call user operation from the data
    template<std::size_t...S>
    void  call_op_from_tuple(const input_key_type& key, const input_values_tuple_type& params, output_type& output, std::index_sequence<S...>) {
        static_cast<derivedT*>(this)->op(key, std::get<S>(params)..., output);
    }

public:
    Op(const std::string& name = std::string("unamed arg op"))
        : opT(name) {}

    template <typename arg_input_flowsT>
    Op(const arg_input_flowsT& inputs, const output_flowsT& outputs,
       const std::string& name = std::string("unnamed arg op"))
        : opT(inputs, outputs, name) {
      static_assert(std::is_same<typename arg_input_flowsT::key_type,input_key_type>::value,
                    "Op::Op(inputs,outputs,name) received nonconforming inputs object");
    }

    void op(const input_key_type& key, const input_values_tuple_type& t, output_type& output) {
        call_op_from_tuple(key, t, output, std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{});
        
    };
};

// Class to wrap a callable with signature
//
// void op(const input_keyT&, const std::tuple<valuesT...>&, outputT&)
//
template <typename funcT, typename input_flowsT, typename output_flowsT>
class WrapOpTuple : public OpTuple<input_flowsT, output_flowsT, WrapOpTuple<funcT,input_flowsT,output_flowsT>> {
    using baseT = OpTuple<input_flowsT, output_flowsT, WrapOpTuple<funcT,input_flowsT,output_flowsT>>;

    funcT func;

 public:
    WrapOpTuple(const funcT& func, const input_flowsT& inflows, const output_flowsT& outflows, const std::string& name = "wrapper")
        : baseT(inflows,outflows,name)
        , func(func)
    {}

    void op(const typename baseT::input_key_type& key, const typename baseT::input_values_tuple_type& args, output_flowsT& out) {
        func(key, args, out);
    }
};

// Wrap a callable with signature
//
// void op(const input_keyT&, const std::tuple<input_valuesT...>&, outputT&);
template <typename funcT, typename input_flowsT, typename output_flowsT>
auto make_optuple_wrapper(const funcT& func, const input_flowsT& inputs, const output_flowsT& outputs, const std::string& name="wrapper") {
    return WrapOpTuple<funcT, input_flowsT, output_flowsT>(func, inputs, outputs, name);
}

// Class to wrap a callable with signature
//
// void op(const input_keyT&, const valuesT& ..., outputT&)
template <typename funcT, typename input_flowsT, typename output_flowsT>
class WrapOp : public Op<input_flowsT, output_flowsT, WrapOp<funcT,input_flowsT,output_flowsT>> {
    using baseT = Op<input_flowsT, output_flowsT, WrapOp<funcT,input_flowsT,output_flowsT>>;
    using input_keyT = typename input_flowsT::key_type;

    funcT func;
 public:
    WrapOp(const funcT& func, const input_flowsT& inflows, const output_flowsT& outflows, const std::string& name = "wrapper")
        : baseT(inflows,outflows,name)
        , func(func)
    {}

    // Deduction fails for this
    // void op(const input_keyT& key, const input_valuesT& ... values, output_flowsT& out) {
    //     func(key, values..., out);
    // }

    // So for a while go old school
    template <typename value0T>
    void op(const input_keyT& key, const value0T& value0, output_flowsT& out) {
        func(key, value0, out);
    }

    // So for a while go old school
    template <typename value0T, typename value1T>
    void op(const input_keyT& key, const value0T& value0, const value1T& value1, output_flowsT& out) {
        func(key, value0, value1, out);
    }
    
    // So for a while go old school
    template <typename value0T, typename value1T, typename value2T>
    void op(const input_keyT& key, const value0T& value0, const value1T& value1, const value2T& value2, output_flowsT& out) {
        func(key, value0, value1, value2, out);
    }
};

// Wrap a callable convertible to an std::function with signature
//
// void op(const input_keyT&, const input_valuesT& ..., outputT&);
template <typename funcT, typename input_flowsT, typename output_flowsT>
auto make_op_wrapper(const funcT& func,
                  const input_flowsT& inputs,
                  const output_flowsT& outputs,
                  const std::string& name="wrapper")
{
    return WrapOp<funcT, input_flowsT, output_flowsT> (func, inputs, outputs, name);
}




#endif
