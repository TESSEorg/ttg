#ifndef MADNESS_EDGE_H_INCLUDED
#define MADNESS_EDGE_H_INCLUDED

#include <string>
#include <iostream>
#include <tuple>
#include <vector>
#include <functional>
#include <memory>
#include <map>
#include <cassert>
#include <algorithm>

#include <parsec.h>
#include <parsec/parsec_internal.h>
#include <parsec/devices/device.h>
#include <parsec/data_internal.h>
#include <parsec/scheduling.h>

extern "C" parsec_execution_unit_t* eu;
extern "C" parsec_handle_t* handle;

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


template <typename keyT, typename valueT>
class BaseEdge {

protected:

    // Can bring these back when store InEdges inside Op as planned
    //BaseEdge(const BaseEdge& a) = delete;
    //BaseEdge(BaseEdge&& a) = delete;
    BaseEdge& operator=(const BaseEdge& a) = delete;

private:


public:
    const parsec_flow_t* flow;
    int function_id;

    BaseEdge() : flow(NULL), function_id(-1) { }

    BaseEdge(const parsec_flow_t* f, int fid) : flow(f), function_id(fid) { }

    void init(const parsec_flow_t* f, int fid) {
        flow = f;
        function_id = fid;
    }
    
    template <typename rangeT>
    void broadcast(const rangeT& keylist, const valueT& value) const {
        for (auto key : keylist) send(key,value);
    }
};


// The graph edge connecting to an input/argument of a task template.
//
// It is presumably connected with an output/result of a predecessor
// task, but you can also inject data directly using the
// send/broadcast methods.
template <typename keyT, typename valueT>
class InEdge : public BaseEdge<keyT,valueT> {

protected:

    template <typename kT, typename iT, typename oT, typename dT> friend class Op;
    template <typename kT, typename vT> friend class Merge;

public:
    void (*set_arg_static_fct)(const keyT& key, const valueT& value, int function_id);

    InEdge(const parsec_flow_t* f, int fid,
           void (*set_arg_static_fct_)(const keyT& key, const valueT& value, int function_id)) :
        BaseEdge<keyT, valueT>(f, fid), set_arg_static_fct(set_arg_static_fct_) {}

    void send(const keyT& key, const valueT& value) const {
        std::cout << " Send: " << key << std::endl;

        this->set_arg_static_fct(key, value, this->flow->dep_out[0]->flow->dep_in[0]->function_id);

        // DO PaRSEC magic
    }
};


// The graph edge connecting an output/result of a task with an
// input/argument of a successor task.
//
// It is connected to an input using the connect method (i.e., outedge.connect(inedge))
template <typename keyT, typename valueT>
    class OutEdge : public BaseEdge<keyT,valueT> {
    bool initialized;
 public:
    void (*set_arg_static_fct)(const keyT& key, const valueT& value, int function_id);

    OutEdge() : initialized(false) {}
    
    void init(const parsec_flow_t* f, int fid) {
        BaseEdge<keyT, valueT>::init(f, fid);
        initialized = true;
    }
    
    bool is_initialized(void) {
        return initialized;
    }
 
    void connect(const InEdge<keyT,valueT>& in) {
        int i;
        dep_t* indep = new dep_t;
        indep->cond = NULL;
        indep->ctl_gather_nb = NULL;
        indep->function_id = this->function_id;
        indep->direct_data = NULL;
        indep->flow = in.flow;
        indep->dep_index = 0;
        indep->dep_datatype_index = 0;
        indep->belongs_to = this->flow;
        
        dep_t* outdep = new dep_t;
        outdep->cond = NULL;
        outdep->ctl_gather_nb = NULL;
        outdep->function_id = in.function_id;
        outdep->direct_data = NULL;
        outdep->flow = this->flow;
        outdep->dep_index = 0;
        outdep->dep_datatype_index = 0;
        outdep->belongs_to = in.flow;

        this->set_arg_static_fct = in.set_arg_static_fct;
        
        (*(parsec_flow_t**)&(this->flow))->dep_out[0] = indep;
        for( i = 0; NULL != in.flow->dep_in[i]; i++);
        (*(parsec_flow_t**)&(in.flow))->dep_in[i] = outdep;
    }
    
    void send(const keyT& key, const valueT& value) const {
        std::cout << " Send: " << key << std::endl;

        this->set_arg_static_fct(key, value, this->flow->dep_out[0]->flow->dep_in[0]->function_id);

        // DO PaRSEC magic
    }
};

// Data/functionality common to all Ops
class BaseOp {
    static bool trace; // If true prints trace of all assignments and all op invocations
    static int count;  // Counts number of instances (to explore if cycles are inhibiting garbage collection)
    std::string name;
    static std::map<int, BaseOp*> function_id_to_instance;
public:

    BaseOp(const std::string& name) : name(name) {count++;}
    
    // Sets trace to value and returns previous setting
    static bool set_trace(bool value) {std::swap(trace,value); return value;}

    static bool get_trace() {return trace;}

    static bool tracing() {return trace;}

    static int get_count() {return count;}

    void set_name(const std::string& name) {this->name = name;}
    
    const std::string& get_name() const {return name;}
    
    ~BaseOp() {count--;}
};

static parsec_hook_return_t do_nothing(parsec_execution_unit_t *eu, parsec_execution_context_t *task)
{
    (void)eu;
    (void)task;
    return PARSEC_HOOK_RETURN_DONE;
}

// With more than one source file this will need to be moved
bool BaseOp::trace = false;
int BaseOp::count = 0;

template <typename keyT, typename input_valuesT, typename output_edgesT, typename derivedT>
class Op : private BaseOp {
protected:
    parsec_function_t self;

private:
    static constexpr int numargs = std::tuple_size<input_valuesT>::value; // Number of input arguments

    struct OpArgs{
        int counter;                     // Tracks the number of arguments set
        std::array<bool,numargs> argset; // Tracks if a given arg is already set;
        input_valuesT t;                 // The input values
        
        OpArgs() : counter(numargs), argset(), t() {}
    };

    std::map<keyT, OpArgs> cache; // Contains tasks waiting for input to become complete

    output_edgesT outedges_cache;
    
public:
    template <std::size_t i, typename valueT>
    static void set_arg_static(const keyT& key, const valueT& value, int function_id) {
        Op* op2 = static_cast<Op*>(function_id_to_instance[function_id]); // error checking!
        
        op2->set_arg<i,valueT>(key, value);
    }
private:
    // Used to set the i'th argument
    template <std::size_t i, typename valueT>
    void set_arg(const keyT& key, const valueT& value) {
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
                reinterpret_cast<void (*)(void*)>(&Op::static_op);
            task->object_ptr = static_cast<derivedT*>(this);
            task->key = key;
            task->data[0].data_in = static_cast<parsec_data_copy_t*>(malloc(sizeof(value)));
            memcpy(task->data[0].data_in, &value, sizeof(value));
            __parsec_schedule(eu, task, 0);
        }
    }
    
    Op(const Op& other) = delete;
    
    Op& operator=(const Op& other) = delete;
    
    Op (Op&& other) = delete;
    
public:
    Op(const std::string& name = std::string("unnamed op")) : BaseOp(name) {
        int i;

        memset(&self, 0, sizeof(parsec_function_t));
        
        self.name = name.c_str();
        self.function_id = static_global_function_id++;
        self.nb_parameters = 0;
        self.nb_locals = 0;
        self.nb_flows = std::max((int)numargs, (int)std::tuple_size<output_edgesT>::value);

        function_id_to_instance[self.function_id] = this;
        
        self.incarnations = (__parsec_chore_t *) malloc(2 * sizeof(__parsec_chore_t));
        ((__parsec_chore_t*)self.incarnations)[0].type = PARSEC_DEV_CPU;
        ((__parsec_chore_t*)self.incarnations)[0].evaluate = NULL;
        ((__parsec_chore_t*)self.incarnations)[0].hook = hook;
        ((__parsec_chore_t*)self.incarnations)[1].type = PARSEC_DEV_NONE;
        ((__parsec_chore_t*)self.incarnations)[1].evaluate = NULL;
        ((__parsec_chore_t*)self.incarnations)[1].hook = NULL;

        self.release_task = do_nothing;

        for( i = 0; i < numargs; i++ ) {
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

        for( i = 0; i < (int)std::tuple_size<output_edgesT>::value; i++ ) {
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
    };
    
    // Destructor checks for unexecuted tasks
    ~Op() {
        if (cache.size() != 0) {
            std::cerr << "warning: unprocessed tasks in destructor of operation '" << get_name() << "'" << std::endl;
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

    static void static_op(parsec_execution_context_t* my_task) {
        my_op_t* task = static_cast<my_op_t*>(my_task);
        /*        
                  struct data_repo_entry_s     *data_repo;
                  struct parsec_data_copy_s    *data_in;
                  struct parsec_data_copy_s    *data_out;
        */

        ((derivedT*)task->object_ptr)->op((keyT)task->key,
                                          *reinterpret_cast<input_valuesT*>(task->data[0].data_in));
#ifdef OR_BETTER
            *static_cast<input_valuesT*>(task->data[0].data_in->device_private));
#endif
    }

#if 0
    template<std::size_t I = 0, typename... Tp>
    inline typename std::enable_if<I == sizeof...(Tp), void>::type
    set_ids(std::tuple<Tp...> &) // Unused arguments are given no names.
        { }

    template<std::size_t I = 0, typename... Tp>
    inline typename std::enable_if<I < sizeof...(Tp), void>::type
    set_ids(std::tuple<Tp...>& t)
    {
        std::get<I>(t).set_id(this->function_id, I);
        self.out[I] = &std::get<I>(t).flow;
        set_ids<I + 1, Tp...>(t);
    }
#endif
    // Returns input edge i to facilitate connection
    template <std::size_t i>
    InEdge<keyT,typename std::tuple_element<i, input_valuesT>::type>
    in () {
        using edgeT = InEdge<keyT,typename std::tuple_element<i, input_valuesT>::type>;
        return edgeT(this->self.in[i], this->self.function_id,
                     &Op::set_arg_static<i, typename std::tuple_element<i, input_valuesT>::type>);
    }
    // Returns the output edge i to facilitate connection
    template <int i>
    typename std::tuple_element<i, output_edgesT>::type&
    out() {
        if( ! (std::get<i>(outedges_cache).is_initialized()) ) {
            std::get<i>(outedges_cache).init(this->self.out[i], this->self.function_id);
        }
        return std::get<i>(outedges_cache);
    }

    // Send result to successor task of output i
    template <int i, typename outkeyT, typename outvalT>
    void send(const outkeyT& key, const outvalT& value) {out<i>().send(key,value);}

    // Broadcast result to successor tasks of output i
    template <int i, typename outkeysT, typename outvalT>
    void broadcast(const outkeysT& keys, const outvalT& value) {out<i>().broadcast(keys,value);}
};

static parsec_hook_return_t hook(struct parsec_execution_unit_s* eu,
                                 parsec_execution_context_t* task)
{
    my_op_t* me = static_cast<my_op_t*>(task);
    me->function_template_class_ptr(task);
    (void)eu;
    return PARSEC_HOOK_RETURN_DONE;
}

std::map<int, BaseOp*> BaseOp::function_id_to_instance = {};

template <typename keyT, typename valueT>
class Merge : public Op<keyT, std::tuple<valueT,valueT>, std::tuple<OutEdge<keyT, valueT>>, Merge<keyT, valueT>> {
    using baseT = Op<keyT, std::tuple<valueT,valueT>, std::tuple<OutEdge<keyT, valueT>>, Merge<keyT,valueT>>;
public:
    Merge(const std::string& name = std::string("unnamed op")) : baseT(name) {}
    
    // Returns input edge i to facilitate connection
    template <std::size_t i>
    InEdge<keyT,valueT>
    in () {
        using edgeT = InEdge<keyT,valueT>;
        return edgeT(this->self.in[i], this->self.function_id,
                     &baseT::template set_arg_static<i, valueT>);
    }
    // Returns the output edge i to facilitate connection
    template <std::size_t i>
    OutEdge<keyT,valueT>
    out() {
        using edgeT = OutEdge<keyT,valueT>;
        return edgeT(this->self.out[i], this->self.function_id);
    }

    // Used to set the i'th argument
    template <std::size_t i>
    void set_arg(const keyT& key, const valueT& value) {
        //if (this->tracing()) std::cout << get_name() << " : " << key << ": invoking op " << std::endl;

        //create PaRSEC task
        // and give it to the scheduler
        my_op_t* task = static_cast<my_op_t*>(calloc(1, sizeof(my_op_t)));
        task->function_template_class_ptr = &baseT::static_op;
        task->object_ptr = this;
        task->key = key;
        task->data[0].data_in = malloc(sizeof(value));
        memcpy(task->data[0].data_in, &value, sizeof(value));
        __parsec_schedule(eu, task, 0);
    }
};



#endif
