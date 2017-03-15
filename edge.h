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

static int static_global_function_id = 0;

template <typename keyT, typename valueT>
class BaseEdge {

protected:

    // Can bring these back when store InEdges inside Op as planned
    //BaseEdge(const BaseEdge& a) = delete;
    //BaseEdge(BaseEdge&& a) = delete;
    BaseEdge& operator=(const BaseEdge& a) = delete;

private:

    bool connected = false;

public:
    const parsec_flow_t* flow;
    int function_id;

    BaseEdge(const parsec_flow_t* f, int fid) : flow(f), function_id(fid) { }
    
    void send(const keyT& key, const valueT& value) const {
        assert(connected);
        std::cout << " Send: " << key << std::endl;
        // DO PaRSEC magic
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

    InEdge(const parsec_flow_t* f, int fid) : BaseEdge<keyT, valueT>(f, fid) {}
};


// The graph edge connecting an output/result of a task with an
// input/argument of a successor task.
//
// It is connected to an input using the connect method (i.e., outedge.connect(inedge))
template <typename keyT, typename valueT>
    class OutEdge : public BaseEdge<keyT,valueT> {
 public:

    OutEdge(const parsec_flow_t* f, int fid) : BaseEdge<keyT,valueT>(f, fid) {}
 
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

        (*(parsec_flow_t**)&(this->flow))->dep_out[0] = indep;
        for( i = 0; NULL != in.flow->dep_in[i]; i++);
        (*(parsec_flow_t**)&(in.flow))->dep_in[i] = outdep;
    }
};

// Data/functionality common to all Ops
class BaseOp {
    static bool trace; // If true prints trace of all assignments and all op invocations
    static int count;  // Counts number of instances (to explore if cycles are inhibiting garbage collection)
    std::string name;

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
    
    // Used to set the i'th argument
    template <std::size_t i, typename valueT>
    void set_arg(const keyT& key, const valueT& value) {
        if (tracing()) std::cout << get_name() << " : " << key << ": setting argument : " << i << std::endl;
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
            static_cast<derivedT*>(this)->op(key, args.t);
            cache.erase(key);
            //create PaRSEC task
            // and give it to the scheduler
        }
    }
    
    Op(const Op& other) = delete;
    
    Op& operator=(const Op& other) = delete;
    
    Op (Op&& other) = delete;
    
public:
    Op(const std::string& name = std::string("unnamed op")) : BaseOp(name) {
        int i;
        
        self.name = name.c_str();
        self.function_id = static_global_function_id++;
        self.nb_parameters = 1;
        self.nb_locals = 0;
        self.nb_flows = std::max((int)numargs, (int)std::tuple_size<output_edgesT>::value);
        
        self.incarnations = (__parsec_chore_t *) malloc(2 * sizeof(__parsec_chore_t));
        self.incarnations[0].type = PARSEC_DEV_CPU;
        self.incarnations[0].evaluate = NULL;
        self.incarnations[0].hook = 0;
        self.incarnations[1].type = PARSEC_DEV_NONE;
        self.incarnations[1].evaluate = NULL;
        self.incarnations[1].hook = NULL;

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

        for( i = 0; i < std::tuple_size<output_edgesT>::value; i++ ) {
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
    
         *(int*)&self.flags = 0;
         *(int*)&self.dependencies_goal = 0;
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

    static void static_op(void* derived_ptr, void* key_ptr, void* dc_ptr_list) {
        ((*derivedT)derived_ptr)->op(*(keyT*)key_ptr, *(tuple<inputs>)(dc_ptr_list));
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
        return edgeT(this->self.in[i], this->self.function_id);
    }
    // Returns the output edge i to facilitate connection
    template <int i>
    typename std::tuple_element<i, output_edgesT>::type
    out() {
        using edgeT = typename std::tuple_element<i, output_edgesT>::type;
        return edgeT(this->self.out[i], this->self.function_id);
    }

    // Send result to successor task of output i
    template <int i, typename outkeyT, typename outvalT>
    void send(const outkeyT& key, const outvalT& value) {out<i>().send(key,value);}

    // Broadcast result to successor tasks of output i
    template <int i, typename outkeysT, typename outvalT>
    void broadcast(const outkeysT& keys, const outvalT& value) {out<i>().broadcast(keys,value);}
};

extern "C"{
    typedef struct my_op_s {
        PARSEC_MINIMAL_EXECUTION_CONTEXT
#if defined(PARSEC_PROF_TRACE)
        parsec_profile_ddesc_info_t prof_info;
#endif /* defined(PARSEC_PROF_TRACE) */
        void* function_template_class_ptr;
        void* object_ptr;
        int value;
    } my_op_t;

static parsec_hook_return_t hook(struct parsec_execution_unit_s* eu,
                                 parsec_execution_context_t* task);
}

static parsec_hook_return_t hook(struct parsec_execution_unit_s* eu,
                                 parsec_execution_context_t* task)
{
    my_op_t* me = (my_op_t*)task;
    Op::static_op(me->object_ptr, NULL, NULL);
    (void)eu;
}

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
        return edgeT(this->self.in[i], this->self.function_id);
    }
    // Returns the output edge i to facilitate connection
    template <std::size_t i>
    OutEdge<keyT,valueT>
    out() {
        using edgeT = OutEdge<keyT,valueT>;
        return edgeT(this->self.out[i], this->self.function_id);
    }
};



#endif
