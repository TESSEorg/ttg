#ifndef MADNESS_TTG_H_INCLUDED
#define MADNESS_TTG_H_INCLUDED

#include <string>
#include <iostream>
#include <tuple>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>
#include <map>

#include <madness/world/MADworld.h>
#include <madness/world/worldhashmap.h>
#include <madness/world/worldtypes.h>
#include "ttgbase.h"

template <typename keyT, typename valueT> class EdgeArray;

template <typename keyT, typename valueT>
class TTGIn : public TTGTerminalBase {
public:
    typedef valueT value_type;
    typedef keyT key_type;
    typedef EdgeArray<keyT,valueT> edge_type;
    using send_callback_type = std::function<void(const keyT&, const valueT&)>;

private:
    bool initialized;
    send_callback_type send_callback;

    TTGIn(TTGIn&& other) = delete;
    TTGIn(const TTGIn& other) = delete;
    TTGIn& operator=(const TTGIn& other) = delete; 

public:
    
    TTGIn() : initialized(false) {}

    TTGIn(const send_callback_type& send_callback)
        : initialized(true)
        , send_callback(send_callback)
    {}

    // callback (std::function) is used to erase the operator type and argument index
    void set_callback(const send_callback_type& send_callback) {
        initialized = true;
        this->send_callback = send_callback;
    }

    void send(const keyT& key, const valueT& value) {
        if (!initialized) throw "sending to uninitialzed callback";
        send_callback(key, value);
    }

    // An optimized implementation will need a separate callback for broadcast with a specific value for rangeT
    template <typename rangeT>
    void broadcast(const rangeT& keylist, const valueT& value) {
        if (!initialized) throw "broadcasting to uninitialzed callback";
        for (auto key : keylist) send(key,value);
    }
};


// Output terminal
template <typename keyT, typename valueT>
class TTGOut : public TTGTerminalBase {
public:
    typedef valueT value_type;
    typedef keyT key_type;
    typedef TTGIn<keyT,valueT> input_terminal_type;
    typedef EdgeArray<keyT,valueT> edge_type;

private:
    std::vector<input_terminal_type*> successors;
    
    TTGOut(TTGOut&& other) = delete;
    TTGOut(const TTGOut& other) = delete;
    TTGOut& operator=(const TTGOut& other) = delete;

public:

    TTGOut() {}

    void connect(input_terminal_type& successor) {
        std::cout << "actually connecting\n";
        successors.push_back(&successor);
        static_cast<TTGTerminalBase*>(this)->connect_base(&successor);
    }

    void send(const keyT& key, const valueT& value) {
        for (auto successor : successors) successor->send(key,value);
    }

    // An optimized implementation will need a separate callback for broadcast with a specific value for rangeT
    template <typename rangeT>
    void broadcast(const rangeT& keylist, const valueT& value) {
        for (auto successor : successors) successor->broadcast(keylist,value);
    }
};

template <typename keyT, typename valueT>
class Edge {
private:

    struct EdgePimpl {
        TTGIn<keyT,valueT>* out;
        TTGOut<keyT,valueT>* in;

        EdgePimpl() : out(0), in(0) {}


        ~EdgePimpl() {
            if ((!in) || (!out)) {
                std::cerr << "Edge: destroying edge pimpl with either in or out not assigned --- DAG may be incomplete" << std::endl;
            }
        }
        
    };

    mutable std::shared_ptr<EdgePimpl> p; // Need shallow copy semantics

    void try_to_connect() const {
        if (p->in && p->out) {
            p->in->connect(*(p->out)); // Easy to get your inputs and outputs mixed up!
        }
    }

public:
    Edge() : p(new EdgePimpl) {}

    void set_in(TTGOut<keyT,valueT>* in) const {
        if (!p) throw "Edge: not constructed before use!";
        if (p->in) throw "Edge: setting input twice";
        p->in = in;
        try_to_connect();
    }

    void set_out(TTGIn<keyT,valueT>* out) const {
        if (!p) throw "Edge: not constructed before use!";
        if (p->out) throw "Edge: setting output twice";
        p->out = out;
        try_to_connect();
    }

    template <typename stream>
    void print(stream& s) const {
        TTGIn<keyT,valueT>* out = 0;
        TTGOut<keyT,valueT>* in = 0;
        if (p) {
            out = p->out;
            in  = p->in;
        }
        std::cout << "Edge(pimpl=" << (void*) p.get() << ", in=" << (void*) in << ", out=" << (void*) out << ")";
    }
};

// Multiple edges that have the same type since they are originating
// from or terminating upon the same terminal ... behaves just like
// a single edge in terms of connecting
template <typename keyT, typename valueT>
class EdgeArray {
private:
    mutable std::vector<Edge<keyT,valueT>> edges;

public:
    template <typename...argsT>
    EdgeArray(const argsT&... args) : edges({args...}) {}

    EdgeArray() : edges() {}

    void set_in(TTGOut<keyT,valueT>* in) const {
        for (auto& edge : edges) edge.set_in(in);
    }

    void set_out(TTGIn<keyT,valueT>* out) const {
        for (auto& edge : edges) edge.set_out(out);
    }

    template <typename stream>
    void print(stream& s) const {
        for (auto& edge : edges) edge.print(s);
    }
};

template <typename keyT, typename valueT>
std::ostream& operator<<(std::ostream& s, const Edge<keyT,valueT>& edge) {
    edge.print(s);
    return s;
}

template <typename keyT, typename valueT>
std::ostream& operator<<(std::ostream& s, const EdgeArray<keyT,valueT>& edges) {
    edges.print(s);
    return s;
}

// All the types have to be the same ... just using valuesT for variadic args
template <typename keyT, typename...valuesT>
auto
fuse(const Edge<keyT,valuesT>&...args) {
    using valueT = typename std::tuple_element<0, std::tuple<valuesT...>>::type; // grab first type
    // std::vector<Edge<keyT,valueT>> v = {args...};
    // return EdgeArray<keyT,valueT>(v);
    return EdgeArray<keyT,valueT>(args...);
}

std::tuple<> empty() {return std::make_tuple<>();}

template <typename...inedgesT>
auto edges(const inedgesT& ... args) {
    return std::make_tuple(args...);
}

template <size_t i, typename keyT, typename valueT, typename...output_terminalsT>
void send(const keyT& key, const valueT& value, std::tuple<output_terminalsT...>& t) {
    std::get<i>(t).send(key,value);
}

template <size_t i, typename rangeT, typename valueT, typename...output_terminalsT>
void broadcast(const rangeT& keylist, const valueT& value, std::tuple<output_terminalsT...>& t) {
    std::get<i>(t).broadcast(keylist,value);
}

template <typename keyT, typename output_terminalsT, typename derivedT, typename...input_valueTs>
class TTGOp
    : public TTGOpBase
    , public madness::WorldObject< TTGOp<keyT,output_terminalsT,derivedT,input_valueTs...> > {
private:
    madness::World& world;
    std::shared_ptr<madness::WorldDCPmapInterface<keyT>> pmap;
    
    // Make type of tuple of edges from type of tuple of terminals
    template <typename termsT> struct munge {};
    template <typename...termsT> struct munge<std::tuple<termsT...>> {
        typedef std::tuple<typename termsT::edge_type...> type;
    };

    using opT = TTGOp<keyT, output_terminalsT, derivedT, input_valueTs...>;
    using worldobjT = madness::WorldObject<opT>;

public:
    static constexpr int numins = sizeof...(input_valueTs);  // number of input arguments
    static constexpr int numouts = std::tuple_size<output_terminalsT>::value; // number of outputs or results

    using input_values_tuple_type = std::tuple<input_valueTs...>;
    using input_terminals_type = std::tuple<TTGIn<keyT,input_valueTs>...>;
    using input_edges_type = typename munge<input_terminals_type>::type;

    using output_terminals_type = output_terminalsT;
    using output_edges_type = typename munge<output_terminalsT>::type;

private:

    input_terminals_type input_terminals;
    output_terminalsT output_terminals;

    struct OpArgs : madness::TaskInterface {
        int counter;                     // Tracks the number of arguments set
        std::array<bool,numins> argset; // Tracks if a given arg is already set;
        input_values_tuple_type t;           // The input values
        derivedT* derived;               // Pointer to derived class instance
        keyT key;                        // Task key
        
        OpArgs() : counter(numins), argset(), t() {}

        void run(madness::World& world) {
            derived->op(key, t, derived->output_terminals);            
        }

        virtual ~OpArgs()  {}            // Will be deleted via TaskInterface*
    };

    using cacheT = madness::ConcurrentHashMap<keyT, OpArgs*>;
    using accessorT = typename cacheT::accessor;
    cacheT cache;

    // Used to set the i'th argument
    template <std::size_t i>
    void set_arg(const keyT& key, const typename std::tuple_element<i,input_values_tuple_type>::type& value) {

        using valueT = typename std::tuple_element<i,input_terminals_type>::type;

        ProcessID owner = pmap->owner(key);

        if (owner != world.rank()) {
            if (tracing()) std::cout << world.rank() << ":" << get_name() << " : " << key << ": forwarding setting argument : " << i << std::endl;
            worldobjT::send(owner, &opT:: template set_arg<i>, key, value);
        }
        else {
            if (tracing()) std::cout << world.rank() << ":" << get_name() << " : " << key << ": setting argument : " << i << std::endl;
            
            accessorT acc;
            if (cache.insert(acc, key)) acc->second = new OpArgs(); // It will be deleted by the task q
            OpArgs* args = acc->second;

            if (args->argset[i]) {
                std::cerr << world.rank() << ":" << get_name() << " : " << key << ": error argument is already set : " << i << std::endl;
                throw "bad set arg";
            }
            args->argset[i] = true;        
            std::get<i>(args->t) = value;
            args->counter--;
            if (args->counter == 0) {
                if (tracing()) std::cout << world.rank() << ":" << get_name() << " : " << key << ": submitting task for op " << std::endl;
                args->derived = static_cast<derivedT*>(this);
                args->key = key;

                world.taskq.add(args);

                //world.taskq.add(static_cast<derivedT*>(this), &derivedT::op, key, args.t);

                // if (tracing()) std::cout << world.rank() << ":" << get_name() << " : " << key << ": invoking op " << std::endl;
                // static_cast<derivedT*>(this)

                cache.erase(key);
            }
        }
    }

    // Used to generate tasks with no input arguments
    void set_arg_empty(const keyT& key) {
        ProcessID owner = pmap->owner(key);

        if (owner != world.rank()) {
            if (tracing()) std::cout << world.rank() << ":" << get_name() << " : " << key << ": forwarding no-arg task: " << std::endl;
            worldobjT::send(owner, &opT::set_arg_empty, key);
        }
        else {
            accessorT acc;
            if (cache.insert(acc, key)) acc->second = new OpArgs(); // It will be deleted by the task q
            OpArgs* args = acc->second;

            if (tracing()) std::cout << world.rank() << ":" << get_name() << " : " << key << ": submitting task for op " << std::endl;
            args->derived = static_cast<derivedT*>(this);
            args->key = key;

            world.taskq.add(args);

            cache.erase(key);
        }
    }
    
    // Used by invoke to set all arguments associated with a task
    template <size_t...IS>
    void set_args(std::index_sequence<IS...>, const keyT& key, const input_values_tuple_type& args) {
        int junk[] = {0,(set_arg<IS>(key,std::get<IS>(args)),0)...}; junk[0]++;
    }
    
    // Copy/assign forbidden
    TTGOp(const TTGOp& other) = delete;
    TTGOp& operator=(const TTGOp& other) = delete;
    
    // Moving will be supported eventually
    TTGOp (TTGOp&& other) = delete;
    TTGOp& operator=(TTGOp&& other) = delete;

    // Registers the callback for the i'th input terminal
    template <typename terminalT, std::size_t i>
    void register_input_callback(terminalT& input) {
        using callbackT = std::function<void(const typename terminalT::key_type&, const typename terminalT::value_type&)>;
        auto callback = [this](const typename terminalT::key_type& key, const typename terminalT::value_type& value){set_arg<i>(key,value);};
        input.set_callback(callbackT(callback));
    }
    
    template<std::size_t...IS>
    void register_input_callbacks(std::index_sequence<IS...>) {
        int junk[] = {0,(register_input_callback<typename std::tuple_element<IS,input_terminals_type>::type, IS>(std::get<IS>(input_terminals)),
                         0)...}; junk[0]++;
    }

    template<std::size_t...IS, typename inedgesT>
    void connect_my_inputs_to_incoming_edge_outputs(std::index_sequence<IS...>, inedgesT& inedges) {
        int junk[] = {0,(std::get<IS>(inedges).set_out(&std::get<IS>(input_terminals)),0)...}; junk[0]++;
    }

    template<std::size_t...IS, typename outedgesT>
    void connect_my_outputs_to_outgoing_edge_inputs(std::index_sequence<IS...>, outedgesT& outedges) {
        int junk[] = {0,(std::get<IS>(outedges).set_in(&std::get<IS>(output_terminals)),0)...}; junk[0]++;
    }

public:
    TTGOp(const std::string& name,
          const std::vector<std::string>& innames,
          const std::vector<std::string>& outnames)
        : TTGOpBase(name, numins, numouts)
        , worldobjT(madness::World::get_default())
        , world(madness::World::get_default())
        , pmap(std::make_shared<madness::WorldDCDefaultPmap<keyT>>(world))
    {
        // Cannot call in base constructor since terminals not yet constructed

        if (innames.size() != std::tuple_size<input_terminals_type>::value) throw "TTGOP: #input names != #input terminals";
        if (outnames.size() != std::tuple_size<output_terminalsT>::value) throw "TTGOP: #output names != #output terminals";
        
        register_input_terminals(input_terminals,  innames);
        register_output_terminals(output_terminals, outnames);
        
        register_input_callbacks(std::make_index_sequence<numins>{});
        
        this->process_pending();
    }
    
    TTGOp(const input_edges_type& inedges,
          const output_edges_type& outedges,
          const std::string& name,
          const std::vector<std::string>& innames,
          const std::vector<std::string>& outnames)
        : TTGOpBase(name, numins, numouts)
        , worldobjT(madness::World::get_default())
        , world(madness::World::get_default())
        , pmap(std::make_shared<madness::WorldDCDefaultPmap<keyT>>(world))
    {
        // Cannot call in base constructor since terminals not yet constructed
        if (innames.size() != std::tuple_size<input_terminals_type>::value) throw "TTGOP: #input names != #input terminals";
        if (outnames.size() != std::tuple_size<output_terminalsT>::value) throw "TTGOP: #output names != #output terminals";
        
        register_input_terminals(input_terminals,  innames);
        register_output_terminals(output_terminals, outnames);
        
        register_input_callbacks(std::make_index_sequence<numins>{});

        connect_my_inputs_to_incoming_edge_outputs(std::make_index_sequence<numins>{}, inedges);
        connect_my_outputs_to_outgoing_edge_inputs(std::make_index_sequence<numouts>{}, outedges);
        
        this->process_pending();
    }
    
    // Destructor checks for unexecuted tasks
    ~TTGOp() {
        if (cache.size() != 0) {
            std::cerr << world.rank() << ":" << "warning: unprocessed tasks in destructor of operation '" << get_name() << "'" << std::endl;
            std::cerr << world.rank() << ":" << "   T => argument assigned     F => argument unassigned" << std::endl;
            int nprint=0;
            for (auto item : cache) {
                if (nprint++ > 10) {
                    std::cerr << "   etc." << std::endl;
                    break;
                }
                std::cerr << world.rank() << ":" << "   unused: " << item.first << " : ( ";
                for (std::size_t i=0; i<numins; i++) std::cerr << (item.second->argset[i] ? "T" : "F") << " ";
                std::cerr << ")" << std::endl;
            }
        }
    }
    
    // Returns reference to input terminal i to facilitate connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    typename std::tuple_element<i, input_terminals_type>::type&
    in () {
        return std::get<i>(input_terminals);
    }

    // Returns reference to output terminal for purpose of connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    typename std::tuple_element<i, output_terminalsT>::type&
    out () {
        return std::get<i>(output_terminals);
    }

    // Manual injection of a task with all input arguments specified as a tuple
    void invoke(const keyT& key, const input_values_tuple_type& args) {
        set_args(std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{}, key, args);
    }

    // Manual injection of a task that has no arguments
    void invoke(const keyT& key) {
        set_arg_empty(key);
    }
};

// // Class to wrap a callable with signature
// //
// // void op(const input_keyT&, const std::tuple<input_valuesT...>&, std::tuple<output_terminalsT...>&)
// //
// template <typename keyT, typename funcT, typename output_terminalsT, typename...input_valuesT>
// class WrapOp : public TTGOp<keyT, output_terminalsT, WrapOp<keyT,funcT,output_terminalsT,input_valuesT...>> {
//     using baseT =     TTGOp<keyT, output_terminalsT, WrapOp<keyT,funcT,output_terminalsT,input_valuesT...>>;

//     funcT func;

//  public:
//     WrapOp(const funcT& func,
//            const baseT::input_edges_type& inedges,
//            const baseT::output_edges_type& outedges,
//            const std::string& name = "wrapper",
//            const std::vector<std::string>& innames = std::vector<std::string>(baseT::numins, "input"),
//            const std::vector<std::string>& outnames= std::vector<std::string>(baseT::numins, "output"))
//         : baseT(inedges, outedges, name, innames, outnames)
//         , func(func)
//     {}
    
//     void op(const keyT& key, const typename baseT::input_values_tuple_type& args, output_terminalsT& out) {
//         func(key, args, out);
//     }
// };

// // Factory function to assist in wrapping a callable with signature
// //
// // void op(const input_keyT&, const std::tuple<input_valuesT...>&, std::tuple<output_terminalsT...>&)
// template <typename funcT, typename input_flowsT, typename output_flowsT>
// auto make_optuple_wrapper(const funcT& func, const input_flowsT& inputs, const output_flowsT& outputs, const std::string& name="wrapper") {
//     return WrapOpTuple<funcT, input_flowsT, output_flowsT>(func, inputs, outputs, name);
// }




#endif // MADNESS_TTG_H_INCLUDED
