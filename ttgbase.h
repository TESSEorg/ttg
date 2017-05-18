#ifndef MADNESS_TTG_BASE_H_INCLUDED
#define MADNESS_TTG_BASE_H_INCLUDED

#include <string>
#include <iostream>
#include <tuple>
#include <vector>
#include <functional>
#include <memory>
#include <cassert>

#include "demangle.h"

    class TTGOpBase; // forward decl
    
    /// Provides basic information and DAG connectivity (eventually statistics, etc.)
    class TTGTerminalBase {
        TTGOpBase* op;     //< Pointer to containing operation
        size_t n;          //< Index of terminal
        std::string name;  //< Name of terminal
        std::string key_type_str; //< String describing key type
        std::string value_type_str; //< String describing value type

        std::vector<TTGTerminalBase*> successors;
        
        TTGTerminalBase(const TTGTerminalBase&) = delete;
        TTGTerminalBase(TTGTerminalBase&&);
        
        
    public:
        TTGTerminalBase() : op(0), n(0),name("") {}
        
        void set(TTGOpBase* op, size_t index, const std::string& name, const std::string& key_type_str, const std::string& value_type_str) {
            this->op = op;
            this->n = index;
            this->name = name;
            this->key_type_str = key_type_str;
            this->value_type_str = value_type_str;
        }
        
        /// Return ptr to containing op
        TTGOpBase* get_op() const {
            if (!op) throw "TTGTerminalBase:get_op() but op is null";
            return op;
        }
        
        /// Returns index of terminal
        size_t get_index() const {
            if (!op) throw "TTGTerminalBase:get_index() but op is null";
            return n;
        }
        
        /// Returns name of terminal
        const std::string& get_name() const {
            if (!op) throw "TTGTerminalBase:get_name() but op is null";
            return name;
        }

        const std::string& get_key_type_str() const {
            if (!op) throw "TTGTerminalBase:get_key_type_str() but op is null";
            return key_type_str;
        }
        const std::string& get_value_type_str() const {
            if (!op) throw "TTGTerminalBase:get_value_type_str() but op is null";
            return value_type_str;
        }
        
        /// Add directed connection (this --> successor)
        void connect_base(TTGTerminalBase* successor) {successors.push_back(successor);}

        /// Get connections to successors
        const std::vector<TTGTerminalBase*>& get_connections() const {return successors;}
        
        virtual ~TTGTerminalBase() {}
    };
    
    /// Provides basic information and DAG connectivity (eventually statistics, etc.)
    class TTGOpBase {
        static bool trace; //< If true prints trace of all assignments and all op invocations
        
        std::string name;
        std::vector<TTGTerminalBase*> inputs;
        std::vector<TTGTerminalBase*> outputs;
        bool trace_instance; //< If true traces just this instance
        
        // Default copy/move/assign all OK
        
        void set_input(size_t i, TTGTerminalBase* t) {
            if (i >= inputs.size()) throw("out of range i setting input");
            inputs[i] = t;
        }
        
        void set_output(size_t i, TTGTerminalBase* t) {
            if (i >= outputs.size()) throw("out of range i setting output");
            outputs[i] = t;
        }
        
        template <typename terminalT, std::size_t i, typename setfuncT>
        void register_terminal(terminalT& term, const std::string& name, const setfuncT setfunc) {
            term.set(this, i, name, demangled_type_name<typename terminalT::key_type>(), demangled_type_name<typename terminalT::value_type>());
            (this->*setfunc)(i, &term);
        }
        
        template<std::size_t...IS, typename terminalsT, typename namesT, typename setfuncT>
        void register_terminals(std::index_sequence<IS...>, terminalsT& terms, const namesT& names, const setfuncT setfunc) {
            int junk[] = {0,(register_terminal<typename std::tuple_element<IS,terminalsT>::type, IS>(std::get<IS>(terms),names[IS],setfunc),0)...}; junk[0]++;
        }

        
    public:

        TTGOpBase(const std::string& name,
                  size_t numins,
                  size_t numouts)
            : name(name)
            , inputs(numins)
            , outputs(numouts)
            , trace_instance(false)
        {
        }
        
        /// Sets trace to value and returns previous setting
        static bool set_trace_all(bool value) {std::swap(trace,value); return value;}
        bool set_trace_instance(bool value) {std::swap(trace_instance,value); return value;}
        bool get_trace() {return trace || trace_instance;}
        bool tracing() {return get_trace();}
        
        void set_name(const std::string& name) {this->name = name;}
        const std::string& get_name() const {return name;}
        
        const std::vector<TTGTerminalBase*>& get_inputs() const {return inputs;}
        const std::vector<TTGTerminalBase*>& get_outputs() const {return outputs;}

        template <typename terminalsT, typename namesT>
        void register_input_terminals(terminalsT& terms, const namesT& names) {
            register_terminals(std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms, names, &TTGOpBase::set_input);
        }
        
        template <typename terminalsT, typename namesT>
        void register_output_terminals(terminalsT& terms, const namesT& names) {
            register_terminals(std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms, names, &TTGOpBase::set_output);
        }

        virtual ~TTGOpBase() {}
    };
    
    // With more than one source file this will need to be moved
    bool TTGOpBase::trace = false;

#endif // MADNESS_TTG_BASE_H_INCLUDED
