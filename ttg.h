#ifndef MADNESS_TTG_BASE_H_INCLUDED
#define MADNESS_TTG_BASE_H_INCLUDED

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

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
        static constexpr bool is_a_terminal = true;
        
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

class TTGTraverse {
    std::set<const TTGOpBase*> seen;

    bool visited(const TTGOpBase* p) {
        return !seen.insert(p).second;
    }

public:
    virtual void opfunc(const TTGOpBase* op) = 0;
    
    virtual void infunc(const TTGTerminalBase* in) = 0;

    virtual void outfunc(const TTGTerminalBase* out) = 0;
    
    void reset() {seen.clear();}

    // Returns true if no null pointers encountered (i.e., if all
    // encountered terminals/operations are connected)
    bool traverse(const TTGOpBase* op) {
        if (!op) {
            std::cout << "TTGTraverse: got a null op!\n";
            return false;
        }
        
        if (visited(op)) return true;

        bool status = true;
        
        opfunc(op);

        for (auto in : op->get_inputs()) {
            if (!in) {
                std::cout << "TTGTraverse: got a null in!\n";
                status = false;
            }
            else {
                infunc(in);
            }
        }

        for (auto out: op->get_outputs()) {
            if (!out) {
                std::cout << "TTGTraverse: got a null out!\n";
                status = false;
            }
            else {
                outfunc(out);
            }
        }

        for (auto out: op->get_outputs()) {
            if (out) {
                for (auto successor : out->get_connections()) {
                    if (!successor) {
                        std::cout << "TTGTraverse: got a null successor!\n";
                        status = false;
                    }
                    else {
                        status = status && traverse(successor->get_op());
                    }
                }
            }
        }

        return status;
    }

};

class TTGVerify : private TTGTraverse {
    void opfunc(const TTGOpBase* op) {}
    void infunc(const TTGTerminalBase* in) {}
    void outfunc(const TTGTerminalBase* out) {}
public:

    bool operator()(const TTGOpBase* op) {
        reset();
        bool status = traverse(op);
        reset();
        return status;
    }
};


class TTGPrint : private TTGTraverse {
    void opfunc(const TTGOpBase* op) {
        std::cout << "op: " << (void*) op << " " << op->get_name() << " numin " << op->get_inputs().size() << " numout " << op->get_outputs().size() << std::endl;
    }
    
    void infunc(const TTGTerminalBase* in) {
        std::cout << "  in: " << in->get_index() << " " << in->get_name() << " " << in->get_key_type_str() << " " << in->get_value_type_str() << std::endl;
    }
    
    void outfunc(const TTGTerminalBase* out) {
        std::cout << " out: " << out->get_index() << " " << out->get_name() << " " << out->get_key_type_str() << " " << out->get_value_type_str() << std::endl;
    }
public:

    bool operator()(const TTGOpBase* op) {
        reset();
        bool status = traverse(op);
        reset();
        return status;
    }
};

#include <sstream>
class TTGDot : private TTGTraverse {
    std::stringstream buf;

    // Insert backslash before characters that dot is interpreting
    std::string escape(const std::string& in) {
        std::stringstream s;
        for (char c : in) {
            if (c == '<' || c == '>' || c == '"') s << "\\" << c;
            else s << c;
        }
        return s.str();
    }

    // A unique name for the node derived from the pointer
    std::string nodename(const TTGOpBase* op) {
        std::stringstream s;
        s << "n" << (void*) op;
        return s.str();
    }

    void opfunc(const TTGOpBase* op) {
        std::string opnm = nodename(op);

        buf << "        " << opnm << " [shape=record,style=filled,fillcolor=gray90,label=\"{";

        size_t count = 0;
        if (op->get_inputs().size() > 0) buf << "{";
        for (auto in : op->get_inputs()) {
            if (in) {
                if (count != in->get_index()) throw "TTGDot: lost count of ins";
                buf << " <in"
                    << count
                    << ">"
                    << " "
                    << escape("<" + in->get_key_type_str() + "," + in->get_value_type_str() + ">")
                    << " "
                    << in->get_name();
            }
            else {
                buf << " <in" << count << ">" << " unknown ";
            }
            count++;
            if (count < op->get_inputs().size()) buf << " |";
        }
        if (op->get_inputs().size() > 0) buf << "} |";

        buf << op->get_name() << " ";

        if (op->get_outputs().size() > 0) buf << " | {";

        count = 0;
        for (auto out: op->get_outputs()) {
            if (out) {
                if (count != out->get_index()) throw "TTGDot: lost count of outs";
                buf << " <out"
                    << count
                    << ">"
                    << " "
                    << escape("<" + out->get_key_type_str() + "," + out->get_value_type_str() + ">")
                    << " "
                    << out->get_name();
            }
            else {
                buf << " <out" << count << ">" << " unknown ";
            }
            count++;
            if (count < op->get_outputs().size()) buf << " |";
        }

        if (op->get_outputs().size() > 0) buf << "}";
        
        buf << " } \"];\n";

        for (auto out: op->get_outputs()) {
            if (out) {
                for (auto successor : out->get_connections()) {
                    if (successor) {
                        buf << opnm << ":out" << out->get_index() << ":s -> " << nodename(successor->get_op()) << ":in" << successor->get_index() << ":n;\n";
                    }
                }
            }
        }
    }
 
    void infunc(const TTGTerminalBase* in) {}
 
    void outfunc(const TTGTerminalBase* out) {}

public:

    std::string operator()(const TTGOpBase* op) {
        reset();
        buf.str( std::string() );
        buf.clear();

        buf << "digraph G {\n";
        buf << "        ranksep=1.5;\n";
        traverse(op);
        buf << "}\n";
        
        reset();
        std::string result = buf.str();
        buf.str( std::string() );
        buf.clear();        
        
        return result;
    }
};
    

#endif // MADNESS_TTG_BASE_H_INCLUDED
