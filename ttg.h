#ifndef TTG_H_INCLUDED
#define TTG_H_INCLUDED

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "demangle.h"

namespace ttg {

class OpBase;  // forward decl

/// Provides basic information and DAG connectivity (eventually statistics,
/// etc.)
class TerminalBase {
  OpBase* op;                  //< Pointer to containing operation
  size_t n;                    //< Index of terminal
  std::string name;            //< Name of terminal
  std::string key_type_str;    //< String describing key type
  std::string value_type_str;  //< String describing value type

  std::vector<TerminalBase*> successors;

  TerminalBase(const TerminalBase&) = delete;
  TerminalBase(TerminalBase&&);

 public:
  static constexpr bool is_a_terminal = true;

  TerminalBase() : op(0), n(0), name("") {}

  void set(OpBase* op, size_t index, const std::string& name,
           const std::string& key_type_str, const std::string& value_type_str) {
    this->op = op;
    this->n = index;
    this->name = name;
    this->key_type_str = key_type_str;
    this->value_type_str = value_type_str;
  }

  /// Return ptr to containing op
  OpBase* get_op() const {
    if (!op) throw "ttg::TerminalBase:get_op() but op is null";
    return op;
  }

  /// Returns index of terminal
  size_t get_index() const {
    if (!op) throw "ttg::TerminalBase:get_index() but op is null";
    return n;
  }

  /// Returns name of terminal
  const std::string& get_name() const {
    if (!op) throw "ttg::TerminalBase:get_name() but op is null";
    return name;
  }

  const std::string& get_key_type_str() const {
    if (!op) throw "ttg::TerminalBase:get_key_type_str() but op is null";
    return key_type_str;
  }
  const std::string& get_value_type_str() const {
    if (!op) throw "ttg::TerminalBase:get_value_type_str() but op is null";
    return value_type_str;
  }

  /// Add directed connection (this --> successor)
  void connect_base(TerminalBase* successor) {
    successors.push_back(successor);
  }

  /// Get connections to successors
  const std::vector<TerminalBase*>& get_connections() const {
    return successors;
  }

  virtual ~TerminalBase() {}
};

/// Provides basic information and DAG connectivity (eventually statistics,
/// etc.)
class OpBase {
  static bool
      trace;  //< If true prints trace of all assignments and all op invocations

  std::string name;
  std::vector<TerminalBase*> inputs;
  std::vector<TerminalBase*> outputs;
  bool trace_instance;  //< If true traces just this instance

  // Default copy/move/assign all OK

  void set_input(size_t i, TerminalBase* t) {
    if (i >= inputs.size()) throw("out of range i setting input");
    inputs[i] = t;
  }

  void set_output(size_t i, TerminalBase* t) {
    if (i >= outputs.size()) throw("out of range i setting output");
    outputs[i] = t;
  }

  template <typename terminalT, std::size_t i, typename setfuncT>
  void register_terminal(terminalT& term, const std::string& name,
                         const setfuncT setfunc) {
    term.set(this, i, name, demangled_type_name<typename terminalT::key_type>(),
             demangled_type_name<typename terminalT::value_type>());
    (this->*setfunc)(i, &term);
  }

  template <std::size_t... IS, typename terminalsT, typename namesT,
            typename setfuncT>
  void register_terminals(std::index_sequence<IS...>, terminalsT& terms,
                          const namesT& names, const setfuncT setfunc) {
    int junk[] = {
        0, (register_terminal<typename std::tuple_element<IS, terminalsT>::type,
                              IS>(std::get<IS>(terms), names[IS], setfunc),
            0)...};
    junk[0]++;
  }

 public:
  OpBase(const std::string& name, size_t numins, size_t numouts)
      : name(name), inputs(numins), outputs(numouts), trace_instance(false) {}

  /// Sets trace to value and returns previous setting
  static bool set_trace_all(bool value) {
    std::swap(trace, value);
    return value;
  }
  bool set_trace_instance(bool value) {
    std::swap(trace_instance, value);
    return value;
  }
  bool get_trace() { return trace || trace_instance; }
  bool tracing() { return get_trace(); }

  void set_name(const std::string& name) { this->name = name; }
  const std::string& get_name() const { return name; }

  const std::vector<TerminalBase*>& get_inputs() const { return inputs; }
  const std::vector<TerminalBase*>& get_outputs() const { return outputs; }

  template <typename terminalsT, typename namesT>
  void register_input_terminals(terminalsT& terms, const namesT& names) {
    register_terminals(
        std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms,
        names, &OpBase::set_input);
  }

  template <typename terminalsT, typename namesT>
  void register_output_terminals(terminalsT& terms, const namesT& names) {
    register_terminals(
        std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms,
        names, &OpBase::set_output);
  }

  virtual ~OpBase() {}
};

// With more than one source file this will need to be moved
bool OpBase::trace = false;

class Traverse {
  std::set<const OpBase*> seen;

  bool visited(const OpBase* p) { return !seen.insert(p).second; }

 public:
  virtual void opfunc(const OpBase* op) = 0;

  virtual void infunc(const TerminalBase* in) = 0;

  virtual void outfunc(const TerminalBase* out) = 0;

  void reset() { seen.clear(); }

  // Returns true if no null pointers encountered (i.e., if all
  // encountered terminals/operations are connected)
  bool traverse(const OpBase* op) {
    if (!op) {
      std::cout << "ttg::Traverse: got a null op!\n";
      return false;
    }

    if (visited(op)) return true;

    bool status = true;

    opfunc(op);

    for (auto in : op->get_inputs()) {
      if (!in) {
        std::cout << "ttg::Traverse: got a null in!\n";
        status = false;
      } else {
        infunc(in);
      }
    }

    for (auto out : op->get_outputs()) {
      if (!out) {
        std::cout << "ttg::Traverse: got a null out!\n";
        status = false;
      } else {
        outfunc(out);
      }
    }

    for (auto out : op->get_outputs()) {
      if (out) {
        for (auto successor : out->get_connections()) {
          if (!successor) {
            std::cout << "ttg::Traverse: got a null successor!\n";
            status = false;
          } else {
            status = status && traverse(successor->get_op());
          }
        }
      }
    }

    return status;
  }
};

class Verify : private Traverse {
  void opfunc(const OpBase* op) {}
  void infunc(const TerminalBase* in) {}
  void outfunc(const TerminalBase* out) {}

 public:
  bool operator()(const OpBase* op) {
    reset();
    bool status = traverse(op);
    reset();
    return status;
  }
};

class Print : private Traverse {
  void opfunc(const OpBase* op) {
    std::cout << "op: " << (void*)op << " " << op->get_name() << " numin "
              << op->get_inputs().size() << " numout "
              << op->get_outputs().size() << std::endl;
  }

  void infunc(const TerminalBase* in) {
    std::cout << "  in: " << in->get_index() << " " << in->get_name() << " "
              << in->get_key_type_str() << " " << in->get_value_type_str()
              << std::endl;
  }

  void outfunc(const TerminalBase* out) {
    std::cout << " out: " << out->get_index() << " " << out->get_name() << " "
              << out->get_key_type_str() << " " << out->get_value_type_str()
              << std::endl;
  }

 public:
  bool operator()(const OpBase* op) {
    reset();
    bool status = traverse(op);
    reset();
    return status;
  }
};

class Dot : private Traverse {
  std::stringstream buf;

  // Insert backslash before characters that dot is interpreting
  std::string escape(const std::string& in) {
    std::stringstream s;
    for (char c : in) {
      if (c == '<' || c == '>' || c == '"')
        s << "\\" << c;
      else
        s << c;
    }
    return s.str();
  }

  // A unique name for the node derived from the pointer
  std::string nodename(const OpBase* op) {
    std::stringstream s;
    s << "n" << (void*)op;
    return s.str();
  }

  void opfunc(const OpBase* op) {
    std::string opnm = nodename(op);

    buf << "        " << opnm
        << " [shape=record,style=filled,fillcolor=gray90,label=\"{";

    size_t count = 0;
    if (op->get_inputs().size() > 0) buf << "{";
    for (auto in : op->get_inputs()) {
      if (in) {
        if (count != in->get_index()) throw "ttg::Dot: lost count of ins";
        buf << " <in" << count << ">"
            << " "
            << escape("<" + in->get_key_type_str() + "," +
                      in->get_value_type_str() + ">")
            << " " << in->get_name();
      } else {
        buf << " <in" << count << ">"
            << " unknown ";
      }
      count++;
      if (count < op->get_inputs().size()) buf << " |";
    }
    if (op->get_inputs().size() > 0) buf << "} |";

    buf << op->get_name() << " ";

    if (op->get_outputs().size() > 0) buf << " | {";

    count = 0;
    for (auto out : op->get_outputs()) {
      if (out) {
        if (count != out->get_index()) throw "ttg::Dot: lost count of outs";
        buf << " <out" << count << ">"
            << " "
            << escape("<" + out->get_key_type_str() + "," +
                      out->get_value_type_str() + ">")
            << " " << out->get_name();
      } else {
        buf << " <out" << count << ">"
            << " unknown ";
      }
      count++;
      if (count < op->get_outputs().size()) buf << " |";
    }

    if (op->get_outputs().size() > 0) buf << "}";

    buf << " } \"];\n";

    for (auto out : op->get_outputs()) {
      if (out) {
        for (auto successor : out->get_connections()) {
          if (successor) {
            buf << opnm << ":out" << out->get_index() << ":s -> "
                << nodename(successor->get_op()) << ":in"
                << successor->get_index() << ":n;\n";
          }
        }
      }
    }
  }

  void infunc(const TerminalBase* in) {}

  void outfunc(const TerminalBase* out) {}

 public:
  std::string operator()(const OpBase* op) {
    reset();
    buf.str(std::string());
    buf.clear();

    buf << "digraph G {\n";
    buf << "        ranksep=1.5;\n";
    traverse(op);
    buf << "}\n";

    reset();
    std::string result = buf.str();
    buf.str(std::string());
    buf.clear();

    return result;
  }
};

template <typename keyT, typename valueT>
class Edge;  // Forward decl.

template <typename keyT, typename valueT>
class In : public TerminalBase {
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
  In(In&& other) = delete;
  In(const In& other) = delete;
  In& operator=(const In& other) = delete;
  In& operator=(const In&& other) = delete;

 public:
  In() : initialized(false) {}

  In(const send_callback_type& send_callback)
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
class Out : public TerminalBase {
 public:
  typedef valueT value_type;
  typedef keyT key_type;
  typedef In<keyT, valueT> input_terminal_type;
  typedef Edge<keyT, valueT> edge_type;
  static constexpr bool is_an_output_terminal = true;

 private:
  std::vector<input_terminal_type*> successors;

  // No moving, copying, assigning permitted
  Out(Out&& other) = delete;
  Out(const Out& other) = delete;
  Out& operator=(const Out& other) = delete;
  Out& operator=(const Out&& other) = delete;

 public:
  Out() {}

  void connect(input_terminal_type& successor) {
    successors.push_back(&successor);
    static_cast<TerminalBase*>(this)->connect_base(&successor);
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
    std::vector<In<keyT, valueT>*> outs;
    std::vector<Out<keyT, valueT>*> ins;

    EdgeImpl() : name(""), outs(), ins() {}

    EdgeImpl(const std::string& name) : name(name), outs(), ins() {}

    void set_in(Out<keyT, valueT>* in) {
      if (ins.size())
        std::cout << "Edge: " << name << " : has multiple inputs" << std::endl;
      ins.push_back(in);
      try_to_connect_new_in(in);
    }

    void set_out(In<keyT, valueT>* out) {
      if (outs.size())
        std::cout << "Edge: " << name << " : has multiple outputs" << std::endl;
      outs.push_back(out);
      try_to_connect_new_out(out);
    }

    void try_to_connect_new_in(Out<keyT, valueT>* in) const {
      for (auto out : outs)
        if (in && out) in->connect(*out);
    }

    void try_to_connect_new_out(In<keyT, valueT>* out) const {
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
  typedef In<keyT, valueT> input_terminal_type;
  typedef Out<keyT, valueT> output_terminal_type;
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

  void set_in(Out<keyT, valueT>* in) const {
    for (auto& edge : p) edge->set_in(in);
  }

  void set_out(In<keyT, valueT>* out) const {
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

}  // namespace ttg

#endif  // TTG_H_INCLUDED
