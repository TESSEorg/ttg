#ifndef TTG_H_INCLUDED
#define TTG_H_INCLUDED

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <boost/callable_traits.hpp>  // needed for wrap.h
#include <boost/core/demangle.hpp>

#include "util/demangle.h"
#include "util/meta.h"
#include "util/runtimes.h"
#include "util/hash.h"

#if __has_include(<mpi.h>)
#  include <mpi.h>
#endif

#define TTGUNUSED(x) ((void)(x))

namespace ttg {

  inline int rank() {
    int me = -1;
#if __has_include(<mpi.h>)
    int inited;
    MPI_Initialized(&inited);
    if (inited) {
      int fini;
      MPI_Finalized(&fini);
      if (!fini) {
        auto errcod = MPI_Comm_rank(MPI_COMM_WORLD, &me);
        assert(errcod == 0);
      }
    }
#endif
    return me;
  }

/// @brief A complete version of void

/// Void can be used interchangeably with void as key or value type, but is also hashable, etc.
/// May reduce the amount of metaprogramming relative to void.
class Void {
 public:
  Void() = default;
  template <typename T> Void(T&&) {}
};

bool operator==(const Void&, const Void&) { return true; }

bool operator!=(const Void&, const Void&) { return false; }

std::ostream& operator<<(std::ostream& os, const ttg::Void&) {
  return os;
}

static_assert(meta::is_empty_tuple_v<std::tuple<>>,"ouch");
static_assert(meta::is_empty_tuple_v<std::tuple<Void>>,"ouch");

}  // namespace ttg

namespace std {
template <>
struct hash<ttg::Void> {
  template <typename ... Args> int64_t operator()(Args&& ... args) const { return 0; }
};
}

namespace ttg {
  namespace detail {

    /// the default keymap implementation requires ttg::hash{}(key) ... use SFINAE
    /// TODO improve error messaging via more elaborate techniques e.g.
    /// https://gracicot.github.io/tricks/2017/07/01/deleted-function-diagnostic.html
    template <typename keyT, typename Enabler = void>
    struct default_keymap_impl;
    template <typename keyT>
    struct default_keymap_impl<
        keyT, std::enable_if_t<meta::has_ttg_hash_specialization_v<keyT> || meta::is_void_v<keyT>>> {
      default_keymap_impl() = default;
      default_keymap_impl(int world_size) : world_size(world_size) {}

      template <typename Key = keyT>
      std::enable_if_t<!meta::is_void_v<Key>,int>
      operator()(const Key &key) const { return ttg::hash<keyT>{}(key) % world_size; }
      template <typename Key = keyT>
      std::enable_if_t<meta::is_void_v<Key>,int>
      operator()() const { return 0; }

     private:
      int world_size;
    };

  }  // namespace detail

  namespace detail {
    bool &trace_accessor() {
      static bool trace = false;
      return trace;
    }
  }  // namespace detail

  bool tracing() { return detail::trace_accessor(); }
  void trace_on() { detail::trace_accessor() = true; }
  void trace_off() { detail::trace_accessor() = false; }

  namespace detail {
    inline std::ostream &print_helper(std::ostream &out) { return out; }
    template <typename T, typename... Ts>
    inline std::ostream &print_helper(std::ostream &out, const T &t, const Ts &... ts) {
      out << ' ' << t;
      return print_helper(out, ts...);
    }
    //
    enum class StdOstreamTag { Cout, Cerr };
    template <StdOstreamTag>
    inline std::mutex &print_mutex_accessor() {
      static std::mutex mutex;
      return mutex;
    }
  }  // namespace detail

  template <typename T, typename... Ts>
  void print(const T &t, const Ts &... ts) {
    std::lock_guard<std::mutex> lock(detail::print_mutex_accessor<detail::StdOstreamTag::Cout>());
    std::cout << t;
    detail::print_helper(std::cout, ts...) << std::endl;
  }

  template <typename T, typename... Ts>
  void print_error(const T &t, const Ts &... ts) {
    std::lock_guard<std::mutex> lock(detail::print_mutex_accessor<detail::StdOstreamTag::Cout>()); // don't mix cerr and cout
    std::cerr << t;
    detail::print_helper(std::cerr, ts...) << std::endl;
  }

  class OpBase;  // forward decl
  template <typename keyT, typename valueT>
  class In;  // forward decl
  template <typename keyT, typename valueT>
  class Out;  // forward decl

  /// Provides basic information and graph connectivity (eventually statistics,
  /// etc.)
  class TerminalBase {
   public:
    static constexpr bool is_a_terminal = true;

    /// describes the terminal type
    enum class Type {
      Write,   /// can only be written to
      Read,    /// can only be read from
      Consume  /// provides consumable data
    };

   private:
    OpBase *op;                  //< Pointer to containing operation
    size_t n;                    //< Index of terminal
    std::string name;            //< Name of terminal
    bool connected;              //< True if is connected
    std::string key_type_str;    //< String describing key type
    std::string value_type_str;  //< String describing value type

    std::vector<TerminalBase *> successors_;

    TerminalBase(const TerminalBase &) = delete;
    TerminalBase(TerminalBase &&) = delete;

    friend class OpBase;
    template <typename keyT, typename valueT>
    friend class In;
    template <typename keyT, typename valueT>
    friend class Out;

   protected:
      TerminalBase() : op(0), n(0), name(""), connected(false) {}

    void set(OpBase *op, size_t index, const std::string &name, const std::string &key_type_str,
             const std::string &value_type_str, Type type) {
      this->op = op;
      this->n = index;
      this->name = name;
      this->key_type_str = key_type_str;
      this->value_type_str = value_type_str;
    }

    /// Add directed connection (this --> successor) in internal representation of the TTG.
    /// This is called by the derived class's connect method
    void connect_base(TerminalBase *successor) { successors_.push_back(successor); connected = true; successor->connected = true;}

   public:
    /// Return ptr to containing op
    OpBase *get_op() const {
      if (!op) throw "ttg::TerminalBase:get_op() but op is null";
      return op;
    }

    /// Returns index of terminal
    size_t get_index() const {
      if (!op) throw "ttg::TerminalBase:get_index() but op is null";
      return n;
    }

    /// Returns name of terminal
    const std::string &get_name() const {
      if (!op) throw "ttg::TerminalBase:get_name() but op is null";
      return name;
    }

    /// Returns string representation of key type
    const std::string &get_key_type_str() const {
      if (!op) throw "ttg::TerminalBase:get_key_type_str() but op is null";
      return key_type_str;
    }

    /// Returns string representation of value type
    const std::string &get_value_type_str() const {
      if (!op) throw "ttg::TerminalBase:get_value_type_str() but op is null";
      return value_type_str;
    }

    /// Returns the terminal type
    virtual Type get_type() const = 0;

    /// Get connections to successors
    const std::vector<TerminalBase *> &get_connections() const { return successors_; }

    /// Returns true if this terminal (input or output) is connected
    bool is_connected() const {return connected;}

    /// Connect this (a TTG output terminal) to a TTG input terminal.
    /// The base class method forwards to the the derived class connect method and so
    /// type checking for the key/value will be done at runtime when performing the
    /// dynamic down cast from TerminalBase* to In<keyT,valueT>.
    virtual void connect(TerminalBase *in) = 0;

    virtual ~TerminalBase() = default;
  };

  /// Provides basic information and graph connectivity (eventually statistics,
  /// etc.)
  class OpBase {
   private:
    uint64_t instance_id;  //< Unique ID for object
    static bool trace;     //< If true prints trace of all assignments and all op invocations

    std::string name;
    std::vector<TerminalBase *> inputs;
    std::vector<TerminalBase *> outputs;
    bool trace_instance;              //< If true traces just this instance
    bool is_composite;                //< True if the operator is composite
    bool is_within_composite;         //< True if the operator is part of a composite
    OpBase *containing_composite_op;  //< If part of a composite, points to composite operator

    bool executable;

    // Default copy/move/assign all OK
    static uint64_t next_instance_id() {
      static uint64_t id = 0;
      return id++;
    }

   protected:
    void set_input(size_t i, TerminalBase *t) {
      if (i >= inputs.size()) throw(name+":OpBase: out of range i setting input");
      inputs[i] = t;
    }

    void set_output(size_t i, TerminalBase *t) {
      if (i >= outputs.size()) throw(name+":OpBase: out of range i setting output");
      outputs[i] = t;
    }

    template <bool out, typename terminalT, std::size_t i, typename setfuncT>
    void register_terminal(terminalT &term, const std::string &name, const setfuncT setfunc) {
      term.set(this, i, name, detail::demangled_type_name<typename terminalT::key_type>(),
               detail::demangled_type_name<typename terminalT::value_type>(),
               out ? TerminalBase::Type::Write
                   : (std::is_const<typename terminalT::value_type>::value ? TerminalBase::Type::Read
                                                                           : TerminalBase::Type::Consume));
      (this->*setfunc)(i, &term);
    }

    template <bool out, std::size_t... IS, typename terminalsT, typename namesT, typename setfuncT>
    void register_terminals(std::index_sequence<IS...>, terminalsT &terms, const namesT &names,
                            const setfuncT setfunc) {
      int junk[] = {0, (register_terminal<out, typename std::tuple_element<IS, terminalsT>::type, IS>(
                            std::get<IS>(terms), names[IS], setfunc),
                        0)...};
      junk[0]++;
    }

    // Used by op ... terminalsT will be a tuple of terminals
    template <typename terminalsT, typename namesT>
    void register_input_terminals(terminalsT &terms, const namesT &names) {
      register_terminals<false>(std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms, names,
                                &OpBase::set_input);
    }

    // Used by op ... terminalsT will be a tuple of terminals
    template <typename terminalsT, typename namesT>
    void register_output_terminals(terminalsT &terms, const namesT &names) {
      register_terminals<true>(std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms, names,
                               &OpBase::set_output);
    }

    // Used by composite op ... terminalsT will be a tuple of pointers to terminals
    template <std::size_t... IS, typename terminalsT, typename setfuncT>
    void set_terminals(std::index_sequence<IS...>, terminalsT &terms, const setfuncT setfunc) {
      int junk[] = {0, ((this->*setfunc)(IS, std::get<IS>(terms)), 0)...};
      junk[0]++;
    }

    // Used by composite op ... terminalsT will be a tuple of pointers to terminals
    template <typename terminalsT, typename setfuncT>
    void set_terminals(const terminalsT &terms, const setfuncT setfunc) {
      set_terminals(std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms, setfunc);
    }

   private:
    OpBase(const OpBase &) = delete;
    OpBase &operator=(const OpBase &) = delete;
    OpBase(OpBase &&) = delete;
    OpBase &operator=(OpBase &&) = delete;

   public:
    OpBase(const std::string &name, size_t numins, size_t numouts)
        : instance_id(next_instance_id())
        , name(name)
        , inputs(numins)
        , outputs(numouts)
        , trace_instance(false)
        , is_composite(false)
        , is_within_composite(false)
        , containing_composite_op(0)
        , executable(false) {
      // std::cout << name << "@" << (void *)this << " -> " << instance_id << std::endl;
    }

    virtual ~OpBase() = default;

    virtual void release() { }

    /// Sets trace for all operations to value and returns previous setting
    static bool set_trace_all(bool value) {
      std::swap(trace, value);
      return value;
    }

    /// Sets trace for just this instance to value and returns previous setting
    bool set_trace_instance(bool value) {
      std::swap(trace_instance, value);
      return value;
    }

    /// Returns true if tracing set for either this instance or all instances
    bool get_trace() { return trace || trace_instance; }
    bool tracing() { return get_trace(); }

    void set_is_composite(bool value) { is_composite = value; }
    bool get_is_composite() const { return is_composite; }
    void set_is_within_composite(bool value, OpBase *op) {
      is_within_composite = value;
      containing_composite_op = op;
    }
    bool get_is_within_composite() const { return is_within_composite; }
    OpBase *get_containing_composite_op() const { return containing_composite_op; }

    /// Sets the name of this operation
    void set_name(const std::string &name) { this->name = name; }

    /// Gets the name of this operation
    const std::string &get_name() const { return name; }

    /// Gets the demangled class name (uses RTTI)
    std::string get_class_name() const {
      return boost::core::demangle(typeid(*this).name());
    }

    /// Returns the vector of input terminals
    const std::vector<TerminalBase *> &get_inputs() const { return inputs; }

    /// Returns the vector of output terminals
    const std::vector<TerminalBase *> &get_outputs() const { return outputs; }

    /// Returns a pointer to the i'th input terminal
    TerminalBase *in(size_t i) {
      if (i >= inputs.size()) throw name + ":OpBase: you are requesting an input terminal that does not exist";
      return inputs[i];
    }

    /// Returns a pointer to the i'th output terminal
    TerminalBase *out(size_t i) {
      if (i >= outputs.size()) throw name + "OpBase: you are requesting an output terminal that does not exist";
      return outputs[i];
    }

    /// Returns a pointer to the i'th input terminal ... to make API consistent with Op
    template <std::size_t i>
    TerminalBase *in() {
      return in(i);
    }

    /// Returns a pointer to the i'th output terminal ... to make API consistent with Op
    template <std::size_t i>
    TerminalBase *out() {
      return out(i);
    }

    uint64_t get_instance_id() const { return instance_id; }

    /// Waits for the entire TTG associated with this op to be completed (collective)
    virtual void fence() = 0;

    /// Marks this executable
    /// @return nothing
    virtual void make_executable() = 0;

    /// Queries if this ready to execute
    /// @return true is this object is executable
    bool is_executable() const { return executable; }

    /// Asserts that this is executable
    /// Use this macro from inside a derived class
    /// @throw std::logic_error if this is not executable
#define TTG_OP_ASSERT_EXECUTABLE() \
      do { \
        if (!this->is_executable()) { \
          std::ostringstream oss; \
          oss << "Op is not executable at " << __FILE__ << ":" << __LINE__; \
          throw std::logic_error(oss.str().c_str()); \
        } \
      } while (0);

  };

  // With more than one source file this will need to be moved
  bool OpBase::trace = false;

  void OpBase::make_executable() { executable = true; }


  template <typename input_terminalsT, typename output_terminalsT>
  class CompositeOp : public OpBase {
   public:
    static constexpr int numins = std::tuple_size<input_terminalsT>::value;    // number of input arguments
    static constexpr int numouts = std::tuple_size<output_terminalsT>::value;  // number of outputs or results

    using input_terminals_type = input_terminalsT;    // should be a tuple of pointers to input terminals
    using output_terminals_type = output_terminalsT;  // should be a tuple of pointers to output terminals

   private:
    std::vector<std::unique_ptr<OpBase>> ops;
    input_terminals_type ins;
    output_terminals_type outs;

    CompositeOp(const CompositeOp &) = delete;
    CompositeOp &operator=(const CompositeOp &) = delete;
    CompositeOp(const CompositeOp &&) = delete;  // Move should be OK

   public:
    template <typename opsT>
    CompositeOp(opsT &&ops_take_ownership,
                const input_terminals_type &ins,    // tuple of pointers to input terminals
                const output_terminals_type &outs,  // tuple of pointers to output terminals
                const std::string &name = "compositeop")
        : OpBase(name, numins, numouts), ops(std::forward<opsT>(ops_take_ownership)), ins(ins), outs(outs) {
      if (ops.size() == 0) throw name + ":CompositeOp: need to wrap at least one op";  // see fence

      set_is_composite(true);
      for (auto &op : ops) op->set_is_within_composite(true, this);
      set_terminals(ins, &CompositeOp<input_terminalsT, output_terminalsT>::set_input);
      set_terminals(outs, &CompositeOp<input_terminalsT, output_terminalsT>::set_output);

      // traversal is still broken ... need to add checking for composite
    }

    /// Return a pointer to i'th input terminal
    template <std::size_t i>
    typename std::tuple_element<i, input_terminals_type>::type in() {
      return std::get<i>(ins);
    }

    /// Return a pointer to i'th output terminal
    template <std::size_t i>
    typename std::tuple_element<i, output_terminalsT>::type out() {
      return std::get<i>(outs);
    }

    OpBase *get_op(std::size_t i) { return ops.at(i).get(); }

    void fence() { ops[0]->fence(); }

    void make_executable() {
      for (auto &op : ops) op->make_executable();
    }
  };

  template <typename opsT, typename input_terminalsT, typename output_terminalsT>
  std::unique_ptr<CompositeOp<input_terminalsT, output_terminalsT>> make_composite_op(
      opsT &&ops, const input_terminalsT &ins, const output_terminalsT &outs, const std::string &name = "compositeop") {
    return std::make_unique<CompositeOp<input_terminalsT, output_terminalsT>>(std::forward<opsT>(ops), ins, outs, name);
  }

  namespace detail {
    /// Traverses a graph of ops in depth-first manner following out edges
    class Traverse {
      std::set<OpBase *> seen;

      bool visited(OpBase *p) { return !seen.insert(p).second; }

     public:
      virtual void opfunc(OpBase *op) = 0;

      virtual void infunc(TerminalBase *in) = 0;

      virtual void outfunc(TerminalBase *out) = 0;

      void reset() { seen.clear(); }

      // Returns true if no null pointers encountered (i.e., if all
      // encountered terminals/operations are connected)
      bool traverse(OpBase *op) {
        if (!op) {
          std::cout << "ttg::Traverse: got a null op!\n";
          return false;
        }

        if (visited(op)) return true;

        bool status = true;

        opfunc(op);

      int count = 0;
      for (auto in : op->get_inputs()) {
          if (!in) {
              std::cout << "ttg::Traverse: got a null in!\n";
              status = false;
          } else {
              infunc(in);
              if (!in->is_connected()) {
                  std::cout << "ttg::Traverse: " << op->get_name() << " input terminal #" << count << " " << in->get_name() << " is not connected\n";
                  status = false;
              }
          }
          count++;
      }

      count = 0;
      for (auto out : op->get_outputs()) {
          if (!out) {
              std::cout << "ttg::Traverse: got a null out!\n";
              status = false;
          } else {
              outfunc(out);
              if (!out->is_connected()) {
                  std::cout << "ttg::Traverse: " << op->get_name() << " output terminal #" << count << " " << out->get_name() << " is not connected\n";
                  status = false;
              }
          }
          count++;
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

      // converters to OpBase*
      static OpBase* to_OpBase_ptr(OpBase* op) { return op; }
      static OpBase* to_OpBase_ptr(const OpBase* op) { return const_cast<OpBase*>(op); }

      /// visitor that does nothing
      /// @tparam Visitable any type
      template <typename Visitable>
      struct null_visitor {
        /// visits a non-const Visitable object
        void operator()(Visitable*) {};
        /// visits a const Visitable object
        void operator()(const Visitable*) {};
      };

    };
  }  // namespace detail

  /// @brief Traverses a graph of ops in depth-first manner following out edges
  /// @tparam OpVisitor A Callable type that visits each Op
  /// @tparam InVisitor A Callable type that visits each In terminal
  /// @tparam OutVisitor A Callable type that visits each Out terminal
  template <typename OpVisitor = detail::Traverse::null_visitor<OpBase>,
      typename InVisitor = detail::Traverse::null_visitor<TerminalBase>,
      typename OutVisitor = detail::Traverse::null_visitor<TerminalBase>>
  class Traverse : private detail::Traverse {
   public:
    static_assert(
        std::is_void<meta::void_t<decltype(std::declval<OpVisitor>()(std::declval<OpBase *>()))>>::value,
        "Traverse<OpVisitor,...>: OpVisitor(OpBase *op) must be a valid expression");
    static_assert(
        std::is_void<meta::void_t<decltype(std::declval<InVisitor>()(std::declval<TerminalBase *>()))>>::value,
        "Traverse<,InVisitor,>: InVisitor(TerminalBase *op) must be a valid expression");
    static_assert(
        std::is_void<meta::void_t<decltype(std::declval<OutVisitor>()(std::declval<TerminalBase *>()))>>::value,
        "Traverse<...,OutVisitor>: OutVisitor(TerminalBase *op) must be a valid expression");

    template <typename OpVisitor_ = detail::Traverse::null_visitor<OpBase>, typename InVisitor_ = detail::Traverse::null_visitor<TerminalBase>, typename OutVisitor_ = detail::Traverse::null_visitor<TerminalBase>>
    Traverse(OpVisitor_ &&op_v = OpVisitor_{}, InVisitor_ &&in_v = InVisitor_{}, OutVisitor_ &&out_v = OutVisitor_{})
        : op_visitor_(std::forward<OpVisitor_>(op_v))
        , in_visitor_(std::forward<InVisitor_>(in_v))
        , out_visitor_(std::forward<OutVisitor_>(out_v)){};

    const OpVisitor &op_visitor() const { return op_visitor_; }
    const InVisitor &in_visitor() const { return in_visitor_; }
    const OutVisitor &out_visitor() const { return out_visitor_; }

    /// Traverses graph starting at one or more Ops
    template <typename ... OpBasePtrs>
    std::enable_if_t<(std::is_convertible_v<std::remove_reference_t<OpBasePtrs>,OpBase*> && ...),bool>
        operator()(OpBase* op, OpBasePtrs && ... ops) {
      reset();
      bool result = traverse(op);
      result &= (traverse(std::forward<OpBasePtrs>(ops)) && ... );
      reset();
      return result;
    }

   private:
    OpVisitor op_visitor_;
    InVisitor in_visitor_;
    OutVisitor out_visitor_;

    void opfunc(OpBase *op) { op_visitor_(op); }

    void infunc(TerminalBase *in) { in_visitor_(in); }

    void outfunc(TerminalBase *out) { out_visitor_(out); }
  };

  namespace {
    auto trivial_1param_lambda = [](auto &&op) {};
  }
  template <typename OpVisitor = decltype(trivial_1param_lambda)&, typename InVisitor = decltype(trivial_1param_lambda)&, typename OutVisitor = decltype(trivial_1param_lambda)&>
  auto make_traverse(OpVisitor &&op_v = trivial_1param_lambda, InVisitor &&in_v = trivial_1param_lambda, OutVisitor &&out_v = trivial_1param_lambda) {
    return Traverse<std::remove_reference_t<OpVisitor>, std::remove_reference_t<InVisitor>,
                    std::remove_reference_t<OutVisitor>>{std::forward<OpVisitor>(op_v), std::forward<InVisitor>(in_v),
                                                         std::forward<OutVisitor>(out_v)};
  };

  /// verifies connectivity of the Graph
  static Traverse<> verify{};

  /// Prints the graph to std::cout in an ad hoc format
  static auto print_ttg = make_traverse(
      [](auto *op) {
        std::cout << "op: " << (void *)op << " " << op->get_name() << " numin " << op->get_inputs().size() << " numout "
                  << op->get_outputs().size() << std::endl;
      },
      [](auto *in) {
        std::cout << "  in: " << in->get_index() << " " << in->get_name() << " " << in->get_key_type_str() << " "
                  << in->get_value_type_str() << std::endl;
      },
      [](auto *out) {
        std::cout << " out: " << out->get_index() << " " << out->get_name() << " " << out->get_key_type_str() << " "
                  << out->get_value_type_str() << std::endl;
      });

  /// Prints the graph to a std::string in the format understood by GraphViz's dot program
  class Dot : private detail::Traverse {
    std::stringstream buf;

    // Insert backslash before characters that dot is interpreting
    std::string escape(const std::string &in) {
      std::stringstream s;
      for (char c : in) {
        if (c == '<' || c == '>' || c == '"' || c == '|')
          s << "\\" << c;
        else
          s << c;
      }
      return s.str();
    }

    // A unique name for the node derived from the pointer
    std::string nodename(const OpBase *op) {
      std::stringstream s;
      s << "n" << (void *)op;
      return s.str();
    }

    void opfunc(OpBase *op) {
      std::string opnm = nodename(op);

      buf << "        " << opnm << " [shape=record,style=filled,fillcolor=gray90,label=\"{";

      size_t count = 0;
      if (op->get_inputs().size() > 0) buf << "{";
      for (auto in : op->get_inputs()) {
        if (in) {
          if (count != in->get_index()) throw "ttg::Dot: lost count of ins";
          buf << " <in" << count << ">"
              << " " << escape("<" + in->get_key_type_str() + "," + in->get_value_type_str() + ">") << " "
              << escape(in->get_name());
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
              << " " << escape("<" + out->get_key_type_str() + "," + out->get_value_type_str() + ">") << " "
              << out->get_name();
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
              buf << opnm << ":out" << out->get_index() << ":s -> " << nodename(successor->get_op()) << ":in"
                  << successor->get_index() << ":n;\n";
            }
          }
        }
      }
    }

    void infunc(TerminalBase *in) {}

    void outfunc(TerminalBase *out) {}

   public:
    /// @return string containing the graph specification in the format understood by GraphViz's dot program
    template <typename... OpBasePtrs>
    std::enable_if_t<(std::is_convertible_v<std::remove_const_t<std::remove_reference_t<OpBasePtrs>>, OpBase *> && ...),
                     std::string>
    operator()(OpBasePtrs &&... ops) {
      reset();
      buf.str(std::string());
      buf.clear();

      buf << "digraph G {\n";
      buf << "        ranksep=1.5;\n";
      bool t = true;
      t &= (traverse(std::forward<OpBasePtrs>(ops)) && ... );
      buf << "}\n";

      reset();
      std::string result = buf.str();
      buf.str(std::string());
      buf.clear();

      return result;
    }
  };

  /// applies @c make_executable method to every op in the graph
  /// return true if there are no dangling out terminals
  template <typename... OpBasePtrs>
  std::enable_if_t<(std::is_convertible_v<std::remove_const_t<std::remove_reference_t<OpBasePtrs>>, OpBase *> && ...),
                   bool>
  make_graph_executable(OpBasePtrs &&... ops) {
    return ::ttg::make_traverse([](auto&&x) { std::forward<decltype(x)>(x)->make_executable(); })(std::forward<OpBasePtrs>(ops)...);
  }

  template <typename keyT = void, typename valueT = void>
  class Edge;  // Forward decl.

  template <typename keyT = void, typename valueT = void>
  class In : public TerminalBase {
   public:
    typedef valueT value_type;
    typedef keyT key_type;
    static_assert(std::is_same<keyT, std::decay_t<keyT>>::value,
                  "In<keyT,valueT> assumes keyT is a non-decayable type");
    // valueT can be T or const T
    static_assert(std::is_same<std::remove_const_t<valueT>, std::decay_t<valueT>>::value,
                  "In<keyT,valueT> assumes std::remove_const<T> is a non-decayable type");
    using edge_type = Edge<keyT, valueT>;
    using send_callback_type = meta::detail::send_callback_t<keyT, std::decay_t<valueT>>;
    using move_callback_type = meta::detail::move_callback_t<keyT, std::decay_t<valueT>>;
    using setsize_callback_type = meta::detail::setsize_callback_t<keyT>;
    using finalize_callback_type = meta::detail::finalize_callback_t<keyT>;
    static constexpr bool is_an_input_terminal = true;

   private:
    send_callback_type send_callback;
    move_callback_type move_callback;
    setsize_callback_type setsize_callback;
    finalize_callback_type finalize_callback;

    // No moving, copying, assigning permitted
    In(In &&other) = delete;
    In(const In &other) = delete;
    In &operator=(const In &other) = delete;
    In &operator=(const In &&other) = delete;

    void connect(TerminalBase *p) override {
      throw "Edge: to connect terminals use out->connect(in) rather than in->connect(out)";
    }

   public:
    In() {}

    void set_callback(const send_callback_type &send_callback, const move_callback_type &move_callback,
                      const setsize_callback_type &setsize_callback = setsize_callback_type{},
                      const finalize_callback_type &finalize_callback = finalize_callback_type{}) {
      this->send_callback = send_callback;
      this->move_callback = move_callback;
      this->setsize_callback = setsize_callback;
      this->finalize_callback = finalize_callback;
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key,Value>,void>
    send(const Key &key, const Value &value) {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback(key, value);
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key,Value> && std::is_same_v<Value,std::remove_reference_t<Value>>,void>
    send(const Key &key, Value &&value) {
      if (!move_callback) throw std::runtime_error("move callback not initialized");
      move_callback(key, std::forward<valueT>(value));
    }

    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>,void>
    sendk(const Key &key) {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback(key);
    }

    template <typename Value = valueT>
    std::enable_if_t<!meta::is_void_v<Value>,void>
    sendv(const Value &value) {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback(value);
    }

    template <typename Value = valueT>
    std::enable_if_t<!meta::is_void_v<Value> && std::is_same_v<Value,std::remove_reference_t<Value>>,void>
    sendv(Value &&value) {
      if (!move_callback) throw std::runtime_error("move callback not initialized");
      move_callback(std::forward<valueT>(value));
    }

    void send() {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback();
    }

    // An optimized implementation will need a separate callback for broadcast
    // with a specific value for rangeT
    template <typename rangeT, typename Value = valueT>
    std::enable_if_t<!meta::is_void_v<Value>,void>
    broadcast(const rangeT &keylist, const Value &value) {
      for (auto key : keylist) send(key, value);
    }

    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>,void>
    set_size(const Key &key, std::size_t size) {
      if (!setsize_callback) throw std::runtime_error("set_size callback not initialized");
      setsize_callback(key, size);
    }

    template <typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>,void>
    set_size(std::size_t size) {
      if (!setsize_callback) throw std::runtime_error("set_size callback not initialized");
      setsize_callback(size);
    }

    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>,void>
    finalize(const Key &key) {
      // std::cout << "In::finalize::\n";
      if (!finalize_callback) throw std::runtime_error("finalize callback not initialized");
      finalize_callback(key);
    }

    template <typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>,void>
    finalize() {
      if (!finalize_callback) throw std::runtime_error("finalize callback not initialized");
      finalize_callback();
    }

    Type get_type() const override {
      return std::is_const<valueT>::value ? TerminalBase::Type::Read : TerminalBase::Type::Consume;
    }
  };

  // Output terminal
  template <typename keyT = void, typename valueT = void>
  class Out : public TerminalBase {
   public:
    typedef valueT value_type;
    typedef keyT key_type;
    static_assert(std::is_same<keyT, std::decay_t<keyT>>::value,
                  "Out<keyT,valueT> assumes keyT is a non-decayable type");
    static_assert(std::is_same<valueT, std::decay_t<valueT>>::value,
                  "Out<keyT,valueT> assumes valueT is a non-decayable type");
    typedef Edge<keyT, valueT> edge_type;
    static constexpr bool is_an_output_terminal = true;

   private:
    // No moving, copying, assigning permitted
    Out(Out &&other) = delete;
    Out(const Out &other) = delete;
    Out &operator=(const Out &other) = delete;
    Out &operator=(const Out &&other) = delete;

   public:
    Out() {}

    /// \note will check data types unless macro \c NDEBUG is defined
    void connect(TerminalBase *in) override {
#ifndef NDEBUG
      if (in->get_type() == TerminalBase::Type::Read) {
        typedef In<keyT, std::add_const_t<valueT>> input_terminal_type;
        if (!dynamic_cast<input_terminal_type *>(in))
          throw std::invalid_argument(
              std::string("you are trying to connect terminals with incompatible types:\ntype of this Terminal = ") +
              detail::demangled_type_name(this) + "\ntype of other Terminal" + detail::demangled_type_name(in));
      } else if (in->get_type() == TerminalBase::Type::Consume) {
        typedef In<keyT, valueT> input_terminal_type;
        if (!dynamic_cast<input_terminal_type *>(in))
          throw std::invalid_argument(
              std::string("you are trying to connect terminals with incompatible types:\ntype of this Terminal = ") +
              detail::demangled_type_name(this) + "\ntype of other Terminal" + detail::demangled_type_name(in));
      } else  // successor->type() == TerminalBase::Type::Write
        throw std::invalid_argument(std::string("you are trying to connect an Out terminal to another Out terminal"));
      if (tracing()) {
        print(rank(), ": connected Out<> ", get_name(), "(ptr=", this, ") to In<> ", in->get_name(), "(ptr=", in, ")");
      }
#endif
      this->connect_base(in);
    }

    auto nsuccessors() const {
      return get_connections().size();
    }
    const auto& successors() const {
      return get_connections();
    }

    template<typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key,Value>,void> send(const Key &key, const Value &value) {
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->send(key, value);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->send(key, value);
        }
      }
    }

    template<typename Key = keyT, typename Value = valueT>
    std::enable_if_t<!meta::is_void_v<Key> && meta::is_void_v<Value>,void> sendk(const Key &key) {
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->sendk(key);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->sendk(key);
        }
      }
    }

    template<typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_void_v<Key> && !meta::is_void_v<Value>,void> sendv(const Value &value) {
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->sendv(value);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->sendv(value);
        }
      }
    }

    template<typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_all_void_v<Key,Value>,void> send() {
      if (tracing()) {
        print(rank(), ": in ", get_name(), "(ptr=", this, ") Out<>::send: #successors=", successors().size());
      }
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->send();
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->send();
        }
        else {
          throw std::logic_error("Out<>: invalid successor type");
        }
        if (tracing()) {
          print("Out<> ", get_name(), "(ptr=", this, ") send to In<> ", successor->get_name(), "(ptr=", successor, ")");
        }
      }
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key,Value> && std::is_same_v<Value,std::remove_reference_t<Value>>,void>
    send(const Key &key, Value &&value) {
      std::size_t N = successors().size();
      // find the first terminal that can consume the value
      std::size_t move_terminal = N - 1;
      for (std::size_t i = 0; i != N; ++i) {
        if (successors().at(i)->get_type() == TerminalBase::Type::Consume) {
          move_terminal = i;
          break;
        }
      }
      if (N > 0) {
        // send copies to every terminal except the one we will move the results to
        for (std::size_t i = 0; i != N; ++i) {
          if (i != move_terminal) {
            TerminalBase *successor = successors().at(i);
            if (successor->get_type() == TerminalBase::Type::Read) {
              static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->send(key, value);
            } else if (successor->get_type() == TerminalBase::Type::Consume) {
              static_cast<In<keyT, valueT> *>(successor)->send(key, value);
            }
          }
        }
        {
          TerminalBase *successor = successors().at(move_terminal);
          static_cast<In<keyT, valueT> *>(successor)->send(key, std::forward<Value>(value));
        }
      }
    }

    // An optimized implementation will need a separate callback for broadcast
    // with a specific value for rangeT
    template<typename rangeT, typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key,Value>,void>
    broadcast(const rangeT &keylist, const Value &value) {  // NO MOVE YET
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->broadcast(keylist, value);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->broadcast(keylist, value);
        }
      }
    }

    template<typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>,void>
    set_size(const Key &key, std::size_t size) {
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->set_size(key, size);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->set_size(key, size);
        }
      }
    }

    template<typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>,void>
    set_size(std::size_t size) {
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->set_size(size);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->set_size(size);
        }
      }
    }

    template<typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>,void>
    finalize(const Key &key) {
      for (auto && successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->finalize(key);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->finalize(key);
        }
      }
    }

    template<typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>,void>
    finalize() {
      for (auto successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->finalize();
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->finalize();
        }
      }
    }

    Type get_type() const override { return TerminalBase::Type::Write; }
  };

  /// Connect output terminal to successor input terminal
  template <typename out_terminalT, typename in_terminalT>
  void connect(out_terminalT *out, in_terminalT *in) {
    out->connect(in);
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via unique or otherwise wrapped pointers to Ops)
  template <std::size_t outindex, std::size_t inindex, typename producer_op_ptr, typename successor_op_ptr>
  void connect(producer_op_ptr &p, successor_op_ptr &s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via bare pointers to Ops)
  template <std::size_t outindex, std::size_t inindex, typename producer_op_ptr, typename successor_op_ptr>
  void connect(producer_op_ptr *p, successor_op_ptr *s) {
    connect(p->template out<outindex>(), s->template in<inindex>());
  }

  /// Connected producer output terminal outindex to consumer input terminal inindex (via OpBase pointers)
  void connect(size_t outindex, size_t inindex, OpBase* producer, OpBase* consumer) {
      connect(producer->out(outindex), consumer->in(inindex));
  }

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
      std::vector<TerminalBase *> outs;  // In<keyT, valueT> or In<keyT, const valueT>
      std::vector<Out<keyT, valueT> *> ins;

      EdgeImpl() : name(""), outs(), ins() {}

      EdgeImpl(const std::string &name) : name(name), outs(), ins() {}

      void set_in(Out<keyT, valueT> *in) {
        if (ins.size() && tracing()) {
          print("Edge: ", name, " : has multiple inputs");
        }
        ins.push_back(in);
        try_to_connect_new_in(in);
      }

      void set_out(TerminalBase *out) {
        if (outs.size() && tracing()) {
          print("Edge: ", name, " : has multiple outputs");
        }
        outs.push_back(out);
        try_to_connect_new_out(out);
      }

      void try_to_connect_new_in(Out<keyT, valueT> *in) const {
        for (auto out : outs)
          if (in && out) in->connect(out);
      }

      void try_to_connect_new_out(TerminalBase *out) const {
        assert(out->get_type() != TerminalBase::Type::Write);  // out must be an In<>
        for (auto in : ins)
          if (in && out) in->connect(out);
      }

      ~EdgeImpl() {
        if (ins.size() == 0 || outs.size() == 0) {
            std::cerr << "Edge: destroying edge pimpl ('" << name << "') with either in or out not "
                       "assigned --- graph may be incomplete"
                    << std::endl;
        }
      }
    };

    // We have a vector here to accomodate fusing multiple edges together
    // when connecting them all to a single terminal.
    mutable std::vector<std::shared_ptr<EdgeImpl>> p;  // Need shallow copy semantics

   public:
    typedef Out<keyT, valueT> output_terminal_type;
    typedef keyT key_type;
    typedef valueT value_type;
    static_assert(std::is_same<keyT, std::decay_t<keyT>>::value,
                  "Edge<keyT,valueT> assumes keyT is a non-decayable type");
    static_assert(std::is_same<valueT, std::decay_t<valueT>>::value,
                  "Edge<keyT,valueT> assumes valueT is a non-decayable type");
    static constexpr bool is_an_edge = true;

    Edge(const std::string name = "anonymous edge") : p(1) { p[0] = std::make_shared<EdgeImpl>(name); }

    template <typename... valuesT>
    Edge(const Edge<keyT, valuesT> &... edges) : p(0) {
      std::vector<Edge<keyT, valueT>> v = {edges...};
      for (auto &edge : v) {
        p.insert(p.end(), edge.p.begin(), edge.p.end());
      }
    }

    /// probes if this is already has at least one input
    bool live() const {
      bool result = false;
      for(const auto& edge: p) {
        if (!edge->ins.empty())
          return true;
      }
      return result;
    }

    void set_in(Out<keyT, valueT> *in) const {
      for (auto &edge : p) edge->set_in(in);
    }

    void set_out(TerminalBase *out) const {
      for (auto &edge : p) edge->set_out(out);
    }

    // this is currently just a hack, need to understand better whether this is a good idea
    Out<keyT, valueT> *in(size_t edge_index = 0, size_t terminal_index = 0) {
      return p.at(edge_index)->ins.at(terminal_index);
    }
  };

  // Fuse edges into one ... all the types have to be the same ... just using
  // valuesT for variadic args
  template <typename keyT, typename... valuesT>
  auto fuse(const Edge<keyT, valuesT> &... args) {
    using valueT = typename std::tuple_element<0, std::tuple<valuesT...>>::type;  // grab first type
    return Edge<keyT, valueT>(args...);  // This will force all valuesT to be the same
  }

  // Make a tuple of Edges ... needs some type checking injected
  template <typename... inedgesT>
  auto edges(const inedgesT &... args) {
    return std::make_tuple(args...);
  }

  template <typename keyT, typename valueT, typename output_terminalT>
  void send(const keyT &key, valueT &&value, output_terminalT &t) {
    t.send(key, std::forward<valueT>(value));
  }

  template <typename keyT, typename output_terminalT>
  void sendk(const keyT &key, output_terminalT &t) {
    t.sendk(key);
  }

  // TODO if sendk is removed, rename to send
  template <typename valueT, typename output_terminalT>
  void sendv(valueT &&value, output_terminalT &t) {
    t.sendv(std::forward<valueT>(value));
  }

  template <typename keyT, typename valueT, typename output_terminalT>
  void send(output_terminalT &t) {
    t.send();
  }

  template <size_t i, typename keyT, typename valueT, typename... output_terminalsT>
  std::enable_if_t<meta::is_none_void_v<keyT,std::decay_t<valueT>>,void>
      send(const keyT &key, valueT &&value, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).send(key, std::forward<valueT>(value));
  }

  // TODO decide whether we need this ... (how common will be pure control flow?)
  template <size_t i, typename keyT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<keyT>,void>
  sendk(const keyT &key, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).sendk(key);
  }

  // TODO if sendk is removed, rename to send
  template <size_t i, typename valueT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<valueT>,void>
  sendv(valueT &&value, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).sendv(std::forward<valueT>(value));
  }

  template <size_t i, typename... output_terminalsT>
  void send(std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).send();
  }

  template <size_t i, typename rangeT, typename valueT, typename... output_terminalsT>
  void broadcast(const rangeT &keylist, valueT &&value, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).broadcast(keylist, std::forward<valueT>(value));
  }

  template <typename keyT, typename output_terminalT>
  void set_size(const keyT &key, const std::size_t size, output_terminalT &t) {
    t.set_size(key, size);
  }

  template <size_t i, typename keyT, typename... output_terminalsT>
  void set_size(const keyT &key, const std::size_t size, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).set_size(key, size);
  }

  template <typename keyT, typename output_terminalT>
  std::enable_if_t<!meta::is_void_v<keyT>,void>
  finalize(const keyT &key, output_terminalT &t) {
    t.finalize(key);
  }

  template <size_t i, typename keyT, typename... output_terminalsT>
  std::enable_if_t<!meta::is_void_v<keyT>,void>
  finalize(const keyT &key, std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).finalize(key);
  }

  template <typename output_terminalT>
  void finalize(output_terminalT &t) {
    t.finalize();
  }

  template <size_t i, typename... output_terminalsT>
  void finalize(std::tuple<output_terminalsT...> &t) {
    std::get<i>(t).finalize();
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

  /// A data sink for one input
    template <typename keyT, typename input_valueT>
    class OpSink : public ::ttg::OpBase {
        static constexpr int numins = 1;
        static constexpr int numouts = 0;
        
        using input_terminals_type = std::tuple<::ttg::In<keyT, input_valueT>>;
        using input_edges_type = std::tuple<::ttg::Edge<keyT, std::decay_t<input_valueT>>>;
        using output_terminals_type = std::tuple<>;
        
    private:
        input_terminals_type input_terminals;
        output_terminals_type output_terminals;
        
        OpSink(const OpSink &other) = delete;
        OpSink &operator=(const OpSink &other) = delete;
        OpSink(OpSink &&other) = delete;
        OpSink &operator=(OpSink &&other) = delete;
        
        template <typename terminalT>
        void register_input_callback(terminalT &input) {
            using valueT = std::decay_t<typename terminalT::value_type>;
            auto move_callback = [](const keyT &key, valueT &&value) {};
            auto send_callback = [](const keyT &key, const valueT &value) {};
            auto setsize_callback = [](const keyT &key, std::size_t size) {};
            auto finalize_callback = [](const keyT &key) {};
            
            input.set_callback(send_callback, move_callback, setsize_callback, finalize_callback);
        }
        
    public:
        OpSink(const std::string& inname="junk") : ::ttg::OpBase("sink", numins, numouts) {
            register_input_terminals(input_terminals, std::vector<std::string>{inname});
            register_input_callback(std::get<0>(input_terminals));
        }
        
        OpSink(const input_edges_type &inedges, const std::string& inname="junk") : ::ttg::OpBase("sink", numins, numouts) {
            register_input_terminals(input_terminals, std::vector<std::string>{inname});
            register_input_callback(std::get<0>(input_terminals));
            std::get<0>(inedges).set_out(&std::get<0>(input_terminals));
        }
        
        virtual ~OpSink() {}

        void fence() {}
        
        void make_executable() {
            OpBase::make_executable();
        }
        
        /// Returns pointer to input terminal i to facilitate connection --- terminal cannot be copied, moved or assigned
        template <std::size_t i>
        typename std::tuple_element<i, input_terminals_type>::type *in() {
            static_assert(i==0);
            return &std::get<i>(input_terminals);
        }
    };

}  // namespace ttg

// This provides an efficent API for serializing/deserializing a data type.
// An object of this type will need to be provided for each serializable type.
// The default implementation, in serialization.h, works only for primitive/POD data types;
// backend-specific implementations may be available in backend/serialization.h .
extern "C" struct ttg_data_descriptor {
  const char *name;
  void (*get_info)(const void *object, uint64_t *hs, uint64_t *ps, int *is_contiguous_mask, void **buf);
  void (*pack_header)(const void *object, uint64_t header_size, void **buf);
  uint64_t (*pack_payload)(const void *object, uint64_t chunk_size, uint64_t pos, void *buf);
  void (*unpack_header)(void *object, uint64_t header_size, const void *buf);
  void (*unpack_payload)(void *object, uint64_t chunk_size, uint64_t pos, const void *buf);
  void (*print)(const void *object);
};

namespace ttg {

  template <typename T, typename Enabler>
  struct default_data_descriptor;

  // Returns a pointer to a constant static instance initialized
  // once at run time.
  template <typename T>
  const ttg_data_descriptor *get_data_descriptor();

}  // namespace ttg

#endif  // TTG_H_INCLUDED
