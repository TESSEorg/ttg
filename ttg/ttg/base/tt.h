#ifndef TTG_BASE_OP_H
#define TTG_BASE_OP_H

#include <cstdint>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#include "ttg/base/terminal.h"
#include "ttg/util/demangle.h"

namespace ttg {

  namespace detail {
    // If true prints trace of all assignments and all TT invocations
    inline bool &tt_base_trace_accessor(void) {
      static bool trace = false;
      return trace;
    }
  } // namespace detail

  /// Provides basic information and graph connectivity (eventually statistics,
  /// etc.)
  class TTBase {
  private:
    uint64_t instance_id;  //< Unique ID for object

    std::string name;
    std::vector<TerminalBase *> inputs;
    std::vector<TerminalBase *> outputs;
    bool trace_instance;              //< If true traces just this instance
    bool is_composite;                //< True if the operator is composite
    bool is_within_composite;         //< True if the operator is part of a composite
    TTBase *containing_composite_tt;  //< If part of a composite, points to composite operator

    bool executable;

    // Default copy/move/assign all OK
    static uint64_t next_instance_id() {
      static uint64_t id = 0;
      return id++;
    }

  protected:
    void set_input(size_t i, TerminalBase *t) {
      if (i >= inputs.size()) throw(name+":TTBase: out of range i setting input");
      inputs[i] = t;
    }

    void set_output(size_t i, TerminalBase *t) {
      if (i >= outputs.size()) throw(name+":TTBase: out of range i setting output");
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
                                &TTBase::set_input);
    }

    // Used by op ... terminalsT will be a tuple of terminals
    template <typename terminalsT, typename namesT>
    void register_output_terminals(terminalsT &terms, const namesT &names) {
      register_terminals<true>(std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms, names,
                              &TTBase::set_output);
    }

    // Used by composite TT ... terminalsT will be a tuple of pointers to terminals
    template <std::size_t... IS, typename terminalsT, typename setfuncT>
    void set_terminals(std::index_sequence<IS...>, terminalsT &terms, const setfuncT setfunc) {
      int junk[] = {0, ((this->*setfunc)(IS, std::get<IS>(terms)), 0)...};
      junk[0]++;
    }

    // Used by composite TT ... terminalsT will be a tuple of pointers to terminals
    template <typename terminalsT, typename setfuncT>
    void set_terminals(const terminalsT &terms, const setfuncT setfunc) {
      set_terminals(std::make_index_sequence<std::tuple_size<terminalsT>::value>{}, terms, setfunc);
    }

  private:
   TTBase(const TTBase &) = delete;
   TTBase &operator=(const TTBase &) = delete;
   TTBase(TTBase &&) = delete;
   TTBase &operator=(TTBase &&) = delete;

  public:
   TTBase(const std::string &name, size_t numins, size_t numouts)
        : instance_id(next_instance_id())
        , name(name)
        , inputs(numins)
        , outputs(numouts)
        , trace_instance(false)
        , is_composite(false)
        , is_within_composite(false)
        , containing_composite_tt(0)
        , executable(false) {
      // std::cout << name << "@" << (void *)this << " -> " << instance_id << std::endl;
    }

    virtual ~TTBase() = default;

    virtual void release() { }

    /// Sets trace for all operations to value and returns previous setting
    static bool set_trace_all(bool value) {
      std::swap(ttg::detail::tt_base_trace_accessor(), value);
      return value;
    }

    /// Sets trace for just this instance to value and returns previous setting
    bool set_trace_instance(bool value) {
      std::swap(trace_instance, value);
      return value;
    }

    /// Returns true if tracing set for either this instance or all instances
    bool get_trace() { return ttg::detail::tt_base_trace_accessor() || trace_instance; }
    bool tracing() { return get_trace(); }

    void set_is_composite(bool value) { is_composite = value; }
    bool get_is_composite() const { return is_composite; }
    void set_is_within_composite(bool value, TTBase *op) {
      is_within_composite = value;
      containing_composite_tt = op;
    }
    bool get_is_within_composite() const { return is_within_composite; }
    TTBase *get_containing_composite_op() const { return containing_composite_tt; }

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
    ttg::TerminalBase *in(size_t i) {
      if (i >= inputs.size()) throw name + ":TTBase: you are requesting an input terminal that does not exist";
      return inputs[i];
    }

    /// Returns a pointer to the i'th output terminal
    ttg::TerminalBase *out(size_t i) {
      if (i >= outputs.size()) throw name + "TTBase: you are requesting an output terminal that does not exist";
      return outputs[i];
    }

    /// Returns a pointer to the i'th input terminal ... to make API consistent with Op
    template <std::size_t i>
    ttg::TerminalBase *in() {
      return in(i);
    }

    /// Returns a pointer to the i'th output terminal ... to make API consistent with Op
    template <std::size_t i>
    ttg::TerminalBase *out() {
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


  inline void TTBase::make_executable() { executable = true; }

} // namespace ttg

#endif // TTG_BASE_OP_H
