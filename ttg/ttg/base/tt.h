#ifndef TTG_BASE_OP_H
#define TTG_BASE_OP_H

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "ttg/base/terminal.h"
#include "ttg/util/demangle.h"
#include "ttg/util/trace.h"

namespace ttg {

  namespace detail {
    // If true prints trace of all assignments and all TT invocations
    inline bool &tt_base_trace_accessor(void) {
      static bool trace = false;
      return trace;
    }

    inline bool &op_base_lazy_pull_accessor(void) {
      static bool lazy_pull = false;
      return lazy_pull;
    }
  }  // namespace detail

  /// A base class for all template tasks
  class TTBase {
   private:
    int64_t instance_id;  //!< Unique ID for object; in after-move state will be -1

    std::string name;
    std::vector<TerminalBase *> inputs;
    std::vector<TerminalBase *> outputs;
    bool trace_instance = false;         //!< If true traces just this instance
    const TTBase *owning_ttg = nullptr;  //!< the containing TTG, if any
    template <typename input_terminalsT, typename output_terminalsT>
    friend class TTG;  // TTG needs to be able to control owning_ttg

    bool executable = false;  //!< ready to execute?
    bool is_ttg_ = false;
    bool lazy_pull_instance = false;

    // Default copy/move/assign all OK
    static uint64_t next_instance_id() {
      static uint64_t id = 0;
      return id++;
    }

   protected:
    void set_input(size_t i, TerminalBase *t) {
      if (i >= inputs.size()) throw(name + ":TTBase: out of range i setting input");
      inputs[i] = t;
    }

    void set_output(size_t i, TerminalBase *t) {
      if (i >= outputs.size()) throw(name + ":TTBase: out of range i setting output");
      outputs[i] = t;
    }

    template <bool out, typename terminalT, std::size_t i, typename setfuncT>
    void register_terminal(terminalT &term, const std::string &name, const setfuncT setfunc) {
      term.set(this, i, name, detail::demangled_type_name<typename terminalT::key_type>(),
               detail::demangled_type_name<typename terminalT::value_type>(),
               out ? TerminalBase::Type::Write
                   : (std::is_const_v<typename terminalT::value_type> ? TerminalBase::Type::Read
                                                                      : TerminalBase::Type::Consume));
      (this->*setfunc)(i, &term);
    }

    template <bool out, std::size_t... IS, typename terminalsT, typename namesT, typename setfuncT>
    void register_terminals(std::index_sequence<IS...>, terminalsT &terms, const namesT &names,
                            const setfuncT setfunc) {
      int junk[] = {
          0, (register_terminal<out, std::tuple_element_t<IS, terminalsT>, IS>(std::get<IS>(terms), names[IS], setfunc),
              0)...};
      junk[0]++;
    }

    // Used by op ... terminalsT will be a tuple of terminals
    template <typename terminalsT, typename namesT>
    void register_input_terminals(terminalsT &terms, const namesT &names) {
      register_terminals<false>(std::make_index_sequence<std::tuple_size_v<terminalsT>>{}, terms, names,
                                &TTBase::set_input);
    }

    // Used by op ... terminalsT will be a tuple of terminals
    template <typename terminalsT, typename namesT>
    void register_output_terminals(terminalsT &terms, const namesT &names) {
      register_terminals<true>(std::make_index_sequence<std::tuple_size_v<terminalsT>>{}, terms, names,
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
      set_terminals(std::make_index_sequence<std::tuple_size_v<terminalsT>>{}, terms, setfunc);
    }

   private:
    // non-copyable, but movable
    TTBase(const TTBase &) = delete;
    TTBase &operator=(const TTBase &) = delete;

   protected:
    TTBase(TTBase &&other)
        : instance_id(other.instance_id)
        , is_ttg_(std::move(other.is_ttg_))
        , name(std::move(other.name))
        , inputs(std::move(other.inputs))
        , outputs(std::move(other.outputs)) {
      other.instance_id = -1;
    }
    TTBase &operator=(TTBase &&other) {
      instance_id = other.instance_id;
      is_ttg_ = std::move(other.is_ttg_);
      name = std::move(other.name);
      inputs = std::move(other.inputs);
      outputs = std::move(other.outputs);
      other.instance_id = -1;
      return *this;
    }

    TTBase(const std::string &name, size_t numins, size_t numouts)
        : instance_id(next_instance_id()), is_ttg_(false), name(name), inputs(numins), outputs(numouts) {}

    static const std::vector<TerminalBase *> *&outputs_tls_ptr_accessor() {
      static thread_local const std::vector<TerminalBase *> *outputs_tls_ptr = nullptr;
      return outputs_tls_ptr;
    }
    void set_outputs_tls_ptr() { outputs_tls_ptr_accessor() = &this->outputs; }
    void set_outputs_tls_ptr(const std::vector<TerminalBase *> *ptr) { outputs_tls_ptr_accessor() = ptr; }

   public:
    virtual ~TTBase() = default;

    /// Use this to create a task that takes no data "manually"
    /// @warning calls ttg::abort() if the derived class TT did not override this;
    ///          only makes sense to override this if the derived TT uses void for key or data
    virtual void invoke() {
      std::cerr << "TTBase::invoke() invoked on a TT that did not override it" << std::endl;
      ttg::abort();
    }

    /// Sets trace for all operations to value and returns previous setting.
    /// This has no effect unless `trace_enabled()==true`
    static bool set_trace_all(bool value) {
      if constexpr (trace_enabled()) std::swap(ttg::detail::tt_base_trace_accessor(), value);
      return value;
    }

    //Sets lazy pulling on.
    //Lazy pulling delays invoking pull terminals until all inputs from push terminals for a task have arrived.
    //Default is false.
    static bool set_lazy_pull(bool value) {
      std::swap(ttg::detail::op_base_lazy_pull_accessor(), value);
      return value;
    }

    /// Sets trace for just this instance to value and returns previous setting
    /// This has no effect unless `trace_enabled()==true`
    bool set_trace_instance(bool value) {
      if constexpr (trace_enabled()) std::swap(trace_instance, value);
      return value;
    }

    /// @return false if `trace_enabled()==false`, else true if tracing set for either this instance or all instances
    bool tracing() const {
      if constexpr (trace_enabled())
        return ttg::detail::tt_base_trace_accessor() || trace_instance;
      else
        return false;
    }

    /// Like ttg::trace(), but only produces tracing output if `this->tracing()==true`
    template <typename T, typename... Ts>
    inline void trace(const T &t, const Ts &...ts) {
      if constexpr (trace_enabled()) {
        if (this->tracing()) {
          log(t, ts...);
        }
      }
    }

    bool set_lazy_pull_instance(bool value) {
      std::swap(lazy_pull_instance, value);
      return value;
    }

    bool is_lazy_pull() { return ttg::detail::op_base_lazy_pull_accessor() || lazy_pull_instance; }

    std::optional<std::reference_wrapper<const TTBase>> ttg() const {
      return owning_ttg ? std::cref(*owning_ttg) : std::optional<std::reference_wrapper<const TTBase>>{};
    }

    const TTBase *ttg_ptr() const {
      return owning_ttg;
    }

    bool is_ttg() const {
      return is_ttg_;
    }

    /// Sets the name of this operation
    void set_name(const std::string &name) { this->name = name; }

    /// Gets the name of this operation
    const std::string &get_name() const { return name; }

    /// Gets the demangled class name (uses RTTI)
    std::string get_class_name() const { return ttg::detail::demangled_type_name(this); }

    /// Returns the vector of input terminals
    const std::vector<TerminalBase *> &get_inputs() const { return inputs; }

    /// Returns the vector of output terminals
    const std::vector<TerminalBase *> &get_outputs() const { return outputs; }

    /// Returns this thread's pointer to the vector of output terminals
    static const std::vector<TerminalBase *> *get_outputs_tls_ptr() { return outputs_tls_ptr_accessor(); }

    /// @return World in which this lives
    virtual ttg::World get_world() const = 0;

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

    /// Returns a pointer to the i'th input terminal ... to make API consistent with TT
    template <std::size_t i>
    ttg::TerminalBase *in() {
      return in(i);
    }

    /// Returns a pointer to the i'th output terminal ... to make API consistent with TT
    template <std::size_t i>
    ttg::TerminalBase *out() {
      return out(i);
    }

    auto get_instance_id() const { return instance_id; }

    /// Waits for the entire TTG that contains this object to be completed (collective); if not contained by a
    /// TTG this is a no-op
    virtual void fence() = 0;

    virtual void release() {}

    /// Marks this executable
    virtual void make_executable() = 0;

    /// Queries if this ready to execute
    /// @return true is this object is executable
    bool is_executable() const { return executable; }

    /// Asserts that this is executable
    /// Use this macro from inside a derived class
    /// @throw std::logic_error if this is not executable
#define TTG_OP_ASSERT_EXECUTABLE()                                      \
  do {                                                                  \
    if (!this->is_executable()) {                                       \
      std::ostringstream oss;                                           \
      oss << "TT is not executable at " << __FILE__ << ":" << __LINE__; \
      throw std::logic_error(oss.str().c_str());                        \
    }                                                                   \
  } while (0);
  };

  inline void TTBase::make_executable() { executable = true; }

}  // namespace ttg

#endif  // TTG_BASE_OP_H
