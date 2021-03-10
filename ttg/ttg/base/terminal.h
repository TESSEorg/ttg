#ifndef TTG_BASE_TERMINAL_H
#define TTG_BASE_TERMINAL_H

#include <string>
#include <vector>

namespace ttg {

  // forward-decl
  class OpBase;

  /// Provides basic information and graph connectivity (eventually statistics,
  /// etc.)
  class TerminalBase {
  public:
    static constexpr bool is_a_terminal = true;
		bool is_pull_terminal = false; //< Default is push terminal

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
    std::vector<TerminalBase *> predecessors_; //This is required for pull terminals.

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
    
    void connect_pull(TerminalBase *predecessor) {
      //std::cout << "set_out : " << this->get_name() << "-> has predecessor " << predecessor->get_name() << std::endl; 
      predecessors_.push_back(predecessor); 
      connected = true; 
      predecessor->connected = true; 
    }

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

    // Get connections to predecessors
    const std::vector<TerminalBase *> &get_predecessors() const {return predecessors_; }

    /// Returns true if this terminal (input or output) is connected
    bool is_connected() const {return connected;}


    /// Connect this (a TTG output terminal) to a TTG input terminal.
    /// The base class method forwards to the the derived class connect method and so
    /// type checking for the key/value will be done at runtime when performing the
    /// dynamic down cast from TerminalBase* to In<keyT,valueT>.
    virtual void connect(TerminalBase *in) = 0;

    virtual ~TerminalBase() = default;
  };
} // namespace ttg

#endif // TTG_BASE_TERMINAL_H
