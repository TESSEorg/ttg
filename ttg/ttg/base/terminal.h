#ifndef TTG_BASE_TERMINAL_H
#define TTG_BASE_TERMINAL_H

#include <string>
#include <vector>
#include "ttg/fwd.h"

namespace ttg {

  /// Provides basic information and graph connectivity (eventually statistics,
  /// etc.)
  class TerminalBase {
   public:
    static constexpr bool is_a_terminal = true;

    /// describes the terminal type
    enum class Type {
      Write,   //!< can only be written to
      Read,    //!< can only be used to read immutable data
      Consume  //!< can only be used to read consumable data
    };

   private:
    TTBase *tt;              //< Pointer to containing operation
    size_t n = 0;            //< Index of terminal
    std::string name = "";   //< Name of terminal
    bool connected = false;  //< True if is connected
    Type type;
    std::string key_type_str;    //< String describing key type
    std::string value_type_str;  //< String describing value type

    std::vector<TerminalBase *> successors_;

    TerminalBase(const TerminalBase &) = delete;
    TerminalBase(TerminalBase &&) = delete;

    friend class TTBase;
    template <typename keyT, typename valueT>
    friend class In;
    template <typename keyT, typename valueT>
    friend class Out;

   protected:
    TerminalBase(Type type) : type(type) {}

    void set(TTBase *tt, size_t index, const std::string &name, const std::string &key_type_str,
             const std::string &value_type_str, Type type) {
      this->tt = tt;
      this->n = index;
      this->name = name;
      this->key_type_str = key_type_str;
      this->value_type_str = value_type_str;
      this->type = type;
    }

    /// Add directed connection (this --> successor) in internal representation of the TTG.
    /// This is called by the derived class's connect method
    void connect_base(TerminalBase *successor) {
      successors_.push_back(successor);
      connected = true;
      successor->connected = true;
    }

   public:
    /// Return ptr to containing tt
    TTBase *get_tt() const {
      if (!tt) throw "ttg::TerminalBase:get_tt() but tt is null";
      return tt;
    }

    /// Returns index of terminal
    size_t get_index() const {
      if (!tt) throw "ttg::TerminalBase:get_index() but tt is null";
      return n;
    }

    /// Returns name of terminal
    const std::string &get_name() const {
      if (!tt) throw "ttg::TerminalBase:get_name() but tt is null";
      return name;
    }

    /// Returns string representation of key type
    const std::string &get_key_type_str() const {
      if (!tt) throw "ttg::TerminalBase:get_key_type_str() but tt is null";
      return key_type_str;
    }

    /// Returns string representation of value type
    const std::string &get_value_type_str() const {
      if (!tt) throw "ttg::TerminalBase:get_value_type_str() but tt is null";
      return value_type_str;
    }

    /// Returns the terminal type
    Type get_type() const { return this->type; }

    /// Get connections to successors
    const std::vector<TerminalBase *> &get_connections() const { return successors_; }

    /// Returns true if this terminal (input or output) is connected
    bool is_connected() const { return connected; }

    /// Connect this (a TTG output terminal) to a TTG input terminal.
    /// The base class method forwards to the the derived class connect method and so
    /// type checking for the key/value will be done at runtime when performing the
    /// dynamic down cast from TerminalBase* to In<keyT,valueT>.
    virtual void connect(TerminalBase *in) = 0;

    virtual ~TerminalBase() = default;
  };
}  // namespace ttg

#endif  // TTG_BASE_TERMINAL_H
