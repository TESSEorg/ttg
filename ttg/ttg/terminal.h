#ifndef TTG_TERMINALS_H
#define TTG_TERMINALS_H

#include <exception>
#include <stdexcept>
#include <type_traits>

#include "ttg/base/terminal.h"
#include "ttg/fwd.h"
#include "ttg/util/demangle.h"
#include "ttg/util/meta.h"
#include "ttg/util/trace.h"
#include "ttg/world.h"

namespace ttg {

  template <typename keyT = void>
  class InTerminalBase : public TerminalBase {
   public:
    typedef keyT key_type;
    static_assert(std::is_same_v<keyT, std::decay_t<keyT>>,
                  "InTerminalBase<keyT,valueT> assumes keyT is a non-decayable type");
    using setsize_callback_type = meta::detail::setsize_callback_t<keyT>;
    using finalize_callback_type = meta::detail::finalize_callback_t<keyT>;
    static constexpr bool is_an_input_terminal = true;

   protected:
    InTerminalBase(TerminalBase::Type t) : TerminalBase(t) {}

    setsize_callback_type setsize_callback;
    finalize_callback_type finalize_callback;

    void set_callback(const setsize_callback_type &setsize_callback = setsize_callback_type{},
                      const finalize_callback_type &finalize_callback = finalize_callback_type{}) {
      this->setsize_callback = setsize_callback;
      this->finalize_callback = finalize_callback;
    }

   private:
    // No moving, copying, assigning permitted
    InTerminalBase(InTerminalBase &&other) = delete;
    InTerminalBase(const InTerminalBase &other) = delete;
    InTerminalBase &operator=(const InTerminalBase &other) = delete;
    InTerminalBase &operator=(const InTerminalBase &&other) = delete;

   public:
    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>, void> set_size(const Key &key, std::size_t size) {
      if (!setsize_callback) throw std::runtime_error("set_size callback not initialized");
      setsize_callback(key, size);
    }

    template <typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>, void> set_size(std::size_t size) {
      if (!setsize_callback) throw std::runtime_error("set_size callback not initialized");
      setsize_callback(size);
    }

    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>, void> finalize(const Key &key) {
      // std::cout << "In::finalize::\n";
      if (!finalize_callback) throw std::runtime_error("finalize callback not initialized");
      finalize_callback(key);
    }

    template <typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>, void> finalize() {
      if (!finalize_callback) throw std::runtime_error("finalize callback not initialized");
      finalize_callback();
    }
  };

  /// Input terminal
  template <typename keyT = void, typename valueT = void>
  class In : public InTerminalBase<keyT> {
   public:
    using base_type = InTerminalBase<keyT>;
    typedef valueT value_type;
    typedef keyT key_type;
    static_assert(std::is_same_v<keyT, std::decay_t<keyT>>, "In<keyT,valueT> assumes keyT is a non-decayable type");
    // valueT can be T or const T
    static_assert(std::is_same_v<std::remove_const_t<valueT>, std::decay_t<valueT>>,
                  "In<keyT,valueT> assumes std::remove_const<T> is a non-decayable type");
    using edge_type = Edge<keyT, valueT>;
    using send_callback_type = meta::detail::send_callback_t<keyT, std::decay_t<valueT>>;
    using move_callback_type = meta::detail::move_callback_t<keyT, std::decay_t<valueT>>;
    using broadcast_callback_type = meta::detail::broadcast_callback_t<keyT, std::decay_t<valueT>>;
    using setsize_callback_type = typename base_type::setsize_callback_type;
    using finalize_callback_type = typename base_type::finalize_callback_type;
    static constexpr bool is_an_input_terminal = true;

   private:
    send_callback_type send_callback;
    move_callback_type move_callback;
    broadcast_callback_type broadcast_callback;

    // No moving, copying, assigning permitted
    In(In &&other) = delete;
    In(const In &other) = delete;
    In &operator=(const In &other) = delete;
    In &operator=(const In &&other) = delete;

    void connect(TerminalBase *p) override {
      throw "Edge: to connect terminals use out->connect(in) rather than in->connect(out)";
    }

   public:
    In() : InTerminalBase<keyT>(std::is_const_v<valueT> ? TerminalBase::Type::Read : TerminalBase::Type::Consume){};

    void set_callback(const send_callback_type &send_callback, const move_callback_type &move_callback,
                      const broadcast_callback_type &bcast_callback = broadcast_callback_type{},
                      const setsize_callback_type &setsize_callback = setsize_callback_type{},
                      const finalize_callback_type &finalize_callback = finalize_callback_type{}) {
      this->send_callback = send_callback;
      this->move_callback = move_callback;
      this->broadcast_callback = bcast_callback;
      base_type::set_callback(setsize_callback, finalize_callback);
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key, Value>, void> send(const Key &key, const Value &value) {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback(key, value);
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key, Value> && std::is_same_v<Value, std::remove_reference_t<Value>>, void>
    send(const Key &key, Value &&value) {
      if (!move_callback) throw std::runtime_error("move callback not initialized");
      move_callback(key, std::forward<valueT>(value));
    }

    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>, void> sendk(const Key &key) {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback(key);
    }

    template <typename Value = valueT>
    std::enable_if_t<!meta::is_void_v<Value>, void> sendv(const Value &value) {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback(value);
    }

    template <typename Value = valueT>
    std::enable_if_t<!meta::is_void_v<Value> && std::is_same_v<Value, std::remove_reference_t<Value>>, void> sendv(
        Value &&value) {
      if (!move_callback) throw std::runtime_error("move callback not initialized");
      move_callback(std::forward<valueT>(value));
    }

    void send() {
      if (!send_callback) throw std::runtime_error("send callback not initialized");
      send_callback();
    }

    // An optimized implementation will need a separate callback for broadcast
    // with a specific value for rangeT
    template <typename rangeT, typename Value>
    std::enable_if_t<!meta::is_void_v<Value>, void> broadcast(const rangeT &keylist, const Value &value) {
      if (broadcast_callback) {
        if constexpr (ttg::meta::is_iterable_v<rangeT>) {
          broadcast_callback(ttg::span(&(*std::begin(keylist)), std::distance(std::begin(keylist), std::end(keylist))),
                             value);
        } else {
          /* got something we cannot iterate over (single element?) so put one element in the span */
          broadcast_callback(ttg::span<const keyT>(&keylist, 1), value);
        }
      } else {
        if constexpr (ttg::meta::is_iterable_v<rangeT>) {
          for (auto &&key : keylist) send(key, value);
        } else {
          /* single element */
          send(keylist, value);
        }
      }
    }

    template <typename rangeT, typename Value>
    std::enable_if_t<!meta::is_void_v<Value>, void> broadcast(const rangeT &keylist, Value &&value) {
      const Value &v = value;
      if (broadcast_callback) {
        if constexpr (ttg::meta::is_iterable_v<rangeT>) {
          broadcast_callback(
              ttg::span<const keyT>(&(*std::begin(keylist)), std::distance(std::begin(keylist), std::end(keylist))), v);
        } else {
          /* got something we cannot iterate over (single element?) so put one element in the span */
          broadcast_callback(ttg::span<const keyT>(&keylist, 1), v);
        }
      } else {
        if constexpr (ttg::meta::is_iterable_v<rangeT>) {
          for (auto &&key : keylist) send(key, v);
        } else {
          /* got something we cannot iterate over (single element?) so put one element in the span */
          broadcast_callback(ttg::span<const keyT>(&keylist, 1), v);
        }
      }
    }

    template <typename rangeT, typename Value = valueT>
    std::enable_if_t<meta::is_void_v<Value>, void> broadcast(const rangeT &keylist) {
      if (broadcast_callback) {
        if constexpr (ttg::meta::is_iterable_v<rangeT>) {
          broadcast_callback(
              ttg::span<const keyT>(&(*std::begin(keylist)), std::distance(std::begin(keylist), std::end(keylist))));
        } else {
          /* got something we cannot iterate over (single element?) so put one element in the span */
          broadcast_callback(ttg::span<const keyT>(&keylist, 1));
        }
      } else {
        if constexpr (ttg::meta::is_iterable_v<rangeT>) {
          for (auto &&key : keylist) sendk(key);
        } else {
          /* got something we cannot iterate over (single element?) so put one element in the span */
          broadcast_callback(ttg::span<const keyT>(&keylist, 1));
        }
      }
    }
  };

  template <typename T>
  inline constexpr bool is_input_terminal_v = false;
  template <typename keyT>
  inline constexpr bool is_input_terminal_v<InTerminalBase<keyT>> = true;
  template <typename keyT, typename valueT>
  inline constexpr bool is_input_terminal_v<In<keyT, valueT>> = true;
  template <>
  inline constexpr bool is_input_terminal_v<TerminalBase> = true;

  namespace detail {
    template <typename keyT, typename... valuesT>
    struct input_terminals_tuple {
      using type = std::tuple<ttg::In<keyT, ttg::meta::remove_wrapper_t<valuesT>>...>;
    };

    template <typename keyT, typename... valuesT>
    struct input_terminals_tuple<keyT, std::tuple<valuesT...>>
    : input_terminals_tuple<keyT, valuesT...>
    { };

    template <typename keyT, typename... valuesT>
    using input_terminals_tuple_t = typename input_terminals_tuple<keyT, valuesT...>::type;
  }  // namespace detail

  template <typename keyT = void>
  class OutTerminalBase : public TerminalBase {
   public:
    using key_type = keyT;
    static_assert(std::is_same_v<keyT, std::decay_t<keyT>>, "Out<keyT,valueT> assumes keyT is a non-decayable type");
    static constexpr bool is_an_output_terminal = true;

   private:
    // No moving, copying, assigning permitted
    OutTerminalBase(OutTerminalBase &&other) = delete;
    OutTerminalBase(const OutTerminalBase &other) = delete;
    OutTerminalBase &operator=(const OutTerminalBase &other) = delete;
    OutTerminalBase &operator=(const OutTerminalBase &&other) = delete;

   public:
    OutTerminalBase() : TerminalBase(TerminalBase::Type::Write) {}

    auto nsuccessors() const { return get_connections().size(); }
    const auto &successors() const { return get_connections(); }

    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>, void> set_size(const Key &key, std::size_t size) {
      for (auto &&successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        static_cast<InTerminalBase<keyT> *>(successor)->set_size(key, size);
      }
    }

    template <typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>, void> set_size(std::size_t size) {
      for (auto &&successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        static_cast<InTerminalBase<keyT> *>(successor)->set_size(size);
      }
    }

    template <typename Key = keyT>
    std::enable_if_t<!meta::is_void_v<Key>, void> finalize(const Key &key) {
      for (auto &&successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        static_cast<InTerminalBase<keyT> *>(successor)->finalize(key);
      }
    }

    template <typename Key = keyT>
    std::enable_if_t<meta::is_void_v<Key>, void> finalize() {
      for (auto successor : successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        static_cast<InTerminalBase<keyT> *>(successor)->finalize();
      }
    }
  };

  /// Output terminal
  template <typename keyT = void, typename valueT = void>
  class Out : public OutTerminalBase<keyT> {
   public:
    using value_type = valueT;
    using key_type = typename OutTerminalBase<keyT>::key_type;
    static_assert(std::is_same_v<valueT, std::decay_t<valueT>>,
                  "Out<keyT,valueT> assumes valueT is a non-decayable type");
    using edge_type = Edge<keyT, valueT>;
    static constexpr bool is_an_output_terminal = true;

   private:
    // No moving, copying, assigning permitted
    Out(Out &&other) = delete;
    Out(const Out &other) = delete;
    Out &operator=(const Out &other) = delete;
    Out &operator=(const Out &&other) = delete;

   public:
    Out() = default;

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
      trace(rank(), ": connected Out<> ", this->get_name(), "(ptr=", this, ") to In<> ", in->get_name(), "(ptr=", in,
            ")");
#endif
      this->connect_base(in);
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key, Value>, void> send(const Key &key, const Value &value) {
      for (auto &&successor : this->successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->send(key, value);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->send(key, value);
        }
      }
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<!meta::is_void_v<Key> && meta::is_void_v<Value>, void> sendk(const Key &key) {
      for (auto &&successor : this->successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->sendk(key);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->sendk(key);
        }
      }
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_void_v<Key> && !meta::is_void_v<Value>, void> sendv(const Value &value) {
      for (auto &&successor : this->successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->sendv(value);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->sendv(value);
        }
      }
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_all_void_v<Key, Value>, void> send() {
      trace(rank(), ": in ", this->get_name(), "(ptr=", this, ") Out<>::send: #successors=", this->successors().size());
      for (auto &&successor : this->successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->send();
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->send();
        } else {
          throw std::logic_error("Out<>: invalid successor type");
        }
        trace("Out<> ", this->get_name(), "(ptr=", this, ") send to In<> ", successor->get_name(), "(ptr=", successor,
              ")");
      }
    }

    template <typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key, Value> && std::is_same_v<Value, std::remove_reference_t<Value>>, void>
    send(const Key &key, Value &&value) {
      const std::size_t N = this->nsuccessors();
      TerminalBase *move_successor = nullptr;
      // send copies to every terminal except the one we will move the results to
      for (std::size_t i = 0; i != N; ++i) {
        TerminalBase *successor = this->successors().at(i);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->send(key, value);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          if (nullptr == move_successor) {
            move_successor = successor;
          } else {
            static_cast<In<keyT, valueT> *>(successor)->send(key, value);
          }
        }
      }
      if (nullptr != move_successor) {
        static_cast<In<keyT, valueT> *>(move_successor)->send(key, std::forward<Value>(value));
      }
    }

    // An optimized implementation will need a separate callback for broadcast
    // with a specific value for rangeT
    template <typename rangeT, typename Key = keyT, typename Value = valueT>
    std::enable_if_t<meta::is_none_void_v<Key, Value>, void> broadcast(const rangeT &keylist,
                                                                       const Value &value) {  // NO MOVE YET
      for (auto &&successor : this->successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, std::add_const_t<valueT>> *>(successor)->broadcast(keylist, value);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, valueT> *>(successor)->broadcast(keylist, value);
        }
      }
    }

    template <typename rangeT, typename Key = keyT>
    std::enable_if_t<meta::is_none_void_v<Key> && meta::is_void_v<valueT>, void> broadcast(const rangeT &keylist) {
      for (auto &&successor : this->successors()) {
        assert(successor->get_type() != TerminalBase::Type::Write);
        if (successor->get_type() == TerminalBase::Type::Read) {
          static_cast<In<keyT, void> *>(successor)->broadcast(keylist);
        } else if (successor->get_type() == TerminalBase::Type::Consume) {
          static_cast<In<keyT, void> *>(successor)->broadcast(keylist);
        }
      }
    }
  };

  template <typename T>
  inline constexpr bool is_output_terminal_v = false;
  template <typename keyT>
  inline constexpr bool is_output_terminal_v<OutTerminalBase<keyT>> = true;
  template <typename keyT, typename valueT>
  inline constexpr bool is_output_terminal_v<Out<keyT, valueT>> = true;
  template <>
  inline constexpr bool is_output_terminal_v<TerminalBase> = true;

}  // namespace ttg

#endif  // TTG_TERMINALS_H
