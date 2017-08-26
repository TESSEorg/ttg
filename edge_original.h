#ifndef MADNESS_EDGE_H_INCLUDED
#define MADNESS_EDGE_H_INCLUDED

#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

template <typename keyT, typename valueT>
class BaseEdge {
 protected:
  using callbackT = std::function<void(const keyT&, const valueT&)>;

  void set_callback(const callbackT& callback) {
    assert(!connected);
    connected = true;
    this->callback = callback;
  }

  // Can bring these back when store InEdges inside Op as planned
  // BaseEdge(const BaseEdge& a) = delete;
  // BaseEdge(BaseEdge&& a) = delete;
  BaseEdge& operator=(const BaseEdge& a) = delete;

 private:
  bool connected = false;
  mutable callbackT callback;

 public:
  BaseEdge() {}

  void send(const keyT& key, const valueT& value) const {
    assert(connected);
    callback(key, value);
  }

  template <typename rangeT>
  void broadcast(const rangeT& keylist, const valueT& value) const {
    for (auto key : keylist) send(key, value);
  }

  const callbackT& get_callback() const {  // Temporarily not protected but friend status needs fixing
    assert(connected);
    return callback;
  }
};

// The graph edge connecting to an input/argument of a task template.
//
// It is presumably connected with an output/result of a predecessor
// task, but you can also inject data directly using the
// send/broadcast methods.
template <typename keyT, typename valueT>
class InEdge : public BaseEdge<keyT, valueT> {
 protected:
  template <typename kT, typename iT, typename oT, typename dT>
  friend class Op;
  template <typename kT, typename vT>
  friend class Merge;

  InEdge(const typename BaseEdge<keyT, valueT>::callbackT& callback) { this->set_callback(callback); }

 public:
  InEdge() {}
};

// The graph edge connecting an output/result of a task with an
// input/argument of a successor task.
//
// It is connected to an input using the connect method (i.e., outedge.connect(inedge))
template <typename keyT, typename valueT>
class OutEdge : public BaseEdge<keyT, valueT> {
 public:
  OutEdge() {}

  void connect(const InEdge<keyT, valueT>& in) { this->set_callback(in.get_callback()); }
};

// Data/functionality common to all Ops
class BaseOp {
  static bool trace;  // If true prints trace of all assignments and all op invocations
  static int count;   // Counts number of instances (to explore if cycles are inhibiting garbage collection)
  std::string name;

 public:
  BaseOp(const std::string& name) : name(name) { count++; }

  // Sets trace to value and returns previous setting
  static bool set_trace(bool value) {
    std::swap(trace, value);
    return value;
  }

  static bool get_trace() { return trace; }

  static bool tracing() { return trace; }

  static int get_count() { return count; }

  void set_name(const std::string& name) { this->name = name; }

  const std::string& get_name() const { return name; }

  ~BaseOp() { count--; }
};

// With more than one source file this will need to be moved
bool BaseOp::trace = false;
int BaseOp::count = 0;

template <typename keyT, typename input_valuesT, typename output_edgesT, typename derivedT>
class Op : private BaseOp {
 private:
  static constexpr int numargs = std::tuple_size<input_valuesT>::value;  // Number of input arguments

  struct OpArgs {
    int counter;                       // Tracks the number of arguments set
    std::array<bool, numargs> argset;  // Tracks if a given arg is already set;
    input_valuesT t;                   // The input values

    OpArgs() : counter(numargs), argset(), t() {}
  };

  output_edgesT output_edges;

  // inputs_edgesT input_edges;

  std::map<keyT, OpArgs> cache;  // Contains tasks waiting for input to become complete

  // Used to set the i'th argument
  template <std::size_t i, typename valueT>
  void set_arg(const keyT& key, const valueT& value) {
    if (tracing()) std::cout << get_name() << " : " << key << ": setting argument : " << i << std::endl;
    OpArgs& args = cache[key];
    if (args.argset[i]) {
      std::cerr << get_name() << " : " << key << ": error argument is already set : " << i << std::endl;
      throw "bad set arg";
    }
    args.argset[i] = true;
    std::get<i>(args.t) = value;
    args.counter--;
    if (args.counter == 0) {
      if (tracing()) std::cout << get_name() << " : " << key << ": invoking op " << std::endl;
      static_cast<derivedT*>(this)->op(key, args.t);
      cache.erase(key);
    }
  }

  Op(const Op& other) = delete;

  Op& operator=(const Op& other) = delete;

  Op(Op&& other) = delete;

 public:
  Op(const std::string& name = std::string("unnamed op")) : BaseOp(name) {}

  // Destructor checks for unexecuted tasks
  ~Op() {
    if (cache.size() != 0) {
      std::cerr << "warning: unprocessed tasks in destructor of operation '" << get_name() << "'" << std::endl;
      std::cerr << "   T => argument assigned     F => argument unassigned" << std::endl;
      int nprint = 0;
      for (auto item : cache) {
        if (nprint++ > 10) {
          std::cerr << "   etc." << std::endl;
          break;
        }
        std::cerr << "   unused: " << item.first << " : ( ";
        for (std::size_t i = 0; i < numargs; i++) std::cerr << (item.second.argset[i] ? "T" : "F") << " ";
        std::cerr << ")" << std::endl;
      }
    }
  }

  // Returns input edge i to facilitate connection
  template <std::size_t i>
  InEdge<keyT, typename std::tuple_element<i, input_valuesT>::type> in() {
    using edgeT = InEdge<keyT, typename std::tuple_element<i, input_valuesT>::type>;
    using valueT = typename std::tuple_element<i, input_valuesT>::type;
    using callbackT = std::function<void(const keyT&, const valueT&)>;
    auto callback = [this](const keyT& key, const valueT& value) { set_arg<i, valueT>(key, value); };
    return edgeT(callbackT(callback));
  }

  // Returns the output edge i to facilitate connection
  template <int i>
  typename std::tuple_element<i, output_edgesT>::type& out() {
    return std::get<i>(output_edges);
  }

  // Send result to successor task of output i
  template <int i, typename outkeyT, typename outvalT>
  void send(const outkeyT& key, const outvalT& value) {
    out<i>().send(key, value);
  }

  // Broadcast result to successor tasks of output i
  template <int i, typename outkeysT, typename outvalT>
  void broadcast(const outkeysT& keys, const outvalT& value) {
    out<i>().broadcast(keys, value);
  }
};

template <typename keyT, typename valueT>
class Merge : private BaseOp {
 private:
  static constexpr int numargs = 2;

  OutEdge<keyT, valueT> outedge;

  template <std::size_t i>
  void set_arg(const keyT& key, const valueT& value) {
    if (tracing()) std::cout << get_name() << " : " << key << ": setting argument : " << i << std::endl;
    outedge.send(key, value);
  }

 public:
  Merge(const std::string& name = std::string("unnamed op")) : BaseOp(name) {}

  // Returns input edge i to facilitate connection
  template <std::size_t i>
  InEdge<keyT, valueT> in() {
    using edgeT = InEdge<keyT, valueT>;
    using callbackT = std::function<void(const keyT&, const valueT&)>;
    auto callback = [this](const keyT& key, const valueT& value) { set_arg<i>(key, value); };
    return edgeT(callbackT(callback));
  }

  // Returns the output edge i to facilitate connection
  template <int i>
  OutEdge<keyT, valueT>& out() {
    static_assert(i == 0);
    return outedge;
  }
};

#endif
