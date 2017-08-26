#ifndef MADNESS_EDGE_H_INCLUDED
#define MADNESS_EDGE_H_INCLUDED

#include <madness/world/MADworld.h>
#include <madness/world/worldhashmap.h>
#include <madness/world/worldtypes.h>
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
class Op : protected BaseOp, public madness::WorldObject<Op<keyT, input_valuesT, output_edgesT, derivedT>> {
 private:
  static constexpr int numargs = std::tuple_size<input_valuesT>::value;  // Number of input arguments
  using opT = Op<keyT, input_valuesT, output_edgesT, derivedT>;
  using worldobjT = madness::WorldObject<Op<keyT, input_valuesT, output_edgesT, derivedT>>;

  madness::World& world;
  std::shared_ptr<madness::WorldDCPmapInterface<keyT>> pmap;

  struct OpArgs : madness::TaskInterface {
    int counter;                       // Tracks the number of arguments set
    std::array<bool, numargs> argset;  // Tracks if a given arg is already set;
    input_valuesT t;                   // The input values
    derivedT* derived;                 // Pointer to derived class instance
    keyT key;                          // Task key

    OpArgs() : counter(numargs), argset(), t() {}

    void run(madness::World& world) { derived->op(key, t); }

    virtual ~OpArgs() {}  // Will be deleted via TaskInterface*
  };

  output_edgesT output_edges;

  using cacheT = madness::ConcurrentHashMap<keyT, OpArgs*>;
  using accessorT = typename cacheT::accessor;
  cacheT cache;

  // Used to set the i'th argument
  template <std::size_t i, typename valueT>
  void set_arg(const keyT& key, const valueT& value) {
    ProcessID owner = pmap->owner(key);

    if (owner != world.rank()) {
      if (tracing())
        std::cout << world.rank() << ":" << get_name() << " : " << key << ": forwarding setting argument : " << i
                  << std::endl;
      worldobjT::send(owner, &opT::template set_arg<i, valueT>, key, value);
    } else {
      if (tracing())
        std::cout << world.rank() << ":" << get_name() << " : " << key << ": setting argument : " << i << std::endl;

      accessorT acc;
      if (cache.insert(acc, key)) acc->second = new OpArgs();  // It will be deleted by the task q
      OpArgs* args = acc->second;

      if (args->argset[i]) {
        std::cerr << world.rank() << ":" << get_name() << " : " << key << ": error argument is already set : " << i
                  << std::endl;
        throw "bad set arg";
      }
      args->argset[i] = true;
      std::get<i>(args->t) = value;
      args->counter--;
      if (args->counter == 0) {
        if (tracing())
          std::cout << world.rank() << ":" << get_name() << " : " << key << ": submitting task for op " << std::endl;
        args->derived = static_cast<derivedT*>(this);
        args->key = key;

        world.taskq.add(args);

        // world.taskq.add(static_cast<derivedT*>(this), &derivedT::op, key, args.t);

        // if (tracing()) std::cout << world.rank() << ":" << get_name() << " : " << key << ": invoking op " <<
        // std::endl; static_cast<derivedT*>(this)

        cache.erase(key);
      }
    }
  }

  Op(const Op& other) = delete;

  Op& operator=(const Op& other) = delete;

  Op(Op&& other) = delete;

 public:
  Op(madness::World& world, const std::string& name = std::string("unnamed op"))
      : BaseOp(name)
      , madness::WorldObject<Op<keyT, input_valuesT, output_edgesT, derivedT>>(world)
      , world(world)
      , pmap(std::make_shared<madness::WorldDCDefaultPmap<keyT>>(world)) {
    this->process_pending();
  }

  // Destructor checks for unexecuted tasks
  ~Op() {
    if (cache.size() != 0) {
      std::cerr << world.rank() << ":"
                << "warning: unprocessed tasks in destructor of operation '" << get_name() << "'" << std::endl;
      std::cerr << world.rank() << ":"
                << "   T => argument assigned     F => argument unassigned" << std::endl;
      int nprint = 0;
      for (auto item : cache) {
        if (nprint++ > 10) {
          std::cerr << "   etc." << std::endl;
          break;
        }
        std::cerr << world.rank() << ":"
                  << "   unused: " << item.first << " : ( ";
        for (std::size_t i = 0; i < numargs; i++) std::cerr << (item.second->argset[i] ? "T" : "F") << " ";
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
class Merge : private BaseOp, public madness::WorldObject<Merge<keyT, valueT>> {
 private:
  static constexpr int numargs = 2;
  using opT = Merge<keyT, valueT>;
  using worldobjT = madness::WorldObject<Merge<keyT, valueT>>;

  madness::World& world;
  std::shared_ptr<madness::WorldDCPmapInterface<keyT>> pmap;

  OutEdge<keyT, valueT> outedge;

  template <std::size_t i>
  void set_arg(const keyT& key, const valueT& value) {
    ProcessID owner = pmap->owner(key);

    if (owner != world.rank()) {
      if (tracing())
        std::cout << world.rank() << ":" << get_name() << " : " << key << ": forwarding setting argument : " << i
                  << std::endl;
      worldobjT::send(owner, &opT::template set_arg<i>, key, value);
    } else {
      if (tracing())
        std::cout << world.rank() << ":" << get_name() << " : " << key << ": setting argument : " << i << std::endl;
      outedge.send(key, value);
    }
  }

 public:
  Merge(madness::World& world, const std::string& name = std::string("unnamed op"))
      : BaseOp(name)
      , madness::WorldObject<Merge<keyT, valueT>>(world)
      , world(world)
      , pmap(std::make_shared<madness::WorldDCDefaultPmap<keyT>>(world)) {
    this->process_pending();
  }

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
    static_assert(i == 0, "why?");
    return outedge;
  }
};

#endif
