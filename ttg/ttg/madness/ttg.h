#ifndef MADNESS_TTG_H_INCLUDED
#define MADNESS_TTG_H_INCLUDED

/* set up env if this header was included directly */
#if !defined(TTG_IMPL_NAME)
#define TTG_USE_MADNESS 1
#endif  // !defined(TTG_IMPL_NAME)

#include "ttg/impl_selector.h"

/* include ttg header to make symbols available in case this header is included directly */
#include "../../ttg.h"
#include "ttg/base/keymap.h"
#include "ttg/base/op.h"
#include "ttg/func.h"
#include "ttg/op.h"
#include "ttg/runtimes.h"
#include "ttg/util/bug.h"
#include "ttg/util/hash.h"
#include "ttg/util/macro.h"
#include "ttg/util/meta.h"
#include "ttg/util/void.h"
#include "ttg/world.h"

#include <array>
#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <madness/world/MADworld.h>
#include <madness/world/world_object.h>
#include <madness/world/worldhashmap.h>
#include <madness/world/worldtypes.h>

#include <madness/world/world_task_queue.h>

#include <boost/callable_traits.hpp>  // needed for wrap.h

namespace ttg_madness {

#if 0
    class Control;
    class Graph;
  /// Graph is a collection of Op objects
  class Graph {
   public:
    Graph() {
      world_ = get_default_world();
    }
    Graph(World& w) : world_(w) {}


   private:
    World& world_;
  };
#endif

  class WorldImpl final : public ttg::base::WorldImplBase {
   private:
    ::madness::World &m_impl;
    bool m_allocated = false;

    ttg::Edge<> m_ctl_edge;

   public:
    WorldImpl(::madness::World &world) : m_impl(world) {}

    WorldImpl(const SafeMPI::Intracomm &comm) : m_impl(*new ::madness::World(comm)), m_allocated(true) {}

    /* Deleted copy ctor */
    WorldImpl(const WorldImpl &other) = delete;

    /* Deleted move ctor */
    WorldImpl(WorldImpl &&other) = delete;

    virtual ~WorldImpl() override { destroy(); }

    /* Deleted copy assignment */
    WorldImpl &operator=(const WorldImpl &other) = delete;

    /* Deleted move assignment */
    WorldImpl &operator=(WorldImpl &&other) = delete;

    virtual int size(void) const override { return m_impl.size(); }

    virtual int rank(void) const override { return m_impl.rank(); }

    MPI_Comm comm() const { return m_impl.mpi.Get_mpi_comm(); }

    virtual void fence_impl(void) override { m_impl.gop.fence(); }

    ttg::Edge<> &ctl_edge() { return m_ctl_edge; }

    const ttg::Edge<> &ctl_edge() const { return m_ctl_edge; }

    virtual void destroy(void) override {
      if (is_valid()) {
        release_ops();
        ttg::detail::deregister_world(*this);
        if (m_allocated) {
          delete &m_impl;
          m_allocated = false;
        }
        mark_invalid();
      }
    }

    /* Return an unmanaged reference to the implementation object */
    ::madness::World &impl() { return m_impl; }

    const ::madness::World &impl() const { return m_impl; }

#ifdef ENABLE_PARSEC
    parsec_context_t *context() { return ::madness::ThreadPool::instance()->parsec; }
#endif
  };

  template <typename... RestOfArgs>
  inline void ttg_initialize(int argc, char **argv, RestOfArgs &&...) {
    ::madness::World &madworld = ::madness::initialize(argc, argv);
    auto *world_ptr = new ttg_madness::WorldImpl{madworld};
    std::shared_ptr<ttg::base::WorldImplBase> world_sptr{static_cast<ttg::base::WorldImplBase *>(world_ptr)};
    ttg::World world{std::move(world_sptr)};
    ttg::detail::set_default_world(std::move(world));
  }
  inline void ttg_finalize() {
    ttg::detail::set_default_world(ttg::World{});  // reset the default world
    ttg::detail::destroy_worlds<ttg_madness::WorldImpl>();
    ::madness::finalize();
  }
  inline void ttg_abort() { MPI_Abort(MPI_COMM_WORLD, 1); }
  inline ttg::World ttg_default_execution_context() { return ttg::get_default_world(); }
  inline void ttg_execute(ttg::World world) {
    // World executes tasks eagerly
  }
  inline void ttg_fence(ttg::World world) { world.impl().fence(); }

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr) {
    world.impl().register_ptr(ptr);
  }

  inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr) {
    world.impl().register_status(status_ptr);
  }

  template <typename Callback>
  inline void ttg_register_callback(ttg::World world, Callback &&callback) {
    world.impl().register_callback(std::forward<Callback>(callback));
  }

  inline ttg::Edge<> &ttg_ctl_edge(ttg::World world) { return world.impl().ctl_edge(); }

  template <typename T>
  inline void ttg_sum(ttg::World world, T &value) {
    world.impl().impl().gop.sum(value);
  }
  /// broadcast
  /// @tparam T a serializable type
  template <typename T>
  inline void ttg_broadcast(ttg::World world, T &data, int source_rank) {
    world.impl().impl().gop.broadcast_serializable(data, source_rank);
  }

  /// CRTP base for MADNESS-based Op classes
  /// \tparam keyT a Key type
  /// \tparam output_terminalsT
  /// \tparam derivedT
  /// \tparam input_valueTs pack of *value* types (no references; pointers are OK) encoding the types of input values
  ///         flowing into this Op; a const type indicates nonmutating (read-only) use, nonconst type
  ///         indicates mutating use (e.g. the corresponding input can be used as scratch, moved-from, etc.)
  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  class Op : public ttg::OpBase,
             public ::madness::WorldObject<Op<keyT, output_terminalsT, derivedT, input_valueTs...>> {
   public:
    /// preconditions
    static_assert((!std::is_reference_v<input_valueTs> && ...), "input_valueTs cannot contain reference types");

   private:
    ttg::World world;
    ttg::meta::detail::keymap_t<keyT> keymap;
    ttg::meta::detail::keymap_t<keyT> priomap;
    // For now use same type for unary/streaming input terminals, and stream reducers assigned at runtime
    ttg::meta::detail::input_reducers_t<input_valueTs...>
        input_reducers;  //!< Reducers for the input terminals (empty = expect single value)

    std::array<std::size_t, sizeof...(input_valueTs)> static_streamsize;

   public:
    ttg::World get_world() const { return world; }

   protected:
    using opT = Op<keyT, output_terminalsT, derivedT, input_valueTs...>;
    using worldobjT = ::madness::WorldObject<opT>;

    static constexpr int numins = sizeof...(input_valueTs);                    // number of input arguments
    static constexpr int numouts = std::tuple_size<output_terminalsT>::value;  // number of outputs or
    // results

    // This to support op fusion
    inline static __thread struct {
      uint64_t key_hash = 0;  // hash of current key
      size_t call_depth = 0;  // how deep calls are nested
    } threaddata;

    using input_terminals_type = std::tuple<ttg::In<keyT, input_valueTs>...>;
    using input_edges_type = std::tuple<ttg::Edge<keyT, std::decay_t<input_valueTs>>...>;
    static_assert(ttg::meta::is_none_Void_v<input_valueTs...>, "ttg::Void is for internal use only, do not use it");
    static_assert(ttg::meta::is_none_void_v<input_valueTs...> || ttg::meta::is_last_void_v<input_valueTs...>,
                  "at most one void input can be handled, and it must come last");
    // if have data inputs and (always last) control input, convert last input to Void to make logic easier
    using input_values_full_tuple_type = std::tuple<ttg::meta::void_to_Void_t<std::decay_t<input_valueTs>>...>;
    using input_refs_full_tuple_type =
        std::tuple<std::add_lvalue_reference_t<ttg::meta::void_to_Void_t<input_valueTs>>...>;
    using input_values_tuple_type =
        std::conditional_t<ttg::meta::is_none_void_v<input_valueTs...>, input_values_full_tuple_type,
                           typename ttg::meta::drop_last_n<input_values_full_tuple_type, std::size_t{1}>::type>;
    using input_refs_tuple_type =
        std::conditional_t<ttg::meta::is_none_void_v<input_valueTs...>, input_refs_full_tuple_type,
                           typename ttg::meta::drop_last_n<input_refs_full_tuple_type, std::size_t{1}>::type>;

    using output_terminals_type = output_terminalsT;
    using output_edges_type = typename ttg::terminals_to_edges<output_terminalsT>::type;

    template <std::size_t i, typename resultT, typename InTuple>
    static resultT get(InTuple &&intuple) {
      return static_cast<resultT>(std::get<i>(std::forward<InTuple>(intuple)));
    };
    template <std::size_t i, typename InTuple>
    static auto &get(InTuple &&intuple) {
      return std::get<i>(std::forward<InTuple>(intuple));
    };

   private:
    input_terminals_type input_terminals;
    output_terminalsT output_terminals;

    struct OpArgs : ::madness::TaskInterface {
     private:
      using TaskInterface = ::madness::TaskInterface;

     public:
      int counter;  // Tracks the number of arguments finalized
      std::array<std::size_t, numins>
          nargs;  // Tracks the number of expected values
                  // for any type of input: 0 = finalized;
                  // for a streaming input the following values are possible:
                  // - 0: finalized
                  // - std::numeric_limits<std::size_t>::max(): initial state (there is no value yet)
                  // - 1: if streaming: have a value, waiting for more
                  // - n: if nonstreaming: expect this many more values
      std::array<std::size_t, numins> stream_size;  // Expected number of values to receive, to be used for streaming
                                                    // inputs (0 = unbounded stream, >0 = bounded stream)
      input_values_tuple_type input_values;         // The input values (does not include control)
      derivedT *derived;                            // Pointer to derived class instance
      std::conditional_t<ttg::meta::is_void_v<keyT>, ttg::Void, keyT> key;  // Task key

      /// makes a tuple of references out of tuple of
      template <typename Tuple, std::size_t... Is>
      static input_refs_tuple_type make_input_refs_impl(Tuple &&inputs, std::index_sequence<Is...>) {
        return input_refs_tuple_type{
            get<Is, std::tuple_element_t<Is, input_refs_tuple_type>>(std::forward<Tuple>(inputs))...};
      }

      /// makes a tuple of references out of input_values
      input_refs_tuple_type make_input_refs() {
        return make_input_refs_impl(this->input_values,
                                    std::make_index_sequence<std::tuple_size_v<input_values_tuple_type>>{});
      }

      OpArgs(int prio = 0)
          : TaskInterface(TaskAttributes(prio ? TaskAttributes::HIGHPRIORITY : 0))
          , counter(numins)
          , nargs()
          , stream_size()
          , input_values() {
        std::fill(nargs.begin(), nargs.end(), std::numeric_limits<std::size_t>::max());
      }

      virtual void run(::madness::World &world) override {
        // ttg::print("starting task");

        using ttg::hash;
        opT::threaddata.key_hash = hash<decltype(key)>{}(key);
        opT::threaddata.call_depth++;

        if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          derived->op(key, this->make_input_refs(),
                      derived->output_terminals);  // !!! NOTE converting input values to refs
        } else if constexpr (!ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          derived->op(key, derived->output_terminals);
        } else if constexpr (ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          derived->op(this->make_input_refs(),
                      derived->output_terminals);  // !!! NOTE converting input values to refs
        } else if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
          derived->op(derived->output_terminals);
        } else
          abort();

        opT::threaddata.call_depth--;

        // ttg::print("finishing task",opT::threaddata.call_depth);
      }

      virtual ~OpArgs() {}  // Will be deleted via TaskInterface*

     private:
      ::madness::Spinlock lock_;  // synchronizes access to data
     public:
      void lock() { lock_.lock(); }
      void unlock() { lock_.unlock(); }
    };

    using hashable_keyT = std::conditional_t<ttg::meta::is_void_v<keyT>, int, keyT>;
    using cacheT = ::madness::ConcurrentHashMap<hashable_keyT, OpArgs *, ttg::hash<hashable_keyT>>;
    using accessorT = typename cacheT::accessor;
    cacheT cache;

   protected:
    // there are 6 types of set_arg:
    // - case 1: nonvoid Key, complete Value type
    // - case 2: nonvoid Key, void Value, mixed (data+control) inputs
    // - case 3: nonvoid Key, void Value, no inputs
    // - case 4:    void Key, complete Value type
    // - case 5:    void Key, void Value, mixed (data+control) inputs
    // - case 6:    void Key, void Value, no inputs
    // cases 2 and 5 will be implemented by passing dummy ttg::Void object to reduce the number of code branches

    // case 1:
    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg(const Key &key,
                                                                                                       Value &&value) {
      using valueT = typename std::tuple_element<i, input_values_full_tuple_type>::type;  // Should be T or const T
      static_assert(std::is_same_v<std::decay_t<Value>, std::decay_t<valueT>>,
                    "Op::set_arg(key,value) given value of type incompatible with Op");

      const auto owner = keymap(key);
      if (owner != world.rank()) {
        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding setting argument : ", i);
        // should be able on the other end to consume value (since it is just a temporary byproduct of serialization)
        // BUT compiler vomits when const std::remove_reference_t<Value>& -> std::decay_t<Value>
        // this exposes bad design in MemFuncWrapper (probably similar bugs elsewhere?) whose generic operator()
        // should use memfun's argument types (since that's what will be called) rather than misautodeduce in a
        // particular context P.S. another issue is in send_am which can execute both remotely (where one can always
        // move arguments) and locally
        //      here we know that this will be a remove execution, so we prepare to take rvalues;
        //      send_am will need to separate local and remote paths to deal with this
        worldobjT::send(owner, &opT::template set_arg<i, Key, const std::remove_reference_t<Value> &>, key, value);
      } else {
        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : ", key, ": received value for argument : ", i);

        accessorT acc;
        if (cache.insert(acc, key)) acc->second = new OpArgs(this->priomap(key));  // It will be deleted by the task q
        OpArgs *args = acc->second;

        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error argument is already finalized : ", i);
          throw std::runtime_error("Op::set_arg called for a finalized stream");
        }

        auto reducer = std::get<i>(input_reducers);
        if (reducer) {  // is this a streaming input? reduce the received value
          // N.B. Right now reductions are done eagerly, without spawning tasks
          //      this means we must lock
          args->lock();
          if constexpr (!ttg::meta::is_void_v<valueT>) {  // for data values
            // have a value already? if not, set, otherwise reduce
            if (args->nargs[i] == std::numeric_limits<std::size_t>::max()) {
              this->get<i, std::decay_t<valueT> &>(args->input_values) = std::forward<Value>(value);

              // now have a value, reset nargs
              // check if we have a stream size for the op, which has precedence over the global setting.
              if (args->stream_size[i] != 0) {
                args->nargs[i] = args->stream_size[i];
              } else if (static_streamsize[i] != 0) {
                args->stream_size[i] = static_streamsize[i];
                args->nargs[i] = static_streamsize[i];
              } else {
                args->nargs[i] = 1;
              }
            } else {
              // Why do we need to move the value here??
              valueT value_copy = value;  // use constexpr if to avoid making a copy if given nonconst rvalue
              this->get<i, std::decay_t<valueT> &>(args->input_values) =
                  std::move(reducer(this->get<i, std::decay_t<valueT> &&>(args->input_values), std::move(value_copy)));
            }
          } else {
            reducer();  // even if this was a control input, must execute the reducer for possible side effects
          }
          // update the counter if the stream is bounded
          // this assumes that the stream size is set before data starts flowing ... strong-typing streams will solve
          // this
          if (args->stream_size[i] != 0) {
            args->nargs[i]--;
            if (args->nargs[i] == 0) args->counter--;
          }
          args->unlock();
        } else {                                          // this is a nonstreaming input => set the value
          if constexpr (!ttg::meta::is_void_v<valueT>) {  // for data values
            this->get<i, std::decay_t<valueT> &>(args->input_values) = std::forward<Value>(value);
          }
          args->nargs[i] = 0;
          args->counter--;
        }

        // ready to run the task?
        if (args->counter == 0) {
          if (tracing()) ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
          args->derived = static_cast<derivedT *>(this);
          args->key = key;

          using ttg::hash;
          auto curhash = hash<keyT>{}(key);

          if (curhash == threaddata.key_hash && threaddata.call_depth < 6) {  // Needs to be externally configurable

            // ttg::print("directly invoking:", get_name(), key, curhash, threaddata.key_hash, threaddata.call_depth);
            opT::threaddata.call_depth++;
            if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(key, args->make_input_refs(), output_terminals);  // Runs immediately
            } else if constexpr (!ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(key, output_terminals);  // Runs immediately
            } else if constexpr (ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(args->make_input_refs(), output_terminals);  // Runs immediately
            } else if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(output_terminals);  // Runs immediately
            } else
              abort();
            opT::threaddata.call_depth--;

          } else {
            // ttg::print("enqueuing task", get_name(), key, curhash, threaddata.key_hash, threaddata.call_depth);
            world.impl().impl().taskq.add(args);
          }

          cache.erase(acc);
        }
      }
    }

    // case 2
    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && std::is_void_v<Value>, void> set_arg(const Key &key) {
      set_arg<i>(key, ttg::Void{});
    }

    // case 4
    template <std::size_t i, typename Key = keyT, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg(Value &&value) {
      using valueT = typename std::tuple_element<i, input_values_full_tuple_type>::type;  // Should be T or const T
      static_assert(std::is_same<std::decay_t<Value>, std::decay_t<valueT>>::value,
                    "Op::set_arg(key,value) given value of type incompatible with Op");

      const int owner = keymap();

      if (owner != world.rank()) {
        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : forwarding setting argument : ", i);
        // CAVEAT see comment above in set_arg re:
        worldobjT::send(owner, &opT::template set_arg<i, keyT, const std::remove_reference_t<Value> &>, value);
      } else {
        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : received value for argument : ", i);

        accessorT acc;
        if (cache.insert(acc, 0)) acc->second = new OpArgs();  // It will be deleted by the task q
        OpArgs *args = acc->second;

        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error argument is already finalized : ", i);
          throw std::runtime_error("Op::set_arg called for a finalized stream");
        }

        auto reducer = std::get<i>(input_reducers);
        if (reducer) {  // is this a streaming input? reduce the received value
          // N.B. Right now reductions are done eagerly, without spawning tasks
          //      this means we must lock
          args->lock();
          if (tracing()) {
            ttg::print_error(world.rank(), ":", get_name(), " : reducing value into argument : ", i);
          }
          // have a value already? if not, set, otherwise reduce
          if (args->nargs[i] == std::numeric_limits<std::size_t>::max()) {
            this->get<i, std::decay_t<valueT> &>(args->input_values) = std::forward<Value>(value);
            // now have a value, reset nargs
            if (args->stream_size[i] != 0) {
              args->nargs[i] = args->stream_size[i];
            } else if (static_streamsize[i] != 0) {
              args->stream_size[i] = static_streamsize[i];
              args->nargs[i] = static_streamsize[i];
            } else {
              args->nargs[i] = 1;
            }
          } else {
            valueT value_copy = value;  // use constexpr if to avoid making a copy if given nonconst rvalue
            // once Future<>::operator= semantics is cleaned up will avoid Future<>::get()
            this->get<i, std::decay_t<valueT> &>(args->input_values) =
                std::move(reducer(this->get<i, std::decay_t<valueT> &&>(args->input_values), std::move(value_copy)));
          }
          // update the counter if the stream is bounded
          // this assumes that the stream size is set before data starts flowing ... strong-typing streams will solve
          // this
          if (args->stream_size[i] != 0) {
            args->nargs[i]--;
            if (tracing()) {
              ttg::print_error(world.rank(), ":", get_name(), " : stream ", i, " has size ", args->stream_size[i],
                               " current nargs", args->nargs[i]);
            }
            if (args->nargs[i] == 0) args->counter--;
          }
          args->unlock();
        } else {  // this is a nonstreaming input => set the value
          this->get<i, std::decay_t<valueT> &>(args->input_values) = std::forward<Value>(value);
          args->nargs[i] = 0;
          args->counter--;
        }

        // ready to run the task?
        if (args->counter == 0) {
          if (tracing()) ttg::print(world.rank(), ":", get_name(), " : submitting task for op ");
          args->derived = static_cast<derivedT *>(this);

          world.impl().impl().taskq.add(args);

          cache.erase(acc);
        }
      }
    }

    // case 5
    template <std::size_t i, typename Key = keyT, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && std::is_void_v<Value>, void> set_arg() {
      set_arg<i>(ttg::Void{});
    }

    // case 3
    template <typename Key = keyT>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, void> set_arg(const Key &key) {
      static_assert(ttg::meta::is_empty_tuple_v<input_values_tuple_type>,
                    "set_arg called without a value but valueT!=void");
      const int owner = keymap(key);

      if (owner != world.rank()) {
        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding no-arg task: ");
        worldobjT::send(owner, &opT::set_arg<keyT>, key);
      } else {
        accessorT acc;
        if (cache.insert(acc, key)) acc->second = new OpArgs(this->priomap(key));  // It will be deleted by the task q
        OpArgs *args = acc->second;

        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
        args->derived = static_cast<derivedT *>(this);
        args->key = key;

        world.impl().impl().taskq.add(args);
        // static_cast<derivedT*>(this)->op(key, std::move(args->t), output_terminals);// runs immediately

        cache.erase(acc);
      }
    }

    // case 6
    template <typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key>, void> set_arg() {
      static_assert(ttg::meta::is_empty_tuple_v<input_values_tuple_type>,
                    "set_arg called without a value but valueT!=void");
      const int owner = keymap();

      if (owner != world.rank()) {
        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : forwarding no-arg task: ");
        worldobjT::send(owner, &opT::set_arg<keyT>);
      } else {
        auto task = new OpArgs();  // It will be deleted by the task q

        if (tracing()) ttg::print(world.rank(), ":", get_name(), " : submitting task for op ");
        task->derived = static_cast<derivedT *>(this);

        world.impl().impl().taskq.add(task);
      }
    }

    // Used by invoke to set all arguments associated with a task
    template <typename Key, size_t... IS>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, void> set_args(std::index_sequence<IS...>, const Key &key,
                                                                const input_values_tuple_type &args) {
      int junk[] = {0, (set_arg<IS>(key, Op::get<IS>(args)), 0)...};
      junk[0]++;
    }

   public:
    /// sets stream size for input \c i
    /// \param size positive integer that specifies the stream size
    template <std::size_t i, bool key_is_void = ttg::meta::is_void_v<keyT>>
    std::enable_if_t<key_is_void, void> set_argstream_size(std::size_t size) {
      // preconditions
      assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "Op::set_argstream_size(size) called with size=0");

      // body
      const auto owner = keymap();
      if (owner != world.rank()) {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : forwarding stream size for terminal ", i);
        }
        worldobjT::send(owner, &opT::template set_argstream_size<i, true>, size);
      } else {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : setting stream size to ", size, " for terminal ", i);
        }

        accessorT acc;
        if (cache.insert(acc, 0)) acc->second = new OpArgs();  // It will be deleted by the task q
        OpArgs *args = acc->second;

        args->lock();

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error stream is already bounded : ", i);
          throw std::runtime_error("Op::set_argstream_size called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error stream is already finalized : ", i);
          throw std::runtime_error("Op::set_argstream_size called for a finalized stream");
        }

        // commit changes
        args->stream_size[i] = size;

        args->unlock();
      }
    }

    template <std::size_t i>
    void set_static_argstream_size(std::size_t size) {
      assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "Op::set_static_argstream_size(key,size) called with size=0");

      if (tracing()) {
        ttg::print(world.rank(), ":", get_name(), ": setting global stream size for terminal ", i);
      }

      // Check if stream is already bounded
      if (static_streamsize[i] > 0) {
        ttg::print_error(world.rank(), ":", get_name(), " : error stream is already bounded : ", i);
        throw std::runtime_error("Op::set_static_argstream_size called for a bounded stream");
      }

      // commit changes
      static_streamsize[i] = size;
    }

    /// sets stream size for input \c i
    /// \param size positive integer that specifies the stream size
    template <std::size_t i, typename Key = keyT, bool key_is_void = ttg::meta::is_void_v<Key>>
    std::enable_if_t<!key_is_void, void> set_argstream_size(const Key &key, std::size_t size) {
      // preconditions
      assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "Op::set_argstream_size(key,size) called with size=0");

      // body
      const auto owner = keymap(key);
      if (owner != world.rank()) {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding stream size for terminal ", i);
        }
        worldobjT::send(owner, &opT::template set_argstream_size<i>, key, size);
      } else {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : ", key, ": setting stream size for terminal ", i);
        }

        accessorT acc;
        if (cache.insert(acc, key)) acc->second = new OpArgs(this->priomap(key));  // It will be deleted by the task q
        OpArgs *args = acc->second;

        args->lock();

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already bounded : ", i);
          throw std::runtime_error("Op::set_argstream_size called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already finalized : ", i);
          throw std::runtime_error("Op::set_argstream_size called for a finalized stream");
        }

        // commit changes
        args->stream_size[i] = size;

        args->unlock();
      }
    }

    /// finalizes stream for input \c i
    template <std::size_t i, typename Key = keyT, bool key_is_void = ttg::meta::is_void_v<Key>>
    std::enable_if_t<!key_is_void, void> finalize_argstream(const Key &key) {
      // preconditions
      assert(std::get<i>(input_reducers) && "Op::finalize_argstream called on nonstreaming input terminal");

      // body
      const auto owner = keymap(key);
      if (owner != world.rank()) {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding stream finalize for terminal ", i);
        }
        worldobjT::send(owner, &opT::template finalize_argstream<i>, key);
      } else {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : ", key, ": finalizing stream for terminal ", i);
        }

        accessorT acc;
        const auto found = cache.find(acc, key);
        assert(found && "Op::finalize_argstream called but no values had been received yet for this key");
        TTGUNUSED(found);
        OpArgs *args = acc->second;

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error finalize called on bounded stream: ", i);
          throw std::runtime_error("Op::finalize called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already finalized : ", i);
          throw std::runtime_error("Op::finalize called for a finalized stream");
        }

        // commit changes
        args->nargs[i] = 0;
        args->counter--;
        // ready to run the task?
        if (args->counter == 0) {
          if (tracing()) {
            ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
          }
          args->derived = static_cast<derivedT *>(this);
          args->key = key;

          world.impl().impl().taskq.add(args);
          // static_cast<derivedT*>(this)->op(key, std::move(args->t), output_terminals); // Runs immediately

          cache.erase(acc);
        }
      }
    }

    /// finalizes stream for input \c i
    template <std::size_t i, bool key_is_void = ttg::meta::is_void_v<keyT>>
    std::enable_if_t<key_is_void, void> finalize_argstream() {
      // preconditions
      assert(std::get<i>(input_reducers) && "Op::finalize_argstream called on nonstreaming input terminal");

      // body
      const int owner = keymap();
      if (owner != world.rank()) {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : forwarding stream finalize for terminal ", i);
        }
        worldobjT::send(owner, &opT::template finalize_argstream<i, true>);
      } else {
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : finalizing stream for terminal ", i);
        }

        accessorT acc;
        const auto found = cache.find(acc, 0);
        assert(found && "Op::finalize_argstream called but no values had been received yet for this key");
        TTGUNUSED(found);
        OpArgs *args = acc->second;

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error finalize called on bounded stream: ", i);
          throw std::runtime_error("Op::finalize called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error stream is already finalized : ", i);
          throw std::runtime_error("Op::finalize called for a finalized stream");
        }

        // commit changes
        args->nargs[i] = 0;
        args->counter--;
        // ready to run the task?
        if (args->counter == 0) {
          if (tracing()) {
            ttg::print(world.rank(), ":", get_name(), " : submitting task for op ");
          }
          args->derived = static_cast<derivedT *>(this);

          world.impl().impl().taskq.add(args);
          // static_cast<derivedT*>(this)->op(key, std::move(args->t), output_terminals); // Runs immediately

          cache.erase(acc);
        }
      }
    }

   private:
    // Copy/assign/move forbidden ... we could make it work using
    // PIMPL for this base class.  However, this instance of the base
    // class is tied to a specific instance of a derived class a
    // pointer to which is captured for invoking derived class
    // functions.  Thus, not only does the derived class have to be
    // involved but we would have to do it in a thread safe way
    // including for possibly already running tasks and remote
    // references.  This is not worth the effort ... wherever you are
    // wanting to move/assign an Op you should be using a pointer.
    Op(const Op &other) = delete;
    Op &operator=(const Op &other) = delete;
    Op(Op &&other) = delete;
    Op &operator=(Op &&other) = delete;

    // Registers the callback for the i'th input terminal
    template <typename terminalT, std::size_t i>
    void register_input_callback(terminalT &input) {
      static_assert(std::is_same<keyT, typename terminalT::key_type>::value,
                    "Op::register_input_callback(terminalT) -- incompatible terminalT");
      using valueT = std::decay_t<typename terminalT::value_type>;

      //////////////////////////////////////////////////////////////////
      // case 1: nonvoid key, nonvoid value
      //////////////////////////////////////////////////////////////////
      if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type> &&
                    !std::is_void_v<valueT>) {
        auto move_callback = [this](const keyT &key, valueT &&value) {
          set_arg<i, keyT, valueT>(key, std::forward<valueT>(value));
        };
        auto send_callback = [this](const keyT &key, const valueT &value) {
          set_arg<i, keyT, const valueT &>(key, value);
        };
        auto setsize_callback = [this](const keyT &key, std::size_t size) { set_argstream_size<i>(key, size); };
        auto finalize_callback = [this](const keyT &key) { finalize_argstream<i>(key); };
        input.set_callback(send_callback, move_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 4: void key, nonvoid value
      //////////////////////////////////////////////////////////////////
      else if constexpr (ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type> &&
                         !std::is_void_v<valueT>) {
        auto move_callback = [this](valueT &&value) { set_arg<i, keyT, valueT>(std::forward<valueT>(value)); };
        auto send_callback = [this](const valueT &value) { set_arg<i, keyT, const valueT &>(value); };
        auto setsize_callback = [this](std::size_t size) { set_argstream_size<i>(size); };
        auto finalize_callback = [this]() { finalize_argstream<i>(); };
        input.set_callback(send_callback, move_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 2: nonvoid key, void value, mixed inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type> &&
                         std::is_void_v<valueT>) {
        auto send_callback = [this](const keyT &key) { set_arg<i, keyT, void>(key); };
        auto setsize_callback = [this](const keyT &key, std::size_t size) { set_argstream_size<i>(key, size); };
        auto finalize_callback = [this](const keyT &key) { finalize_argstream<i>(key); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 5: void key, void value, mixed inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (ttg::meta::is_all_void_v<keyT, valueT> &&
                         !ttg::meta::is_empty_tuple_v<input_values_tuple_type> && std::is_void_v<valueT>) {
        auto send_callback = [this]() { set_arg<i, keyT, void>(); };
        auto setsize_callback = [this](std::size_t size) { set_argstream_size<i>(size); };
        auto finalize_callback = [this]() { finalize_argstream<i>(); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 3: nonvoid key, void value, no inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (!ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type> &&
                         std::is_void_v<valueT>) {
        auto send_callback = [this](const keyT &key) { set_arg<keyT>(key); };
        auto setsize_callback = [this](const keyT &key, std::size_t size) { set_argstream_size<i>(key, size); };
        auto finalize_callback = [this](const keyT &key) { finalize_argstream<i>(key); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 6: void key, void value, no inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (ttg::meta::is_all_void_v<keyT, valueT> &&
                         ttg::meta::is_empty_tuple_v<input_values_tuple_type> && std::is_void_v<valueT>) {
        auto send_callback = [this]() { set_arg<keyT>(); };
        auto setsize_callback = [this](std::size_t size) { set_argstream_size<i>(size); };
        auto finalize_callback = [this]() { finalize_argstream<i>(); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
        if (tracing()) {
          ttg::print(world.rank(), ":", get_name(), " : set callbacks for terminal ", input.get_name(),
                     " assuming void {key,value} and no input");
        }
      } else
        abort();
    }

    template <std::size_t... IS>
    void register_input_callbacks(std::index_sequence<IS...>) {
      int junk[] = {0, (register_input_callback<typename std::tuple_element<IS, input_terminals_type>::type, IS>(
                            std::get<IS>(input_terminals)),
                        0)...};
      junk[0]++;
    }

    template <std::size_t... IS, typename inedgesT>
    void connect_my_inputs_to_incoming_edge_outputs(std::index_sequence<IS...>, inedgesT &inedges) {
      static_assert(sizeof...(IS) == std::tuple_size_v<input_terminals_type>);
      static_assert(std::tuple_size_v<inedgesT> == std::tuple_size_v<input_terminals_type>);
      int junk[] = {0, (std::get<IS>(inedges).set_out(&std::get<IS>(input_terminals)), 0)...};
      junk[0]++;
      if (tracing()) {
        ttg::print(world.rank(), ":", get_name(), " : connected ", sizeof...(IS), " Op inputs to ", sizeof...(IS),
                   " Edges");
      }
    }

    template <std::size_t... IS, typename outedgesT>
    void connect_my_outputs_to_outgoing_edge_inputs(std::index_sequence<IS...>, outedgesT &outedges) {
      static_assert(sizeof...(IS) == std::tuple_size_v<output_terminalsT>);
      static_assert(std::tuple_size_v<outedgesT> == std::tuple_size_v<output_terminalsT>);
      int junk[] = {0, (std::get<IS>(outedges).set_in(&std::get<IS>(output_terminals)), 0)...};
      junk[0]++;
      if (tracing()) {
        ttg::print(world.rank(), ":", get_name(), " : connected ", sizeof...(IS), " Op outputs to ", sizeof...(IS),
                   " Edges");
      }
    }

   public:
    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       ttg::World world, keymapT &&keymap_ = keymapT(), priomapT &&priomap_ = priomapT())
        : ttg::OpBase(name, numins, numouts)
        , static_streamsize()
        , worldobjT(world.impl().impl())
        , world(world)
        // if using default keymap, rebind to the given world
        , keymap(std::is_same<keymapT, ttg::detail::default_keymap<keyT>>::value
                     ? decltype(keymap)(ttg::detail::default_keymap<keyT>(world))
                     : decltype(keymap)(std::forward<keymapT>(keymap_)))
        , priomap(decltype(keymap)(std::forward<priomapT>(priomap_))) {
      // Cannot call these in base constructor since terminals not yet constructed
      if (innames.size() != std::tuple_size<input_terminals_type>::value) {
        ttg::print_error(world.rank(), ":", get_name(), "#input_names", innames.size(), "!= #input_terminals",
                         std::tuple_size<input_terminals_type>::value);
        throw this->get_name() + ":madnessttg::Op: #input names != #input terminals";
      }
      if (outnames.size() != std::tuple_size_v<output_terminalsT>)
        throw this->get_name() + ":madnessttg::Op: #output names != #output terminals";

      register_input_terminals(input_terminals, innames);
      register_output_terminals(output_terminals, outnames);

      register_input_callbacks(std::make_index_sequence<numins>{});
    }

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       keymapT &&keymap = keymapT(ttg::get_default_world()), priomapT &&priomap = priomapT())
        : Op(name, innames, outnames, ttg::get_default_world(), std::forward<keymapT>(keymap),
             std::forward<priomapT>(priomap)) {}

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
       const std::vector<std::string> &innames, const std::vector<std::string> &outnames, ttg::World world,
       keymapT &&keymap_ = keymapT(), priomapT &&priomap_ = priomapT())
        : ttg::OpBase(name, numins, numouts)
        , static_streamsize()
        , worldobjT(ttg::get_default_world().impl().impl())
        , world(ttg::get_default_world())
        // if using default keymap, rebind to the given world
        , keymap(std::is_same<keymapT, ttg::detail::default_keymap<keyT>>::value
                     ? decltype(keymap)(ttg::detail::default_keymap<keyT>(world))
                     : decltype(keymap)(std::forward<keymapT>(keymap_)))
        , priomap(decltype(keymap)(std::forward<priomapT>(priomap_))) {
      // Cannot call in base constructor since terminals not yet constructed
      if (innames.size() != std::tuple_size<input_terminals_type>::value) {
        ttg::print_error(world.rank(), ":", get_name(), "#input_names", innames.size(), "!= #input_terminals",
                         std::tuple_size<input_terminals_type>::value);
        throw this->get_name() + ":madnessttg::Op: #input names != #input terminals";
      }
      if (outnames.size() != std::tuple_size<output_terminalsT>::value)
        throw this->get_name() + ":madnessttg::Op: #output names != #output terminals";

      register_input_terminals(input_terminals, innames);
      register_output_terminals(output_terminals, outnames);

      register_input_callbacks(std::make_index_sequence<numins>{});

      connect_my_inputs_to_incoming_edge_outputs(std::make_index_sequence<numins>{}, inedges);
      connect_my_outputs_to_outgoing_edge_inputs(std::make_index_sequence<numouts>{}, outedges);
    }

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
       const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       keymapT &&keymap = keymapT(ttg::get_default_world()), priomapT &&priomap = priomapT())
        : Op(inedges, outedges, name, innames, outnames, ttg::get_default_world(), std::forward<keymapT>(keymap),
             std::forward<priomapT>(priomap)) {}

    // Destructor checks for unexecuted tasks
    virtual ~Op() {
      if (cache.size() != 0) {
        std::cerr << world.rank() << ":"
                  << "warning: unprocessed tasks in destructor of operation '" << get_name()
                  << "' (class name = " << get_class_name() << ")" << std::endl;
        std::cerr << world.rank() << ":"
                  << "   T => argument assigned     F => argument unassigned" << std::endl;
        int nprint = 0;
        for (auto item : cache) {
          if (nprint++ > 10) {
            std::cerr << "   etc." << std::endl;
            break;
          }
          using ::madness::operators::operator<<;
          std::cerr << world.rank() << ":"
                    << "   unused: " << item.first << " : ( ";
          for (std::size_t i = 0; i < numins; i++) std::cerr << (item.second->nargs[i] == 0 ? "T" : "F") << " ";
          std::cerr << ")" << std::endl;
        }
        abort();
      }
    }

    template <std::size_t i, typename Reducer>
    void set_input_reducer(Reducer &&reducer) {
      if (tracing()) {
        ttg::print(world.rank(), ":", get_name(), " : setting reducer for terminal ", i);
      }
      std::get<i>(input_reducers) = reducer;
    }

    template <typename Keymap>
    void set_keymap(Keymap &&km) {
      keymap = km;
    }

    auto get_priomap(void) const { return priomap; }

    /// Set the priority map, mapping a Key to an integral value.
    /// Higher values indicate higher priority. The default priority is 0, higher
    /// values are treated as high priority tasks in the MADNESS backend.
    template <typename Priomap>
    void set_priomap(Priomap &&pm) {
      priomap = pm;
    }

    /// implementation of OpBase::make_executable()
    void make_executable() {
      this->process_pending();
      OpBase::make_executable();
    }

    /// Waits for the entire TTG associated with this op to be completed (collective)

    /// This is a collective operation and must be invoked by the main
    /// thread on all processes.  In the MADNESS implementation it
    /// fences the entire world associated with the TTG.  If you wish to
    /// fence TTGs independently, then give each its own world.
    void fence() { ttg_fence(world); }

    /// Returns pointer to input terminal i to facilitate connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    typename std::tuple_element<i, input_terminals_type>::type *in() {
      return &std::get<i>(input_terminals);
    }

    /// Returns pointer to output terminal for purpose of connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    typename std::tuple_element<i, output_terminalsT>::type *out() {
      return &std::get<i>(output_terminals);
    }

    /// Manual injection of a task with all input arguments specified as a tuple
    template <typename Key = keyT>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const Key &key, const input_values_tuple_type &args) {
      TTG_OP_ASSERT_EXECUTABLE();
      set_args(std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{}, key, args);
    }

    /// Manual injection of a key-free task with all input arguments specified as a tuple
    template <typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const input_values_tuple_type &args) {
      TTG_OP_ASSERT_EXECUTABLE();
      set_args(std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{}, args);
    }

    /// Manual injection of a task that has no arguments
    template <typename Key = keyT>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const Key &key) {
      TTG_OP_ASSERT_EXECUTABLE();
      set_arg<Key>(key);
    }

    /// Manual injection of a task that has no key or arguments
    template <typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke() {
      TTG_OP_ASSERT_EXECUTABLE();
      set_arg<Key>();
    }

    /// keymap accessor
    /// @return the keymap
    const decltype(keymap) &get_keymap() const { return keymap; }

    /// computes the owner of key @c key
    /// @param[in] key the key
    /// @return the owner of @c key
    template <typename Key>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, int> owner(const Key &key) const {
      return keymap(key);
    }

    /// computes the owner of void key
    /// @return the owner of void key
    template <typename Key>
    std::enable_if_t<ttg::meta::is_void_v<Key>, int> owner() const {
      return keymap();
    }
  };

#include "ttg/wrap.h"

}  // namespace ttg_madness

#include "ttg/madness/watch.h"

#endif  // MADNESS_TTG_H_INCLUDED
