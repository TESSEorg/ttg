#ifndef MADNESS_TTG_H_INCLUDED
#define MADNESS_TTG_H_INCLUDED

/* set up env if this header was included directly */
#if !defined(TTG_IMPL_NAME)
#define TTG_USE_MADNESS 1
#endif  // !defined(TTG_IMPL_NAME)

#include "ttg/impl_selector.h"

#include "ttg/base/keymap.h"
#include "ttg/base/tt.h"
#include "ttg/func.h"
#include "ttg/madness/device.h"
#include "ttg/madness/devicefunc.h"

/* needed for make_tt */
#include "ttg/device/task.h"

#include "ttg/runtimes.h"
#include "ttg/tt.h"
#include "ttg/util/bug.h"
#include "ttg/util/env.h"
#include "ttg/util/hash.h"
#include "ttg/util/macro.h"
#include "ttg/util/meta.h"
#include "ttg/util/meta/callable.h"
#include "ttg/util/scope_exit.h"
#include "ttg/util/void.h"
#include "ttg/world.h"
#include "ttg/coroutine.h"

/* include ttg header to make symbols available in case this header is included directly */
#include "../../ttg.h"

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

namespace ttg_madness {

#if 0
    class Control;
    class Graph;
  /// Graph is a collection of Op objects
  class Graph {
   public:
    Graph() {
      world_ = default_execution_context();
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
    WorldImpl(::madness::World &world) : WorldImplBase(world.size(), world.rank()), m_impl(world) {}

    WorldImpl(const SafeMPI::Intracomm &comm)
        : WorldImplBase(comm.Get_size(), comm.Get_rank()), m_impl(*new ::madness::World(comm)), m_allocated(true) {}

    /* Deleted copy ctor */
    WorldImpl(const WorldImpl &other) = delete;

    /* Deleted move ctor */
    WorldImpl(WorldImpl &&other) = delete;

    virtual ~WorldImpl() override { destroy(); }

    /* Deleted copy assignment */
    WorldImpl &operator=(const WorldImpl &other) = delete;

    /* Deleted move assignment */
    WorldImpl &operator=(WorldImpl &&other) = delete;

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

  inline void make_executable_hook(ttg::World& world) { }

  inline void ttg_initialize(int argc, char **argv, int num_threads) {
    if (num_threads < 1) num_threads = ttg::detail::num_threads();
    ::madness::World &madworld = ::madness::initialize(argc, argv, num_threads, /* quiet = */ true);
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
  inline ttg::World ttg_default_execution_context() { return ttg::get_default_world(); }
  inline void ttg_abort() {
    MPI_Abort(ttg_default_execution_context().impl().impl().mpi.Get_mpi_comm(), 1);
    assert(0); // make sure we abort
  }
  inline void ttg_execute(ttg::World world) {
    // World executes tasks eagerly
  }
  inline void ttg_fence(ttg::World world) { world.impl().fence(); }

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr) {
    world.impl().register_ptr(ptr);
  }

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, std::unique_ptr<T> &&ptr) {
    world.impl().register_ptr(std::move(ptr));
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

  /// CRTP base for MADNESS-based TT classes
  /// \tparam keyT a Key type
  /// \tparam output_terminalsT
  /// \tparam derivedT
  /// \tparam input_valueTs ttg::typelist of *value* types (no references; pointers are OK) encoding the types of input
  /// values
  ///         flowing into this TT; a const type indicates nonmutating (read-only) use, nonconst type
  ///         indicates mutating use (e.g. the corresponding input can be used as scratch, moved-from, etc.)
  template <typename keyT, typename output_terminalsT, typename derivedT, typename input_valueTs, ttg::ExecutionSpace Space>
  class TT : public ttg::TTBase, public ::madness::WorldObject<TT<keyT, output_terminalsT, derivedT, input_valueTs, Space>> {
    static_assert(Space == ttg::ExecutionSpace::Host, "MADNESS backend only supports Host Execution Space");
    static_assert(ttg::meta::is_typelist_v<input_valueTs>,
                  "The fourth template for ttg::TT must be a ttg::typelist containing the input types");
    using input_tuple_type = ttg::meta::typelist_to_tuple_t<input_valueTs>;
    // create a virtual control input if the input list is empty, to be used in invoke()
    using actual_input_tuple_type = std::conditional_t<!ttg::meta::typelist_is_empty_v<input_valueTs>,
                                                       ttg::meta::typelist_to_tuple_t<input_valueTs>, std::tuple<void>>;

   public:
    using ttT = TT;
    using key_type = keyT;
    /// preconditions
    static_assert((ttg::meta::none_has_reference_v<input_valueTs>), "input_valueTs cannot contain reference types");

   private:
    ttg::World world;
    ttg::meta::detail::keymap_t<keyT> keymap;
    ttg::meta::detail::keymap_t<keyT> priomap;
    // For now use same type for unary/streaming input terminals, and stream reducers assigned at runtime
    ttg::meta::detail::input_reducers_t<actual_input_tuple_type>
        input_reducers;  //!< Reducers for the input terminals (empty = expect single value)
    int num_pullins = 0;

    std::array<std::size_t, std::tuple_size_v<actual_input_tuple_type>> static_streamsize;

   public:
    ttg::World get_world() const override final { return world; }

    /// @return true if derivedT::have_cuda_op exists and is defined to true
    static constexpr bool derived_has_cuda_op() {
      return false;
    }

    /// @return true if derivedT::have_hip_op exists and is defined to true
    static constexpr bool derived_has_hip_op() {
      return false;
    }

    /// @return true if derivedT::have_hip_op exists and is defined to true
    static constexpr bool derived_has_level_zero_op() {
      return false;
    }

    /// @return true if the TT supports device execution
    static constexpr bool derived_has_device_op() {
      return false;
    }

   protected:
    using worldobjT = ::madness::WorldObject<ttT>;

    static constexpr int numinedges = std::tuple_size_v<input_tuple_type>;     // number of input edges
    static constexpr int numins = std::tuple_size_v<actual_input_tuple_type>;  // number of input arguments
    static constexpr int numouts = std::tuple_size_v<output_terminalsT>;       // number of outputs

    // This to support tt fusion
    inline static __thread struct {
      uint64_t key_hash = 0;  // hash of current key
      size_t call_depth = 0;  // how deep calls are nested
    } threaddata;

   public:
    using input_terminals_type = ttg::detail::input_terminals_tuple_t<keyT, input_tuple_type>;
    using input_edges_type = ttg::detail::edges_tuple_t<keyT, ttg::meta::decayed_typelist_t<input_tuple_type>>;
    static_assert(ttg::meta::is_none_Void_v<input_valueTs>, "ttg::Void is for internal use only, do not use it");
    static_assert(ttg::meta::is_none_void_v<input_valueTs> || ttg::meta::is_last_void_v<input_valueTs>,
                  "at most one void input can be handled, and it must come last");
    // if have data inputs and (always last) control input, convert last input to Void to make logic easier
    using input_values_full_tuple_type =
        ttg::meta::void_to_Void_tuple_t<ttg::meta::decayed_typelist_t<actual_input_tuple_type>>;
    using input_refs_full_tuple_type =
        ttg::meta::add_glvalue_reference_tuple_t<ttg::meta::void_to_Void_tuple_t<actual_input_tuple_type>>;

    using input_args_type = actual_input_tuple_type;

    using input_values_tuple_type = ttg::meta::drop_void_t<ttg::meta::decayed_typelist_t<input_tuple_type>>;
    using input_refs_tuple_type = ttg::meta::drop_void_t<ttg::meta::add_glvalue_reference_tuple_t<input_tuple_type>>;
    static_assert(!ttg::meta::is_any_void_v<input_values_tuple_type>);

    using output_terminals_type = output_terminalsT;
    using output_edges_type = typename ttg::terminals_to_edges<output_terminalsT>::type;

   protected:
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

   protected:
    const auto &get_output_terminals() const { return output_terminals; }

   private:
    struct TTArgs : ::madness::TaskInterface {
     private:
      using TaskInterface = ::madness::TaskInterface;

     public:
      int counter;  // Tracks the number of arguments finalized
      std::array<std::int64_t, numins>
          nargs;  // Tracks the number of expected values minus the number of received values
                  // 0 = finalized
                  // for a streaming input initialized to std::numeric_limits<std::int64_t>::max()
                  // which indicates that the value needs to be initialized
      std::array<std::size_t, numins> stream_size;  // Expected number of values to receive, to be used for streaming
                                                    // inputs (0 = unbounded stream, >0 = bounded stream)
      input_values_tuple_type input_values;         // The input values (does not include control)
      derivedT *derived;                            // Pointer to derived class instance
      bool pull_terminals_invoked = false;
      std::conditional_t<ttg::meta::is_void_v<keyT>, ttg::Void, keyT> key;  // Task key
#ifdef TTG_HAVE_COROUTINE
      void *suspended_task_address = nullptr;  // if not null the function is suspended
      ttg::TaskCoroutineID coroutine_id = ttg::TaskCoroutineID::Invalid;
#endif // TTG_HAVE_COROUTINE

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

      TTArgs(int prio = 0)
          : TaskInterface(TaskAttributes(prio ? TaskAttributes::HIGHPRIORITY : 0))
          , counter(numins)
          , nargs()
          , stream_size()
          , input_values() {
        std::fill(nargs.begin(), nargs.end(), std::numeric_limits<std::int64_t>::max());
      }

      virtual void run(::madness::World &world) override {
        using ttg::hash;
        ttT::threaddata.key_hash = hash<decltype(key)>{}(key);
        ttT::threaddata.call_depth++;

        void *suspended_task_address =
#ifdef TTG_HAVE_COROUTINE
            this->suspended_task_address;  // non-null = need to resume the task
#else  // TTG_HAVE_COROUTINE
            nullptr;
#endif // TTG_HAVE_COROUTINE
        if (suspended_task_address == nullptr) {  // task is a coroutine that has not started or an ordinary function
          // ttg::print("starting task");
          if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
            TTG_PROCESS_TT_OP_RETURN(
                suspended_task_address,
                coroutine_id,
                derived->op(key, this->make_input_refs(),
                            derived->output_terminals));  // !!! NOTE converting input values to refs
          } else if constexpr (!ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
            TTG_PROCESS_TT_OP_RETURN(suspended_task_address, coroutine_id, derived->op(key, derived->output_terminals));
          } else if constexpr (ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
            TTG_PROCESS_TT_OP_RETURN(
                suspended_task_address,
                coroutine_id,
                derived->op(this->make_input_refs(),
                            derived->output_terminals));  // !!! NOTE converting input values to refs
          } else if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
            TTG_PROCESS_TT_OP_RETURN(suspended_task_address, coroutine_id, derived->op(derived->output_terminals));
          } else  // unreachable
            ttg::abort();
        } else {  // resume suspended coroutine
#ifdef TTG_HAVE_COROUTINE
          auto ret = static_cast<ttg::resumable_task>(ttg::coroutine_handle<ttg::resumable_task_state>::from_address(suspended_task_address));
          assert(ret.ready());
          ret.resume();
          if (ret.completed()) {
            ret.destroy();
            suspended_task_address = nullptr;
          } else {  // not yet completed
            // leave suspended_task_address as is
          }
          this->suspended_task_address = suspended_task_address;
#else  // TTG_HAVE_COROUTINE
          ttg::abort();  // should not happen
#endif // TTG_HAVE_COROUTINE
        }

        ttT::threaddata.call_depth--;

        // if (suspended_task_address == nullptr) {
        //   ttg::print("finishing task",ttT::threaddata.call_depth);
        // }

#ifdef TTG_HAVE_COROUTINE
        if (suspended_task_address) {
          // TODO implement handling of suspended coroutines properly

          // only resumable_task is recognized at the moment
          assert(coroutine_id == ttg::TaskCoroutineID::ResumableTask);

          // right now can events are not properly implemented, we are only testing the workflow with dummy events
          // so mark the events finished manually
          // proper thing to do is to process event queue and resubmit this task again
          auto events =
              static_cast<ttg::resumable_task>(ttg::coroutine_handle<ttg::resumable_task_state>::from_address(suspended_task_address)).events();
          for (auto &event_ptr : events) {
            event_ptr->finish();
          }
          assert(ttg::coroutine_handle<ttg::resumable_task_state>::from_address(suspended_task_address).promise().ready());

          // resume the coroutine
          auto ret = static_cast<ttg::resumable_task>(ttg::coroutine_handle<ttg::resumable_task_state>::from_address(suspended_task_address));
          assert(ret.ready());
          ret.resume();
          if (ret.completed()) {
            ret.destroy();
            suspended_task_address = nullptr;
          } else {  // not yet completed
            ttg::abort();
          }
        }
#endif  // TTG_HAVE_COROUTINE
      }

      virtual ~TTArgs() {}  // Will be deleted via TaskInterface*

     private:
      ::madness::Spinlock lock_;  // synchronizes access to data
     public:
      void lock() { lock_.lock(); }
      void unlock() { lock_.unlock(); }
    };

    using hashable_keyT = std::conditional_t<ttg::meta::is_void_v<keyT>, int, keyT>;
    using cacheT = ::madness::ConcurrentHashMap<hashable_keyT, TTArgs *, ttg::hash<hashable_keyT>>;
    using accessorT = typename cacheT::accessor;
    cacheT cache;

   protected:
    template <typename terminalT, std::size_t i, typename Key>
    void invoke_pull_terminal(terminalT &in, const Key &key, TTArgs *args) {
      if (in.is_pull_terminal) {
        int owner;
        if constexpr (!ttg::meta::is_void_v<Key>) {
          owner = in.container.owner(key);
        } else {
          owner = in.container.owner();
        }

        if (owner != world.rank()) {
          get_terminal_data<i, Key>(owner, key);
        } else {
          if constexpr (!ttg::meta::is_void_v<Key>) {
            auto value = (in.container).get(key);
            if (args->nargs[i] == 0) {
              ::ttg::print_error(world.rank(), ":", get_name(), " : ", key,
                                 ": error argument is already finalized : ", i);
              throw std::runtime_error("Op::set_arg called for a finalized stream");
            }

            if (typeid(value) != typeid(std::nullptr_t) && i < std::tuple_size_v<input_values_tuple_type>) {
              this->get<i, std::decay_t<decltype(value)> &>(args->input_values) = std::forward<decltype(value)>(value);
              args->nargs[i] = 0;
              args->counter--;
            }
          } else {
            auto value = (in.container).get();
            if (args->nargs[i] == 0) {
              ::ttg::print_error(world.rank(), ":", get_name(), " : ", key,
                                 ": error argument is already finalized : ", i);
              throw std::runtime_error("Op::set_arg called for a finalized stream");
            }

            if (typeid(value) != typeid(std::nullptr_t) && i < std::tuple_size_v<input_values_tuple_type>) {
              this->get<i, std::decay_t<decltype(value)> &>(args->input_values) = std::forward<decltype(value)>(value);
              args->nargs[i] = 0;
              args->counter--;
            }
          }
        }
      }
    }

    template <std::size_t i, typename Key>
    void get_terminal_data(const int owner, const Key &key) {
      if (owner != world.rank()) {
        worldobjT::send(owner, &ttT::template get_terminal_data<i, Key>, owner, key);
      } else {
        auto &in = std::get<i>(input_terminals);
        if constexpr (!ttg::meta::is_void_v<Key>) {
          auto value = (in.container).get(key);
          worldobjT::send(keymap(key), &ttT::template set_arg<i, Key, const std::remove_reference_t<decltype(value)> &>,
                          key, value);
        } else {
          auto value = (in.container).get();
          worldobjT::send(keymap(), &ttT::template set_arg<i, void, const std::remove_reference_t<decltype(value)> &>,
                          value);
        }
      }
    }

    template <std::size_t... IS, typename Key = keyT>
    void invoke_pull_terminals(std::index_sequence<IS...>, const Key &key, TTArgs *args) {
      int junk[] = {0, (invoke_pull_terminal<typename std::tuple_element<IS, input_terminals_type>::type, IS>(
                            std::get<IS>(input_terminals), key, args),
                        0)...};
      junk[0]++;
    }

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
    void set_arg(const Key &key, Value &&value) {
      using valueT = std::tuple_element_t<i, input_values_full_tuple_type>;  // Should be T or const T
      static_assert(std::is_same_v<std::decay_t<Value>, std::decay_t<valueT>>,
                    "TT::set_arg(key,value) given value of type incompatible with TT");

      int owner;
      if constexpr (!ttg::meta::is_void_v<Key>) {
        owner = keymap(key);
      } else {
        owner = keymap();
      }

      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": forwarding setting argument : ", i);
        // should be able on the other end to consume value (since it is just a temporary byproduct of serialization)
        // BUT compiler vomits when const std::remove_reference_t<Value>& -> std::decay_t<Value>
        // this exposes bad design in MemFuncWrapper (probably similar bugs elsewhere?) whose generic operator()
        // should use memfun's argument types (since that's what will be called) rather than misautodeduce in a
        // particular context P.S. another issue is in send_am which can execute both remotely (where one can always
        // move arguments) and locally
        //      here we know that this will be a remove execution, so we prepare to take rvalues;
        //      send_am will need to separate local and remote paths to deal with this
        if constexpr (!ttg::meta::is_void_v<Key>) {
          if constexpr (!ttg::meta::is_void_v<Value>) {
            worldobjT::send(owner, &ttT::template set_arg<i, Key, const std::remove_reference_t<Value> &>, key, value);
          } else {
            worldobjT::send(owner, &ttT::template set_arg<i, Key, void>, key);
          }
        } else {
          if constexpr (!ttg::meta::is_void_v<Value>) {
            worldobjT::send(owner, &ttT::template set_arg<i, void, const std::remove_reference_t<Value> &>, value);
          } else {
            worldobjT::send(owner, &ttT::template set_arg<i, void, void>);
          }
        }
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": received value for argument : ", i);

        bool pullT_invoked = false;
        accessorT acc;

        int prio;
        if constexpr (!ttg::meta::is_void_v<Key>) {
          prio = this->priomap(key);
          if (cache.insert(acc, key)) {
            acc->second = new TTArgs(prio);  // It will be deleted by the task q
            if (!is_lazy_pull()) {
              // Invoke pull terminals for only the terminals with non-void values.
              invoke_pull_terminals(std::make_index_sequence<std::tuple_size_v<input_values_tuple_type>>{}, key,
                                    acc->second);
              pullT_invoked = true;
            }
          }
        } else {
          prio = this->priomap();
          if (cache.insert(acc, 0)) acc->second = new TTArgs(prio);  // It will be deleted by the task q
        }

        TTArgs *args = acc->second;
        if (!is_lazy_pull() && pullT_invoked) args->pull_terminals_invoked = true;

        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error argument is already finalized : ", i);
          throw std::runtime_error("TT::set_arg called for a finalized stream");
        }

        const auto &reducer = std::get<i>(input_reducers);
        if (reducer) {  // is this a streaming input? reduce the received value
          // N.B. Right now reductions are done eagerly, without spawning tasks
          //      this means we must lock
          args->lock();

          bool initialize_not_reduce = false;
          if (args->nargs[i] == std::numeric_limits<std::int64_t>::max()) {
            // upon first datum initialize, if needed
            if constexpr (!ttg::meta::is_void_v<valueT>) {
              initialize_not_reduce = true;
            }

            // initialize nargs
            // if we have a stream size for the op, use it first
            if (args->stream_size[i] != 0) {
              assert(args->stream_size[i] <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
              args->nargs[i] = args->stream_size[i];
            } else if (static_streamsize[i] != 0) {
              assert(static_streamsize[i] <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
              args->stream_size[i] = static_streamsize[i];
              args->nargs[i] = static_streamsize[i];
            } else {
              args->nargs[i] = 0;
            }
          }

          if constexpr (!ttg::meta::is_void_v<valueT>) {  // for data values
            if (initialize_not_reduce)
              this->get<i, std::decay_t<valueT> &>(args->input_values) = std::forward<Value>(value);
            else
              reducer(this->get<i, std::decay_t<valueT> &>(args->input_values), value);
          } else {
            reducer();  // even if this was a control input, must execute the reducer for possible side effects
          }

          // update the counter
          args->nargs[i]--;

          // is this the last message?
          if (args->nargs[i] == 0) args->counter--;

          args->unlock();
        } else {                                          // this is a nonstreaming input => set the value
          if constexpr (!ttg::meta::is_void_v<valueT>) {  // for data values
            this->get<i, std::decay_t<valueT> &>(args->input_values) = std::forward<Value>(value);
          }
          args->nargs[i] = 0;
          args->counter--;
        }

        // If lazy pulling in enabled, check it here.
        if (numins - args->counter == num_pullins) {
          if (is_lazy_pull() && !args->pull_terminals_invoked) {
            // Invoke pull terminals for only the terminals with non-void values.
            invoke_pull_terminals(std::make_index_sequence<std::tuple_size_v<input_values_tuple_type>>{}, key, args);
          }
        }

        // ready to run the task?
        if (args->counter == 0) {
          ttg::trace(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
          args->derived = static_cast<derivedT *>(this);
          args->key = key;

          using ttg::hash;
          auto curhash = hash<keyT>{}(key);

          if (curhash == threaddata.key_hash && threaddata.call_depth < 6) {  // Needs to be externally configurable

            // ttg::print("directly invoking:", get_name(), key, curhash, threaddata.key_hash, threaddata.call_depth);
            ttT::threaddata.call_depth++;
            if constexpr (!ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(key, args->make_input_refs(), output_terminals);  // Runs immediately
            } else if constexpr (!ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(key, output_terminals);  // Runs immediately
            } else if constexpr (ttg::meta::is_void_v<keyT> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(args->make_input_refs(), output_terminals);  // Runs immediately
            } else if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
              static_cast<derivedT *>(this)->op(output_terminals);  // Runs immediately
            } else
              ttg::abort();
            ttT::threaddata.call_depth--;

          } else {
            // ttg::print("enqueuing task", get_name(), key, curhash, threaddata.key_hash, threaddata.call_depth);
            world.impl().impl().taskq.add(args);
          }

          cache.erase(acc);
        }
      }
    }

    // case 2 and 3
    template <std::size_t i, typename Key, typename Value>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && std::is_void_v<Value>, void> set_arg(const Key &key) {
      set_arg<i>(key, ttg::Void{});
    }

    // case 4
    template <std::size_t i, typename Key = keyT, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !std::is_void_v<std::decay_t<Value>>, void> set_arg(Value &&value) {
      return set_arg<i>(ttg::Void{}, std::forward<Value>(value));
    }

    // case 5 and 6
    template <std::size_t i, typename Key = keyT, typename Value>
    std::enable_if_t<ttg::meta::is_void_v<Key> && std::is_void_v<Value>, void> set_arg() {
      set_arg<i, ttg::Void, ttg::Void>(ttg::Void{}, ttg::Void{});
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of elements in args
    // Js: index sequence of input terminals to set
    template <typename Key, typename... Ts, size_t... Is, size_t... Js>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, void> set_args(std::index_sequence<Is...>, std::index_sequence<Js...>,
                                                                const Key &key, const std::tuple<Ts...> &args) {
      static_assert(sizeof...(Js) == sizeof...(Is));
      constexpr std::size_t js[] = {Js...};
      int junk[] = {0, (set_arg<js[Is]>(key, TT::get<Is>(args)), 0)...};
      junk[0]++;
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of input terminals to set
    template <typename Key, typename... Ts, size_t... Is>
    std::enable_if_t<!ttg::meta::is_void_v<Key>, void> set_args(std::index_sequence<Is...> is, const Key &key,
                                                                const std::tuple<Ts...> &args) {
      set_args(std::index_sequence_for<Ts...>{}, is, key, args);
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of elements in args
    // Js: index sequence of input terminals to set
    template <typename Key = keyT, typename... Ts, size_t... Is, size_t... Js>
    std::enable_if_t<ttg::meta::is_void_v<Key>, void> set_args(std::index_sequence<Is...>, std::index_sequence<Js...>,
                                                               const std::tuple<Ts...> &args) {
      static_assert(sizeof...(Js) == sizeof...(Is));
      constexpr std::size_t js[] = {Js...};
      int junk[] = {0, (set_arg<js[Is], void>(TT::get<Is>(args)), 0)...};
      junk[0]++;
    }

    // Used by invoke to set all arguments associated with a task
    // Is: index sequence of input terminals to set
    template <typename Key = keyT, typename... Ts, size_t... Is>
    std::enable_if_t<ttg::meta::is_void_v<Key>, void> set_args(std::index_sequence<Is...> is,
                                                               const std::tuple<Ts...> &args) {
      set_args(std::index_sequence_for<Ts...>{}, is, args);
    }

   public:
    /// sets stream size for input \c i
    /// \param size positive integer that specifies the stream size
    template <std::size_t i, bool key_is_void = ttg::meta::is_void_v<keyT>>
    std::enable_if_t<key_is_void, void> set_argstream_size(std::size_t size) {
      // preconditions
      assert(std::get<i>(input_reducers) && "TT::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "TT::set_argstream_size(size) called with size=0");

      // body
      const auto owner = keymap();
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), " : forwarding stream size for terminal ", i);
        worldobjT::send(owner, &ttT::template set_argstream_size<i, true>, size);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : setting stream size to ", size, " for terminal ", i);

        accessorT acc;
        if (cache.insert(acc, 0)) acc->second = new TTArgs();  // It will be deleted by the task q
        TTArgs *args = acc->second;

        args->lock();

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error stream is already bounded : ", i);
          throw std::runtime_error("TT::set_argstream_size called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error stream is already finalized : ", i);
          throw std::runtime_error("TT::set_argstream_size called for a finalized stream");
        }

        // commit changes
        args->stream_size[i] = size;
        // if messages already received for this key update the expected-received counter
        const auto messages_received_already = args->nargs[i] != std::numeric_limits<std::int64_t>::max();
        if (messages_received_already) {
          // cannot have received more messages than expected
          if (-(args->nargs[i]) > size) {
            ttg::print_error(world.rank(), ":", get_name(),
                             " : error stream received more messages than specified via set_argstream_size : ", i);
            throw std::runtime_error("TT::set_argstream_size(n): n less than the number of messages already received");
          }
          args->nargs[i] += size;
        }
        // if done, update the counter
        if (args->nargs[i] == 0) args->counter--;
        args->unlock();

        // ready to run the task?
        if (args->counter == 0) {
          ttg::trace(world.rank(), ":", get_name(), " : submitting task for op ");
          args->derived = static_cast<derivedT *>(this);

          world.impl().impl().taskq.add(args);

          cache.erase(acc);
        }
      }
    }

    template <std::size_t i>
    void set_static_argstream_size(std::size_t size) {
      assert(std::get<i>(input_reducers) && "TT::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "TT::set_static_argstream_size(key,size) called with size=0");

      ttg::trace(world.rank(), ":", get_name(), ": setting global stream size for terminal ", i);

      // Check if stream is already bounded
      if (static_streamsize[i] > 0) {
        ttg::print_error(world.rank(), ":", get_name(), " : error stream is already bounded : ", i);
        throw std::runtime_error("TT::set_static_argstream_size called for a bounded stream");
      }

      // commit changes
      static_streamsize[i] = size;
    }

    /// sets stream size for input \c i and key \c key
    /// \tparam <i>: index of the input terminal to set
    /// \param key the task identifier
    /// \param size positive integer that specifies the stream size
    template <std::size_t i, typename Key = keyT, bool key_is_void = ttg::meta::is_void_v<Key>>
    std::enable_if_t<!key_is_void, void> set_argstream_size(const Key &key, std::size_t size) {
      // preconditions
      assert(std::get<i>(input_reducers) && "TT::set_argstream_size called on nonstreaming input terminal");
      assert(size > 0 && "TT::set_argstream_size(key,size) called with size=0");

      // body
      const auto owner = keymap(key);
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": forwarding stream size for terminal ", i);
        worldobjT::send(owner, &ttT::template set_argstream_size<i>, key, size);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": setting stream size for terminal ", i);

        accessorT acc;
        if (cache.insert(acc, key)) acc->second = new TTArgs(this->priomap(key));  // It will be deleted by the task q
        TTArgs *args = acc->second;

        args->lock();

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already bounded : ", i);
          throw std::runtime_error("TT::set_argstream_size called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already finalized : ", i);
          throw std::runtime_error("TT::set_argstream_size called for a finalized stream");
        }

        // commit changes
        args->stream_size[i] = size;
        // if messages already received for this key update the expected-received counter
        const auto messages_received_already = args->nargs[i] != std::numeric_limits<std::int64_t>::max();
        if (messages_received_already) args->nargs[i] += size;
        // if done, update the counter
        if (args->nargs[i] == 0) args->counter--;

        args->unlock();

        // ready to run the task?
        if (args->counter == 0) {
          ttg::trace(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
          args->derived = static_cast<derivedT *>(this);
          args->key = key;

          world.impl().impl().taskq.add(args);

          cache.erase(acc);
        }
      }
    }

    /// finalizes stream for input \c i
    template <std::size_t i, typename Key = keyT, bool key_is_void = ttg::meta::is_void_v<Key>>
    std::enable_if_t<!key_is_void, void> finalize_argstream(const Key &key) {
      // preconditions
      assert(std::get<i>(input_reducers) && "TT::finalize_argstream called on nonstreaming input terminal");

      // body
      const auto owner = keymap(key);
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": forwarding stream finalize for terminal ", i);
        worldobjT::send(owner, &ttT::template finalize_argstream<i>, key);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : ", key, ": finalizing stream for terminal ", i);

        accessorT acc;
        const auto found = cache.find(acc, key);
        assert(found && "TT::finalize_argstream called but no values had been received yet for this key");
        TTGUNUSED(found);
        TTArgs *args = acc->second;

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error finalize called on bounded stream: ", i);
          throw std::runtime_error("TT::finalize called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already finalized : ", i);
          throw std::runtime_error("TT::finalize called for a finalized stream");
        }

        // commit changes
        args->nargs[i] = 0;
        args->counter--;
        // ready to run the task?
        if (args->counter == 0) {
          ttg::trace(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
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
      assert(std::get<i>(input_reducers) && "TT::finalize_argstream called on nonstreaming input terminal");

      // body
      const int owner = keymap();
      if (owner != world.rank()) {
        ttg::trace(world.rank(), ":", get_name(), " : forwarding stream finalize for terminal ", i);
        worldobjT::send(owner, &ttT::template finalize_argstream<i, true>);
      } else {
        ttg::trace(world.rank(), ":", get_name(), " : finalizing stream for terminal ", i);

        accessorT acc;
        const auto found = cache.find(acc, 0);
        assert(found && "TT::finalize_argstream called but no values had been received yet for this key");
        TTGUNUSED(found);
        TTArgs *args = acc->second;

        // check if stream is already bounded
        if (args->stream_size[i] > 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error finalize called on bounded stream: ", i);
          throw std::runtime_error("TT::finalize called for a bounded stream");
        }

        // check if stream is already finalized
        if (args->nargs[i] == 0) {
          ttg::print_error(world.rank(), ":", get_name(), " : error stream is already finalized : ", i);
          throw std::runtime_error("TT::finalize called for a finalized stream");
        }

        // commit changes
        args->nargs[i] = 0;
        args->counter--;
        // ready to run the task?
        if (args->counter == 0) {
          ttg::trace(world.rank(), ":", get_name(), " : submitting task for op ");
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
    // wanting to move/assign an TT you should be using a pointer.
    TT(const TT &other) = delete;
    TT &operator=(const TT &other) = delete;
    TT(TT &&other) = delete;
    TT &operator=(TT &&other) = delete;

    // Registers the callback for the i'th input terminal
    template <typename terminalT, std::size_t i>
    void register_input_callback(terminalT &input) {
      static_assert(std::is_same_v<keyT, typename terminalT::key_type>,
                    "TT::register_input_callback(terminalT) -- incompatible terminalT");
      using valueT = std::decay_t<typename terminalT::value_type>;

      if (input.is_pull_terminal) {
        num_pullins++;
      }

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
      // case 3: nonvoid key, void value, no inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (!ttg::meta::is_void_v<keyT> && std::is_void_v<valueT>) {
        auto send_callback = [this](const keyT &key) { set_arg<i, keyT, void>(key); };
        auto setsize_callback = [this](const keyT &key, std::size_t size) { set_argstream_size<i>(key, size); };
        auto finalize_callback = [this](const keyT &key) { finalize_argstream<i>(key); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
      }
      //////////////////////////////////////////////////////////////////
      // case 5: void key, void value, mixed inputs
      // case 6: void key, void value, no inputs
      //////////////////////////////////////////////////////////////////
      else if constexpr (ttg::meta::is_all_void_v<keyT, valueT> && std::is_void_v<valueT>) {
        auto send_callback = [this]() { set_arg<i, keyT, void>(); };
        auto setsize_callback = [this](std::size_t size) { set_argstream_size<i>(size); };
        auto finalize_callback = [this]() { finalize_argstream<i>(); };
        input.set_callback(send_callback, send_callback, {}, setsize_callback, finalize_callback);
      } else
        ttg::abort();
    }

    template <std::size_t... IS>
    void register_input_callbacks(std::index_sequence<IS...>) {
      int junk[] = {
          0,
          (register_input_callback<std::tuple_element_t<IS, input_terminals_type>, IS>(std::get<IS>(input_terminals)),
           0)...};
      junk[0]++;
    }

    template <std::size_t... IS, typename inedgesT>
    void connect_my_inputs_to_incoming_edge_outputs(std::index_sequence<IS...>, inedgesT &inedges) {
      static_assert(sizeof...(IS) == std::tuple_size_v<input_terminals_type>);
      static_assert(std::tuple_size_v<inedgesT> == std::tuple_size_v<input_terminals_type>);
      int junk[] = {0, (std::get<IS>(inedges).set_out(&std::get<IS>(input_terminals)), 0)...};
      junk[0]++;
      ttg::trace(world.rank(), ":", get_name(), " : connected ", sizeof...(IS), " TT inputs to ", sizeof...(IS),
                 " Edges");
    }

    template <std::size_t... IS, typename outedgesT>
    void connect_my_outputs_to_outgoing_edge_inputs(std::index_sequence<IS...>, outedgesT &outedges) {
      static_assert(sizeof...(IS) == numouts);
      static_assert(std::tuple_size_v<outedgesT> == numouts);
      int junk[] = {0, (std::get<IS>(outedges).set_in(&std::get<IS>(output_terminals)), 0)...};
      junk[0]++;
      ttg::trace(world.rank(), ":", get_name(), " : connected ", sizeof...(IS), " TT outputs to ", sizeof...(IS),
                 " Edges");
    }

   public:
    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       ttg::World world, keymapT &&keymap_ = keymapT(), priomapT &&priomap_ = priomapT())
        : ttg::TTBase(name, numinedges, numouts)
        , static_streamsize()
        , worldobjT(world.impl().impl())
        , world(world)
        // if using default keymap, rebind to the given world
        , keymap(std::is_same_v<keymapT, ttg::detail::default_keymap<keyT>>
                     ? decltype(keymap)(ttg::detail::default_keymap<keyT>(world))
                     : decltype(keymap)(std::forward<keymapT>(keymap_)))
        , priomap(decltype(keymap)(std::forward<priomapT>(priomap_))) {
      // Cannot call these in base constructor since terminals not yet constructed
      if (innames.size() != numinedges) {
        ttg::print_error(world.rank(), ":", get_name(), "#input_names", innames.size(), "!= #input_terminals",
                         numinedges);
        throw this->get_name() + ":madness::ttg::TT: #input names != #input terminals";
      }
      if (outnames.size() != numouts) throw this->get_name() + ":madness::ttg::TT: #output names != #output terminals";

      register_input_terminals(input_terminals, innames);
      register_output_terminals(output_terminals, outnames);

      register_input_callbacks(std::make_index_sequence<numinedges>{});
    }

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       keymapT &&keymap = keymapT(ttg::default_execution_context()), priomapT &&priomap = priomapT())
        : TT(name, innames, outnames, ttg::default_execution_context(), std::forward<keymapT>(keymap),
             std::forward<priomapT>(priomap)) {}

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
       const std::vector<std::string> &innames, const std::vector<std::string> &outnames, ttg::World world,
       keymapT &&keymap_ = keymapT(), priomapT &&priomap_ = priomapT())
        : ttg::TTBase(name, numinedges, numouts)
        , static_streamsize()
        , worldobjT(ttg::default_execution_context().impl().impl())
        , world(ttg::default_execution_context())
        // if using default keymap, rebind to the given world
        , keymap(std::is_same_v<keymapT, ttg::detail::default_keymap<keyT>>
                     ? decltype(keymap)(ttg::detail::default_keymap<keyT>(world))
                     : decltype(keymap)(std::forward<keymapT>(keymap_)))
        , priomap(decltype(keymap)(std::forward<priomapT>(priomap_))) {
      // Cannot call in base constructor since terminals not yet constructed
      if (innames.size() != numinedges) {
        ttg::print_error(world.rank(), ":", get_name(), "#input_names", innames.size(), "!= #input_terminals",
                         numinedges);
        throw this->get_name() + ":madness::ttg::TT: #input names != #input terminals";
      }
      if (outnames.size() != numouts) throw this->get_name() + ":madness::ttg::T: #output names != #output terminals";

      register_input_terminals(input_terminals, innames);
      register_output_terminals(output_terminals, outnames);

      connect_my_inputs_to_incoming_edge_outputs(std::make_index_sequence<numinedges>{}, inedges);
      connect_my_outputs_to_outgoing_edge_inputs(std::make_index_sequence<numouts>{}, outedges);
      // DO NOT MOVE THIS - information about the number of pull terminals is only available after connecting the edges.
      register_input_callbacks(std::make_index_sequence<numinedges>{});
    }

    template <typename keymapT = ttg::detail::default_keymap<keyT>,
              typename priomapT = ttg::detail::default_priomap<keyT>>
    TT(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
       const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
       keymapT &&keymap = keymapT(ttg::default_execution_context()), priomapT &&priomap = priomapT())
        : TT(inedges, outedges, name, innames, outnames, ttg::default_execution_context(),
             std::forward<keymapT>(keymap), std::forward<priomapT>(priomap)) {}

    // Destructor checks for unexecuted tasks
    virtual ~TT() {
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
        ttg::abort();
      }
    }

    /// define the reducer function to be called when additional inputs are
    /// received on a streaming terminal
    ///   @tparam <i> the index of the input terminal that is used as a streaming terminal
    ///   @param[in] reducer: a function of prototype `void(input_type<i> &a, const input_type<i> &b)`
    ///                       that function should aggregate b into a
    template <std::size_t i, typename Reducer>
    void set_input_reducer(Reducer &&reducer) {
      ttg::trace(world.rank(), ":", get_name(), " : setting reducer for terminal ", i);
      std::get<i>(input_reducers) = reducer;
    }

    /// define the reducer function to be called when additional inputs are
    /// received on a streaming terminal
    ///   @tparam <i> the index of the input terminal that is used as a streaming terminal
    ///   @param[in] reducer: a function of prototype `void(input_type<i> &a, const input_type<i> &b)`
    ///                       that function should aggregate b into a
    ///   @param[in] size: the default number of inputs that are received in this streaming terminal,
    ///                    for each task
    template <std::size_t i, typename Reducer>
    void set_input_reducer(Reducer &&reducer, std::size_t size) {
      set_input_reducer<i>(std::forward<Reducer>(reducer));
      set_static_argstream_size<i>(size);
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
      priomap = std::forward<Priomap>(pm);
    }

    /// add a constraint
    /// the constraint must provide a valid override of `check_key(key)`
    template<typename Constraint>
    void add_constraint(Constraint&& c) {
      /* currently a noop */
    }

    template<typename Constraint, typename Mapper>
    void add_constraint(std::shared_ptr<Constraint> c, Mapper&& map) {
      /* currently a noop */
    }

    template<typename Constraint, typename Mapper>
    void add_constraint(Constraint c, Mapper&& map) {
      /* currently a noop */
    }

    /// implementation of TTBase::make_executable()
    void make_executable() override {
      TTBase::make_executable();
      this->process_pending();
    }

    /// Waits for the entire TTG associated with this TT to be completed (collective)

    /// This is a collective operation and must be invoked by the main
    /// thread on all processes.  In the MADNESS implementation it
    /// fences the entire world associated with the TTG.  If you wish to
    /// fence TTGs independently, then give each its own world.
    void fence() override { ttg_fence(world); }

    /// Returns pointer to input terminal i to facilitate connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    std::tuple_element_t<i, input_terminals_type> *in() {
      return &std::get<i>(input_terminals);
    }

    /// Returns pointer to output terminal for purpose of connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    std::tuple_element_t<i, output_terminalsT> *out() {
      return &std::get<i>(output_terminals);
    }

    /// Manual injection of a task with all input arguments specified as a tuple
    template <typename Key = keyT>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const Key &key, const input_values_tuple_type &args) {
      TTG_OP_ASSERT_EXECUTABLE();
      if constexpr(!std::is_same_v<Key, key_type>) {
        key_type k = key; /* cast that type into the key type we know */
        invoke(k, args);
      } else {
        /* trigger non-void inputs */
        set_args(ttg::meta::nonvoid_index_seq<actual_input_tuple_type>{}, key, args);
        /* trigger void inputs */
        using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
        set_args(void_index_seq{}, key, ttg::detail::make_void_tuple<void_index_seq::size()>());
      }
    }

    /// Manual injection of a key-free task with all input arguments specified as a tuple
    template <typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key> && !ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const input_values_tuple_type &args) {
      TTG_OP_ASSERT_EXECUTABLE();
      /* trigger non-void inputs */
      set_args(ttg::meta::nonvoid_index_seq<actual_input_tuple_type>{}, args);
      /* trigger void inputs */
      using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
      set_args(void_index_seq{}, ttg::detail::make_void_tuple<void_index_seq::size()>());
    }

    /// Manual injection of a task that has no arguments
    template <typename Key = keyT>
    std::enable_if_t<!ttg::meta::is_void_v<Key> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke(
        const Key &key) {
      TTG_OP_ASSERT_EXECUTABLE();
      if constexpr(!std::is_same_v<Key, key_type>) {
        key_type k = key; /* cast that type into the key type we know */
        invoke(k);
      } else {
        /* trigger void inputs */
        using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
        set_args(void_index_seq{}, key, ttg::detail::make_void_tuple<void_index_seq::size()>());
      }
    }

    /// Manual injection of a task that has no key or arguments
    template <typename Key = keyT>
    std::enable_if_t<ttg::meta::is_void_v<Key> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>, void> invoke() {
      TTG_OP_ASSERT_EXECUTABLE();
      /* trigger void inputs */
      using void_index_seq = ttg::meta::void_index_seq<actual_input_tuple_type>;
      set_args(void_index_seq{}, ttg::detail::make_void_tuple<void_index_seq::size()>());
    }

    void invoke() override {
      if constexpr (ttg::meta::is_void_v<keyT> && ttg::meta::is_empty_tuple_v<input_values_tuple_type>)
        invoke<keyT>();
      else
        TTBase::invoke();
    }

    void set_defer_writer(bool _) {}

    bool get_defer_writer(bool _) { return false; }

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

#include "ttg/make_tt.h"

}  // namespace ttg_madness

#include "ttg/madness/watch.h"
#include "ttg/madness/buffer.h"
#include "ttg/madness/ttvalue.h"

#endif  // MADNESS_TTG_H_INCLUDED
