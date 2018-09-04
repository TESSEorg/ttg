#ifndef MADNESS_TTG_H_INCLUDED
#define MADNESS_TTG_H_INCLUDED

#include "../ttg.h"
#include "../ttg/util/bug.h"
#include "../util/meta.h"

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

namespace madness {
  namespace ttg {

    namespace detail {
      World *&default_world_accessor() {
        static World *world_ptr = nullptr;
        return world_ptr;
      }
    }  // namespace detail

    inline World &get_default_world() {
      if (detail::default_world_accessor() != nullptr) {
        return *detail::default_world_accessor();
      } else {
        throw "madness::ttg::set_default_world() must be called before use";
      }
    }
    inline void set_default_world(World &world) { detail::default_world_accessor() = &world; }
    inline void set_default_world(World *world) { detail::default_world_accessor() = world; }

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

  namespace detail {
  static inline std::map<World*, std::set<std::shared_ptr<void>>>& ptr_registry_accessor() {
    static std::map<World*, std::set<std::shared_ptr<void>>> registry;
    return registry;
  };
  static inline std::map<World*, ::ttg::Edge<>>& clt_edge_registry_accessor() {
    static std::map<World*, ::ttg::Edge<>> registry;
    return registry;
  };
  static inline std::map<World*, std::set<std::shared_ptr<std::promise<void>>>>& status_registry_accessor() {
    static std::map<World*, std::set<std::shared_ptr<std::promise<void>>>> registry;
    return registry;
  };
  }

  template <typename... RestOfArgs>
    inline void ttg_initialize(int argc, char **argv, RestOfArgs &&...) {
      World &world = madness::initialize(argc, argv);
      set_default_world(world);
    }
    inline void ttg_finalize() { madness::finalize(); }
    inline World &ttg_default_execution_context() { return get_default_world(); }
    inline void ttg_execute(World &world) {
      // World executes tasks eagerly
    }
    inline void ttg_fence(World &world) {
      world.gop.fence();

      // flag registered statuses
      {
        auto& registry = detail::status_registry_accessor();
        auto iter = registry.find(&world);
        if (iter != registry.end()) {
          auto &statuses = iter->second;
          for (auto &status: statuses) {
            status->set_value();
          }
          statuses.clear();  // clear out the statuses
        }
      }

    }

    template <typename T>
    inline void ttg_register_ptr(World& world, const std::shared_ptr<T>& ptr) {
      auto& registry = detail::ptr_registry_accessor();
      auto iter = registry.find(&world);
      if (iter != registry.end()) {
        auto& ptr_set = iter->second;
        assert(ptr_set.find(ptr) == ptr_set.end());  // prevent duplicate registration
        ptr_set.insert(ptr);
      } else {
        registry.insert(std::make_pair(&world, std::set<std::shared_ptr<void>>({ptr})));
      }
    }

    inline void ttg_register_status(World& world, const std::shared_ptr<std::promise<void>>& status_ptr) {
      auto& registry = detail::status_registry_accessor();
      auto iter = registry.find(&world);
      if (iter != registry.end()) {
        auto& ptr_set = iter->second;
        assert(ptr_set.find(status_ptr) == ptr_set.end());  // prevent duplicate registration
        ptr_set.insert(status_ptr);
      } else {
        registry.insert(std::make_pair(&world, std::set<std::shared_ptr<std::promise<void>>>({status_ptr})));
      }
    }

    inline ::ttg::Edge<>& ttg_ctl_edge(World& world) {
      auto& registry = detail::clt_edge_registry_accessor();
      auto iter = registry.find(&world);
      if (iter != registry.end()) {
        return iter->second;
      } else {
        registry.insert(std::make_pair(&world, ::ttg::Edge<>{}));
        return registry[&world];
      }
    }

  template <typename T>
    void ttg_sum(World &world, T &value) {
      world.gop.sum(value);
    }
    /// broadcast
    /// @tparam T a serializable type
    template <typename T>
    void ttg_broadcast(World &world, T &data, int source_rank) {
      world.gop.broadcast_serializable(data, source_rank);
    }

    template <typename keyT>
    struct default_keymap : ::ttg::detail::default_keymap_impl<keyT> {
     public:
      default_keymap() = default;
      default_keymap(World &world) : ::ttg::detail::default_keymap_impl<keyT>(world.mpi.size()) {}
    };

    template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
    class Op : public ::ttg::OpBase, public WorldObject<Op<keyT, output_terminalsT, derivedT, input_valueTs...>> {
     private:
      World &world;
      ::ttg::meta::detail::keymap_t<keyT> keymap;
      // For now use same type for unary/streaming input terminals, and stream reducers assigned at runtime
      ::ttg::meta::detail::input_reducers_t<input_valueTs...>
          input_reducers;  //!< Reducers for the input terminals (empty = expect single value)

     public:
      World &get_world() const { return world; }

     protected:
      using opT = Op<keyT, output_terminalsT, derivedT, input_valueTs...>;
      using worldobjT = WorldObject<opT>;

      static constexpr int numins = sizeof...(input_valueTs);                    // number of input arguments
      static constexpr int numouts = std::tuple_size<output_terminalsT>::value;  // number of outputs or
      // results

      // MADNESS tasks pass data directly, as values
      template <typename T>
      using data_wrapper_t = T;
      template <typename Wrapper>
      static auto &&unwrap(Wrapper &&wrapper) {
        return std::forward<Wrapper>(wrapper);
      }
      template <typename Result, typename Wrapper>
      static Result unwrap_to(Wrapper &&wrapper) {
        return static_cast<Result>(std::forward<Wrapper>(wrapper));
      }
      template <typename T>
      static auto &&wrap(T &&data) {
        return std::forward<T>(data);
      }

      // This to support op fusion
      inline static __thread struct {
        uint64_t key_hash = 0; // hash of current key
        size_t call_depth = 0; // how deep calls are nested
      } threaddata;

      using input_terminals_type = std::tuple<::ttg::In<keyT, input_valueTs>...>;
      using input_edges_type = std::tuple<::ttg::Edge<keyT, std::decay_t<input_valueTs>>...>;
      static_assert(::ttg::meta::is_none_void_v<input_valueTs...> || std::tuple_size_v<input_terminals_type> == 1, "only single void input can be handled (i.e. can't mix void and nonvoid inputs)");
      using input_values_tuple_type = std::conditional_t<::ttg::meta::is_none_void_v<input_valueTs...>,std::tuple<data_wrapper_t<std::decay_t<input_valueTs>>...>,std::tuple<>>;
      using input_unwrapped_values_tuple_type = input_values_tuple_type;

      using output_terminals_type = output_terminalsT;
      using output_edges_type = typename ::ttg::terminals_to_edges<output_terminalsT>::type;

      template <std::size_t i, typename resultT, typename InTuple>
      static resultT get(InTuple &&intuple) {
        return unwrap_to<resultT>(std::get<i>(intuple));
      };
      template <std::size_t i, typename InTuple>
      static auto &get(InTuple &&intuple) {
        return unwrap(std::get<i>(intuple));
      };

     private:
      input_terminals_type input_terminals;
      output_terminalsT output_terminals;

      struct OpArgs : TaskInterface {
       public:
        int counter;                            // Tracks the number of arguments finalized
        std::array<std::size_t, numins> nargs;  // Tracks the number of expected values (0 = finalized)
        std::array<std::size_t, numins>
            stream_size;             // Expected number of values to receive, only used for streaming inputs
                                     // (0 = unbounded stream)
        input_values_tuple_type t;   // The input values
        derivedT *derived;           // Pointer to derived class instance
        std::conditional_t<::ttg::meta::is_void_v<keyT>,::ttg::Void,keyT> key;                    // Task key

        OpArgs() : counter(numins), nargs(), stream_size(), t() { std::fill(nargs.begin(), nargs.end(), 1); }

        void run(World &world) {
          // ::ttg::print("starting task");

          opT::threaddata.key_hash = std::hash<decltype(key)>()(key);
          opT::threaddata.call_depth++;
            
          if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
            derived->op(key, std::move(t), derived->output_terminals);  // !!! NOTE moving t into op
          } else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
            derived->op(key, derived->output_terminals);
          } else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
            derived->op(std::move(t), derived->output_terminals);  // !!! NOTE moving t into op
          } else {
            derived->op(derived->output_terminals);  // !!! NOTE moving t into op
          }

          opT::threaddata.call_depth--;
          
          //::ttg::print("finishing task",opT::threaddata.call_depth);
        }

        virtual ~OpArgs() {}  // Will be deleted via TaskInterface*

       private:
        madness::Spinlock lock_;                          // sychronizes access to data
       public:
        void lock() {
          lock_.lock();
        }
        void unlock() {
          lock_.unlock();
        }

      };

      using hashable_keyT = std::conditional_t<::ttg::meta::is_void_v<keyT>,int,keyT>;
      using cacheT = ConcurrentHashMap<hashable_keyT, OpArgs *, std::hash<hashable_keyT>>;
      using accessorT = typename cacheT::accessor;
      cacheT cache;

     protected:

      template <std::size_t i, typename Key, typename Value>
      std::enable_if_t<::ttg::meta::is_none_void_v<Key,std::decay_t<Value>>,void>
      set_arg(const Key &key, Value &&value) {
        using valueT = typename std::tuple_element<i, input_values_tuple_type>::type;  // Should be T or const T
        static_assert(std::is_same<std::decay_t<Value>, std::decay_t<valueT>>::value,
                      "Op::set_arg(key,value) given value of type incompatible with Op");

        const auto owner = keymap(key);
        if (owner != world.rank()) {
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding setting argument : ", i);
          // should be able on the other end to consume value (since it is just a temporary byproduct of serialization)
          // BUT compiler vomits when const std::remove_reference_t<Value>& -> std::decay_t<Value>
          // this exposes bad design in MemFuncWrapper (probably similar bugs elsewhere?) whose generic operator()
          // should use memfun's argument types (since that's what will be called) rather than misautodeduce in a particular context
          // P.S. another issue is in send_am which can execute both remotely (where one can always move arguments) and locally
          //      here we know that this will be a remove execution, so we prepare to take rvalues;
          //      send_am will need to separate local and remote paths to deal with this
          worldobjT::send(owner, &opT::template set_arg<i, Key, const std::remove_reference_t<Value>&>, key, value);
        } else {
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": received value for argument : ", i);

          accessorT acc;
          if (cache.insert(acc, key)) acc->second = new OpArgs();  // It will be deleted by the task q
          OpArgs *args = acc->second;

          if (args->nargs[i] == 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : ", key,
                               ": error argument is already finalized : ", i);
            throw "bad set arg";
          }

          auto reducer = std::get<i>(input_reducers);
          if (reducer) {  // is this a streaming input? reduce the received value
            // N.B. Right now reductions are done eagerly, without spawning tasks
            //      this means we must lock
            args->lock();
            // have a value already? if not, set, otherwise reduce
            if ((args->stream_size[i] == 0 && args->nargs[i] == 1) || (args->stream_size[i] == args->nargs[i])) {
              this->get<i, std::decay_t<valueT> &>(args->t) = std::forward<Value>(value);
            } else {
              valueT value_copy = value;  // use constexpr if to avoid making a copy if given nonconst rvalue
              // once Future<>::operator= semantics is cleaned up will avoid Future<>::get()
              this->get<i, std::decay_t<valueT> &>(args->t) =
                  std::move(reducer(this->get<i, std::decay_t<valueT> &&>(args->t), std::move(value_copy)));
            }
            // update the counter if the stream is bounded
            // this assumes that the stream size is set before data starts flowing ... strong-typing streams will solve
            // this
            if (args->stream_size[i] != 0) {
              args->nargs[i]--;
              if (args->nargs[i] == 0) args->counter--;
            }
            args->unlock();
          } else {  // this is a nonstreaming input => set the value
            this->get<i, std::decay_t<valueT> &>(args->t) = std::forward<Value>(value);
            args->nargs[i]--;
            args->counter--;
          }

          // ready to run the task?
          if (args->counter == 0) {
            if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
            args->derived = static_cast<derivedT *>(this);
            args->key = key;

            auto curhash = std::hash<keyT>{}(key);

            if (curhash == threaddata.key_hash && threaddata.call_depth<6) { // Needs to be externally configurable
                
                //::ttg::print("directly invoking:", get_name(), key, curhash, threaddata.key_hash, threaddata.call_depth);
                opT::threaddata.call_depth++;
                if constexpr (!::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
                  static_cast<derivedT*>(this)->op(key, std::move(args->t), output_terminals); // Runs immediately
                } else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
                  static_cast<derivedT *>(this)->op(key, output_terminals); // Runs immediately
                } else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_empty_tuple_v<input_values_tuple_type>) {
                  static_cast<derivedT *>(this)->op(std::move(args->t), output_terminals); // Runs immediately
                } else {
                  static_cast<derivedT *>(this)->op(output_terminals); // Runs immediately
                }
                opT::threaddata.call_depth--;
                
            }
            else {
                //::ttg::print("enqueuing task", get_name(), key, curhash, threaddata.key_hash, threaddata.call_depth);
                world.taskq.add(args);
            }
            

            cache.erase(acc);
          }
        }
      }

      template <std::size_t i, typename Key = keyT, typename Value>
      std::enable_if_t<::ttg::meta::is_void_v<Key> && !::ttg::meta::is_void_v<std::decay_t<Value>>,void>
      set_arg(Value &&value) {
        using valueT = typename std::tuple_element<i, input_values_tuple_type>::type;  // Should be T or const T
        static_assert(std::is_same<std::decay_t<Value>, std::decay_t<valueT>>::value,
                      "Op::set_arg(key,value) given value of type incompatible with Op");

        const int owner = keymap();

        if (owner != world.rank()) {
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : forwarding setting argument : ", i);
          // CAVEAT see comment above in set_arg re:
          worldobjT::send(owner, &opT::template set_arg<i, keyT, const std::remove_reference_t<Value>&>, value);
        } else {
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : received value for argument : ", i);

          accessorT acc;
          if (cache.insert(acc, 0)) acc->second = new OpArgs();  // It will be deleted by the task q
          OpArgs *args = acc->second;

          if (args->nargs[i] == 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : error argument is already finalized : ", i);
            throw "bad set arg";
          }

          auto reducer = std::get<i>(input_reducers);
          if (reducer) {  // is this a streaming input? reduce the received value
            // N.B. Right now reductions are done eagerly, without spawning tasks
            //      this means we must lock
            args->lock();
            // have a value already? if not, set, otherwise reduce
            if ((args->stream_size[i] == 0 && args->nargs[i] == 1) || (args->stream_size[i] == args->nargs[i])) {
              this->get<i, std::decay_t<valueT> &>(args->t) = std::forward<Value>(value);
            } else {
              valueT value_copy = value;  // use constexpr if to avoid making a copy if given nonconst rvalue
              // once Future<>::operator= semantics is cleaned up will avoid Future<>::get()
              this->get<i, std::decay_t<valueT> &>(args->t) =
                  std::move(reducer(this->get<i, std::decay_t<valueT> &&>(args->t), std::move(value_copy)));
            }
            // update the counter if the stream is bounded
            // this assumes that the stream size is set before data starts flowing ... strong-typing streams will solve
            // this
            if (args->stream_size[i] != 0) {
              args->nargs[i]--;
              if (args->nargs[i] == 0) args->counter--;
            }
            args->unlock();
          } else {  // this is a nonstreaming input => set the value
            this->get<i, std::decay_t<valueT> &>(args->t) = std::forward<Value>(value);
            args->nargs[i]--;
            args->counter--;
          }

          // ready to run the task?
          if (args->counter == 0) {
            if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : submitting task for op ");
            args->derived = static_cast<derivedT *>(this);

            world.taskq.add(args);

            cache.erase(acc);
          }
        }
      }

      // Used to generate tasks with no input arguments
      template <typename Key = keyT>
      std::enable_if_t<!::ttg::meta::is_void_v<Key>,void>
      set_arg(const Key &key) {
        static_assert(::ttg::meta::is_empty_tuple_v<input_values_tuple_type>, "set_arg called without a value but valueT!=void");
        const int owner = keymap(key);

        if (owner != world.rank()) {
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding no-arg task: ");
          worldobjT::send(owner, &opT::set_arg<keyT>, key);
        } else {
          accessorT acc;
          if (cache.insert(acc, key)) acc->second = new OpArgs();  // It will be deleted by the task q
          OpArgs *args = acc->second;

          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
          args->derived = static_cast<derivedT *>(this);
          args->key = key;

          world.taskq.add(args);
          // static_cast<derivedT*>(this)->op(key, std::move(args->t), output_terminals);// runs immediately

          cache.erase(acc);
        }
      }

      // Used to generate tasks with no input arguments
      template <typename Key = keyT>
      std::enable_if_t<::ttg::meta::is_void_v<Key>,void> set_arg() {
        static_assert(::ttg::meta::is_empty_tuple_v<input_values_tuple_type>, "set_arg called without a value but valueT!=void");
        const int owner = keymap();

        if (owner != world.rank()) {
          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : forwarding no-arg task: ");
          worldobjT::send(owner, &opT::set_arg<keyT>);
        } else {
          auto task = new OpArgs();  // It will be deleted by the task q

          if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : submitting task for op ");
          task->derived = static_cast<derivedT *>(this);

          world.taskq.add(task);
        }
      }

      // Used by invoke to set all arguments associated with a task
      template <typename Key, size_t... IS>
      std::enable_if_t<!::ttg::meta::is_void_v<Key>,void>
      set_args(std::index_sequence<IS...>, const Key &key, const input_values_tuple_type &args) {
        int junk[] = {0, (set_arg<IS>(key, Op::get<IS>(args)), 0)...};
        junk[0]++;
      }

     public:
      /// sets stream size for input \c i
      /// \param size positive integer that specifies the stream size
      template <std::size_t i, bool key_is_void = ::ttg::meta::is_void_v<keyT>>
      std::enable_if_t<key_is_void,void> set_argstream_size(std::size_t size) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
        assert(size > 0 && "Op::set_argstream_size(size) called with size=0");

        // body
        const auto owner = keymap();
        if (owner != world.rank()) {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : forwarding stream size for terminal ", i);
            worldobjT::send(owner, &opT::template set_argstream_size<i, true>, size);
        } else {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : setting stream size for terminal ", i);

          accessorT acc;
          if (cache.insert(acc, 0)) acc->second = new OpArgs();  // It will be deleted by the task q
          OpArgs *args = acc->second;

          args->lock();

          // check if stream is already bounded
          if (args->stream_size[i] > 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : error stream is already bounded : ", i);
            throw std::runtime_error("Op::set_argstream_size called for a bounded stream");
          }

          // check if stream is already finalized
          if (args->nargs[i] == 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : error stream is already finalized : ", i);
            throw std::runtime_error("Op::set_argstream_size called for a finalized stream");
          }

          // commit changes
          args->stream_size[i] = size;
          args->nargs[i] = size;

          args->unlock();
        }
      }

      /// sets stream size for input \c i
      /// \param size positive integer that specifies the stream size
      template <std::size_t i, typename Key = keyT, bool key_is_void = ::ttg::meta::is_void_v<Key>>
      std::enable_if_t<!key_is_void,void>
      set_argstream_size(const Key &key, std::size_t size) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::set_argstream_size called on nonstreaming input terminal");
        assert(size > 0 && "Op::set_argstream_size(key,size) called with size=0");

        // body
        const auto owner = keymap(key);
        if (owner != world.rank()) {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding stream size for terminal ", i);
            worldobjT::send(owner, &opT::template set_argstream_size<i>, key, size);
        } else {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": setting stream size for terminal ", i);

          accessorT acc;
          if (cache.insert(acc, key)) acc->second = new OpArgs();  // It will be deleted by the task q
          OpArgs *args = acc->second;

          args->lock();

          // check if stream is already bounded
          if (args->stream_size[i] > 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already bounded : ", i);
            throw std::runtime_error("Op::set_argstream_size called for a bounded stream");
          }

          // check if stream is already finalized
          if (args->nargs[i] == 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already finalized : ", i);
            throw std::runtime_error("Op::set_argstream_size called for a finalized stream");
          }

          // commit changes
          args->stream_size[i] = size;
          args->nargs[i] = size;

          args->unlock();
        }
      }

      /// finalizes stream for input \c i
      template <std::size_t i, typename Key = keyT, bool key_is_void = ::ttg::meta::is_void_v<Key>>
      std::enable_if_t<!key_is_void,void>
      finalize_argstream(const Key &key) {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::finalize_argstream called on nonstreaming input terminal");

        // body
        const auto owner = keymap(key);
        if (owner != world.rank()) {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": forwarding stream finalize for terminal ", i);
          worldobjT::send(owner, &opT::template finalize_argstream<i>, key);
        } else {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": finalizing stream for terminal ", i);

          accessorT acc;
          const auto found = cache.find(acc, key);
          assert(found && "Op::finalize_argstream called but no values had been received yet for this key");
          OpArgs *args = acc->second;

          // check if stream is already bounded
          if (args->stream_size[i] > 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : ", key,
                               ": error finalize called on bounded stream: ", i);
            throw std::runtime_error("Op::finalize called for a bounded stream");
          }

          // check if stream is already finalized
          if (args->nargs[i] == 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : ", key, ": error stream is already finalized : ", i);
            throw std::runtime_error("Op::finalized called for a finalized stream");
          }

          // commit changes
          args->nargs[i] = 0;
          args->counter--;
          // ready to run the task?
          if (args->counter == 0) {
            if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : ", key, ": submitting task for op ");
            args->derived = static_cast<derivedT *>(this);
            args->key = key;

            world.taskq.add(args);
            // static_cast<derivedT*>(this)->op(key, std::move(args->t), output_terminals); // Runs immediately

            cache.erase(acc);
          }
        }
      }

      /// finalizes stream for input \c i
      template <std::size_t i, bool key_is_void = ::ttg::meta::is_void_v<keyT>>
      std::enable_if_t<key_is_void,void> finalize_argstream() {
        // preconditions
        assert(std::get<i>(input_reducers) && "Op::finalize_argstream called on nonstreaming input terminal");

        // body
        const int owner = keymap();
        if (owner != world.rank()) {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : forwarding stream finalize for terminal ", i);
          worldobjT::send(owner, &opT::template finalize_argstream<i, true>);
        } else {
          if (tracing())
            ::ttg::print(world.rank(), ":", get_name(), " : finalizing stream for terminal ", i);

          accessorT acc;
          const auto found = cache.find(acc, 0);
          assert(found && "Op::finalize_argstream called but no values had been received yet for this key");
          OpArgs *args = acc->second;

          // check if stream is already bounded
          if (args->stream_size[i] > 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : error finalize called on bounded stream: ", i);
            throw std::runtime_error("Op::finalize called for a bounded stream");
          }

          // check if stream is already finalized
          if (args->nargs[i] == 0) {
            ::ttg::print_error(world.rank(), ":", get_name(), " : error stream is already finalized : ", i);
            throw std::runtime_error("Op::finalized called for a finalized stream");
          }

          // commit changes
          args->nargs[i] = 0;
          args->counter--;
          // ready to run the task?
          if (args->counter == 0) {
            if (tracing()) ::ttg::print(world.rank(), ":", get_name(), " : submitting task for op ");
            args->derived = static_cast<derivedT *>(this);

            world.taskq.add(args);
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
        // using send_callbackT = typename ::ttg::In<keyT, valueT>::send_callback_type;
        // using move_callbackT = typename ::ttg::In<keyT, valueT>::move_callback_type;
        // using setsize_callbackT = typename ::ttg::In<keyT, valueT>::setsize_callback_type;
        // using finalize_callbackT = typename ::ttg::In<keyT, valueT>::finalize_callback_type;

        //////////////////////////////////////////////////////////////////
        // nonvoid key, nonvoid value
        //////////////////////////////////////////////////////////////////
        if constexpr (::ttg::meta::is_none_void_v<keyT,valueT>) {
          auto move_callback = [this](const keyT &key, valueT &&value) {
            set_arg<i, keyT, valueT>(key, std::forward<valueT>(value));
          };
          auto send_callback = [this](const keyT &key, const valueT &value) {
            set_arg<i, keyT, const valueT &>(key, value);
          };
          auto setsize_callback = [this](const keyT &key, std::size_t size) {
            set_argstream_size<i>(key, size);
          };
          auto finalize_callback = [this](const keyT &key) {
            finalize_argstream<i>(key);
          };
          input.set_callback(send_callback, move_callback, setsize_callback, finalize_callback);
        }
        //////////////////////////////////////////////////////////////////
        // void key, nonvoid value
        //////////////////////////////////////////////////////////////////
        else if constexpr (::ttg::meta::is_void_v<keyT> && !::ttg::meta::is_void_v<valueT>) {
          auto move_callback = [this](valueT &&value) {
            set_arg<i, keyT, valueT>(std::forward<valueT>(value));
          };
          auto send_callback = [this](const valueT &value) {
            set_arg<i, keyT, const valueT &>(value);
          };
          auto setsize_callback = [this](std::size_t size) {
            set_argstream_size<i>(size);
          };
          auto finalize_callback = [this]() {
            finalize_argstream<i>();
          };
          input.set_callback(send_callback, move_callback, setsize_callback, finalize_callback);
        }
        //////////////////////////////////////////////////////////////////
        // nonvoid key, void value
        //////////////////////////////////////////////////////////////////
        else if constexpr (!::ttg::meta::is_void_v<keyT> && ::ttg::meta::is_void_v<valueT>) {
          auto send_callback = [this](const keyT &key) {
            set_arg<keyT>(key);
          };
          auto setsize_callback = [this](const keyT &key, std::size_t size) {
            set_argstream_size<i>(key, size);
          };
          auto finalize_callback = [this](const keyT &key) {
            finalize_argstream<i>(key);
          };
          input.set_callback(send_callback, send_callback, setsize_callback, finalize_callback);
        }
        //////////////////////////////////////////////////////////////////
        // void key, void value
        //////////////////////////////////////////////////////////////////
        else if constexpr (::ttg::meta::is_all_void_v<keyT,valueT>) {
          auto send_callback = [this]() {
            set_arg<keyT>();
          };
          auto setsize_callback = [this](std::size_t size) {
            set_argstream_size<i>(size);
          };
          auto finalize_callback = [this]() {
            finalize_argstream<i>();
          };
          input.set_callback(send_callback, send_callback, setsize_callback, finalize_callback);
        }
        else abort();
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
        int junk[] = {0, (std::get<IS>(inedges).set_out(&std::get<IS>(input_terminals)), 0)...};
        junk[0]++;
      }

      template <std::size_t... IS, typename outedgesT>
      void connect_my_outputs_to_outgoing_edge_inputs(std::index_sequence<IS...>, outedgesT &outedges) {
        int junk[] = {0, (std::get<IS>(outedges).set_in(&std::get<IS>(output_terminals)), 0)...};
        junk[0]++;
      }

     public:
      template <typename keymapT = default_keymap<keyT>>
      Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         World &world, keymapT &&keymap_ = keymapT())
          : ::ttg::OpBase(name, numins, numouts)
          , worldobjT(world)
          , world(world)
          // if using default keymap, rebind to the given world
          , keymap(std::is_same<keymapT, default_keymap<keyT>>::value
                       ? decltype(keymap)(default_keymap<keyT>(world))
                       : decltype(keymap)(std::forward<keymapT>(keymap_))) {
        // Cannot call these in base constructor since terminals not yet constructed
          if (innames.size() != std::tuple_size<input_terminals_type>::value) {
              ::ttg::print_error(world.rank(), ":", get_name(),  "#input_names", innames.size(), "!= #input_terminals", std::tuple_size<input_terminals_type>::value);
              throw this->get_name()+":madness::ttg::Op: #input names != #input terminals";
          }
        if (outnames.size() != std::tuple_size<output_terminalsT>::value)
            throw this->get_name()+":madness::ttg::Op: #output names != #output terminals";

        register_input_terminals(input_terminals, innames);
        register_output_terminals(output_terminals, outnames);

        register_input_callbacks(std::make_index_sequence<numins>{});
      }

      template <typename keymapT = default_keymap<keyT>>
      Op(const std::string &name, const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         keymapT &&keymap = keymapT(get_default_world()))
          : Op(name, innames, outnames, get_default_world(), std::forward<keymapT>(keymap)) {}

      template <typename keymapT = default_keymap<keyT>>
      Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames, World &world,
         keymapT &&keymap_ = keymapT())
          : ::ttg::OpBase(name, numins, numouts)
          , worldobjT(get_default_world())
          , world(get_default_world())
          // if using default keymap, rebind to the given world
          , keymap(std::is_same<keymapT, default_keymap<keyT>>::value
                       ? decltype(keymap)(default_keymap<keyT>(world))
                       : decltype(keymap)(std::forward<keymapT>(keymap_))) {
        // Cannot call in base constructor since terminals not yet constructed
          if (innames.size() != std::tuple_size<input_terminals_type>::value) {
              ::ttg::print_error(world.rank(), ":", get_name(),  "#input_names", innames.size(), "!= #input_terminals", std::tuple_size<input_terminals_type>::value);
              throw this->get_name()+":madness::ttg::Op: #input names != #input terminals";
          }
        if (outnames.size() != std::tuple_size<output_terminalsT>::value)
          throw this->get_name()+":madness::ttg::Op: #output names != #output terminals";

        register_input_terminals(input_terminals, innames);
        register_output_terminals(output_terminals, outnames);

        register_input_callbacks(std::make_index_sequence<numins>{});

        connect_my_inputs_to_incoming_edge_outputs(std::make_index_sequence<numins>{}, inedges);
        connect_my_outputs_to_outgoing_edge_inputs(std::make_index_sequence<numouts>{}, outedges);
      }

      template <typename keymapT = default_keymap<keyT>>
      Op(const input_edges_type &inedges, const output_edges_type &outedges, const std::string &name,
         const std::vector<std::string> &innames, const std::vector<std::string> &outnames,
         keymapT &&keymap = keymapT(get_default_world()))
          : Op(inedges, outedges, name, innames, outnames, get_default_world(), std::forward<keymapT>(keymap)) {}

      // Destructor checks for unexecuted tasks
      virtual ~Op() {
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
            for (std::size_t i = 0; i < numins; i++) std::cerr << (item.second->nargs[i] == 0 ? "T" : "F") << " ";
            std::cerr << ")" << std::endl;
          }
          abort();
        }
      }

      static constexpr const ::ttg::Runtime runtime = ::ttg::Runtime::MADWorld;

      template <std::size_t i, typename Reducer>
      void set_input_reducer(Reducer &&reducer) {
        std::get<i>(input_reducers) = reducer;
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
      template <typename Key = keyT> std::enable_if_t<!::ttg::meta::is_void_v<Key>,void>
      invoke(const Key &key, const input_values_tuple_type &args) {
        set_args(std::make_index_sequence<std::tuple_size<input_values_tuple_type>::value>{}, key, args);
      }

      /// Manual injection of a task that has no arguments
      template <typename Key = keyT> std::enable_if_t<!::ttg::meta::is_void_v<Key>,void>
      invoke(const Key &key) { set_arg<Key>(key); }

      /// Manual injection of a task that has no key or arguments
      template <typename Key = keyT> std::enable_if_t<::ttg::meta::is_void_v<Key>,void> invoke() { set_arg<Key>(); }

      /// keymap accessor
      /// @return the keymap
      const decltype(keymap) &get_keymap() const { return keymap; }

      /// computes the owner of key @c key
      /// @param[in] key the key
      /// @return the owner of @c key
      template <typename Key>
      std::enable_if_t<!::ttg::meta::is_void_v<Key>,int>
      owner(const Key& key) const { return keymap(key); }

      /// computes the owner of void key
      /// @return the owner of void key
      template <typename Key>
      std::enable_if_t<::ttg::meta::is_void_v<Key>,int>
      owner() const { return keymap(); }

    };

#include "../wrap.h"

    // clang-format off
/*
 * This allows programmatic control of watchpoints. Requires MADWorld using legacy ThreadPool and macOS. Example:
 * @code
 *   double x = 0.0;
 *   ::madness::ttg::initialize_watchpoints();
 *   ::madness::ttg::watchpoint_set(&x, ::ttg::detail::MemoryWatchpoint_x86_64::kWord,
 *     ::ttg::detail::MemoryWatchpoint_x86_64::kWhenWritten);
 *   x = 1.0;  // this will generate SIGTRAP ...
 *   ttg_default_execution_context().taskq.add([&x](){ x = 1.0; });  // and so will this ...
 *   ::madness::ttg::watchpoint_set(&x, ::ttg::detail::MemoryWatchpoint_x86_64::kWord,
 *     ::ttg::detail::MemoryWatchpoint_x86_64::kWhenWrittenOrRead);
 *   ttg_default_execution_context().taskq.add([&x](){
 *       std::cout << x << std::endl; });  // and even this!
 *
 * @endcode
 */
    // clang-format on

    namespace detail {
      inline const std::vector<const pthread_t *> &watchpoints_threads() {
        static std::vector<const pthread_t *> threads;
        // can set watchpoints only with the legacy MADNESS threadpool
        // TODO improve this when shortsighted MADNESS macro names are strengthened, i.e. HAVE_INTEL_TBB ->
        // MADNESS_HAS_INTEL_TBB
        // TODO also exclude the case of a PARSEC-based backend
#ifndef HAVE_INTEL_TBB
        if (threads.empty()) {
          static pthread_t main_thread_id = pthread_self();
          threads.push_back(&main_thread_id);
          for (int t = 0; t != madness::ThreadPool::size(); ++t) {
            threads.push_back(&(madness::ThreadPool::get_threads()[t].get_id()));
          }
        }
#endif
        return threads;
      }
    }  // namespace detail

    /// must be called from main thread before setting watchpoints
    inline void initialize_watchpoints() {
#if defined(HAVE_INTEL_TBB)
      ::ttg::print_error(ttg_default_execution_context().rank(),
                         "WARNING: watchpoints are only supported with MADWorld using the legacy threadpool");
#endif
#if !defined(__APPLE__)
      ::ttg::print_error(ttg_default_execution_context().rank(), "WARNING: watchpoints are only supported on macOS");
#endif
      ::ttg::detail::MemoryWatchpoint_x86_64::Pool::initialize_instance(detail::watchpoints_threads());
    }

    /// sets a hardware watchpoint for window @c [addr,addr+size) and condition @c cond
    template <typename T>
    inline void watchpoint_set(T *addr, ::ttg::detail::MemoryWatchpoint_x86_64::Size size,
                               ::ttg::detail::MemoryWatchpoint_x86_64::Condition cond) {
      const auto &threads = detail::watchpoints_threads();
      for (auto t : threads) ::ttg::detail::MemoryWatchpoint_x86_64::Pool::instance()->set(addr, size, cond, t);
    }

    /// clears the hardware watchpoint for window @c [addr,addr+size) previously created with watchpoint_set<T>
    template <typename T>
    inline void watchpoint_clear(T *addr) {
      const auto &threads = detail::watchpoints_threads();
      for (auto t : threads) ::ttg::detail::MemoryWatchpoint_x86_64::Pool::instance()->clear(addr, t);
    }

  }  // namespace ttg
}  // namespace madness

#endif  // MADNESS_TTG_H_INCLUDED
