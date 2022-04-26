#ifndef TTG_POLICIES_H
#define TTG_POLICIES_H

#include "ttg/base/keymap.h"
#include "ttg/util/meta.h"

namespace ttg {

  namespace detail {
    template<typename MapT>
    struct is_default_keymap : std::false_type
    { };

    template<typename Key>
    struct is_default_keymap<ttg::meta::detail::keymap<Key>> : std::true_type
    { };

    template<typename T>
    constexpr const bool is_default_keymap_v = is_default_keymap<T>::value;

    template<typename MapT>
    struct map_type {
      using type = std::decay_t<MapT>;
    };

    template<typename Key>
    struct map_type<ttg::meta::detail::keymap<Key>> {
      using type = ttg::meta::detail::keymap_t<Key>;
    };

    template<typename T>
    using map_type_t = typename map_type<T>::type;

    /* query the type of the first argument and the return type of a callable */
    template<typename T>
    struct keymap_signature
    : keymap_signature<decltype(&std::remove_reference_t<T>::operator())>
    {};

    template<typename R, typename Arg1>
    struct keymap_signature<R(*)(Arg1)> {
      using arg_t = std::decay_t<Arg1>;
      using ret_t = std::decay_t<R>;
    };

    template<typename R, typename ClassT, typename Arg1>
    struct keymap_signature<R(ClassT::*)(Arg1)>
    : keymap_signature<R(*)(Arg1)>
    { };

    template<typename R, typename ClassT, typename Arg1>
    struct keymap_signature<R(ClassT::*)(Arg1)const>
    : keymap_signature<R(*)(Arg1)>
    { };

    template<typename R>
    struct keymap_signature<R(*)()> {
      using arg_t = void;
      using ret_t = std::decay_t<R>;
    };

    template<typename R, typename ClassT>
    struct keymap_signature<R(ClassT::*)()>
    : keymap_signature<R(*)()>
    { };

    template<typename R, typename ClassT>
    struct keymap_signature<R(ClassT::*)()const>
    : keymap_signature<R(*)()>
    { };


    template<typename FnT>
    using keymap_arg_t = typename keymap_signature<FnT>::arg_t;

    template<typename FnT>
    using keymap_ret_t = typename keymap_signature<FnT>::ret_t;

  } // namespace detail

  /**
   * \brief Base class for task execution policies.
   *
   * Policies are properties of tasks. Tasks are identified through the key.
   * A policy implementation maps a key to an integer value and can be set per TT.
   * Supported policies include:
   *  * Process mapping: maps a key identifying a task to a process to run on.
   *  * Priority mapping: assigns a priority (positive integer) to a task identified by a key.
   *                      Higher values increase the task's priority.
   *  * Inline mapping: whether a task can be executed inline, i.e., without dispatching the
   *                    task to a scheduler first. The task will be executed in the send or
   *                    broadcast call. The returned value denotes the maximum recirsion depth,
   *                    i.e., how many tasks may be executed inline consecutively. Zero denotes
   *                    no inlining. This is the default.
   *
   * The default mapping functions are not inlined and can be set dynamically
   * in the TT. By inheriting from \c TTPolicyBase and passing callables to its
   * constructor, applications can define policies at compile-time. This may
   * yield improved performance since the compiler is potentially able to inline
   * the calls. In that case, the dynamic setting of mapping functions in the
   * TT will be disabled.
   *
   * \tparam Key The type of the key used to identify tasks.
   * \tparam ProcMap The type of the process map callback.
   * \tparam PrioMap The type of the priority map callback.
   * \tparam InlineMap The type of the inline map callback.
   *
   * \sa ttg::make_policy
   */
  template<typename Key,
           typename ProcMap = typename ttg::meta::detail::keymap_t<Key>,
           typename PrioMap = typename ttg::meta::detail::keymap_t<Key>,
           typename InlineMap = typename ttg::meta::detail::keymap_t<Key>>
  struct TTPolicyBase {

    using PolicyBaseT = TTPolicyBase<Key, ProcMap, PrioMap, InlineMap>;

    using procmap_t = detail::map_type_t<ProcMap>;
    using priomap_t = detail::map_type_t<PrioMap>;
    using inlinemap_t = detail::map_type_t<InlineMap>;

    using key_type = Key;

    procmap_t procmap;
    priomap_t priomap;
    inlinemap_t inlinemap;

    template<typename ProcMap_ = ProcMap,
             typename PrioMap_ = PrioMap,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<std::is_default_constructible_v<ProcMap_> &&
                                         std::is_default_constructible_v<PrioMap_> &&
                                         std::is_default_constructible_v<InlineMap_>>>
    TTPolicyBase()
    { }

    template<typename ProcMap_,
             typename PrioMap_ = PrioMap,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<std::is_constructible_v<procmap_t, ProcMap_> &&
                                         std::is_default_constructible_v<PrioMap_> &&
                                         std::is_default_constructible_v<InlineMap_>>>
    TTPolicyBase(ProcMap_&& procmap)
    : procmap(std::forward<ProcMap_>(procmap))
    { }

    template<typename ProcMap_,
             typename PrioMap_,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<std::is_default_constructible_v<InlineMap_>>>
    TTPolicyBase(ProcMap_&& procmap, PrioMap_&& priomap)
    : procmap(std::forward<ProcMap_>(procmap))
    , priomap(std::forward<PrioMap_>(priomap))
    { }

    template<typename ProcMap_, typename PrioMap_, typename InlineMap_>
    TTPolicyBase(ProcMap_&& procmap,
                 PrioMap_&& priomap,
                 InlineMap_&& im)
    : procmap(std::forward<ProcMap_>(procmap))
    , priomap(std::forward<PrioMap_>(priomap))
    , inlinemap(std::forward<InlineMap_>(im))
    { }

    TTPolicyBase(const PolicyBaseT&) = default;
    TTPolicyBase(PolicyBaseT&&) = default;
  };


  template<typename Key,
           typename ProcMap = typename ttg::detail::default_keymap_impl<Key>,
           typename PrioMap = typename ttg::detail::default_priomap_impl<Key>,
           typename InlineMap = typename ttg::detail::default_inlinemap_impl<Key>>
  struct StaticTTPolicyBase {

    using PolicyBaseT = StaticTTPolicyBase<Key, ProcMap, PrioMap, InlineMap>;

    using procmap_t = detail::map_type_t<ProcMap>;
    using priomap_t = detail::map_type_t<PrioMap>;
    using inlinemap_t = detail::map_type_t<InlineMap>;

    using key_type = Key;

    procmap_t procmap;
    priomap_t priomap;
    inlinemap_t inlinemap;

    template<typename ProcMap_ = ProcMap,
             typename PrioMap_ = PrioMap,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<std::is_default_constructible_v<ProcMap_> &&
                                         std::is_default_constructible_v<PrioMap_> &&
                                         std::is_default_constructible_v<InlineMap_>>>
    StaticTTPolicyBase()
    { }

    template<typename ProcMap_,
             typename PrioMap_ = PrioMap,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<std::is_constructible_v<procmap_t, ProcMap_> &&
                                         std::is_default_constructible_v<PrioMap_> &&
                                         std::is_default_constructible_v<InlineMap_>>>
    StaticTTPolicyBase(ProcMap_&& procmap)
    : procmap(std::forward<ProcMap_>(procmap))
    { }

    template<typename ProcMap_,
             typename PrioMap_,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<std::is_default_constructible_v<InlineMap_>>>
    StaticTTPolicyBase(ProcMap_&& procmap, PrioMap_&& priomap)
    : procmap(std::forward<ProcMap_>(procmap))
    , priomap(std::forward<PrioMap_>(priomap))
    { }

    template<typename ProcMap_, typename PrioMap_, typename InlineMap_>
    StaticTTPolicyBase(ProcMap_&& procmap,
                 PrioMap_&& priomap,
                 InlineMap_&& im)
    : procmap(std::forward<ProcMap_>(procmap))
    , priomap(std::forward<PrioMap_>(priomap))
    , inlinemap(std::forward<InlineMap_>(im))
    { }

    StaticTTPolicyBase(const PolicyBaseT&) = default;
    StaticTTPolicyBase(PolicyBaseT&&) = default;
  };

  namespace detail {

    /**
     * Wrapper around a policy implementation.
     * The wrapper provides default implementations for properties that
     * are not set at compile-time and not yet set at runtime.
     * By using a wrapper object, we can inspect the \c procmap(), \c priomap(),
     * and \c inlinemap() of the policy to see whether a compile-time implementation
     * of them was provided and gracefull catch attempts at setting
     * properties that were provided at compile-time.
     *
     * TT implementations can inherit from this class to get the necessary
     * mapping functions as well as functions to query and set mapping functions.
     */
    template<typename Key, typename TTPolicyImpl>
    struct TTPolicyWrapper {
    private:
      TTPolicyImpl m_policy;

      ttg::detail::default_keymap_impl<Key> m_default_procmap;
      ttg::detail::default_priomap_impl<Key> m_default_priomap;
      ttg::detail::default_inlinemap_impl<Key> m_default_inlinemap;

      static constexpr bool procmap_is_std_function = meta::is_std_function_ptr_v<decltype(&TTPolicyImpl::procmap)>;
      static constexpr bool priomap_is_std_function = meta::is_std_function_ptr_v<decltype(&TTPolicyImpl::priomap)>;
      static constexpr bool inlinemap_is_std_function = meta::is_std_function_ptr_v<decltype(&TTPolicyImpl::inlinemap)>;

    public:

      /**
       * Construct a wrapper from a provided world (needed for some of the defaults)
       * and provided policy implementation.
       */
      template<typename WorldT, typename ImplT>
      TTPolicyWrapper(WorldT world, ImplT&& impl)
      : m_policy(std::forward<ImplT>(impl))
      , m_default_procmap(world.size())
      { }

      /**
       * Return a copy of the used policy, with proper defaults.
       */
      auto get_policy() {
        TTPolicyImpl policy = m_policy;
        if constexpr (procmap_is_std_function) {
          if (!m_policy.procmap) {
            /* create a std::function from the default implementation */
            policy.procmap = m_default_procmap;
          }
        }
        if constexpr (priomap_is_std_function) {
          if (!m_policy.priomap) {
            /* create a std::function from the default implementation */
            policy.priomap = m_default_priomap;
          }
        }
        if constexpr (inlinemap_is_std_function) {
          if (!m_policy.inlinemap) {
            /* create a std::function from the default implementation */
            policy.inlinemap = m_default_inlinemap;
          }
        }
        return policy;
      }

      /**
       * Return a callable for the current process map.
       * Returns a std::function object (not a reference) that can be invoked.
       */
      inline auto get_procmap() const {
        if constexpr (procmap_is_std_function) {
          if (!m_policy.procmap) {
            /* create a std::function from the default implementation */
            return ttg::meta::detail::keymap_t<Key>(m_default_procmap);
          } else {
            /* return the current std::function */
            return m_policy.procmap;
          }
        } else {
          if constexpr (!meta::is_member_fn_ptr_v<decltype(&TTPolicyImpl::procmap)>) {
            /* we can return any callable object directly */
            return m_policy.procmap;
          } else {
            /**
             * wrap the member function in a lambda
             * assuming that a policy is a small object, we'll capture it by value
             * to avoid lifetime issues
             */
            if constexpr (!std::is_void_v<Key>) {
              return [=](const Key& key){ return m_policy.procmap(key); };
            } else {
              return [=](){ return m_policy.procmap(); };
            }
          }
        }
      }

      /**
       * Return a callable for the current priority map.
       * Returns a std::function object (not a reference) that can be invoked.
       */
      inline auto get_priomap() const {
        if constexpr (priomap_is_std_function) {
          if (!m_policy.priomap) {
            /* create a std::function from the default implementation */
            return ttg::meta::detail::keymap_t<Key>(m_default_priomap);
          } else {
            /* return the current std::function */
            return m_policy.priomap;
          }
        } else {
          if constexpr (!meta::is_member_fn_ptr_v<decltype(&TTPolicyImpl::priomap)>) {
            /* we can return any callable object directly */
            return m_policy.priomap;
          } else {
            /**
             * wrap the member function in a lambda
             * assuming that a policy is a small object, we'll capture it by value
             * to avoid lifetime issues
             */
            if constexpr (!std::is_void_v<Key>) {
              return [=](const Key& key){ return m_policy.priomap(key); };
            } else {
              return [=](){ return m_policy.priomap(); };
            }
          }
        }
      }

      inline auto get_inlinemap() const {
        if constexpr (inlinemap_is_std_function) {
          if (!m_policy.inlinemap) {
            /* create a std::function from the default implementation */
            return ttg::meta::detail::keymap_t<Key>(m_default_inlinemap);
          } else {
            /* return the current std::function */
            return m_policy.inlinemap;
          }
        } else {
          if constexpr (!meta::is_member_fn_ptr_v<decltype(&TTPolicyImpl::inlinemap)>) {
            /* we can return any callable object directly */
            return m_policy.inlinemap;
          } else {
            /**
             * wrap the member function in a lambda
             * assuming that a policy is a small object, we'll capture it by value
             * to avoid lifetime issues
             */
            if constexpr (!std::is_void_v<Key>) {
              return [=](const Key& key){ return m_policy.inlinemap(key); };
            } else {
              return [=](){ return m_policy.inlinemap(); };
            }
          }
        }
      }

      template<typename KeyT, typename = std::enable_if_t<!std::is_void_v<KeyT>>>
      inline int procmap(const KeyT& key) const {
        if constexpr (procmap_is_std_function) {
          if (m_policy.procmap) return m_policy.procmap(key);
          else return m_default_procmap(key);
        } else {
          return m_policy.procmap(key);
        }
      }

      template<typename KeyT = Key, typename = std::enable_if_t<std::is_void_v<KeyT>>>
      inline int procmap() const {
        if constexpr (procmap_is_std_function) {
          if (m_policy.procmap) return m_policy.procmap();
          else return m_default_procmap();
        } else {
          return m_policy.procmap();
        }
      }

      /** Deprecated, use procmap instead */
      template<typename KeyT, typename = std::enable_if_t<!std::is_void_v<KeyT>>>
      inline int keymap(const KeyT& key) const {
        return procmap(key);
      }

      /** Deprecated, use procmap instead */
      template<typename KeyT = Key, typename = std::enable_if_t<std::is_void_v<KeyT>>>
      inline int keymap() const {
        return procmap();
      }

      template<typename KeyT, typename = std::enable_if_t<!std::is_void_v<KeyT>>>
      inline int priomap(const KeyT& key) const {
        if constexpr (priomap_is_std_function) {
          if (m_policy.priomap) return m_policy.priomap(key);
          else return m_default_priomap(key);
        } else {
          return m_policy.priomap(key);
        }
      }

      template<typename KeyT = Key, typename = std::enable_if_t<std::is_void_v<KeyT>>>
      inline int priomap() const {
        if constexpr (priomap_is_std_function) {
          if (m_policy.priomap) return m_policy.priomap();
          else return m_default_priomap();
        } else {
          return m_policy.priomap();
        }
      }

      template<typename KeyT, typename = std::enable_if_t<!std::is_void_v<KeyT>>>
      inline int inlinemap(const KeyT& key) const {
        if constexpr (inlinemap_is_std_function) {
          if (m_policy.inlinemap) return m_policy.inlinemap(key);
          else return m_default_inlinemap(key);
        } else {
          return m_policy.inlinemap(key);
        }
      }

      template<typename KeyT = Key, typename = std::enable_if_t<std::is_void_v<KeyT>>>
      inline int inlinemap() const {
        if constexpr (inlinemap_is_std_function) {
          if (m_policy.inlinemap) return m_policy.inlinemap();
          else return m_default_inlinemap();
        } else {
          return m_policy.inlinemap();
        }
      }

      template<typename ProcMap>
      void set_procmap(ProcMap&& pm) {
        static_assert(std::is_assignable_v<decltype(TTPolicyImpl::procmap), ProcMap>,
                      "Cannot set process map on compile-time policy property!");
        m_policy.procmap = std::forward<ProcMap>(pm);
      }

      template<typename KeyMap>
      void set_keymap(KeyMap&& pm) {
        set_procmap(std::forward<KeyMap>(pm));
      }

      template<typename PrioMap>
      void set_priomap(PrioMap&& pm) {
        static_assert(std::is_assignable_v<decltype(TTPolicyImpl::priomap), PrioMap>,
                      "Cannot set process map on compile-time policy property!");
        m_policy.priomap = std::forward<PrioMap>(pm);
      }

      template<typename InlineMap>
      void set_inlinemap(InlineMap&& pm) {
        static_assert(std::is_assignable_v<decltype(TTPolicyImpl::inlinemap), InlineMap>,
                      "Cannot set process map on compile-time policy property!");
        m_policy.inlinemap = std::forward<InlineMap>(pm);
      }

    };
  } // namespace detail

  /**
   * Helper function to create a TT policy from arbitrary function objects.
   * The order of callables is
   *   1) Process map
   *   2) Priority map
   *   3) Inline map
   *
   * Any mapping function provided to make_policy cannot be changed at runtime.
   * All other mapping functions are defaulted and can be set dynamically.
   *
   * Example use:
   *
   * ttg::make_policy(
   *   // Process map: round robin on field i of key
   *   [&](const Key& key){ return key.i % world.size(); },
   *   // Priority map: use key field p as priority
   *   [&](const Key& key){ return key.p; },
   *   // Inline map: never inline
   *   [&](const Key& key){ return 0; });
   *
   * \sa TTPolicy
   */
  template<typename Fn, typename ...Fns>
  auto make_policy(Fn&& fn, Fns&& ...fns)
  {
    using key_t = detail::keymap_arg_t<std::decay_t<Fn>>;
    static_assert(std::is_assignable_v<int&, detail::keymap_ret_t<std::decay_t<Fn>>> &&
                  (std::is_assignable_v<int&, detail::keymap_ret_t<std::decay_t<Fns>>>&&...),
                  "All return types of mapping callables must be assignable to int.");
    if constexpr (!std::is_void_v<key_t>) {
      static_assert(std::is_invocable_v<Fn, std::add_lvalue_reference_t<std::add_const_t<key_t>>> &&
                    (std::is_invocable_v<Fns, std::add_lvalue_reference_t<std::add_const_t<key_t>>> && ...),
                    "All mapping callables must accept keys as const Key&.");
    }
    return TTPolicyBase<key_t, Fn, Fns...>(std::forward<Fn>(fn), std::forward<Fns>(fns)...);
  }

  /**
   * Helper function to create a dynamic TT policy. All mapping functions are
   * defaulted and can be set at runtime.
   */
  template<typename KeyT>
  auto make_policy()
  {
    return TTPolicyBase<KeyT>();
  }


  /**
   * Helper function to create a compile-time static TT policy from arbitrary
   * function objects.
   * The order of callables is
   *   1) Process map
   *   2) Priority map
   *   3) Inline map
   *
   * All mapping functions which were not provided explicitly will be statically
   * defaulted. No mapping function in this policy can be changed at runtime.
   *
   * Example use:
   *
   * ttg::make_policy(
   *   // Process map: round robin on field i of key
   *   [&](const Key& key){ return key.i % world.size(); },
   *   // Priority map: use key field p as priority
   *   [&](const Key& key){ return key.p; },
   *   // Inline map: never inline
   *   [&](const Key& key){ return 0; });
   *
   * \sa TTPolicy
   */
  template<typename Fn, typename ...Fns>
  auto make_static_policy(Fn&& fn, Fns&& ...fns)
  {
    using key_t = detail::keymap_arg_t<std::decay_t<Fn>>;
    static_assert(std::is_assignable_v<int&, detail::keymap_ret_t<std::decay_t<Fn>>> &&
                  (std::is_assignable_v<int&, detail::keymap_ret_t<std::decay_t<Fns>>>&&...),
                  "All return types of mapping callables must be assignable to int.");
    if constexpr (!std::is_void_v<key_t>) {
      static_assert(std::is_invocable_v<Fn, std::add_lvalue_reference_t<std::add_const_t<key_t>>> &&
                    (std::is_invocable_v<Fns, std::add_lvalue_reference_t<std::add_const_t<key_t>>> && ...),
                    "All mapping callables must accept keys as const Key&.");
    }
    return StaticTTPolicyBase<key_t, Fn, Fns...>(std::forward<Fn>(fn), std::forward<Fns>(fns)...);
  }

  /**
   * Helper function to create a TT policy with default implementations statically
   * compiled. No policy mapping function can be changed at runtime.
   */
  template<typename KeyT>
  auto make_static_policy()
  {
    return StaticTTPolicyBase<KeyT>();
  }


  namespace detail {

    /**
     * Generate traits to check policy objects for procmap(), priomap(), and inlinemap() members
     */
#define TTG_POLICY_CREATE_CHECK_FOR(_Pol)                                        \
    /* specialization that does the checking */                                  \
    template<typename KeyT, typename PolicyT>                                    \
    struct has_##_Pol {                                                          \
    private:                                                                     \
        template<typename T>                                                     \
        static constexpr auto check(T*)                                          \
        -> typename                                                              \
            std::is_same<                                                        \
                /* policy function take a key and return int */                  \
                decltype( std::declval<T>(). _Pol ( std::declval<KeyT>() ) ),    \
                int                                                              \
            >::type;                                                             \
        template<typename>                                                       \
        static constexpr std::false_type check(...);                             \
        typedef decltype(check<PolicyT>(0)) type;                                \
    public:                                                                      \
        static constexpr bool value = type::value;                               \
    };                                                                           \
    template<typename PolicyT>                                                   \
    struct has_##_Pol<void, PolicyT> {                                           \
    private:                                                                     \
        template<typename T>                                                     \
        static constexpr auto check(T*)                                          \
        -> typename                                                              \
            std::is_same<                                                        \
                /* policy function for void simply return int */                 \
                decltype( std::declval<T>(). _Pol ( ) ),                         \
                int                                                              \
            >::type;                                                             \
        template<typename>                                                       \
        static constexpr std::false_type check(...);                             \
        typedef decltype(check<PolicyT>(0)) type;                                \
    public:                                                                      \
        static constexpr bool value = type::value;                               \
    };                                                                           \
    template<typename KeyT, typename PolicyT>                                    \
    constexpr const bool has_##_Pol ## _v = has_ ## _Pol<KeyT, PolicyT>::value;

    TTG_POLICY_CREATE_CHECK_FOR(procmap);
    TTG_POLICY_CREATE_CHECK_FOR(priomap);
    TTG_POLICY_CREATE_CHECK_FOR(inlinemap);

    /** Whether PolicyT is a valid policy object using KeyT */
    template<typename KeyT, typename PolicyT>
    struct is_policy {
      static constexpr bool value = has_procmap_v<KeyT, PolicyT> &&
                                    has_priomap_v<KeyT, PolicyT> &&
                                    has_inlinemap_v<KeyT, PolicyT>;
    };

    /** Whether PolicyT is a valid policy object using KeyT */
    template<typename KeyT, typename PolicyT>
    constexpr const bool is_policy_v = is_policy<KeyT, PolicyT>::value;

    /* sanity base check */
    static_assert(is_policy_v<int, ttg::TTPolicyBase<int>>);
    static_assert(is_policy_v<int, ttg::detail::TTPolicyWrapper<int, ttg::TTPolicyBase<int>>>);

    static_assert(is_policy_v<void, ttg::TTPolicyBase<void>>);
    static_assert(is_policy_v<void, ttg::detail::TTPolicyWrapper<void, ttg::TTPolicyBase<void>>>);
  } // namespace detail


} // namespace ttg

#endif // TTG_POLICIES_H
