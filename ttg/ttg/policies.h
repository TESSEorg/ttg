
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
      using type = MapT;
    };

    template<typename Key>
    struct map_type<ttg::meta::detail::keymap<Key>> {
      using type = ttg::meta::detail::keymap_t<Key>;
    };

    template<typename T>
    using map_type_t = typename map_type<T>::type;

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
           typename ProcMap = typename ttg::meta::detail::keymap<Key>,
           typename PrioMap = typename ttg::meta::detail::keymap<Key>,
           typename InlineMap = typename ttg::meta::detail::keymap<Key>>
  struct TTPolicyBase {

    using procmap_t = detail::map_type_t<ProcMap>;
    using priomap_t = detail::map_type_t<PrioMap>;
    using inlinemap_t = detail::map_type_t<InlineMap>;

    procmap_t procmap;
    priomap_t priomap;
    inlinemap_t inlinemap;

    static constexpr bool is_default_procmap = detail::is_default_keymap_v<ProcMap>;
    static constexpr bool is_default_priomap = detail::is_default_keymap_v<PrioMap>;
    static constexpr bool is_default_inlinemap = detail::is_default_keymap_v<InlineMap>;

    template<typename ProcMap_ = ProcMap,
             typename PrioMap_ = PrioMap,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<detail::is_default_keymap_v<ProcMap_> &&
                                         detail::is_default_keymap_v<PrioMap_> &&
                                         detail::is_default_keymap_v<InlineMap_>>>
    TTPolicyBase()
    : TTPolicyBase(ttg::meta::detail::keymap_t<Key>(ttg::detail::default_keymap_impl<Key>()),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_priomap_impl<Key>()),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_inlinemap_impl<Key>()))
    { }

    template<typename ProcMap_,
             typename PrioMap_ = PrioMap,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<detail::is_default_keymap_v<PrioMap_> &&
                                         detail::is_default_keymap_v<InlineMap_>>>
    TTPolicyBase(ProcMap_&& procmap)
    : TTPolicyBase(std::forward<ProcMap_>(procmap),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_priomap_impl<Key>()),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_inlinemap_impl<Key>()))
    { }

    template<typename ProcMap_,
             typename PrioMap_,
             typename InlineMap_ = InlineMap,
             typename = std::enable_if_t<detail::is_default_keymap_v<InlineMap_>>>
    TTPolicyBase(ProcMap_&& procmap, PrioMap_&& priomap)
    : TTPolicyBase(std::forward<ProcMap_>(procmap),
                   std::forward<PrioMap_>(priomap),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_inlinemap_impl<Key>()))
    { }

    template<typename ProcMap_, typename PrioMap_, typename InlineMap_>
    TTPolicyBase(ProcMap_&& procmap,
                 PrioMap_&& priomap,
                 InlineMap_&& im)
    : procmap(std::forward<ProcMap_>(procmap)),
      priomap(std::forward<PrioMap_>(priomap)),
      inlinemap(std::forward<InlineMap_>(im))
    { }

    /**
     * Rebind the default process map to the provided world, if the process map
     * is the default map.
     */
    template<typename WorldT>
    inline
    void rebind(const WorldT& world) {
      if constexpr (is_default_procmap) {
        procmap = ttg::detail::default_keymap_impl<Key>(world.size());
      }
    }

    const auto& get_procmap() const {
      return procmap;
    }

    const auto& get_priomap() const {
      return priomap;
    }

    const auto& get_inlinemap() const {
      return inlinemap;
    }
  };

  /**
   * Helper function to create a TT policy from arbitrary function objects.
   * The order of callables is
   *   1) Process map
   *   2) Priority map
   *   3) Inline map
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
  template<typename Key, typename ...Args>
  auto make_policy(Args&& ...args)
  {
    return TTPolicyBase<Key, Args...>(std::forward<Args>(args)...);
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
  } // namespace detail


} // namespace ttg

#endif // TTG_POLICIES_H
