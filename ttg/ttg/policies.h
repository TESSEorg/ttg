
#ifndef TTG_POLICIES_H
#define TTG_POLICIES_H

#include "ttg/base/keymap.h"
#include "ttg/util/meta.h"
#include "ttg/world.h"

namespace ttg {

  /**
   * The default policy for process maps that allows dynamically
   * setting the mapping function.
   *
   * Derive from this class to acquire the default behavior in your
   * custom policy class.
   */
  template<typename Key>
  struct TTProcmapPolicy {

    using procmap_t = typename ttg::meta::detail::keymap_t<Key>;

    procmap_t m_procmap;

    /**
     * The process mapping. Defaults to round-robin process mapping.
     *
     * This function may not be overriden.
     *
     * \sa ttg::detail::default_keymap
     */
    int procmap(const Key& key) {
      return m_procmap(key);
    }

    virtual void rebind_procmap(ttg::World& world) final {
      m_procmap = ttg::detail::default_keymap<Key>(world);
    }

    /**
     * Dynamically override the default process map by setting the mapping callback.
     */
    template<typename ProcMap>
    int set_procmap(ProcMap&& pm) {
      m_procmap = std::forward<ProcMap>(pm);
    }

    /**
     * Query the dynamic mapping function.
     */
    const procmap_t& get_procmap() const {
      return m_procmap;
    }

  };


  /**
   * Specialization of \sa ttg::TTProcmapPolicy for key type \c void.
   */
  template<>
  struct TTProcmapPolicy<void> {

    using procmap_t = typename ttg::meta::detail::keymap_t<void>;

    procmap_t m_procmap;

    /**
     * The process mapping. Defaults to round-robin process mapping.
     *
     * This function may not be overriden.
     *
     * \sa ttg::detail::default_keymap
     */
    int procmap() const {
      return m_procmap();
    }

    virtual void rebind_procmap(ttg::World& world) final {
      m_procmap = ttg::detail::default_keymap<void>(world);
    }

    /**
     * Dynamically override the default process map by setting the mapping callback.
     */
    template<typename ProcMap>
    int set_procmap(ProcMap&& pm) {
      m_procmap = std::forward<ProcMap>(pm);
    }

    /**
     * Query the dynamic mapping function.
     */
    const procmap_t& get_procmap() const {
      return m_procmap;
    }
  };



  /**
   * The default policy for priority maps that allows dynamically
   * setting the mapping function.
   *
   * Derive from this class to acquire the default behavior in your
   * custom policy class.
   */
  template<typename Key>
  struct TTPriomapPolicy {

    using priomap_t = typename ttg::meta::detail::keymap_t<Key>;

    priomap_t m_priomap;

    /**
     * The priority mapping. Defaults to priority 0.
     *
     * This function may not be overriden.
     *
     * \sa ttg::detail::default_keymap
     */
    int priomap(const Key& key) const {
      return m_priomap(key);
    }

    /**
     * Dynamically override the default priority map by setting the mapping callback.
     */
    template<typename PrioMap>
    int set_priomap(PrioMap&& pm) {
      m_priomap = std::forward<PrioMap>(pm);
    }

    /**
     * Query the dynamic mapping function.
     */
    const priomap_t& get_priomap() const {
      return m_priomap;
    }
  };

  /**
   * Specialization of \sa ttg::TTPriomapPolicy for key type \c void.
   */
  template<>
  struct TTPriomapPolicy<void> {

    using priomap_t = typename ttg::meta::detail::keymap_t<void>;

    priomap_t m_priomap;

    /**
     * The priority mapping. Defaults to priority 0.
     *
     * This function may not be overriden.
     *
     * \sa ttg::detail::default_keymap
     */
    int priomap() const {
      return m_priomap();
    }

    /**
     * Dynamically override the default priority map by setting the mapping callback.
     */
    template<typename PrioMap>
    int set_priomap(PrioMap&& pm) {
      m_priomap = std::forward<PrioMap>(pm);
    }

    /**
     * Query the dynamic mapping function.
     */
    const priomap_t& get_priomap() const {
      return m_priomap;
    }
  };


  /**
   * The default policy for priority maps that allows dynamically
   * setting the mapping function.
   *
   * Derive from this class to acquire the default behavior in your
   * custom policy class.
   */
  template<typename Key>
  struct TTInlinePolicy {

    using inlinemap_t = typename ttg::meta::detail::keymap_t<Key>;

    inlinemap_t m_inlinemap = [](){ return 0; };

    /**
     * The inline mapping. Defaults to no inlining.
     *
     * Whether a task can be executed inline, i.e., without dispatching the
     * task to a scheduler first. The task will be executed in the send or
     * broadcast call once all inputs have been satisfied. The returned value
     * denotes the maximum recirsion depth, i.e., how many tasks may be executed
     * inline consecutively. Zero denotes no inlining. This is the default.
     *
     * This function may not be overriden.
     */
    int inlinemap(const Key& key) const {
      return m_inlinemap(key);
    }

    /**
     * Dynamically override the default inlining map by setting the mapping callback.
     */
    template<typename InlineMap>
    int set_priomap(InlineMap&& im) {
      m_inlinemap = std::forward<InlineMap>(im);
    }

    /**
     * Query the dynamic mapping function.
     */
    const inlinemap_t& get_inlinemap() const {
      return m_inlinemap;
    }
  };

  /**
   * Specialization of \sa ttg::TTInlinePolicy for key type \c void.
   */
  template<>
  struct TTInlinePolicy<void> {

    using inlinemap_t = typename ttg::meta::detail::keymap_t<void>;
    inlinemap_t m_inlinemap = [](){ return 0; };

    /**
     * The inline mapping. Defaults to no inlining.
     *
     * This function may not be overriden.
     *
     * \sa ttg::detail::default_keymap
     */
    virtual int inlinemap() final {
      return m_inlinemap();
    }

    /**
     * Dynamically override the default inlining map by setting the mapping callback.
     */
    template<typename InlineMap>
    int set_priomap(InlineMap&& im) {
      m_inlinemap = std::forward<InlineMap>(im);
    }

    /**
     * Query the dynamic mapping function.
     */
    const inlinemap_t& get_inlinemap() {
      return m_inlinemap;
    }
  };

  /**
   * Policy class used to control aspects of TT execution:
   *  * Process mapping: maps a key identifying a task to a process to run on.
   *  * Priority mapping: assigns a priority (positive integer) to a task identified by a key.
   *                      Higher values increase the task's priority.
   *  * Inline mapping: whether a task can be executed inline, i.e., without dispatching the
   *                    task to a scheduler first. The task will be executed in the send or
   *                    broadcast call. The returned value denotes the maximum recirsion depth,
   *                    i.e., how many tasks may be executed inline consecutively. Zero denotes
   *                    no inlining. This is the default.
   *
   * Custom policy classes can be created by implementing the \c procmap, \c priomap, and
   * \c inlinemap functions. Defaults for the various mappings can be obtained by
   * inheriting from \c TTProcmapPolicy, \c TTPriomapPolicy, or \c TTInlinePolicy.
   *
   * This policy is composed of the default policies and may itself act as base class.
   * If inheriting from \c TTPolicy and the \c priomap, \c procmap, or \c inlinemap are
   * overriden then TTG will not allow setting the respective dynamic mapping function on a TT.
   * Inheriting from TTPolicy allows for custom policies to stay forward compatible
   * by including support for future policy extensions. However, inheriting from
   * \c TTProcmapPolicy, \c TTPriomapPolicy, \c TTInlinePolicy, or \c TTPolicy
   * will include the base class infrastructure that may not be needed in the
   * custom policy implementation.
   *
   * \sa ttg::TTProcmapPolicy
   * \sa ttg::TTPriomapPolicy
   * \sa ttg::TTInlinePolicy
   *
   */
  template<typename Key>
  struct TTPolicy
  : public TTProcmapPolicy<Key>, TTPriomapPolicy<Key>, TTInlinePolicy<Key>
  { };

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
             typename Enabler = std::enable_if_t<is_default_procmap &&
                                                 is_default_priomap &&
                                                 is_default_inlinemap>>
    TTPolicyBase()
    : TTPolicyBase(ttg::meta::detail::keymap_t<Key>(ttg::detail::default_keymap<Key>()),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_priomap<Key>()),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_inlinemap<Key>()))
    { }

    template<typename ProcMap_,
             typename PrioMap_ = PrioMap,
             typename InlineMap_ = InlineMap,
             typename Enabler = std::enable_if_t<is_default_priomap && is_default_inlinemap>>
    TTPolicyBase(ProcMap_&& procmap)
    : TTPolicyBase(std::forward<ProcMap_>(procmap),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_priomap<Key>()),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_inlinemap<Key>()))
    { }

    template<typename ProcMap_,
             typename PrioMap_,
             typename InlineMap_ = InlineMap,
             typename Enabler = std::enable_if_t<is_default_inlinemap>>
    TTPolicyBase(ProcMap_&& procmap, PrioMap_&& priomap)
    : TTPolicyBase(std::forward<ProcMap_>(procmap),
                   std::forward<PrioMap_>(priomap),
                   ttg::meta::detail::keymap_t<Key>(ttg::detail::default_inlinemap<Key>()))
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
    inline
    void rebind(ttg::World& world) {
      if constexpr (is_default_procmap) {
        procmap = ttg::detail::default_keymap<Key>(world);
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


  namespace detail {
    template<typename Key, typename ProcMap, typename PrioMap, typename InlineMap>
    struct TTPolicyWrapper {
      ProcMap procmap;
      PrioMap priomap;
      InlineMap inlinemap;

      template<typename ProcMap_, typename PrioMap_, typename InlineMap_>
      TTPolicyWrapper(ProcMap_&& procmap = ttg::detail::default_keymap<Key>(),
                      PrioMap_&& priomap = ttg::detail::default_priomap<Key>(),
                      InlineMap_&& im = ttg::detail::default_inlinemap<Key>())
      : procmap(std::forward<ProcMap_>(procmap)),
        priomap(std::forward<PrioMap_>(priomap)),
        inlinemap(std::forward<InlineMap_>(im))
      { }

      auto get_procmap() {
        return procmap;
      }

      auto get_priomap() {
        return priomap;
      }

      auto get_inlinemap() {
        return inlinemap;
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

} // namespace ttg

#endif // TTG_POLICIES_H
