#ifndef TTG_REGION_PROBE_H
#define TTG_REGION_PROBE_H

#include <string>
#include <vector>
#include <cassert>

#include <parsec/profiling.h>

namespace ttg {

  namespace detail {

    enum region_probe_types {
      TTG_REGION_PROBE_USER = 0,
      TTG_REGION_PROBE_TASKS = 1,
      TTG_REGION_PROBE_INTERNAL = 2
    };

    template<region_probe_types ProbeType>
    struct ttg_enable_region_probe : std::false_type
    { };

#if defined(TTG_ENABLE_USER_PROBES)
    template<>
    struct ttg_enable_region_probe<TTG_REGION_PROBE_USER> : std::true_type
    { };
#endif

#if defined(TTG_ENABLE_TASK_PROBES)
    template<>
    struct ttg_enable_region_probe<TTG_REGION_PROBE_TASKS> : std::true_type
    { };
#endif

#if defined(TTG_ENABLE_INTERNAL_PROBES)
    template<>
    struct ttg_enable_region_probe<TTG_REGION_PROBE_INTERNAL> : std::true_type
    { };
#endif

    template<region_probe_types RegionType, bool Enabler = ttg_enable_region_probe<RegionType>::value>
    struct region_probe {
    private:
      int enter_, exit_;
      bool initialized = false;

      using deferred_inits_t = std::vector<std::pair<region_probe<RegionType>*, std::string>>;

      static deferred_inits_t& deferred_inits() {
        static deferred_inits_t di;
        return di;
      }

      static bool& defer_inits() {
        static bool v = true;
        return v;
      };

    public:

      static void register_deferred_probes()
      {
        if (defer_inits()) {
          for (auto&& it : deferred_inits()) {
            it.first->init(it.second.c_str());
          }
          deferred_inits().clear();
          defer_inits() = false;
        }
      }

      region_probe()
      { }

      region_probe(const char *name)
      {
        if (defer_inits()) {
          deferred_inits().emplace_back(this, name);
        } else {
          init(name);
        }
      }

      region_probe(const std::string& name) : region_probe(name.c_str())
      { }

      void init(const char *name) {
        assert(!initialized);
        if (!initialized) {
          parsec_profiling_add_dictionary_keyword(name, "#000000", 0, "", &enter_, &exit_);
          initialized = true;
        }
      }

      void init(const std::string& name) {
        init(name.c_str());
      }

      void enter() {
        assert(initialized);
        parsec_profiling_ts_trace(enter_, 0, PROFILE_OBJECT_ID_NULL, NULL);
      }

      void exit() {
        parsec_profiling_ts_trace(exit_, 0, PROFILE_OBJECT_ID_NULL, NULL);
      }
    };

    template<region_probe_types RegionType>
    struct region_probe<RegionType, false>
    {
      static void register_deferred_probes()
      { }

      region_probe()
      { }

      region_probe(const char *)
      { }

      region_probe(const std::string&)
      { }

      void init(const char *)
      { }

      void init(const std::string&)
      { }

      void enter()
      { }

      void exit()
      { }
    };

    template<region_probe_types RegionType>
    struct region_probe_event
    {
    private:
      region_probe<RegionType>& probe_;

    public:
      region_probe_event(region_probe<RegionType>& probe) : probe_(probe)
      {
        probe_.enter();
      }

      ~region_probe_event()
      {
        probe_.exit();
      }
    };

  } // namespace detail

  /**
   * TTG user probe that allows users to define custom probes that
   * are inserted into PaRSEC traces.
   * \see ttg::detail::region_probe for details.
   *
   * The probe may be defined statically with a name and TTG will take care of
   * proper initialization during \ref ttg_initialize.
   * Alternatively, probes can be created after \ref ttg_initialize was called,
   * either with our without a name. In the latter case, the probe remains
   * uninitialized until it is unitilized using the \c init() member function.
   *
   * Once initialized, a the member functions \c enter and \c exit can be
   * used to signal the begin and end of a region. Note that it is the users
   * responsibility to ensure proper balancing of enter and exit events.
   *
   * User probes are disabled by default. Compile with \c -DTTG_ENABLE_USER_PROBES=1
   * to enable them.
   *
   * NOTE: probes must be defined in the same order on all processes!
   *
   */
  using user_probe = detail::region_probe<detail::TTG_REGION_PROBE_USER>;

  /**
   * A scoped user probe event. Upon construction, the \c enter of the
   * \sa user_probe will be called. Once the event goes out of scope,
   * the probe's \c exit will be called.
   */
  using user_probe_event = detail::region_probe_event<detail::TTG_REGION_PROBE_USER>;

} // namespace ttg


#endif // TTG_REGION_PROBE_H
