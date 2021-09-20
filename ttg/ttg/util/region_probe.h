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

      /* Storage: {probe, name, info_length, converter} */
      using deferred_inits_t = std::vector<std::tuple<region_probe<RegionType>*, std::string, size_t, std::string>>;

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
            std::get<0>(it)->init(std::get<1>(it).c_str(), std::get<2>(it), std::get<3>(it).c_str());
          }
          deferred_inits().clear();
          defer_inits() = false;
        }
      }

      /**
       * Default constructor, does not initialize the probe.
       * The probe has to be initialized using \ref init before usage.
       */
      region_probe()
      { }

      /**
       * Create and initialize a probe with the given name.
       */
      region_probe(const char *name) : region_probe(name, 0, "")
      { }

      /**
       * Create and initialize a probe with the given name, the size of an info
       * object, and the converter description.
       * The info object matching the converter description may be passed to
       * enter and exit to provide additional information on the event in the
       * trace.
       * An example for a valid converter description would be:
       * "a{int32_t};b{int64_t}"
       * and the corresponding structure passed to enter/exit might look like
       * struct { int32_t a; int64_t b; };
       */
      region_probe(const char *name, size_t info_length, const char *converter)
      {
        if (defer_inits()) {
          deferred_inits().emplace_back(this, name, info_length, converter);
        } else {
          init(name, info_length, converter);
        }
      }

      region_probe(const std::string& name)
      : region_probe(name.c_str())
      { }

      region_probe(const std::string& name, size_t info_length, const char *converter)
      : region_probe(name.c_str(), info_length, converter)
      { }

      void init(const char *name) {
        init(name, 0, "");
      }

      void init(const char *name, size_t info_length, const char *converter) {
        assert(!initialized);
        if (!initialized) {
          parsec_profiling_add_dictionary_keyword(name, "#000000", info_length, converter, &enter_, &exit_);
          initialized = true;
        }
      }

      void init(const std::string& name) {
        init(name.c_str());
      }

      void init(const std::string& name, size_t info_length, const char *converter) {
        init(name.c_str(), info_length, converter);
      }

      void enter() {
        assert(initialized);
        parsec_profiling_ts_trace(enter_, 0, PROFILE_OBJECT_ID_NULL, NULL);
      }

      void exit() {
        parsec_profiling_ts_trace(exit_, 0, PROFILE_OBJECT_ID_NULL, NULL);
      }

      template<typename Arg>
      void enter(Arg&& arg) {
        assert(initialized);
        parsec_profiling_ts_trace(enter_, 0, PROFILE_OBJECT_ID_NULL, &arg);
      }

      template<typename Arg>
      void exit(Arg&& arg) {
        parsec_profiling_ts_trace(exit_, 0, PROFILE_OBJECT_ID_NULL, &arg);
      }
    };

    /* Fallback implementation if the probe was disabled */
    template<region_probe_types RegionType>
    struct region_probe<RegionType, false>
    {
      static void register_deferred_probes()
      { }

      region_probe()
      { }

      region_probe(const char *)
      { }

      region_probe(const char *, size_t, const char *)
      { }

      region_probe(const std::string&)
      { }

      region_probe(const std::string&, size_t, const char *)
      { }

      void init(const char *)
      { }

      void init(const char *, size_t, const char *)
      { }

      void init(const std::string& )
      { }

      void init(const std::string& , size_t , const char *)
      { }

      void enter()
      { }

      void exit()
      { }

      template<typename Arg>
      void enter(Arg&&)
      { }

      template<typename Arg>
      void exit(Arg&&)
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

      template<typename Arg>
      region_probe_event(region_probe<RegionType>& probe, Arg&& arg) : probe_(probe)
      {
        probe_.enter(std::forward<Arg>(arg));
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
