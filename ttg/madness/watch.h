//
// Created by Eduard Valeyev on 2019-04-05.
//

#ifndef TTG_WATCH_H
#define TTG_WATCH_H

namespace madness {
  namespace ttg {
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
          for (auto t = 0ul; t != madness::ThreadPool::size(); ++t) {
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

  }
}
#endif  // TTG_WATCH_H
