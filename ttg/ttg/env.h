//
// Created by Eduard Valeyev on 5/20/25.
//

#ifndef TTG_ENV_H
#define TTG_ENV_H

#include <umpire_cxx_allocator.hpp>

#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/SizeLimiter.hpp>

#include <cassert>
#include <memory>
#include <mutex>

namespace ttg {

  /**
   * Env maintains the runtime environment of TTG,
   * such as device artifacts, memory allocators, etc.
   *
   * \note this is a Singleton
   */
  class Env {
   public:
    ~Env() {
    }

    Env(const Env&) = delete;
    Env(Env&&) = delete;
    Env& operator=(const Env&) = delete;
    Env& operator=(Env&&) = delete;

    /// access the singleton instance; if not initialized will be
    /// initialized via Env::initialize() with the default params
    static std::unique_ptr<Env>& instance() {
      if (!instance_accessor()) {
        initialize();
      }
      return instance_accessor();
    }

    // clang-format off
    /// initialize the instance using explicit params
    /// \param world the world to use for initialization
    /// \param page_size memory added to the pools supporting `this->um_allocator()`, `this->device_allocator()`, and `this->pinned_allocator()` in chunks of at least
    ///                  this size (bytes) [default=2^25]
    /// \param pinned_alloc_limit the maximum total amount of memory (in bytes) that
    ///        allocator returned by `this->pinned_allocator()` can allocate [default=2^40]
    // clang-format on
    static void initialize(const std::uint64_t page_size = (1ul << 25),
                           const std::uint64_t pinned_alloc_limit = (1ul << 40)) {
      static std::mutex mtx;  // to make initialize() reentrant
      std::scoped_lock lock{mtx};
      // only the winner of the lock race gets to initialize
      if (instance_accessor() == nullptr) {

        // uncomment to debug umpire ops
        //
        //      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
        //          umpire::util::message::Debug);

        auto& rm = umpire::ResourceManager::getInstance();

        // turn off Umpire introspection for non-Debug builds
#ifndef NDEBUG
        constexpr auto introspect = true;
#else
        constexpr auto introspect = false;
#endif

        // allocate pinned_alloc_limit in pinned memory
        auto pinned_size_limited_alloc =
            rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
                "SizeLimited_PINNED", rm.getAllocator("PINNED"),
                pinned_alloc_limit);
        auto pinned_dynamic_pool =
            rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
                "QuickPool_SizeLimited_PINNED", pinned_size_limited_alloc,
                /* first_minimum_pool_allocation_size = */ 0,
                /* next_minimum_pool_allocation_size = */ page_size,
                /* alignment */ 16);

        auto env = std::unique_ptr<Env>(new Env(pinned_dynamic_pool));
        instance_accessor() = std::move(env);
      }
    }

    /// @return an Umpire allocator that allocates from a
    ///         pinned memory pool
    /// @warning this is not a thread-safe allocator, should be only used when
    ///          wrapped into umpire_based_allocator_impl
    auto& pinned_allocator() { return pinned_allocator_; }

    // clang-format off
    /// @return the max actual amount of memory held by pinned_allocator()
    /// @details returns the value provided by `umpire::strategy::QuickPool::getHighWatermark()`
    /// @note if there is only 1 Umpire allocator using PINNED memory this should be identical to the value returned by `umpire::ResourceManager::getInstance().getAllocator("PINNED").getHighWatermark()`
    // clang-format on
    std::size_t pinned_allocator_getActualHighWatermark() {
      assert(dynamic_cast<::umpire::strategy::QuickPool*>(
                    pinned_allocator_.getAllocationStrategy()) != nullptr);
      return dynamic_cast<::umpire::strategy::QuickPool*>(
                 pinned_allocator_.getAllocationStrategy())
          ->getActualHighwaterMark();
    }

   protected:
    Env(::umpire::Allocator pinned_alloc)
        : pinned_allocator_(pinned_alloc) {}

   private:
    // allocates from a dynamic, size-limited pinned memory pool
    // N.B. not thread safe, so must be wrapped into umpire_based_allocator_impl
    ::umpire::Allocator pinned_allocator_;

    inline static std::unique_ptr<Env>& instance_accessor() {
      static std::unique_ptr<Env> instance_{nullptr};
      return instance_;
    }
  };  // class Env

  namespace umpire {

    struct pinned_allocator_getter {
      inline ::umpire::Allocator& operator()() { return Env::instance()->pinned_allocator(); }
    };

  }

  /// pooled thread-safe pinned host memory allocator for device computing
  template <typename T>
  using pinned_allocator_t = ::umpire::default_init_allocator<
      T, ::umpire::allocator<T, ::umpire::detail::MutexLock<::ttg::Env>,
                             umpire::pinned_allocator_getter>>;

}  // namespace ttg

#endif  // TTG_ENV_H
