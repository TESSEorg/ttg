#ifndef TTG_BASE_WORLD_H
#define TTG_BASE_WORLD_H

#include <cassert>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <set>

#include "ttg/base/tt.h"

namespace ttg {

  namespace base {
    // forward decl
    class WorldImplBase;
  }  // namespace base

  /* forward declaration */
  namespace detail {

    /* TODO: how should the MADNESS and PaRSEC init/finalize play together? */
    void register_world(ttg::base::WorldImplBase& world);
    void deregister_world(ttg::base::WorldImplBase& world);
    void destroy_worlds(void);

  }  // namespace detail

  namespace base {

    /// Base class for implementation-specific Worlds
    class WorldImplBase {
     private:
      template <typename T>
      std::function<void(void*)> make_deleter() {
        return {[](void* p) { delete static_cast<T*>(p); }};
      }

      std::list<ttg::TTBase*> m_op_register;
      std::vector<std::shared_ptr<std::promise<void>>> m_statuses;
      std::vector<std::function<void()>> m_callbacks;
      std::vector<std::shared_ptr<void>> m_ptrs;
      std::vector<std::unique_ptr<void, std::function<void(void*)>>> m_unique_ptrs;
      int world_size;
      int world_rank;
      bool m_is_valid = true;

     protected:
      void mark_invalid() { m_is_valid = false; }

      virtual void fence_impl(void) = 0;

      void release_ops(void) {
        while (!m_op_register.empty()) {
          (*m_op_register.begin())->release();
        }
      }

    protected:
      WorldImplBase(int size, int rank)
      : world_size(size), world_rank(rank)
      {}

    public:
      virtual ~WorldImplBase(void) { m_is_valid = false; }

      /**
       * Returns the number of processes that belong this World.
       */
      int size() {
        return world_size;
      }

      /**
       * Returns the rank of the calling process in this World.
       */
      int rank() {
        return world_rank;
      }

      virtual void destroy(void) = 0;

      template <typename T>
      void register_ptr(const std::shared_ptr<T>& ptr) {
        m_ptrs.emplace_back(ptr);
      }

      template <typename T>
      void register_ptr(std::unique_ptr<T>&& ptr) {
        m_unique_ptrs.emplace_back(ptr.release(), make_deleter<T>());
      }

      void register_status(const std::shared_ptr<std::promise<void>>& status_ptr) {
        m_statuses.emplace_back(status_ptr);
      }

      template <typename Callback>
      void register_callback(Callback&& callback) {
        m_callbacks.emplace_back(callback);
      }


      /**
       * Wait for all tasks in this world to complete execution.
       * This is a synchronizing call, even if no active tasks exist
       * (i.e., fence() behaves as a barrier).
       */
      void fence(void) {
        fence_impl();
        for (auto& status : m_statuses) {
          status->set_value();
        }
        m_statuses.clear();  // clear out the statuses
        for (auto&& callback : m_callbacks) {
          callback();
        }
        m_callbacks.clear();  // clear out the statuses
      }

      /**
       * Start the execution of tasks in this world. The call to execute()
       * will return immediately, i.e., it will not wait for all tasks
       * to complete executing.
       *
       * \sa fence
       */
      virtual void execute() {}


      /**
       * Register a TT with this world. All registered TTs will be
       * destroyed during destruction of this world.
       */
      void register_op(ttg::TTBase* op) {
        // TODO: do we need locking here?
        m_op_register.push_back(op);
      }

      /**
       * Deregister a TT from this world. TTs deregister themselves during
       * destruction to avoid dangling references.
       */
      void deregister_op(ttg::TTBase* op) {
        // TODO: do we need locking here?
        m_op_register.remove(op);
      }


      /**
       * Whether this world is valid. A word is marked as invalid during destruction
       * and/or finalization of TTG.
       */
      bool is_valid(void) const { return m_is_valid; }

      virtual void final_task() {}
    };

    /**
     * Slim wrapper around World implementation objects
     * This wrapper should be passed by value, not by reference, to avoid
     * lifetime issues of the world object.
     */
    template <typename WorldImplT>
    class World {
     private:
      std::shared_ptr<ttg::base::WorldImplBase> m_impl;

     public:
      World(void) {}

      World(std::shared_ptr<ttg::base::WorldImplBase> world_impl) : m_impl(world_impl) {}

      /* Defaulted copy ctor */
      World(const World& other) = default;

      /* Defaulted move ctor */
      World(World&& other) = default;

      ~World() {}

      /* Defaulted copy assignment */
      World& operator=(const World& other) = default;

      /* Defaulted move assignment */
      World& operator=(World&& other) = default;

      /* Get the number of ranks in this world */
      int size() const {
        assert(is_valid());
        return m_impl->size();
      }

      /* Get the current rank in this world */
      int rank() const {
        assert(is_valid());
        return m_impl->rank();
      }

      /* Returns true if the World instance is valid, i.e., if it has a valid
       * pointer to a World implementation object */
      bool is_valid(void) const { return static_cast<bool>(m_impl); }

      virtual void final_task() {}

      /* Get an unmanaged reference to the world implementation */
      WorldImplT& impl(void) {
        assert(is_valid());
        return *reinterpret_cast<WorldImplT*>(m_impl.get());
      }

      const WorldImplT& impl(void) const {
        assert(is_valid());
        return *reinterpret_cast<WorldImplT*>(m_impl.get());
      }
    };

  }  // namespace base

}  // namespace ttg
#endif  // TTG_BASE_WORLD_H
