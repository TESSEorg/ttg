#ifndef TTG_BASE_WORLD_H
#define TTG_BASE_WORLD_H

#include <memory>
#include <list>
#include <set>
#include <future>
#include <iostream>
#include <cassert>

#include "ttg/base/op.h"

namespace ttg {

  namespace base {
      // forward decl
      class WorldImplBase;
  }

  /* forward declaration */
  namespace detail {

      /* TODO: how should the MADNESS and PaRSEC init/finalize play together? */
      void register_world(ttg::base::WorldImplBase& world);
      void deregister_world(ttg::base::WorldImplBase& world);
      void destroy_worlds(void);

  } //namespace detail

  namespace base {

      /// Base class for implementation-specific Worlds
      class WorldImplBase {

      private:

        std::list<ttg::OpBase*> m_op_register;
        std::set<std::shared_ptr<std::promise<void>>> m_statuses;
        std::set<std::shared_ptr<void>> m_ptrs;
        bool m_is_valid = true;

      protected:

        void mark_invalid() {
          m_is_valid = false;
        }

        virtual void fence_impl(void) = 0;

        void release_ops(void) {
          while (!m_op_register.empty()) {
              std::cout << "Destroying OpBase " << (*m_op_register.begin()) << std::endl;
              (*m_op_register.begin())->release();
          }
        }

      public:
        WorldImplBase(void)
        { }

        virtual
        ~WorldImplBase(void)
        {
          m_is_valid = false;
        }

        virtual int size(void) const = 0;

        virtual int rank(void) const = 0;

        virtual void destroy(void) = 0;

        template<typename T>
        void register_ptr(const std::shared_ptr<T>& ptr)
        {
            m_ptrs.insert(ptr);
        }

        void register_status(const std::shared_ptr<std::promise<void>>& status_ptr)
        {
            m_statuses.insert(status_ptr);
        }

        void fence(void)
        {
            fence_impl();
            for (auto &status: m_statuses) {
              status->set_value();
            }
            m_statuses.clear();  // clear out the statuses
        }

        virtual void execute()
        { }

        void register_op(ttg::OpBase* op) {
            // TODO: do we need locking here?
            m_op_register.push_back(op);
        }

        void deregister_op(ttg::OpBase* op) {
            // TODO: do we need locking here?
            m_op_register.remove(op);
        }

        bool is_valid(void) const {
            return m_is_valid;
        }
      };

      /**
       * Slim wrapper around World implementation objects
       * This wrapper should be passed by value, not by reference, to avoid
       * lifetime issues of the world object.
       */
      template<typename WorldImplT>
      class World {
      private:
          std::shared_ptr<ttg::base::WorldImplBase> m_impl;
      public:

          World(void)
          { }


          World(std::shared_ptr<ttg::base::WorldImplBase> world_impl) : m_impl(world_impl)
          { }

          /* Defaulted copy ctor */
          World(const World& other) = default;

          /* Defaulted move ctor */
          World(World&& other) = default;

          ~World()
          { }

          /* Defaulted copy assignment */
          World& operator=(const World& other) = default;

          /* Defaulted move assignment */
          World& operator=(World&& other) = default;

          /* Get the number of ranks in this world */
          int size() const
          {
              assert(is_valid());
              return m_impl->size();
          }

          /* Get the current rank in this world */
          int rank() const
          {
              assert(is_valid());
              return m_impl->rank();
          }

          /* Returns true if the World instance is valid, i.e., if it has a valid
           * pointer to a World implementation object */
          bool is_valid(void) const
          {
              return static_cast<bool>(m_impl);
          }

          /* Get an unmanaged reference to the world implementation */
          WorldImplT& impl(void)
          {
              assert(is_valid());
              return *reinterpret_cast<WorldImplT*>(m_impl.get());
          }

          const WorldImplT& impl(void) const
          {
              assert(is_valid());
              return *reinterpret_cast<WorldImplT*>(m_impl.get());
          }
      };

  } // namespace base

} // namespace ttg
#endif // TTG_BASE_WORLD_H
