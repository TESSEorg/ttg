#ifndef TTG_CONSTRAINT_H
#define TTG_CONSTRAINT_H

#include <functional>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <map>

#ifdef TTG_USE_BUNDLED_BOOST_CALLABLE_TRAITS
#include <ttg/external/boost/callable_traits.hpp>
#else
#include <boost/callable_traits.hpp>
#endif

namespace ttg {

  template<typename Key>
  struct ConstraintBase {
    using key_type = Key;
    using listener_t = std::function<void(const std::span<key_type>&)>;

    ConstraintBase()
    { }

    ConstraintBase(ConstraintBase&& cb)
    : m_listeners(std::move(cb.m_listeners))
    { }

    ConstraintBase(const ConstraintBase& cb)
    : m_listeners(cb.m_listeners)
    {}

    ConstraintBase& operator=(ConstraintBase&& cb) {
      m_listeners = std::move(cb.m_listeners);
    }
    ConstraintBase& operator=(const ConstraintBase& cb) {
      m_listeners = cb.m_listeners;
    }

    virtual ~ConstraintBase() = default;

    void add_listener(listener_t l, ttg::TTBase *tt) {
      auto g = this->lock_guard();
      m_listeners.insert_or_assign(tt, std::move(l));
    }

    void notify_listener(const std::span<key_type>& keys, ttg::TTBase* tt) {
      auto& release = m_listeners[tt];
      release(keys);
    }

  protected:

    auto lock_guard() {
      return std::lock_guard{m_mtx};
    }

  private:
    std::map<ttg::TTBase*, listener_t> m_listeners;
    std::mutex m_mtx;
  };

  template<typename Key,
           typename Ordinal = std::size_t,
           typename Compare = std::less<Ordinal>,
           typename Mapper = ttg::Void>
  struct SequencedKeysConstraint : public ConstraintBase<Key> {

    using key_type = std::conditional_t<ttg::meta::is_void_v<Key>, ttg::Void, Key>;
    using ordinal_type = Ordinal;
    using keymap_t = std::function<Ordinal(const key_type&)>;
    using compare_t = Compare;
    using base_t = ConstraintBase<Key>;

  private:
    struct sequence_elem_t {
      std::map<ttg::TTBase*, std::vector<key_type>> m_keys;

      sequence_elem_t() = default;

      void add_key(const key_type& key, ttg::TTBase* tt) {
        auto it = m_keys.find(tt);
        if (it == m_keys.end()) {
          m_keys.insert(std::make_pair(tt, std::vector<key_type>{key}));
        } else {
          it->second.push_back(key);
        }
      }
    };

    void release_next() {
      if (m_stopped) {
        // don't release tasks if we're stopped
        return;
      }
      // trigger the next sequence
      sequence_elem_t elem;
      {
        // extract the next sequence
        auto g = this->lock_guard();
        auto it = m_sequence.begin(); // sequence is ordered by ordinal
        if (it == m_sequence.end()) {
          return; // nothing to be done
        }
        m_current = it->first;
        elem = std::move(it->second);
        m_sequence.erase(it);
      }

      for (auto& seq : elem.m_keys) {
        // account for the newly active keys
        m_active.fetch_add(seq.second.size(), std::memory_order_relaxed);
        this->notify_listener(std::span<key_type>(seq.second.data(), seq.second.size()), seq.first);
      }
    }

    bool check_key_impl(const key_type& key, Ordinal ord, ttg::TTBase *tt) {
      if (!m_stopped) {
        if (m_order(ord, m_current)) {
          // key should be executed
          m_active.fetch_add(1, std::memory_order_relaxed);
          // reset current to the lower ordinal
          m_current = ord;
          return true;
        } else if (m_sequence.empty() && 0 == m_active.load(std::memory_order_relaxed)) {
          // there are no keys (active or blocked) so we execute to avoid a deadlock
          // we don't change the current ordinal because there may be lower ordinals coming in later
          m_active.fetch_add(1, std::memory_order_relaxed);
          return true;
        }
      }
      // key should be deferred
      auto g = this->lock_guard();
      if (!m_stopped && m_order(ord, m_current)) {
        // someone released this ordinal while we took the lock
        return true;
      }
      auto it = m_sequence.find(ord);
      if (it == m_sequence.end()) {
        auto [iter, success] = m_sequence.insert(std::make_pair(ord, sequence_elem_t{}));
        assert(success);
        it = iter;
      }
      it->second.add_key(key, tt);
      return false;
    }

    void complete_key_impl() {
      auto active = m_active.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (0 == active) {
        release_next();
      }
    }


  public:

    /**
     * Used for external key mapper.
     */
    SequencedKeysConstraint()
    : base_t()
    { }

    template<typename Mapper_, typename = std::enable_if_t<std::is_invocable_v<Mapper_, Key>, Mapper_>>
    SequencedKeysConstraint(Mapper_&& map)
    : base_t()
    , m_map(std::forward<Mapper_>(map))
    { }

    SequencedKeysConstraint(SequencedKeysConstraint&& skc)
    : base_t(std::move(skc))
    , m_sequence(std::move(skc.m_sequence))
    , m_active(skc.m_active.load(std::memory_order_relaxed))
    , m_current(std::move(skc.m_current))
    , m_map(std::move(skc.m_map))
    , m_order(std::move(skc.m_order))
    , m_stopped(skc.m_stopped)
    { }

    SequencedKeysConstraint(const SequencedKeysConstraint& skc)
    : base_t(skc)
    , m_sequence(skc.m_sequence)
    , m_active(skc.m_active.load(std::memory_order_relaxed))
    , m_current(skc.m_current)
    , m_map(skc.m_map)
    , m_order(skc.m_order)
    , m_stopped(skc.m_stopped)
    { }

    SequencedKeysConstraint& operator=(SequencedKeysConstraint&& skc) {
      base_t::operator=(std::move(skc));
      m_sequence = std::move(skc.m_sequence);
      m_active = skc.m_active.load(std::memory_order_relaxed);
      m_current = std::move(skc.m_current);
      m_map = std::move(skc.m_map);
      m_order = std::move(skc.m_order);
      m_stopped = skc.m_stopped;
    }
    SequencedKeysConstraint& operator=(const SequencedKeysConstraint& skc) {
      base_t::operator=(skc);
      m_sequence = skc.m_sequence;
      m_active = skc.m_active.load(std::memory_order_relaxed);
      m_current = skc.m_current;
      m_map = skc.m_map;
      m_order = skc.m_order;
      m_stopped = skc.m_stopped;
    }

    virtual ~SequencedKeysConstraint() = default;

    /* Check whether the key may be executed.
     * Returns true if the key may be executed.
     * Otherwise, returns false and  */
    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>, bool>
    check(const key_type& key, ttg::TTBase *tt) {
      ordinal_type ord = m_map(key);
      return check_key_impl(key, ord, tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>, bool>
    check(const key_type& key, Ordinal ord, ttg::TTBase *tt) {
      return check_key_impl(key, ord, tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>, bool>
    check(ttg::TTBase *tt) {
      return check_key_impl(ttg::Void{}, m_map(), tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>, bool>
    check(ordinal_type ord, ttg::TTBase *tt) {
      return check_key_impl(ttg::Void{}, ord, tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>>
    complete(const key_type& key, ttg::TTBase *tt) {
      complete_key_impl();
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>>
    complete(const key_type& key, Ordinal ord, ttg::TTBase *tt) {
      complete_key_impl();
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>>
    complete(Ordinal ord, ttg::TTBase *tt) {
      complete_key_impl();
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>>
    complete(ttg::TTBase *tt) {
      complete_key_impl();
    }

    /**
     * Stop all execution. Call \c start to resume.
     * This constraint is not stopped by default so calls to \c start
     * are only necessary if explictily stopped.
     */
    void stop() {
      m_stopped = true;
    }

    /**
     * Start execution.
     * This constraint is not stopped by default so calls to \c start
     * are only necessary if explictily stopped.
     */
    void start() {
      if (m_stopped) {
        m_stopped = false;
        release_next();
      }
    }


  private:
    std::map<ordinal_type, sequence_elem_t, compare_t> m_sequence;
    std::atomic<std::size_t> m_active;
    ordinal_type m_current;
    [[no_unique_address]]
    Mapper m_map;
    [[no_unique_address]]
    compare_t m_order;
    bool m_stopped = false;
  };

  // deduction guide: take type of first argument to Mapper as the key type
  // TODO: can we use the TTG callable_args instead?
  template<typename Mapper, typename = std::enable_if_t<std::is_invocable_v<Mapper, std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>>>>
  SequencedKeysConstraint(Mapper&&)
    -> SequencedKeysConstraint<
          std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>,
          std::decay_t<boost::callable_traits::return_type_t<Mapper>>,
          std::less<std::decay_t<boost::callable_traits::return_type_t<Mapper>>>,
          std::enable_if_t<std::is_invocable_v<Mapper, std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>>, Mapper>
          >;

  template<typename Key, typename Ordinal, typename Compare, typename Mapper>
  SequencedKeysConstraint(SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>&&)
    -> SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>;

  template<typename Key, typename Ordinal, typename Compare, typename Mapper>
  SequencedKeysConstraint(const SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>&)
    -> SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>;

  template<template<typename...> typename Constraint, typename... Args>
  auto make_shared_constraint(Args&&... args) {
    return std::make_shared<decltype(Constraint(std::forward<Args>(args)...))>(Constraint(std::forward<Args>(args)...));
  }

} // namespace ttg

#endif // TTG_CONSTRAINT_H