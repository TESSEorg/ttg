#ifndef TTG_CONSTRAINT_H
#define TTG_CONSTRAINT_H

#include <functional>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <map>

#include "ttg/util/span.h"

#ifdef TTG_USE_BUNDLED_BOOST_CALLABLE_TRAITS
#include <ttg/external/boost/callable_traits.hpp>
#else
#include <boost/callable_traits.hpp>
#endif

namespace ttg {

  template<typename Key>
  struct ConstraintBase {
    using key_type = Key;
    using listener_t = std::function<void(const ttg::span<key_type>&)>;

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

    void notify_listener(const ttg::span<key_type>& keys, ttg::TTBase* tt) {
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
           typename Compare = std::less_equal<Ordinal>,
           typename Mapper = ttg::Void>
  struct SequencedKeysConstraint : public ConstraintBase<Key> {

    using key_type = std::conditional_t<ttg::meta::is_void_v<Key>, ttg::Void, Key>;
    using ordinal_type = Ordinal;
    using keymap_t = std::function<Ordinal(const key_type&)>;
    using compare_t = Compare;
    using base_t = ConstraintBase<Key>;

  protected:
    struct sequence_elem_t {
      std::map<ttg::TTBase*, std::vector<key_type>> m_keys;

      sequence_elem_t() = default;
      sequence_elem_t(sequence_elem_t&&) = default;
      sequence_elem_t(const sequence_elem_t&) = default;
      sequence_elem_t& operator=(sequence_elem_t&&) = default;
      sequence_elem_t& operator=(const sequence_elem_t&) = default;

      void add_key(const key_type& key, ttg::TTBase* tt) {
        auto it = m_keys.find(tt);
        if (it == m_keys.end()) {
          m_keys.insert(std::make_pair(tt, std::vector<key_type>{key}));
        } else {
          it->second.push_back(key);
        }
      }
    };

    bool check_key_impl(const key_type& key, Ordinal ord, ttg::TTBase *tt) {
      if (!m_stopped) {
        if (m_order(ord, m_current)) {
          // key should be executed
          if (m_auto_release) { // only needed for auto-release
            m_active.fetch_add(1, std::memory_order_relaxed);
            // revert the current ordinal to the lower ordinal
            m_current = ord;
          }
          return true;
        } else if (m_sequence.empty() && m_auto_release && 0 == m_active.load(std::memory_order_relaxed)) {
          // there are no keys (active or blocked) so we execute to avoid a deadlock
          // we don't change the current ordinal because there may be lower ordinals coming in later
          // NOTE: there is a race condition between the check here and the increment above.
          //       This is mostly benign as it can lead to out-of-sequence released tasks.
          //       Avoiding this race would incur significant overheads.
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
      if (m_auto_release) {
        auto active = m_active.fetch_sub(1, std::memory_order_relaxed) - 1;
        if (0 == active) {
          release_next();
        }
      }
    }

    // used in the auto case
    void release_next() {
      if (this->m_stopped) {
        // don't release tasks if we're stopped
        return;
      }
      // trigger the next sequence
      sequence_elem_t elem;
      {
        // extract the next sequence
        auto g = this->lock_guard();
        auto it = this->m_sequence.begin(); // sequence is ordered by ordinal
        if (it == this->m_sequence.end()) {
          return; // nothing to be done
        }
        this->m_current = it->first;
        elem = std::move(it->second);
        this->m_sequence.erase(it);
      }

      for (auto& seq : elem.m_keys) {
        // account for the newly active keys
        this->m_active.fetch_add(seq.second.size(), std::memory_order_relaxed);
        this->notify_listener(ttg::span<key_type>(seq.second.data(), seq.second.size()), seq.first);
      }
    }


    // used in the non-auto case
    void release_next(ordinal_type ord, bool force_check = false) {
      if (this->m_stopped) {
        // don't release tasks if we're stopped but remember that this was released
        this->m_current = ord;
        return;
      }
      if (!force_check && m_order(ord, this->m_current)) {
        return; // already at the provided ordinal, nothing to be done
      }
      // trigger the next sequence(s) (m_sequence is ordered by ordinal)
      std::vector<sequence_elem_t> seqs;
      {
        auto g = this->lock_guard();
        // set current ordinal
        this->m_current = ord;
        {
          for (auto it = this->m_sequence.begin(); it != this->m_sequence.end();) {
            if (!this->m_order(it->first, this->m_current)) break;
            // extract the next sequence
            this->m_current = it->first;
            seqs.push_back(std::move(it->second));
            it = this->m_sequence.erase(it);
          }
        }
      }
      for (auto& elem : seqs) {
        for (auto& e : elem.m_keys) {
          // account for the newly active keys
          this->notify_listener(ttg::span<key_type>(e.second.data(), e.second.size()), e.first);
        }
      }
    }

  public:

    /**
     * Used for external key mapper.
     */
    SequencedKeysConstraint(bool auto_release = false)
    : base_t()
    , m_auto_release(auto_release)
    { }

    template<typename Mapper_>
    requires(std::is_invocable_v<Mapper_, Key>)
    SequencedKeysConstraint(Mapper_&& map, bool auto_release = false)
    : base_t()
    , m_map(std::forward<Mapper_>(map))
    , m_auto_release(auto_release)
    { }

    SequencedKeysConstraint(SequencedKeysConstraint&& skc) = default;

    SequencedKeysConstraint(const SequencedKeysConstraint& skc) = default;

    SequencedKeysConstraint& operator=(SequencedKeysConstraint&& skc) = default;

    SequencedKeysConstraint& operator=(const SequencedKeysConstraint& skc) = default;

    virtual ~SequencedKeysConstraint() = default;

    /* Check whether the key may be executed.
     * Returns true if the key may be executed.
     * Otherwise, returns false and  */
    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>, bool>
    check(const key_type& key, ttg::TTBase *tt) {
      ordinal_type ord = m_map(key);
      return this->check_key_impl(key, ord, tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>, bool>
    check(const key_type& key, Ordinal ord, ttg::TTBase *tt) {
      return this->check_key_impl(key, ord, tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>, bool>
    check(ttg::TTBase *tt) {
      return this->check_key_impl(ttg::Void{}, m_map(), tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>, bool>
    check(ordinal_type ord, ttg::TTBase *tt) {
      return this->check_key_impl(ttg::Void{}, ord, tt);
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>>
    complete(const key_type& key, ttg::TTBase *tt) {
      this->complete_key_impl();
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>>
    complete(const key_type& key, Ordinal ord, ttg::TTBase *tt) {
      this->complete_key_impl();
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && ttg::meta::is_void_v<Mapper_>>
    complete(Ordinal ord, ttg::TTBase *tt) {
      this->complete_key_impl();
    }

    template<typename Key_ = key_type, typename Mapper_ = Mapper>
    std::enable_if_t<!ttg::meta::is_void_v<Key_> && !ttg::meta::is_void_v<Mapper_>>
    complete(ttg::TTBase *tt) {
      this->complete_key_impl();
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
        if (m_auto_release) {
          release_next();
        } else {
          auto ord = m_current;
          // release the first set of available keys if none were set explicitly
          if (ord == std::numeric_limits<ordinal_type>::min() &&
              this->m_sequence.begin() != this->m_sequence.end()) {
            ord = this->m_sequence.begin()->first;
          }
          release_next(ord, true); // force the check for a next release even if the current ordinal hasn't changed
        }
      }
    }

    /**
     * Release tasks up to the ordinal. The provided ordinal is ignored if `auto_release` is enabled.
     */
    void release(ordinal_type ord = 0) {
      if (m_auto_release) {
        // last key for this ordinal, release the next
        // the provided ordinal is ignored
        release_next();
      } else {
        release_next(ord);
      }
    }

    bool is_auto() const {
      return m_auto_release;
    }


  protected:
    std::map<ordinal_type, sequence_elem_t, compare_t> m_sequence;
    ordinal_type m_current = std::numeric_limits<ordinal_type>::min();
    [[no_unique_address]]
    Mapper m_map;
    [[no_unique_address]]
    compare_t m_order;
    std::atomic<std::size_t> m_active;
    bool m_stopped = false;
    bool m_auto_release = false;
  };

  // deduction guides: take type of first argument to Mapper as the key type
  // TODO: can we use the TTG callable_args instead?
  template<typename Mapper, typename = std::enable_if_t<std::is_invocable_v<Mapper, std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>>>>
  SequencedKeysConstraint(Mapper&&)
    -> SequencedKeysConstraint<
          std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>,
          std::decay_t<boost::callable_traits::return_type_t<Mapper>>,
          std::less_equal<std::decay_t<boost::callable_traits::return_type_t<Mapper>>>,
          std::enable_if_t<std::is_invocable_v<Mapper, std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>>, Mapper>
          >;

  template<typename Mapper, typename = std::enable_if_t<std::is_invocable_v<Mapper, std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>>>>
  SequencedKeysConstraint(Mapper&&, bool)
    -> SequencedKeysConstraint<
          std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>,
          std::decay_t<boost::callable_traits::return_type_t<Mapper>>,
          std::less_equal<std::decay_t<boost::callable_traits::return_type_t<Mapper>>>,
          std::enable_if_t<std::is_invocable_v<Mapper, std::decay_t<std::tuple_element_t<0, boost::callable_traits::args_t<Mapper>>>>, Mapper>
          >;

  template<typename Key, typename Ordinal, typename Compare, typename Mapper>
  SequencedKeysConstraint(SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>&&)
    -> SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>;

  template<typename Key, typename Ordinal, typename Compare, typename Mapper>
  SequencedKeysConstraint(const SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>&)
    -> SequencedKeysConstraint<Key, Ordinal, Compare, Mapper>;

  /**
   * Make a constraint that can be shared between multiple TT instances.
   * Overload for incomplete templated constraint types.
   *
   * Example:
   * // SequencedKeysConstraint is incomplete
   * auto c = ttg::make_shared_constraint<SequencedKeysConstraint>([](Key& k){ return k[0]; });
   * auto tt_a = ttg::make_tt<Key>(...);
   * tt_a->add_constraint(c);
   * auto tt_b = ttg::make_tt<Key>(...);
   * tt_b->add_constraint(c);
   *
   * -> the constraint will handle keys from both tt_a and tt_b. Both TTs must have the same key type.
   */
  template<template<typename...> typename Constraint, typename... Args>
  auto make_shared_constraint(Args&&... args) {
    return std::make_shared<decltype(Constraint(std::forward<Args>(args)...))>(std::forward<Args>(args)...);
  }

  /**
   * Make a constraint that can be shared between multiple TT instances.
   * Overload for complete constraint types.
   */
  template<typename Constraint, typename... Args>
  auto make_shared_constraint(Args&&... args) {
    return std::make_shared<Constraint>(std::forward<Args>(args)...);
  }



} // namespace ttg

#endif // TTG_CONSTRAINT_H