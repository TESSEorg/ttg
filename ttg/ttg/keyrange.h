#ifndef TTG_KEYTERATOR_H
#define TTG_KEYTERATOR_H

#include <type_traits>
#include <iterator>
#include <exception>

#include "ttg/util/meta.h"
#include "ttg/serialization.h"

namespace ttg {

  /* Trait providing the diff type of a given key
   * Defaults to the key itself.
   * May be provided as \c diff_type member of the key or by
   * overloading this trait.
   */
  template<typename Key, typename = void>
  struct key_diff_type {
    using type = Key;
  };

  /* Overload for Key::diff_type */
  template<typename Key>
  struct key_diff_type<Key, std::void_t<typename Key::difference_type>>{
    using type = typename Key::difference_type;
  };

  /* Convenience type */
  template<typename Key>
  using key_diff_type_t = typename key_diff_type<Key>::type;

  namespace detail {
    /**
     * Trait checking whether a key is compatible with the LinearKeyRange.
     * Keys must at least support comparison as well as addition or compound addition.
     */
    template<typename Key>
    struct is_range_compatible {
      using key_type = std::decay_t<Key>;
      using difference_type = key_diff_type_t<key_type>;
      constexpr static bool value = ttg::meta::is_comparable_v<key_type>
                                    && (ttg::meta::has_addition_v<key_type, difference_type>
                                      || ttg::meta::has_compound_addition_v<key_type, difference_type>)
                                    && (std::is_trivially_copyable_v<Key> || is_user_buffer_serializable_v<Key>);
    };

    template<typename Key>
    constexpr bool is_range_compatible_v = is_range_compatible<Key>::value;

    /**
    * Represents a range of keys that can be represented as a linear iteration
    * space, i.e., using a start and end (one past the last element) as well as
    * a step increment. An iterator is provided to iterate over the range of keys.
    *
    * The step increment is of type \sa ttg::key_diff_type, which defaults
    * to the key type but can be overridden by either defining a \c diff_type
    * member type or by specializing ttg::key_diff_type.
    */
    template<typename Key>
    struct LinearKeyRange {
      using key_type = std::decay_t<Key>;
      using diff_type = key_diff_type_t<key_type>;

      /* Forward Iterator for the linear key range */
      struct iterator
      {
        /* types for std::iterator_trait */
        using value_type = key_type;
        using difference_type = diff_type;
        using pointer = const value_type*;
        using reference = const value_type&;
        using iterator_category = std::forward_iterator_tag;

        iterator(const key_type& pos, const diff_type& inc)
        : m_pos(pos), m_inc(inc)
        { }

        iterator& operator++() {
          if constexpr (meta::has_compound_addition_v<value_type, difference_type>) {
            m_pos += m_inc;
          } else if constexpr (meta::has_addition_v<value_type, difference_type>) {
            m_pos = m_pos + m_inc;
          } else {
            throw std::logic_error("Key type does not support addition its with difference type");
          }
          return *this;
        }

        iterator operator++(int) {
          iterator retval = *this;
          ++(*this);
          return retval;
        }

        bool operator==(const iterator& other) const {
          if constexpr(meta::is_comparable_v<value_type>) {
            return m_pos == other.m_pos;
          }
          return false;
        }

        bool operator!=(const iterator& other) const {
          return !(*this == other);
        }
        const key_type& operator*() const {
          return m_pos;
        }
        const key_type* operator->() const {
          return &m_pos;
        }

      private:
        key_type m_pos;
        const diff_type& m_inc;
      };

      LinearKeyRange()
      { }

      LinearKeyRange(const Key& begin, const Key& end, const diff_type& inc)
      : m_begin(begin)
      , m_end(end)
      , m_inc(inc)
      { }

      iterator begin() const {
        return iterator(m_begin, m_inc);
      }

      iterator end() const {
        return iterator(m_end, m_inc);
      }

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
      /* Make the LinearKeyRange madness serializable */
      template<typename Archive, typename Key_ = key_type, typename Diff_ = diff_type,
               typename = std::enable_if_t<is_madness_buffer_serializable_v<Key_>
                                            && is_madness_buffer_serializable_v<Diff_>>>
      void serialize(Archive& ar) {
        ar & m_begin;
        ar & m_end;
        ar & m_inc;
      }
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
      /* Make the LinearKeyRange boost serializable */
      template<typename Archive, typename Key_ = key_type, typename Diff_ = diff_type,
               typename = std::enable_if_t<is_boost_buffer_serializable_v<Key_>
                                            && is_boost_buffer_serializable_v<Diff_>>>
      void serialize(Archive& ar, const unsigned int version) {
        ar & m_begin;
        ar & m_end;
        ar & m_inc;
      }
#endif

    friend std::ostream& operator<<(std::ostream& out, LinearKeyRange const& k) {
      out << "LinearKeyRange[" << k.m_begin << "," << k.m_end << "):"<<k.m_inc;
      return out;
    }

    private:
      key_type m_begin, m_end;
      diff_type m_inc;
    };

  } // namespace detail

  /**
   * Create a key range [begin, end) with step size \c inc.
   * \return A representation of the range that can be passed to send/broadcast.
   */
  template<typename Key>
  inline auto make_keyrange(const Key& begin,
                            const Key& end,
                            const key_diff_type_t<Key>& inc) {
    static_assert(detail::is_range_compatible_v<Key>,
                  "Key type does not support all required operations: operator==, "
                  "operator+ or operator+=, and serialization (trivially_copyable, madness, or boost)");
    return detail::LinearKeyRange(begin, end, inc);
  }

  /**
   * Create a key range [begin, end) with unit stride.
   *
   * Requires \c operator++ and \c operator- to be defined on Key.
   * If a difference type (\c Key::diff_type) is defined, \c operator- should return
   * the difference type.
   *
   * \return A representation of the range that can be passed to send/broadcast.
   */
  template<typename Key>
  inline auto make_keyrange(const Key& begin, const Key& end) {
    static_assert(detail::is_range_compatible_v<Key>,
                  "Key type does not support all required operations: operator==, "
                  "operator+ or operator+=, and serialization (trivially_copyable, madness, or boost)");
    static_assert(meta::has_increment_v<Key> && meta::has_difference_v<Key>,
                  "Unit stride key range requires operator++ and operator- on Key");
    if constexpr (meta::has_pre_increment_v<Key>) {
      return detail::LinearKeyRange(begin, end, ++begin - begin);
    } else {
      return detail::LinearKeyRange(begin, end, begin++ - begin);
    }
  }

} // namespace ttg
#endif // TTG_KEYTERATOR_H
