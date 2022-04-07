#ifndef TTG_BITS_H
#define TTG_BITS_H

#ifdef __cpp_lib_bitops
#include <bits>
#endif // __cpp_lib_bitops

#include <cstdint>

/**
 * Implement some functions of the <bits> header introduced in C++20.
 *
 * Also provides a dynamically allocated bitset.
 */

namespace ttg {

#ifdef __cpp_lib_bitops
  template< class T >
  constexpr int popcount( T x ) noexcept {
    return std::popcount(x);
  }
#else
  template< class T >
  constexpr int popcount( T x ) noexcept {
    int res = 0;
    for (int i = 0; i < sizeof(T)*8; ++i) {
      res += !!(x & (1<<i));
    }
    return res;
  }
#endif // __cpp_lib_bitops



  /* Dynamically allocated bitset similar to std::bitset */
  class bitset {
    using storage_type = std::uint_fast8_t;
    std::vector<storage_type> m_storage;
    mutable ssize_t m_popcnt = -1;
    constexpr static size_t storage_size = sizeof(storage_type);

  public:
    bitset(size_t size) : m_storage((size+storage_size-1)/storage_size)
    { }

    void set(size_t i) noexcept {
      m_storage[i/storage_size] |= 1<<(i%storage_size);
      m_popcnt = -1;
    }

    bool get(size_t i) const noexcept {
      return !!(m_storage[i/storage_size] & (1<<(i%storage_size)));
    }

    bool operator[](size_t i) const noexcept {
      return get(i);
    }

    size_t size() const noexcept {
      return m_storage.size()*storage_size;
    }

    size_t popcnt() const noexcept {
      if (m_popcnt == -1) {
        m_popcnt = 0;
        for (const auto& v : m_storage) {
          m_popcnt += ttg::popcount(v);
        }
      }
      return m_popcnt;
    }

    void clear() noexcept {
      size_t size = m_storage.size();
      m_storage.clear();
      m_storage.resize(size);
    }
  };


} // namespace ttg

#endif // TTG_BITS_H
