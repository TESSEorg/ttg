#ifndef TTG_TTG_UTIL_HASH_H
#define TTG_TTG_UTIL_HASH_H

#include <cstddef>
#include <cstdint>

namespace ttg {
  namespace detail {
    /// @brief byte-wise hasher
    class FNVhasher {
      // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
      using result_type = std::size_t;
      static const result_type offset_basis = 14695981039346656037ul;
      static const result_type prime = 1099511628211ul;
      result_type value_ = offset_basis;

     public:
      /// Updates the hash with one byte
      /// @param[in] byte the input value
      void update(std::byte byte) noexcept { value_ = (value_ ^ static_cast<uint_fast8_t>(byte)) * prime; }

      /// Updates the hash with an additional n bytes
      void update(size_t n, const std::byte* bytes) noexcept {
        for (size_t i = 0; i < n; i++) update(bytes[i]);
      }

      /// @return the value of the hash of the stream
      auto value() const noexcept { return value_; }

      /// @return the initial hash value
      static result_type initial_value() { return offset_basis; }
    };
  }  // namespace detail

  /// place for overloading/instantiating hash and other functionality
  namespace overload {

    /// \brief Computes hash values for objects of type T.

    /// Specialize for your type, if needed.
    /// \note Must provide operator()(const Input&)
    template <typename T, typename Enabler = void>
    struct hash;

    /// instantiation of hash for types which have member function hash()
    template <typename T>
    struct hash<T, std::void_t<decltype(std::declval<const T&>().hash())>> {
      auto operator()(const T &t) const { return t.hash(); }
    };

    /// instantiation of hash for types which have member function hash()
    template <>
    struct hash<void, void> {
      auto operator()() const { return detail::FNVhasher::initial_value(); }
    };

    /// default implementation uses the bitwise hasher FNVhasher
    template <typename T, typename Enabler>
    struct hash {
      auto operator()(const T& t) const {
        detail::FNVhasher hasher;
        hasher.update(sizeof(T), reinterpret_cast<const std::byte*>(&t));
        return hasher.value();
      }
    };

  }  // namespace overload

  using namespace ::ttg::overload;
}  // namespace ttg

#endif  // TTG_TTG_UTIL_HASH_H
