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

      /// Returns the value of the hash of the stream
      auto value() const noexcept { return value_; }
    };
  }  // namespace detail
}  // namespace ttg

#endif  // TTG_TTG_UTIL_HASH_H
