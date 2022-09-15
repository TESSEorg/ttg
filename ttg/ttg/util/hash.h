#ifndef TTG_UTIL_HASH_H
#define TTG_UTIL_HASH_H

#include <cstddef>
#include <cstdint>

#include "ttg/util/void.h"

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

  namespace meta {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // has_member_function_hash_v<T> evaluates to true if T::hash() is defined
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enabler = void>
    struct has_member_function_hash : std::false_type {};
    template <typename T>
    struct has_member_function_hash<T, std::void_t<decltype(std::declval<const T&>().hash())>> : std::true_type {};
    template <typename T>
    constexpr bool has_member_function_hash_v = has_member_function_hash<T>::value;
  }  // namespace meta

  /// place for overloading/instantiating hash and other functionality
  namespace overload {

    /// \brief Computes hash values for objects of type T.

    /// Specialize for your type, if needed.
    /// \note Must provide operator()(const Input&)
    template <typename T, typename Enabler = void>
    struct hash;

    /// instantiation of hash for types which have member function hash() that returns
    template <typename T>
    struct hash<T, std::enable_if_t<meta::has_member_function_hash_v<T>>> {
      auto operator()(const T& t) const { return t.hash(); }
    };

    /// instantiation of hash for void
    template <>
    struct hash<void, void> {
      auto operator()() const { return detail::FNVhasher::initial_value(); }
      auto operator()(const ttg::Void&) const { return operator()(); }
    };

    /// instantiation of hash for integral types smaller or equal to size_t
    template <typename T>
    struct hash<T, std::enable_if_t<std::is_integral_v<std::decay_t<T>> && sizeof(T) <= sizeof(std::size_t), void>> {
      auto operator()(T t) const { return static_cast<std::size_t>(t); }
    };

    /// default implementation for types with unique object representation uses the bitwise hasher FNVhasher
    /// \sa https://en.cppreference.com/w/cpp/types/has_unique_object_representations
    template <typename T>
    struct hash<
        T, std::enable_if_t<!(std::is_integral_v<std::decay_t<T>> && sizeof(T) <= sizeof(std::size_t)) &&
                                !(meta::has_member_function_hash_v<T>)&&std::has_unique_object_representations_v<T>,
                            void>> {
      auto operator()(const T& t) const {
        detail::FNVhasher hasher;
        hasher.update(sizeof(T), reinterpret_cast<const std::byte*>(&t));
        return hasher.value();
      }
    };

    /// provide default implementation for error-reporting purposes
    template <typename T, typename Enabler>
    struct hash {
      constexpr static bool NEED_TO_PROVIDE_SPECIALIZATION_OF_TTG_OVERLOAD_HASH_FOR_THIS_TYPE = !std::is_same_v<T, T>;
      static_assert(NEED_TO_PROVIDE_SPECIALIZATION_OF_TTG_OVERLOAD_HASH_FOR_THIS_TYPE);
    };
  }  // namespace overload

  using namespace ttg::overload;

  namespace meta {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // has_ttg_hash_specialization_v<T> evaluates to true if ttg::hash<T> is defined
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enabler = void>
    struct has_ttg_hash_specialization : std::false_type {};
    template <typename T>
    struct has_ttg_hash_specialization<
        T, ttg::meta::void_t<decltype(std::declval<ttg::hash<T>>()(std::declval<const T&>()))>> : std::true_type {};
    template <typename T>
    constexpr bool has_ttg_hash_specialization_v = has_ttg_hash_specialization<T>::value;
  }  // namespace meta

}  // namespace ttg

#endif  // TTG_UTIL_HASH_H
