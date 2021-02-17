#ifndef CXXAPI_SERIALIZATION_H
#define CXXAPI_SERIALIZATION_H

#include <cstring>  // for std::memcpy

#include "ttg/util/meta.h"
#include "ttg/data_descriptor.h"
#include "ttg/util/void.h"

/**
   \file serialization.h

   \brief Provides (de)serialization of C++ data invocable from C primarily to interface with PaRSEC

   The default implementation is only provided for POD data types but is efficient in the sense that
   it does enable zero-copy remote data transfers.   For other data types, optimized implementations
   must be provided as needed or, if available, the MADNESS serialization can be used but this will
   always make a copy and requires that the entire object fit in the message buffer.

 **/

namespace ttg {

  namespace detail {

    template <class, class = void>
    struct is_printable : std::false_type {};

    template <class T>
    struct is_printable<T, ttg::meta::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
        : std::true_type {};

    template <typename T, typename Enabler = void>
    struct printer_helper {
      static void print(const void* object) { std::cout << "[unprintable object]" << std::endl; }
    };

    template <typename T>
    struct printer_helper<T, std::enable_if_t<is_printable<T>::value>> {
      static void print(const void* object) { std::cout << *(static_cast<const T*>(object)) << std::endl; }
    };

  }  // namespace detail

  template <typename T, typename Enabler = void>
  struct default_data_descriptor;

  template <typename T>
  struct default_data_descriptor<T, std::enable_if_t<std::is_trivially_copyable<T>::value>> {
    static constexpr const bool serialize_size_is_const = true;

    static uint64_t payload_size(const void *object) { return static_cast<uint64_t>(sizeof(T)); }

    /// object --- obj to be serialized
    /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
    /// pos --- position in the input buffer to resume serialization
    /// buf[pos] --- place for output
    static uint64_t pack_payload(const void* object, uint64_t chunk_size, uint64_t pos, void* _buf) {
        unsigned char *buf = reinterpret_cast<unsigned char*>(_buf);
        std::memcpy(&buf[pos], object, chunk_size);
        return pos + chunk_size;
    }

    static void unpack_payload(void* object, uint64_t chunk_size, uint64_t pos, const void* _buf) {
      const unsigned char *buf = reinterpret_cast<const unsigned char*>(_buf);
      std::memcpy(object, &buf[pos], chunk_size);
    }
  };

}  // namespace ttg

#if __has_include(<madness/world/archive.h>)
#include <type_traits>

#include <madness/world/archive.h>
#include <madness/world/buffer_archive.h>

namespace ttg {

  namespace detail {

    template <class, class = void>
    struct is_madness_serializable : std::false_type {};

    template <class T>
    struct is_madness_serializable<
        T, ttg::meta::void_t<decltype(std::declval<madness::archive::BufferOutputArchive&>() & std::declval<T>())>>
        : std::true_type {};
  }  // namespace detail

  // The default implementation for non-POD data types that support MADNESS serialization
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<!std::is_trivially_copyable<T>::value && detail::is_madness_serializable<T>::value>> {
    static constexpr const bool serialize_size_is_const = false;

    static uint64_t payload_size(const void *object) {
      madness::archive::BufferOutputArchive ar;
      ar&(*(T*)object);
      return static_cast<uint64_t>(ar.size());
    }

    /// object --- obj to be serialized
    /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
    /// pos --- position in the input buffer to resume serialization
    /// buf[pos] --- place for output
    static uint64_t pack_payload(const void* object, uint64_t chunk_size, uint64_t pos, void* _buf) {
        unsigned char *buf = reinterpret_cast<unsigned char*>(_buf);
        madness::archive::BufferOutputArchive ar(&buf[pos], chunk_size);
        ar&(*(T*)object);
        return pos+chunk_size;
    }

    /// object --- obj to be deserialized
    /// chunk_size --- amount of data for input
    /// pos --- position in the input buffer to resume deserialization
    /// object -- pointer to the object to fill up
    static void unpack_payload(void* object, uint64_t chunk_size, uint64_t pos, const void* _buf) {
      const unsigned char *buf = reinterpret_cast<const unsigned char*>(_buf);
      madness::archive::BufferInputArchive ar(&buf[pos], chunk_size);
      ar&(*(T*)object);
    }
  };

}  // namespace ttg

#endif  // has MADNESS serialization

namespace ttg {

  // Returns a pointer to a constant static instance initialized
  // once at run time.
  template <typename T>
  const ttg_data_descriptor* get_data_descriptor() {
    static const ttg_data_descriptor d = {typeid(T).name(),
                                          &default_data_descriptor<T>::payload_size,
                                          &default_data_descriptor<T>::pack_payload,
                                          &default_data_descriptor<T>::unpack_payload,
                                          &detail::printer_helper<T>::print};
    return &d;
  }

}  // namespace ttg

#endif  // CXXAPI_SERIALIZATION_H
