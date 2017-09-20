#ifndef CXXAPI_SERIALIZATION_H_H
#define CXXAPI_SERIALIZATION_H_H

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

#if __cplusplus >= 201703L
using std::void_t;
#else
template<class...> using void_t = void;
#endif

template<class, class = void>
struct is_printable : std::false_type {};

template<class T>
struct is_printable<T, void_t<decltype(std::declval<std::ostream&>() <<
    std::declval<T>())>> : std::true_type {};

template <typename T, typename Enabler = void>
struct printer_helper {
  static void print(const void* object) {
    std::cout << "[unprintable object]" << std::endl;
  }
};

template <typename T>
struct printer_helper<T, std::enable_if_t<is_printable<T>::value>> {
  static void print(const void* object) {
    std::cout << *(static_cast<const T*>(object)) << std::endl;
  }
};

}  // namespace detail

template<typename T, typename Enabler = void>
struct default_data_descriptor;

template<typename T>
struct default_data_descriptor<T, std::enable_if_t<std::is_pod<T>::value>> {

  static uint64_t header_size(const void *object) { return static_cast<uint64_t>(0); }

  static uint64_t payload_size(const void *object) {
    return static_cast<uint64_t>(sizeof(T));
  }

  static void get_info(const void *object, uint64_t *hs, uint64_t *ps, int *is_contiguous_mask, void **buf) {
    *hs = header_size(object);
    *ps = payload_size(object);
    *is_contiguous_mask = 1;
    // on the receiving side request that payload goes directly to object
    *buf = const_cast<void *>(object);
  }

  static void pack_header(const void *object, uint64_t header_size, void **buf) {}

  /// t --- obj to be serialized
  /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
  /// pos --- position in the input buffer to resume serialization
  /// ptr[chunk_size] --- place for output
  static void pack_payload(const void *object, uint64_t *chunk_size, uint64_t pos, void **buf) {
    std::memcpy(*buf, object, *chunk_size);
  }

  // t points to some memory in which we will construct an object from the header
  static void unpack_header(void *object, uint64_t header_size, const void *buf) {
    assert(header_size == 0);
    new(object) T;
  }

  static void unpack_payload(void *object, uint64_t chunk_size, uint64_t pos, const void *buf) {
    std::memcpy(object, buf, chunk_size);
  }

};

}  // namespace ttg

#if __has_include(<madness/world/archive.h>)

#include <type_traits>

#include <madness/world/archive.h>
#include <madness/world/buffer_archive.h>

namespace ttg {

namespace detail {

template<class, class = void>
struct is_madness_serializable : std::false_type {};

template<class T>
struct is_madness_serializable<T, void_t<decltype(std::declval<madness::archive::BufferOutputArchive&>() &
                                         std::declval<T>())>> : std::true_type {};
}  // namespace detail

// The default implementation for non-POD data types that support MADNESS serialization
template <typename T>
struct default_data_descriptor<T, std::enable_if_t<!std::is_pod<T>::value && detail::is_madness_serializable<T>::value>> {

  static uint64_t header_size(const void* object) { return static_cast<uint64_t>(0); }

  static uint64_t payload_size(const void* object)
  {
    madness::archive::BufferOutputArchive ar;
    ar & (*(T*)object);
    return static_cast<uint64_t>(ar.size());
  }

  static void get_info(const void* object, uint64_t* hs, uint64_t* ps, int* is_contiguous_mask, void** buf) {
    *hs = header_size(object);
    *ps = payload_size(object);
    *is_contiguous_mask = 0;
    *buf = nullptr;
  }

  static void pack_header(const void* object, uint64_t header_size, void**buf) {}

  /// t --- obj to be serialized
  /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
  /// pos --- position in the input buffer to resume serialization
  /// ptr[chunk_size] --- place for output
  static void pack_payload(const void* object, uint64_t* chunk_size, uint64_t pos, void** buf)
  {
    madness::archive::BufferOutputArchive ar(*buf, *chunk_size);
    ar & (*(T*)object);
  }

  // t points to some memory in which we will construct an object from the header
  static void unpack_header(void* object, uint64_t header_size, const void* buf) {
    assert(header_size == 0);
    new (object) T;
  }

  static void unpack_payload(void* object, uint64_t chunk_size, uint64_t pos, const void* buf) {
    madness::archive::BufferInputArchive ar(buf, chunk_size);
    ar & (*(T*)object);
  }

};

}  // namespace ttg

#endif  // has MADNESS serialization

namespace ttg {

// Returns a pointer to a constant static instance initialized
// once at run time. 
template <typename T>
const ttg_data_descriptor*
get_data_descriptor() {
  static const ttg_data_descriptor d = {typeid(T).name(),
                                        &default_data_descriptor<T>::get_info,
                                        &default_data_descriptor<T>::pack_header,
                                        &default_data_descriptor<T>::pack_payload,
                                        &default_data_descriptor<T>::unpack_header,
                                        &default_data_descriptor<T>::unpack_payload,
                                        &detail::printer_helper<T>::print};
  return &d;
}

}  // namespace ttg

#endif //CXXAPI_SERIALIZATION_H_H
