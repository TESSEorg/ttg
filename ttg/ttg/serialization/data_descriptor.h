#ifndef TTG_SERIALIZATION_DATA_DESCRIPTOR_H
#define TTG_SERIALIZATION_DATA_DESCRIPTOR_H

#include <cstdint>

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
#include <madness/world/buffer_archive.h>
#endif

#include "ttg/serialization/traits.h"

#include "ttg/serialization/stream.h"

#include <cstring>  // for std::memcpy

// This provides an efficent API for serializing/deserializing a data type.
// An object of this type will need to be provided for each serializable type.
// The default implementation, in serialization.h, works only for primitive/POD data types;
// backend-specific implementations may be available in backend/serialization.h .
extern "C" struct ttg_data_descriptor {
  const char *name;
  uint64_t (*payload_size)(const void *object);
  uint64_t (*pack_payload)(const void *object, uint64_t chunk_size, uint64_t pos, void *buf);
  void (*unpack_payload)(void *object, uint64_t chunk_size, uint64_t pos, const void *buf);
  void (*print)(const void *object);
};

namespace ttg {

  /**
 *  \brief Provides (de)serialization of C++ data invocable from C primarily to interface with PaRSEC

 * The default implementation is only provided for POD data types but is efficient in the sense that
 * it does enable zero-copy remote data transfers.   For other data types, optimized implementations
 * must be provided as needed or, if available, the MADNESS serialization can be used but this will
 * always make a copy and requires that the entire object fit in the message buffer.
 */
  template <typename T, typename Enabler = void>
  struct default_data_descriptor;

  /// default_data_descriptor for trivially-copyable types
  /// @tparam T a trivially-copyable type
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<std::is_trivially_copyable<T>::value && !detail::is_user_buffer_serializable_v<T>>> {
    static constexpr const bool serialize_size_is_const = true;

    /// @param[in] object pointer to the object to be serialized
    /// @return size of serialized @p object
    static uint64_t payload_size(const void *object) { return static_cast<uint64_t>(sizeof(T)); }

    /// @brief serializes object to a buffer

    /// @param[in] object pointer to the object to be serialized
    /// @param[in] size the size of @p object in bytes
    /// @param[in] begin location in @p buf where the first byte of serialized data will be written
    /// @param[in,out] buf the data buffer that will contain serialized data
    /// @return location in @p buf after the last byte written
    static uint64_t pack_payload(const void *object, uint64_t size, uint64_t begin, void *buf) {
      unsigned char *char_buf = reinterpret_cast<unsigned char *>(buf);
      std::memcpy(&char_buf[begin], object, size);
      return begin + size;
    }

    /// @brief deserializes object from a buffer

    /// @param[in,out] object pointer to the object to be deserialized
    /// @param[in] size the size of @p object in bytes
    /// @param[in] begin location in @p buf where the first byte of serialized data will be read
    /// @param[in] buf the data buffer that contains serialized data
    static void unpack_payload(void *object, uint64_t size, uint64_t begin, const void *buf) {
      const unsigned char *char_buf = reinterpret_cast<const unsigned char *>(buf);
      std::memcpy(object, &char_buf[begin], size);
    }
  };

}  // namespace ttg

#if defined(TTG_SERIALIZATION_SUPPORTS_MADNESS)

namespace ttg {

  /// The default implementation for non-POD data types that are not directly copyable
  /// and support MADNESS serialization
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<(!std::is_trivially_copyable<T>::value && detail::is_madness_buffer_serializable_v<T>) ||
                          detail::is_madness_user_buffer_serializable_v<T>>> {
    static constexpr const bool serialize_size_is_const = false;

    static uint64_t payload_size(const void *object) {
      madness::archive::BufferOutputArchive ar;
      ar &(*(T *)object);
      return static_cast<uint64_t>(ar.size());
    }

    /// object --- obj to be serialized
    /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
    /// pos --- position in the input buffer to resume serialization
    /// buf[pos] --- place for output
    static uint64_t pack_payload(const void *object, uint64_t chunk_size, uint64_t pos, void *_buf) {
      unsigned char *buf = reinterpret_cast<unsigned char *>(_buf);
      madness::archive::BufferOutputArchive ar(&buf[pos], chunk_size);
      ar &(*(T *)object);
      return pos + chunk_size;
    }

    /// object --- obj to be deserialized
    /// chunk_size --- amount of data for input
    /// pos --- position in the input buffer to resume deserialization
    /// object -- pointer to the object to fill up
    static void unpack_payload(void *object, uint64_t chunk_size, uint64_t pos, const void *_buf) {
      const unsigned char *buf = reinterpret_cast<const unsigned char *>(_buf);
      madness::archive::BufferInputArchive ar(&buf[pos], chunk_size);
      ar &(*(T *)object);
    }
  };

}  // namespace ttg

#endif  // has MADNESS serialization

#if defined(TTG_SERIALIZATION_SUPPORTS_BOOST)

#include "ttg/serialization/backends/boost/archive.h"

namespace ttg {

  /// The default implementation for non-POD data types that are not directly copyable,
  /// do not support MADNESS serialization, and support Boost serialization
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<(!std::is_trivially_copyable<T>::value && !detail::is_madness_buffer_serializable_v<T> &&
                           detail::is_boost_buffer_serializable_v<T>) ||
                          (!detail::is_madness_user_buffer_serializable_v<T> &&
                           detail::is_boost_user_buffer_serializable_v<T>)>> {
    static constexpr const bool serialize_size_is_const = false;

    static uint64_t payload_size(const void *object) {
      ttg::detail::boost_counting_oarchive oa;
      oa << (*(T *)object);
      return oa.streambuf().size();
    }

    /// object --- obj to be serialized
    /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
    /// pos --- position in the input buffer to resume serialization
    /// buf[pos] --- place for output
    static uint64_t pack_payload(const void *object, uint64_t chunk_size, uint64_t pos, void *_buf) {
      auto oa = ttg::detail::make_boost_buffer_oarchive(_buf, pos + chunk_size, pos);
      oa << (*(T *)object);
      return pos + chunk_size;
    }

    /// object --- obj to be deserialized
    /// chunk_size --- amount of data for input
    /// pos --- position in the input buffer to resume deserialization
    /// object -- pointer to the object to fill up
    static void unpack_payload(void *object, uint64_t chunk_size, uint64_t pos, const void *_buf) {
      auto ia = ttg::detail::make_boost_buffer_iarchive(_buf, pos + chunk_size, pos);
      ia >> (*(T *)object);
    }
  };

}  // namespace ttg

#endif  // has Boost serialization

#if defined(TTG_SERIALIZATION_SUPPORTS_CEREAL)

namespace ttg {

  /// The default implementation for non-POD data types that are not directly copyable
  /// do not support MADNESS or Boost serialization, and support Cereal serialization
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<(!std::is_trivially_copyable<T>::value && !detail::is_madness_buffer_serializable_v<T> &&
                           !detail::is_boost_buffer_serializable_v<T> && detail::is_cereal_buffer_serializable_v<T>) ||
                          (!detail::is_madness_user_buffer_serializable_v<T> &&
                           !detail::is_boost_user_buffer_serializable_v<T> &&
                           detail::is_cereal_user_buffer_serializable_v<T>)>> {
    static constexpr const bool serialize_size_is_const = false;

    static uint64_t payload_size(const void *object) {
      ttg::detail::counting_streambuf sbuf;
      std::ostream os(&sbuf);
      cereal::BinaryOutputArchive oa(os);
      oa << (*(T *)object);
      return sbuf.size();
    }

    /// object --- obj to be serialized
    /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
    /// pos --- position in the input buffer to resume serialization
    /// buf[pos] --- place for output
    static uint64_t pack_payload(const void *object, uint64_t chunk_size, uint64_t pos, void *_buf) { abort(); }

    /// object --- obj to be deserialized
    /// chunk_size --- amount of data for input
    /// pos --- position in the input buffer to resume deserialization
    /// object -- pointer to the object to fill up
    static void unpack_payload(void *object, uint64_t chunk_size, uint64_t pos, const void *_buf) { abort(); }
  };

}  // namespace ttg

#endif  // has Cereal serialization

namespace ttg {

  // Returns a pointer to a constant static instance initialized
  // once at run time.
  template <typename T>
  const ttg_data_descriptor *get_data_descriptor() {
    static const ttg_data_descriptor d = {
        typeid(T).name(), &default_data_descriptor<T>::payload_size, &default_data_descriptor<T>::pack_payload,
        &default_data_descriptor<T>::unpack_payload, &detail::printer_helper<T>::print};
    return &d;
  }

}  // namespace ttg

#endif  // TTG_SERIALIZATION_DATA_DESCRIPTOR_H
