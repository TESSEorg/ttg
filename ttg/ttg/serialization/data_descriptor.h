#ifndef TTG_SERIALIZATION_DATA_DESCRIPTOR_H
#define TTG_SERIALIZATION_DATA_DESCRIPTOR_H

#include <cstdint>

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
#include <madness/world/buffer_archive.h>
#endif

#include "ttg/serialization/traits.h"

#include "ttg/serialization/stream.h"

#include <cstring>  // for std::memcpy

#include "ttg/serialization/splitmd_data_descriptor.h"

/// This provides an efficient C API for serializing/deserializing a data type to a nonportable contiguous bytestring.
/// An object of this type will need to be provided for each serializable type.
/// The default implementation, in serialization.h, works only for primitive/POD data types;
/// backend-specific implementations may be available in backend/serialization.h .
extern "C" struct ttg_data_descriptor {
  const char *name;

  /// @brief measures the size of the binary representation of @p object
  /// @param[in] object pointer to the object to be serialized
  /// @return the number of bytes needed for binary representation of @p object
  uint64_t (*payload_size)(const void *object);

  /// @brief serializes object to a buffer
  /// @param[in] object pointer to the object to be serialized
  /// @param[in] max_nbytes_to_write the maximum number of bytes to write
  /// @param[in] offset the position in \p buf where the first byte of serialized data will be written
  /// @param[in,out] buf the data buffer that will contain the serialized representation of the object
  /// @return position in \p buf after the last byte written
  uint64_t (*pack_payload)(const void *object, uint64_t max_nbytes_to_write, uint64_t offset, void *buf);

  /// @brief deserializes object from a buffer
  /// @param[in] object pointer to the object to be deserialized
  /// @param[in] max_nbytes_to_read the maximum number of bytes to read
  /// @param[in] offset the position in \p buf where the first byte of serialized data will be read
  /// @param[in] buf the data buffer that contains the serialized representation of the object
  /// @return position in \p buf after the last byte read
  uint64_t (*unpack_payload)(void *object, uint64_t max_nbytes_to_read, uint64_t offset, const void *buf);

  void (*print)(const void *object);
};

namespace ttg {

/**
 *  \brief Provides (de)serialization of C++ data that can be invoked from C via ttg_data_descriptor

 * The default implementation is only provided for POD data types but is efficient in the sense that
 * it does enable zero-copy remote data transfers.   For other data types, optimized implementations
 * must be provided as needed or, if available, the MADNESS serialization can be used but this will
 * always make a copy and requires that the entire object fit in the message buffer.
 */
  template <typename T, typename Enabler = void>
  struct default_data_descriptor;

  /// @brief default_data_descriptor for trivially-copyable types
  /// @tparam T a trivially-copyable type
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<detail::is_memcpyable_v<T> && !detail::is_user_buffer_serializable_v<T> &&
                          !ttg::has_split_metadata<T>::value>> {
    static constexpr const bool serialize_size_is_const = true;

    /// @brief measures the size of the binary representation of @p object
    /// @param[in] object pointer to the object to be serialized
    /// @return the number of bytes needed for binary representation of @p object
    static uint64_t payload_size(const void *object) { return static_cast<uint64_t>(sizeof(T)); }

    /// @brief serializes object to a buffer
    /// @param[in] object pointer to the object to be serialized
    /// @param[in] max_nbytes_to_write the maximum number of bytes to write
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be written
    /// @param[in,out] buf the data buffer that will contain the serialized representation of the object
    /// @return position in \p buf after the last byte written
    static uint64_t pack_payload(const void *object, uint64_t max_nbytes_to_write, uint64_t begin, void *buf) {
      unsigned char *char_buf = reinterpret_cast<unsigned char *>(buf);
      assert(sizeof(T)<=max_nbytes_to_write);
      std::memcpy(&char_buf[begin], object, sizeof(T));
      return begin + sizeof(T);
    }

    /// @brief deserializes object from a buffer
    /// @param[in] object pointer to the object to be deserialized
    /// @param[in] max_nbytes_to_read the maximum number of bytes to read
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be read
    /// @param[in] buf the data buffer that contains the serialized representation of the object
    /// @return position in \p buf after the last byte read
    static uint64_t unpack_payload(void *object, uint64_t max_nbytes_to_read, uint64_t begin, const void *buf) {
      const unsigned char *char_buf = reinterpret_cast<const unsigned char *>(buf);
      assert(sizeof(T)<=max_nbytes_to_read);
      std::memcpy(object, &char_buf[begin], sizeof(T));
      return begin + sizeof(T);
    }
  };

  /// @brief default_data_descriptor for types that support 2-stage serialization (metadata first, then the rest) for implementing zero-copy transfers
  /// @tparam T a type for which `ttg::has_split_metadata<T>::value` is true
  template <typename T>
  struct default_data_descriptor<T, std::enable_if_t<ttg::has_split_metadata<T>::value>> {
    static constexpr const bool serialize_size_is_const = false;

    /// @brief measures the size of the binary representation of @p object
    /// @param[in] object pointer to the object to be serialized
    /// @return the number of bytes needed for binary representation of @p object
    static uint64_t payload_size(const void *object) {
      SplitMetadataDescriptor<T> smd;
      const T *t = reinterpret_cast<T *>(object);
      auto metadata = smd.get_metadata(t);
      size_t size = sizeof(metadata);
      for (auto &&iovec : smd.get_data(t)) {
        size += iovec.num_bytes;
      }

      return static_cast<uint64_t>(size);
    }

    /// @brief serializes object to a buffer
    /// @param[in] object pointer to the object to be serialized
    /// @param[in] max_nbytes_to_write the maximum number of bytes to write
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be written
    /// @param[in,out] buf the data buffer that will contain the serialized representation of the object
    /// @return position in \p buf after the last byte written
    static uint64_t pack_payload(const void *object, uint64_t max_nbytes_to_write, uint64_t begin, void *buf) {
      SplitMetadataDescriptor<T> smd;
      const T *t = reinterpret_cast<T *>(object);

      unsigned char *char_buf = reinterpret_cast<unsigned char *>(buf);
      auto metadata = smd.get_metadata(t);
      assert(sizeof(metadata) <= max_nbytes_to_write);
      std::memcpy(&char_buf[begin], metadata, sizeof(metadata));
      size_t pos = sizeof(metadata);
      for (auto &&iovec : smd.get_data(t)) {
        std::memcpy(&char_buf[begin + pos], iovec.data, iovec.num_bytes);
        pos += iovec.num_bytes;
        assert(pos <= max_nbytes_to_write);
      }
      return begin + pos;
    }

    /// @brief deserializes object from a buffer
    /// @param[in] object pointer to the object to be deserialized
    /// @param[in] max_nbytes_to_read the maximum number of bytes to read
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be read
    /// @param[in] buf the data buffer that contains the serialized representation of the object
    /// @return position in \p buf after the last byte read
    static uint64_t unpack_payload(void *object, uint64_t max_nbytes_to_read, uint64_t begin, const void *buf) {
      SplitMetadataDescriptor<T> smd;
      T *t = reinterpret_cast<T *>(object);

      using metadata_t = decltype(smd.get_metadata(t));
      assert(sizeof(metadata_t) <= max_nbytes_to_read);
      const unsigned char *char_buf = reinterpret_cast<const unsigned char *>(buf);
      const metadata_t *metadata = reinterpret_cast<const metadata_t *>(char_buf + begin);
      T t_created = smd.create_from_metadata();
      size_t pos = sizeof(metadata);
      *t = t_created;
      for (auto &&iovec : smd.get_data(t)) {
        std::memcpy(iovec.data, &char_buf[begin + pos], iovec.num_bytes);
        pos += iovec.num_bytes;
        assert(pos <= max_nbytes_to_read);
      }
      return begin + pos;
    }
  };

}  // namespace ttg

#if defined(TTG_SERIALIZATION_SUPPORTS_MADNESS)

namespace ttg {

  /// @brief default_data_descriptor for non-POD data types that are not directly copyable or 2-stage serializable and support MADNESS serialization
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<((!detail::is_memcpyable_v<T> && detail::is_madness_buffer_serializable_v<T>) ||
                           detail::is_madness_user_buffer_serializable_v<T>)&&!ttg::has_split_metadata<T>::value>> {
    static constexpr const bool serialize_size_is_const = false;

    /// @brief measures the size of the binary representation of @p object
    /// @param[in] object pointer to the object to be serialized
    /// @return the number of bytes needed for binary representation of @p object
    static uint64_t payload_size(const void *object) {
      madness::archive::BufferOutputArchive ar;
      ar & (*static_cast<std::add_pointer_t<std::add_const_t<T>>>(object));
      return static_cast<uint64_t>(ar.size());
    }

    /// @brief serializes object to a buffer
    /// @param[in] object pointer to the object to be serialized
    /// @param[in] max_nbytes_to_write the maximum number of bytes to write
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be written
    /// @param[in,out] buf the data buffer that will contain the serialized representation of the object
    /// @return position in \p buf after the last byte written
    static uint64_t pack_payload(const void *object, uint64_t max_nbytes_to_write, uint64_t pos, void *_buf) {
      unsigned char *buf = reinterpret_cast<unsigned char *>(_buf);
      madness::archive::BufferOutputArchive ar(&buf[pos], max_nbytes_to_write);
      ar & (*static_cast<std::add_pointer_t<std::add_const_t<T>>>(object));
      return pos + ar.size();
    }

    /// @brief deserializes object from a buffer
    /// @param[in] object pointer to the object to be deserialized
    /// @param[in] max_nbytes_to_read the maximum number of bytes to read
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be read
    /// @param[in] buf the data buffer that contains the serialized representation of the object
    /// @return position in \p buf after the last byte read
    static uint64_t unpack_payload(void *object, uint64_t max_nbytes_to_read, uint64_t pos, const void *_buf) {
      const unsigned char *buf = reinterpret_cast<const unsigned char *>(_buf);
      madness::archive::BufferInputArchive ar(&buf[pos], max_nbytes_to_read);
      ar & (*static_cast<std::add_pointer_t<T>>(object));
      return pos + (max_nbytes_to_read - ar.nbyte_avail());
    }
  };

}  // namespace ttg

#endif  // has MADNESS serialization

#if defined(TTG_SERIALIZATION_SUPPORTS_BOOST)

#include "ttg/serialization/backends/boost/archive.h"

namespace ttg {

  /// @brief default_data_descriptor for non-POD data types that are not directly copyable, not 2-stage serializable, do not support MADNESS serialization, and support Boost serialization
  template <typename T>
  struct default_data_descriptor<
      T, std::enable_if_t<(!detail::is_memcpyable_v<T> && !detail::is_madness_buffer_serializable_v<T> &&
                           detail::is_boost_buffer_serializable_v<T>) ||
                          (!detail::is_madness_user_buffer_serializable_v<T> &&
                           detail::is_boost_user_buffer_serializable_v<T>)>> {
    static constexpr const bool serialize_size_is_const = false;

    /// @brief measures the size of the binary representation of @p object
    /// @param[in] object pointer to the object to be serialized
    /// @return the number of bytes needed for binary representation of @p object
    static uint64_t payload_size(const void *object) {
      ttg::detail::boost_counting_oarchive oa;
      oa << (*static_cast<std::add_pointer_t<std::add_const_t<T>>>(object));
      return oa.streambuf().size();
    }

    /// @brief serializes object to a buffer
    /// @param[in] object pointer to the object to be serialized
    /// @param[in] max_nbytes_to_write the maximum number of bytes to write
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be written
    /// @param[in,out] buf the data buffer that will contain the serialized representation of the object
    /// @return position in \p buf after the last byte written
    static uint64_t pack_payload(const void *object, uint64_t max_nbytes_to_write, uint64_t pos, void *buf) {
      auto oa = ttg::detail::make_boost_buffer_oarchive(buf, pos + max_nbytes_to_write, pos);
      oa << (*static_cast<std::add_pointer_t<std::add_const_t<T>>>(object));
      assert(oa.streambuf().size() <= max_nbytes_to_write);
      return pos + oa.streambuf().size();
    }

    /// @brief deserializes object from a buffer
    /// @param[in] object pointer to the object to be deserialized
    /// @param[in] max_nbytes_to_read the maximum number of bytes to read
    /// @param[in] offset the position in \p buf where the first byte of serialized data will be read
    /// @param[in] buf the data buffer that contains the serialized representation of the object
    /// @return position in \p buf after the last byte read
    static uint64_t unpack_payload(void *object, uint64_t max_nbytes_to_read, uint64_t pos, const void *buf) {
      auto ia = ttg::detail::make_boost_buffer_iarchive(buf, pos + max_nbytes_to_read, pos);
      ia >> (*static_cast<std::add_pointer_t<T>>(object));
      assert(ia.streambuf().size() <= max_nbytes_to_read);
      return pos + ia.streambuf().size();
    }
  };

}  // namespace ttg

#endif  // has Boost serialization

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
