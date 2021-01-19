#ifndef TTG_UTIL_DATA_DESCRIPTOR_H
#define TTG_UTIL_DATA_DESCRIPTOR_H

#include <cstdint>

// This provides an efficent API for serializing/deserializing a data type.
// An object of this type will need to be provided for each serializable type.
// The default implementation, in serialization.h, works only for primitive/POD data types;
// backend-specific implementations may be available in backend/serialization.h .
extern "C" struct ttg_data_descriptor {
  const char *name;
  void (*get_info)(const void *object, uint64_t *hs, uint64_t *ps, int *is_contiguous_mask, void **buf);
  void (*pack_header)(const void *object, uint64_t header_size, void **buf);
  uint64_t (*pack_payload)(const void *object, uint64_t chunk_size, uint64_t pos, void *buf);
  void (*unpack_header)(void *object, uint64_t header_size, const void *buf);
  void (*unpack_payload)(void *object, uint64_t chunk_size, uint64_t pos, const void *buf);
  void (*print)(const void *object);
};

namespace ttg {

  template <typename T, typename Enabler>
  struct default_data_descriptor;

  // Returns a pointer to a constant static instance initialized
  // once at run time.
  template <typename T>
  const ttg_data_descriptor *get_data_descriptor();

}  // namespace ttg


#endif // TTG_UTIL_DATA_DESCRIPTOR_H
