#ifndef CXXAPI_SERIALIZATION_H_H
#define CXXAPI_SERIALIZATION_H_H

namespace ttg {

template <typename T, typename Enabler = void>
struct default_data_descriptor;

// The default implementation is only provided for POD data types
template <typename T>
struct default_data_descriptor<T, std::enable_if_t<std::is_pod<T>::value>> {
  static uint64_t payload_size(const void* /* object */)
  {
    return static_cast<int64_t>(sizeof(T));
  }

  static uint64_t header_size(const void* /* object */) {return static_cast<uint64_t>(0);}

  static void get_info(const void* object, uint64_t* hs, uint64_t* ps, int* is_contiguous_mask, void** ptr) {
    *hs = header_size(object);
    *ps = payload_size(object);
    *is_contiguous_mask = 1;
    *ptr = nullptr;
  }

  static void pack_header(const void* object, uint64_t header_size, void**buf) {}

  /// t --- obj to be serialized
  /// chunk_size --- inputs max amount of data to output, and on output returns amount actually output
  /// pos --- position in the input buffer to resume serialization
  /// ptr[chunk_size] --- place for output
  static void pack_payload(const void* object, uint64_t* chunk_size, uint64_t pos, void* buf)
  {
    std::memcpy(buf, object, *chunk_size);
  }

  // t points to some memory in which we will construct an object from the header
  static void unpack_header(void* object, uint64_t header_size, const void* buf) {
    assert(header_size == 0);
    new (object) T;
  }

  static void unpack_payload(void* object, uint64_t chunk_size, uint64_t pos, void* buf) {
    std::memcpy(object, buf, chunk_size);
  }

  static void print(const void *object) {
    std::cout << *(T*)object << std::endl;
  }
};

// Returns a pointer to a constant static instance initialized
// once at run time.  Call this from a piece of C++ code (see
// example below) and return the pointer to C.
template <typename T>
const ttg_data_descriptor*
get_data_descriptor() {
  static const ttg_data_descriptor d = {typeid(T).name(),
                                        &default_data_descriptor<T>::payload_size,
                                        &default_data_descriptor<T>::header_size,
                                        &default_data_descriptor<T>::get_info,
                                        &default_data_descriptor<T>::pack_header,
                                        &default_data_descriptor<T>::pack_payload,
                                        &default_data_descriptor<T>::unpack_header,
                                        &default_data_descriptor<T>::unpack_payload,
                                        &default_data_descriptor<T>::print};
  return &d;
}

}  // namespace ttg

#endif //CXXAPI_SERIALIZATION_H_H
