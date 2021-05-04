#ifndef TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H
#define TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H

#include <type_traits>
#include "ttg/util/meta.h"

namespace ttg {

  /**
   * Used to describe transfer payload in types using the \sa SplitMetadataDescriptor.
   * @member data Pointer to the data to be read from / written to.
   * @member num_bytes The number of bytes to read from / write to the memory location
   *                   \sa data.
   */
  struct iovec {
    size_t num_bytes;
    void*     data;
  };

  /**
   * SplitMetadataDescriptor is a serialization descriptor provided by the user
   * for a user-specified type. I should contain the following public member
   * functions:
   *
   * <metadata_type> get_metadata(const T& t);
   *
   * Returns the metadata that describes the object, e.g., shape information.
   * This data will be passed to
   *
   * auto create_from_metadata(const <metadata_type>& meta);
   *
   * which is expected to return a new instance of T, initialized with the
   * previously provided metadata. This instance will be deserialization target.
   *
   * Both the serialization source and the deserialization target objects will
   * be passed to
   *
   * auto get_data(T& t);
   *
   * which is expected to return a collection of \sa ttg::iovec instances
   * describing the payload data to be transferred from the source to the
   * target object.
   */
  template<typename T>
  struct SplitMetadataDescriptor;

  /* Trait signalling whether metadata and data payload can be transfered separately */
  template<typename T, typename Enabler = void>
  struct has_split_metadata : std::false_type
  { };

  template<typename T>
  struct has_split_metadata<T, ttg::meta::void_t<decltype(std::declval<SplitMetadataDescriptor<T>>().get_metadata(std::declval<T>()))>>
  : std::true_type
  { };

}  // namespace ttg

#endif  // TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H
