#ifndef TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H
#define TTG_SERIALIZATION_SPLITMD_DATA_DESCRIPTOR_H

#include <type_traits>

namespace ttg {

  struct iovec {
    size_t num_bytes;
    void*     data;
  };  

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
