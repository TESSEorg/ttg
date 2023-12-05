#ifndef TTG_PARSEC_BUFFER_H
#define TTG_PARSEC_BUFFER_H

#include <array>
#include <vector>
#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/mca/device/device.h>
#include "ttg/parsec/ttg_data_copy.h"
#include "ttg/parsec/parsec-ext.h"
#include "ttg/util/iovec.h"
#include "ttg/device/device.h"
#include "ttg/parsec/device.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include <cuda_runtime.h>
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT

namespace ttg_parsec {


namespace detail {
  // fwd decl
  template<typename T, typename A>
  parsec_data_t* get_parsec_data(const ttg_parsec::buffer<T, A>& db);
} // namespace detail

/**
 * A buffer that is mirrored between host memory
 * and different devices. The runtime is free to
 * move data between device and host memory based
 * on where the tasks are executing.
 *
 * Note that a buffer is movable and should not
 * be shared between two objects (e.g., through a pointer)
 * in order for TTG to properly facilitate ownership
 * tracking of the containing object.
 */
template<typename T, typename Allocator>
struct buffer : public detail::ttg_parsec_data_wrapper_t
              , private Allocator {

  using element_type = std::decay_t<T>;

  using allocator_traits = std::allocator_traits<Allocator>;
  using allocator_type = typename  allocator_traits::allocator_type;

  static_assert(std::is_trivially_copyable_v<element_type>,
                "Only trivially copyable types are supported for devices.");
  static_assert(std::is_default_constructible_v<element_type>,
                "Only default constructible types are supported for devices.");

private:
  using delete_fn_t = std::function<void(element_type*)>;

  using host_data_ptr   = std::add_pointer_t<element_type>;
  host_data_ptr m_host_data = nullptr;
  std::size_t m_count = 0;
  bool m_owned= false;

  static void delete_non_owned(element_type *ptr) {
    // nothing to be done, we don't own the memory
  }

  friend parsec_data_t* detail::get_parsec_data<T, Allocator>(const ttg_parsec::buffer<T, Allocator>&);

  allocator_type& get_allocator_reference() { return static_cast<allocator_type&>(*this); }

  element_type* allocate(std::size_t n) {
    return allocator_traits::allocate(get_allocator_reference(), n);
  }

  void deallocate() {
    allocator_traits::deallocate(get_allocator_reference(), m_host_data, m_count);
  }

public:

  buffer() : buffer(nullptr, 0)
  { }

  buffer(std::size_t n)
  : ttg_parsec_data_wrapper_t()
  , allocator_type()
  , m_host_data(allocate(n))
  , m_count(n)
  , m_owned(true)
  {
    //std::cout << "buffer " << this << " ctor count "
    //          << m_count << "(" << m_host_data << ") ttg_copy "
    //          << m_ttg_copy
    //          << " parsec_data " << m_data.get() << std::endl;
    this->reset_parsec_data(m_host_data, n*sizeof(element_type));
  }

  /* Constructing a buffer using application-managed memory.
   * The memory pointed to by ptr must be accessible during
   * the life-time of the buffer. */
  buffer(element_type* ptr, std::size_t n = 1)
  : ttg_parsec_data_wrapper_t()
  , allocator_type()
  , m_host_data(ptr)
  , m_count(n)
  , m_owned(false)
  {
    //std::cout << "buffer " << this << " ctor ptr " << ptr << "count "
    //          << m_count << "(" << m_host_data << ") ttg_copy "
    //          << m_ttg_copy
    //          << " parsec_data " << m_data.get() << std::endl;
    this->reset_parsec_data(m_host_data, n*sizeof(element_type));
  }

  virtual ~buffer() {
    if (m_owned) {
      deallocate();
      m_owned = false;
    }
    unpin(); // make sure the copies are not pinned
  }

  /* allow moving device buffers */
  buffer(buffer&& db)
  : ttg_parsec_data_wrapper_t(std::move(db))
  , allocator_type(std::move(db))
  , m_host_data(db.m_host_data)
  , m_count(db.m_count)
  , m_owned(db.m_owned)
  {
    db.m_host_data = nullptr;
    db.m_count = 0;
    db.m_owned = false;
  }

  /* explicitly disable copying of buffers
   * TODO: should we allow this? What data to use?
   */
  buffer(const buffer& db) = delete;

  /* allow moving device buffers */
  buffer& operator=(buffer&& db) {
    ttg_parsec_data_wrapper_t::operator=(std::move(db));
    allocator_type::operator=(std::move(db));
    std::swap(m_host_data, db.m_host_data);
    std::swap(m_count, db.m_count);
    std::swap(m_owned, db.m_owned);
    //std::cout << "buffer " << this << " other " << &db << " mv op ttg_copy " << m_ttg_copy << std::endl;
    //std::cout << "buffer::move-assign from " << &db << " ttg-copy " << db.m_ttg_copy
    //          << " to " << this << " ttg-copy " << m_ttg_copy
    //          << " parsec-data " << m_data.get()
    //          << std::endl;
    /* don't update the ttg_copy, we keep the connection */
    return *this;
  }

  /* explicitly disable copying of buffers
   * TODO: should we allow this? What data to use?
   */
  buffer& operator=(const buffer& db) = delete;

  /* set the current device, useful when a device
   * buffer was modified outside of a TTG */
  void set_current_device(const ttg::device::Device& device) {
    assert(is_valid());
    int parsec_id = detail::ttg_device_to_parsec_device(device);
    /* make sure it's a valid device */
    assert(parsec_nb_devices > parsec_id);
    /* make sure it's a valid copy */
    assert(m_data->device_copies[parsec_id] != nullptr);
    m_data->owner_device = parsec_id;
  }

  /* Get the owner device ID, i.e., the last updated
   * device buffer. */
  ttg::device::Device get_owner_device() const {
    assert(is_valid());
    return detail::parsec_device_to_ttg_device(m_data->owner_device);
  }

  /* Get the pointer on the currently active device. */
  element_type* current_device_ptr() {
    assert(is_valid());
    int device_id = detail::ttg_device_to_parsec_device(ttg::device::current_device());
    return static_cast<element_type*>(m_data->device_copies[device_id]->device_private);
  }

  /* Get the pointer on the currently active device. */
  const element_type* current_device_ptr() const {
    assert(is_valid());
    int device_id = detail::ttg_device_to_parsec_device(ttg::device::current_device());
    return static_cast<element_type*>(m_data->device_copies[device_id]->device_private);
  }

  /* Get the pointer on the owning device.
   * @note: This may not be the device assigned to the currently executing task.
   *        See \ref ttg::device::current_device for that. */
  element_type* owner_device_ptr() {
    assert(is_valid());
    return static_cast<element_type*>(m_data->device_copies[m_data->owner_device]->device_private);
  }

  /* get the current device pointer */
  const element_type* owner_device_ptr() const {
    assert(is_valid());
    return static_cast<element_type*>(m_data->device_copies[m_data->owner_device]->device_private);
  }

  /* get the device pointer at the given device
   */
  element_type* device_ptr_on(const ttg::device::Device& device) {
    assert(is_valid());
    int device_id = detail::ttg_device_to_parsec_device(device);
    return static_cast<element_type*>(parsec_data_get_ptr(m_data.get(), device_id));
  }

  /* get the device pointer at the given device
   */
  const element_type* device_ptr_on(const ttg::device::Device& device) const {
    assert(is_valid());
    int device_id = detail::ttg_device_to_parsec_device(device);
    return static_cast<element_type*>(parsec_data_get_ptr(m_data.get(), device_id));
  }

  element_type* host_ptr() {
    return static_cast<element_type*>(parsec_data_get_ptr(m_data.get(), 0));
  }

  const element_type* host_ptr() const {
    return static_cast<element_type*>(parsec_data_get_ptr(m_data.get(), 0));
  }

  bool is_valid_on(const ttg::device::Device& device) const {
    assert(is_valid());
    int device_id = detail::ttg_device_to_parsec_device(device);
    return (parsec_data_get_ptr(m_data.get(), device_id) != nullptr);
  }

  void allocate_on(const ttg::device::Device& device_id) {
    /* TODO: need exposed PaRSEC memory allocator */
    throw std::runtime_error("not implemented yet");
  }

  /* TODO: can we do this automatically?
   * Pin the memory on all devices we currently track.
   * Pinned memory won't be released by PaRSEC and can be used
   * at any time.
   */
  void pin() {
    for (int i = 1; i < parsec_nb_devices; ++i) {
      pin_on(i);
    }
  }

  /* Unpin the memory on all devices we currently track. */
  void unpin() {
    if (!is_valid()) return;
    for (int i = 0; i < parsec_nb_devices-detail::first_device_id; ++i) {
      unpin_on(i);
    }
  }

  /* Pin the memory on a given device */
  void pin_on(int device_id) {
    /* TODO: how can we pin memory on a device? */
  }

  /* Pin the memory on a given device */
  void unpin_on(int device_id) {
    /* TODO: how can we unpin memory on a device? */
  }

  bool is_valid() const {
    return !!m_data;
  }

  operator bool() const {
    return is_valid();
  }

  std::size_t size() const {
    return m_count;
  }

  /* Reallocate the buffer with count elements */
  void reset(std::size_t n) {
    /* TODO: can we resize if count is smaller than m_count? */
    /* drop the current data and reallocate */
    reset();

    if (m_owned) {
      deallocate();
      m_owned = false;
    }

    if (n == 0) {
      m_host_data = nullptr;
      m_owned = false;
    } else {
      m_host_data = allocate(n);
      m_owned = true;
    }
    reset_parsec_data(m_host_data, n*sizeof(element_type));
    //std::cout << "buffer::reset(" << count << ") ptr " << m_host_data.get()
    //          << " ttg_copy " << m_ttg_copy
    //          << " parsec_data " << m_data.get() << std::endl;
    m_count = n;
  }

  /* Reset the buffer to use the ptr to count elements */
  void reset(T* ptr, std::size_t n = 1) {
    /* TODO: can we resize if count is smaller than m_count? */
    if (n == m_count) {
      return;
    }

    if (m_owned) {
      deallocate();
    }

    if (nullptr == ptr) {
      m_host_data = nullptr;
      m_count = 0;
      m_owned = false;
    } else {
      m_host_data = ptr;
      m_count = n;
      m_owned = false;
    }
    reset_parsec_data(m_host_data, n*sizeof(element_type));
    //std::cout << "buffer::reset(" << ptr << ", " << count << ") ptr " << m_host_data.get()
    //          << " ttg_copy " << m_ttg_copy
    //          << " parsec_data " << m_data.get() << std::endl;
  }

  /* serialization support */

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>) {
      std::size_t s = size();
      ar& s;
      assert(m_ttg_copy != nullptr); // only tracked objects allowed
      m_ttg_copy->iovec_add(ttg::iovec{s*sizeof(T), current_device_ptr()});
    } else {
      std::size_t s;
      ar & s;
      /* initialize internal pointers and then reset */
      reset(s);
      assert(m_ttg_copy != nullptr); // only tracked objects allowed
      m_ttg_copy->iovec_add(ttg::iovec{s*sizeof(T), current_device_ptr()});
    }
  }
#endif // TTG_SERIALIZATION_SUPPORTS_BOOST

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
  template <class Archive>
  std::enable_if_t<std::is_base_of_v<cereal::detail::InputArchiveBase, Archive> ||
                    std::is_base_of_v<cereal::detail::OutputArchiveBase, Archive>>
  serialize(Archive& ar) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>)
      std::size_t s = size();
      assert(m_ttg_copy != nullptr); // only tracked objects allowed
      m_ttg_copy->iovec_add(ttg::iovec{s*sizeof(T), current_device_ptr()});
      ar(s);
    else {
      std::size_t s;
      ar(s);
      reset(s);
      assert(m_ttg_copy != nullptr); // only tracked objects allowed
      m_ttg_copy->iovec_add(ttg::iovec{s*sizeof(T), current_device_ptr()});
    }
    ar(value);
  }
#endif // TTG_SERIALIZATION_SUPPORTS_CEREAL

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive>
  std::enable_if_t<std::is_base_of_v<madness::archive::BufferInputArchive, Archive> ||
                   std::is_base_of_v<madness::archive::BufferOutputArchive, Archive>>
  serialize(Archive& ar) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>) {
      std::size_t s = size();
      ar& s;
      assert(m_ttg_copy != nullptr); // only tracked objects allowed
      /* transfer from the current device
       * note: if the transport layer (MPI) does not support device transfers
       *       the data will have been pushed out */
      m_ttg_copy->iovec_add(ttg::iovec{s*sizeof(T), current_device_ptr()});
    } else {
      std::size_t s;
      ar & s;
      //std::cout << "serialize(IN) buffer " << this << " size " << s << std::endl;
      /* initialize internal pointers and then reset */
      reset(s);
      assert(m_ttg_copy != nullptr); // only tracked objects allowed
      /* transfer to the current device
       * TODO: how can we make sure the device copy is not evicted? */
      m_ttg_copy->iovec_add(ttg::iovec{s*sizeof(T), current_device_ptr()});
    }
  }
#endif // TTG_SERIALIZATION_SUPPORTS_MADNESS


};

template<typename T>
struct is_buffer : std::false_type
{ };

template<typename T, typename A>
struct is_buffer<buffer<T, A>> : std::true_type
{ };

template<typename T, typename A>
struct is_buffer<const buffer<T, A>> : std::true_type
{ };

template<typename T>
constexpr static const bool is_buffer_v = is_buffer<T>::value;

namespace detail {
  template<typename T, typename A>
  parsec_data_t* get_parsec_data(const ttg_parsec::buffer<T, A>& db) {
    return const_cast<parsec_data_t*>(db.m_data.get());
  }
} // namespace detail

} // namespace ttg_parsec

#endif // TTG_PARSEC_BUFFER_H