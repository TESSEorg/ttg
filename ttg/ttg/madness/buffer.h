#ifndef TTG_MADNESS_BUFFER_H
#define TTG_MADNESS_BUFFER_H

#include "ttg/serialization/traits.h"

#include <memory>

namespace ttg_madness {

/// A runtime-managed buffer mirrored between host and device memory
template<typename T, typename Allocator>
struct Buffer : private Allocator {

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
  std::shared_ptr<element_type[]> m_sptr; // to capture smart pointers
  host_data_ptr m_host_data = nullptr;
  std::size_t m_count = 0;
  bool m_owned= false;

  static void delete_non_owned(element_type *ptr) {
    // nothing to be done, we don't own the memory
  }

  allocator_type& get_allocator_reference() { return static_cast<allocator_type&>(*this); }

  element_type* allocate(std::size_t n) {
    return allocator_traits::allocate(get_allocator_reference(), n);
  }

  void deallocate() {
    allocator_traits::deallocate(get_allocator_reference(), m_host_data, m_count);
  }

public:

  Buffer() : Buffer(nullptr, 0)
  { }

  Buffer(std::size_t n, ttg::scope scope = ttg::scope::SyncIn)
  : allocator_type()
  , m_host_data(allocate(n))
  , m_count(n)
  , m_owned(true)
  { }

  /* Constructing a buffer using application-managed memory.
   * The memory pointed to by ptr must be accessible during
   * the life-time of the buffer. */
  template<typename Deleter>
  Buffer(std::unique_ptr<element_type[], Deleter> ptr, std::size_t n, ttg::scope scope = ttg::scope::SyncIn)
  : allocator_type()
  , m_sptr(std::move(ptr))
  , m_host_data(m_sptr.get())
  , m_count(n)
  , m_owned(false)
  { }

  /* Constructing a buffer using application-managed memory.
   * The memory pointed to by ptr must be accessible during
   * the life-time of the buffer. */
  Buffer(std::shared_ptr<element_type[]> ptr, std::size_t n, ttg::scope scope = ttg::scope::SyncIn)
  : allocator_type()
  , m_sptr(std::move(ptr))
  , m_host_data(m_sptr.get())
  , m_count(n)
  , m_owned(false)
  { }

  virtual ~Buffer() {
    if (m_owned) {
      deallocate();
      m_owned = false;
    }
    unpin(); // make sure the copies are not pinned
  }

  /* allow moving device buffers */
  Buffer(Buffer&& db)
  : allocator_type(std::move(db))
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
  Buffer(const Buffer& db) = delete;

  /* allow moving device buffers */
  Buffer& operator=(Buffer&& db) {
    allocator_type::operator=(std::move(db));
    std::swap(m_host_data, db.m_host_data);
    std::swap(m_count, db.m_count);
    std::swap(m_owned, db.m_owned);
    return *this;
  }

  /* explicitly disable copying of buffers
   * TODO: should we allow this? What data to use?
   */
  Buffer& operator=(const Buffer& db) = delete;

  /* set the current device, useful when a device
   * buffer was modified outside of a TTG */
  void set_current_device(const ttg::device::Device& device) {
    assert(is_valid());
    if (!device.is_host()) throw std::runtime_error("MADNESS backend does not support non-host memory!");
    /* no-op */
  }

  /* Get the owner device ID, i.e., the last updated
   * device buffer. */
  ttg::device::Device get_owner_device() const {
    assert(is_valid());
    return {}; // host only
  }

  /* Get the pointer on the currently active device. */
  element_type* current_device_ptr() {
    assert(is_valid());
    return m_host_data;
  }

  /* Get the pointer on the currently active device. */
  const element_type* current_device_ptr() const {
    assert(is_valid());
    return m_host_data;
  }

  /* Get the pointer on the owning device.
   * @note: This may not be the device assigned to the currently executing task.
   *        See \ref ttg::device::current_device for that. */
  element_type* owner_device_ptr() {
    assert(is_valid());
    return m_host_data;
  }

  /* get the current device pointer */
  const element_type* owner_device_ptr() const {
    assert(is_valid());
    return m_host_data;
  }

  /* get the device pointer at the given device
   */
  element_type* device_ptr_on(const ttg::device::Device& device) {
    assert(is_valid());
    if (device.is_device()) throw std::runtime_error("MADNESS missing support for non-host memory!");
    return m_host_data;
  }

  /* get the device pointer at the given device
   */
  const element_type* device_ptr_on(const ttg::device::Device& device) const {
    assert(is_valid());
    if (device.is_device()) throw std::runtime_error("MADNESS missing support for non-host memory!");
    return m_host_data;
  }

  element_type* host_ptr() {
    return m_host_data;
  }

  const element_type* host_ptr() const {
    return m_host_data;
  }

  bool is_valid_on(const ttg::device::Device& device) const {
    assert(is_valid());
    if (device.is_device()) throw std::runtime_error("MADNESS missing support for non-host memory!");
    return true;
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
    // nothing to do
  }

  /* Unpin the memory on all devices we currently track. */
  void unpin() {
    // nothing to do
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
    return true;
  }

  operator bool() const {
    return true;
  }

  std::size_t size() const {
    return m_count;
  }

  /* Reallocate the buffer with count elements */
  void reset(std::size_t n, ttg::scope scope = ttg::scope::SyncIn) {

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
    m_count = n;
  }

  /**
   * Resets the scope of the buffer. Ignored in MADNESS.
   */
  void reset_scope(ttg::scope scope) {
    /* nothing to do here */
  }

  /* serialization support */

#if defined(TTG_SERIALIZATION_SUPPORTS_MADNESS)
  template <typename Archive>
  std::enable_if_t<std::is_base_of_v<madness::archive::BufferInputArchive, Archive> ||
                   std::is_base_of_v<madness::archive::BufferOutputArchive, Archive>>
  serialize(Archive& ar) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>) {
      std::size_t s = size();
      ar& s;
      ar << madness::archive::wrap(host_ptr(), s);
    } else {
      std::size_t s;
      ar & s;
      reset(s);
      ar >> madness::archive::wrap(host_ptr(), s);  // MatrixTile<T>(bm.rows(), bm.cols());
    }
  }
#endif // TTG_SERIALIZATION_SUPPORTS_MADNESS


};

} // namespace ttg_madness

#endif // TTG_MADNESS_BUFFER_H
