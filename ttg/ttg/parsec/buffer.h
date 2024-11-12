#ifndef TTG_PARSEC_BUFFER_H
#define TTG_PARSEC_BUFFER_H

#include <array>
#include <vector>
#include <cassert>
#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/mca/device/device.h>
#include "ttg/parsec/ttg_data_copy.h"
#include "ttg/parsec/parsec-ext.h"
#include "ttg/util/iovec.h"
#include "ttg/device/device.h"
#include "ttg/parsec/device.h"
#include "ttg/devicescope.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include <cuda_runtime.h>
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT

namespace ttg_parsec {


namespace detail {
  // fwd decl
  template<typename T, typename A>
  parsec_data_t* get_parsec_data(const ttg_parsec::Buffer<T, A>& db);

  template<typename T>
  struct empty_allocator {

    using value_type = std::decay_t<T>;

    value_type* allocate(std::size_t size) {
      throw std::runtime_error("Allocate on empty allocator!");
    }

    void deallocate(value_type* ptr, std::size_t size) {
      throw std::runtime_error("Deallocate on empty allocator!");
    }
  };

  /* overloads for pointers and smart pointers */
  template<typename T>
  inline T* to_address(T* ptr) {
    return ptr;
  }

  template<typename T>
  inline auto to_address(T&& ptr) {
    return ptr.get(); // smart pointer
  }

  /**
   * Wrapper type to carry the Allocator into types that are using
   * the PaRSEC object system.
   */
  template<typename PtrT, typename Allocator>
  struct ttg_parsec_data_types {
    using allocator_traits = std::allocator_traits<Allocator>;
    using allocator_type = typename allocator_traits::allocator_type;
    using value_type = typename allocator_traits::value_type;

    /* used as a hook into the PaRSEC object management system
     * so we can release the memory back to the allocator once
     * data copy is destroyed */
    struct data_copy_type : public parsec_data_copy_t
    {
    private:
      [[no_unique_address]]
      allocator_type m_allocator;
      PtrT m_ptr; // keep a reference if PtrT is a shared_ptr
      std::size_t m_size;

      void allocate(std::size_t size) {
        if constexpr (std::is_pointer_v<PtrT>) {
          m_ptr = allocator_traits::allocate(m_allocator, size);
        }
        this->device_private = m_ptr;
        m_size = size;
      }

      void deallocate() {
        allocator_traits::deallocate(m_allocator, static_cast<value_type*>(this->device_private), this->m_size);
        this->device_private = nullptr;
        this->m_size = 0;
      }

    public:

      /* allocation through PARSEC_OBJ_NEW, so no inline constructor */

      void construct(PtrT ptr, std::size_t size) {
        m_allocator = allocator_type{};
        constexpr const bool is_empty_allocator = std::is_same_v<Allocator, empty_allocator<value_type>>;
        assert(is_empty_allocator);
        m_ptr = std::move(ptr);
        this->device_private = to_address(m_ptr);
      }

      void construct(std::size_t size, const allocator_type& alloc = allocator_type()) {
        constexpr const bool is_empty_allocator = std::is_same_v<Allocator, empty_allocator<value_type>>;
        assert(!is_empty_allocator);
        m_allocator = alloc;
        allocate(size);
        this->device_private = m_ptr;
      }

      ~data_copy_type() {
        this->deallocate();
      }
    };

#if 0
    /**
     * derived from parsec_data_t managing an allocator for host memory
     * and creating a host copy through a callback
     */
    struct data_type : public parsec_data_t {
    private:
      [[no_unique_address]]
      Allocator host_allocator;

      allocator_type& get_allocator_reference() { return static_cast<allocator_type&>(*this); }

      element_type* do_allocate(std::size_t n) {
        return allocator_traits::allocate(get_allocator_reference(), n);
      }

    public:

      explicit data_type(const Allocator& host_alloc = Allocator())
      : host_allocator(host_alloc)
      { }

      void* allocate(std::size_t size) {
        return do_allocate(size);
      }

      void deallocate(void* ptr, std::size_t size) {
        allocator_traits::deallocate(get_allocator_reference(), ptr, size);
      }
    };

    /* Allocate the host copy for a given data_t */
    static parsec_data_copy_t* data_copy_allocate(parsec_data_t* pdata, int device) {
      if (device != 0) return nullptr; // we only allocate for the host
      data_type* data = static_cast<data_type*>(pdata); // downcast
      data_copy_type* copy = PARSEC_OBJ_NEW(data_copy_type);
      copy->device_private = data->allocate(data->nb_elts);
      return copy;
    }
    /**
    * Create the PaRSEC object infrastructure for the data copy type
    */
    inline void data_construct(data_type* obj)
    {
      obj->allocate_cb = &data_copy_allocate;
    }

    static void data_destruct(data_type* obj)
    {
      /* cleanup alloctor instance */
      obj->~data_type();
    }

    static constexpr PARSEC_OBJ_CLASS_INSTANCE(data_type, parsec_data_t,
                                               data_copy_construct,
                                               data_copy_destruct);

#endif // 0

    /**
     * Create the PaRSEC object infrastructure for the data copy type
     */
    static void data_copy_construct(data_copy_type* obj)
    {
      /* nothing to do */
    }

    static void data_copy_destruct(data_copy_type* obj)
    {
      obj->~data_copy_type(); // call destructor
    }

    inline static PARSEC_OBJ_CLASS_INSTANCE(data_copy_type, parsec_data_copy_t,
                                            data_copy_construct,
                                            data_copy_destruct);

    static parsec_data_t * create_data(std::size_t size,
                                       const allocator_type& allocator = allocator_type()) {
      parsec_data_t *data = PARSEC_OBJ_NEW(parsec_data_t);
      data->owner_device = 0;
      data->nb_elts = size;

      /* create the host copy and allocate host memory */
      data_copy_type *copy = PARSEC_OBJ_NEW(data_copy_type);
      copy->construct(size, allocator);
      parsec_data_copy_attach(data, copy, 0);

      /* adjust data flags */
      data->device_copies[0]->flags |= PARSEC_DATA_FLAG_PARSEC_MANAGED;
      data->device_copies[0]->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
      data->device_copies[0]->version = 1;

      return data;
    }

    static parsec_data_t * create_data(PtrT& ptr, std::size_t size) {
      parsec_data_t *data = PARSEC_OBJ_NEW(parsec_data_t);
      data->owner_device = 0;
      data->nb_elts = size;

      /* create the host copy and allocate host memory */
      data_copy_type *copy = PARSEC_OBJ_NEW(data_copy_type);
      copy->construct(std::move(ptr), size);
      parsec_data_copy_attach(data, copy, 0);

      /* adjust data flags */
      data->device_copies[0]->flags |= PARSEC_DATA_FLAG_PARSEC_MANAGED;
      data->device_copies[0]->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
      data->device_copies[0]->version = 1;

      return data;
    }
  };
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
struct Buffer {

  /* TODO: add overloads for T[]? */
  using value_type = std::remove_all_extents_t<T>;
  using pointer_type = std::add_pointer_t<value_type>;
  using const_pointer_type = const std::remove_const_t<value_type>*;
  using element_type = std::decay_t<T>;

  static_assert(std::is_trivially_copyable_v<element_type>,
                "Only trivially copyable types are supported for devices.");

private:
  using delete_fn_t = std::function<void(element_type*)>;

  using host_data_ptr = std::add_pointer_t<element_type>;
  parsec_data_t *m_data = nullptr;
  std::size_t m_count = 0;

  friend parsec_data_t* detail::get_parsec_data<T>(const ttg_parsec::Buffer<T, Allocator>&);


  void release_data() {
    if (nullptr == m_data) return;
    /* discard the parsec data so it can be collected by the runtime
     * and the buffer be free'd in the parsec_data_copy_t destructor */
    parsec_data_discard(m_data);
    /* set data to null so we don't repeat the above */
    m_data = nullptr;
  }

public:

  Buffer()
  { }

  Buffer(std::size_t n)
  : m_data(detail::ttg_parsec_data_types<T*, Allocator>::create_data(n*sizeof(element_type)))
  , m_count(n)
  { }

  /**
   * Constructing a buffer using application-managed memory.
   * The shared_ptr will ensure that the memory is not free'd before
   * the runtime has released all of its references.
   */
  Buffer(std::shared_ptr<element_type[]> ptr, std::size_t n)
  : m_data(detail::ttg_parsec_data_types<std::shared_ptr<element_type[]>,
                                         detail::empty_allocator<element_type>>
                                        ::create_data(ptr, n*sizeof(element_type)))
  , m_count(n)
  { }

  template<typename Deleter>
  Buffer(std::unique_ptr<element_type[], Deleter> ptr, std::size_t n)
  : m_data(detail::ttg_parsec_data_types<std::unique_ptr<element_type[], Deleter>,
                                         detail::empty_allocator<element_type>>
                                        ::create_data(ptr, n*sizeof(element_type)))
  , m_count(n)
  { }

  virtual ~Buffer() {
    release_data();
    unpin(); // make sure the copies are not pinned
  }

  /* allow moving device buffers */
  Buffer(Buffer&& db)
  : m_data(db.m_data)
  , m_count(db.m_count)
  {
    db.m_data = nullptr;
    db.m_count = 0;
  }

  /* explicitly disable copying of buffers
   * TODO: should we allow this? What data to use?
   */
  Buffer(const Buffer& db) = delete;

  /* allow moving device buffers */
  Buffer& operator=(Buffer&& db) {
    std::swap(m_data, db.m_data);
    std::swap(m_count, db.m_count);
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
  Buffer& operator=(const Buffer& db) = delete;

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

  bool is_current_on(ttg::device::Device dev) const {
    if (empty()) return true; // empty is current everywhere
    int parsec_id = detail::ttg_device_to_parsec_device(dev);
    uint32_t max_version = 0;
    for (int i = 0; i < parsec_nb_devices; ++i) {
      if (nullptr == m_data->device_copies[i]) continue;
      max_version = std::max(max_version, m_data->device_copies[i]->version);
    }
    return (m_data->device_copies[parsec_id] &&
            m_data->device_copies[parsec_id]->version == max_version);
  }

  /* Get the owner device ID, i.e., the last updated
   * device buffer.
   * NOTE: there may be more than one device with the current
   *       data so the result may not always be what is expected.
   *       Use is_current_on() to check for a specific device. */
  ttg::device::Device get_owner_device() const {
    assert(is_valid());
    if (empty()) return ttg::device::current_device(); // empty is always valid
    return detail::parsec_device_to_ttg_device(m_data->owner_device);
  }

  /* Get the pointer on the currently active device. */
  pointer_type current_device_ptr() {
    assert(is_valid());
    if (empty()) return nullptr;
    int device_id = detail::ttg_device_to_parsec_device(ttg::device::current_device());
    return static_cast<pointer_type>(m_data->device_copies[device_id]->device_private);
  }

  /* Get the pointer on the currently active device. */
  const_pointer_type current_device_ptr() const {
    assert(is_valid());
    if (empty()) return nullptr;
    int device_id = detail::ttg_device_to_parsec_device(ttg::device::current_device());
    return static_cast<const_pointer_type>(m_data->device_copies[device_id]->device_private);
  }

  /* Get the pointer on the owning device.
   * @note: This may not be the device assigned to the currently executing task.
   *        See \ref ttg::device::current_device for that. */
  pointer_type owner_device_ptr() {
    assert(is_valid());
    if (empty()) return nullptr;
    return static_cast<pointer_type>(m_data->device_copies[m_data->owner_device]->device_private);
  }

  /* get the current device pointer */
  const_pointer_type owner_device_ptr() const {
    assert(is_valid());
    if (empty()) return nullptr;
    return static_cast<const_pointer_type>(m_data->device_copies[m_data->owner_device]->device_private);
  }

  /* get the device pointer at the given device
   */
  pointer_type device_ptr_on(const ttg::device::Device& device) {
    assert(is_valid());
    if (empty()) return nullptr;
    int device_id = detail::ttg_device_to_parsec_device(device);
    return static_cast<pointer_type>(parsec_data_get_ptr(m_data, device_id));
  }

  /* get the device pointer at the given device
   */
  const_pointer_type device_ptr_on(const ttg::device::Device& device) const {
    assert(is_valid());
    if (empty()) return nullptr;
    int device_id = detail::ttg_device_to_parsec_device(device);
    return static_cast<const_pointer_type>(parsec_data_get_ptr(m_data, device_id));
  }

  pointer_type host_ptr() {
    return static_cast<pointer_type>(parsec_data_get_ptr(m_data, 0));
  }

  const_pointer_type host_ptr() const {
    return static_cast<const_pointer_type>(parsec_data_get_ptr(m_data, 0));
  }

  bool is_valid_on(const ttg::device::Device& device) const {
    assert(is_valid());
    int device_id = detail::ttg_device_to_parsec_device(device);
    return (parsec_data_get_ptr(m_data, device_id) != nullptr);
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
    return (m_count == 0 || m_data);
  }

  operator bool() const {
    return !empty();
  }

  std::size_t size() const {
    return m_count;
  }

  bool empty() const {
    return m_count == 0;
  }

  /* Reallocate the buffer with count elements */
  void reset(std::size_t n, ttg::scope scope = ttg::scope::SyncIn) {
    release_data();

    if (n > 0) {
      m_data = detail::ttg_parsec_data_types<element_type*, Allocator>
                                            ::create_data(n*sizeof(element_type));
    }

    m_count = n;
  }

  /**
   * Resets the scope of the buffer.
   * If scope is SyncIn then the next time
   * the buffer is made available on a device the host
   * data will be copied from the host.
   * If scope is Allocate then no data will be moved.
   */
  void reset_scope(ttg::scope scope) {
    if (scope == ttg::scope::Allocate) {
      m_data->device_copies[0]->version = 0;
    } else {
      m_data->device_copies[0]->version = 1;
      /* reset all other copies to force a sync-in */
      for (int i = 0; i < parsec_nb_devices; ++i) {
        if (m_data->device_copies[i] != nullptr) {
          m_data->device_copies[i]->version = 0;
        }
      }
    }
    m_data->owner_device = 0;
  }

  ttg::scope scope() const {
    /* if the host owns the data and has a version of zero we only have to allocate data */
    return (m_data->device_copies[0]->version == 0 && m_data->owner_device == 0)
            ? ttg::scope::Allocate : ttg::scope::SyncIn;
  }

  void prefer_device(ttg::device::Device dev) {
    /* only set device if the host has the latest copy as otherwise we might end up with a stale copy */
    if (dev.is_device() && this->parsec_data()->owner_device == 0) {
      parsec_advise_data_on_device(this->parsec_data(), detail::ttg_device_to_parsec_device(dev),
                                   PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE);
    }
  }

  void add_device(ttg::device::Device dev, pointer_type ptr, bool is_current = false) {
    if (is_valid_on(dev)) {
      throw std::runtime_error("Unable to add device that has already a buffer set!");
    }
    add_copy(detail::ttg_device_to_parsec_device(dev), ptr);
    if (is_current) {
      // mark the data as being current on the new device
      parsec_data()->owner_device = detail::ttg_device_to_parsec_device(dev);
    }
  }

  /* serialization support */

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>) {
      std::size_t s = size();
      ar& s;
    } else {
      std::size_t s;
      ar & s;
      /* initialize internal pointers and then reset */
      reset(s);
    }
  }
#endif // TTG_SERIALIZATION_SUPPORTS_BOOST

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive>
  std::enable_if_t<std::is_base_of_v<madness::archive::BufferInputArchive, Archive> ||
                   std::is_base_of_v<madness::archive::BufferOutputArchive, Archive>>
  serialize(Archive& ar) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>) {
      std::size_t s = size();
      ar& s;
    } else {
      std::size_t s;
      ar & s;
      //std::cout << "serialize(IN) buffer " << this << " size " << s << std::endl;
      /* initialize internal pointers and then reset */
      reset(s);
    }
  }
#endif // TTG_SERIALIZATION_SUPPORTS_MADNESS
};

namespace detail {
  template<typename T, typename A>
  parsec_data_t* get_parsec_data(const ttg_parsec::Buffer<T, A>& db) {
    return const_cast<parsec_data_t*>(db.m_data);
  }
} // namespace detail

} // namespace ttg_parsec

#endif // TTG_PARSEC_BUFFER_H
