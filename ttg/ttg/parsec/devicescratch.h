#ifndef TTG_PARSEC_DEVICESCRATCH_H
#define TTG_PARSEC_DEVICESCRATCH_H

// TODO: replace with short vector
#define TTG_PARSEC_MAX_NUM_DEVICES 4

#include <array>
#include <parsec.h>
#include <parsec/data_internal.h>
#include <parsec/mca/device/device.h>
#include <ttg/devicescope.h>

namespace ttg_parsec {

namespace detail {
  // fwd decl
  template<typename T>
  parsec_data_t* get_parsec_data(const ttg_parsec::devicescratch<T>&);
} // namespace detail

/**
 * Scratch-space for task-local variables.
 * TTG will allocate memory on the device
 * and transfer data in and out based on the scope.
 */
template<typename T>
struct devicescratch {

  using element_type = std::decay_t<T>;

  static_assert(std::is_trivially_copyable_v<element_type>,
                "Only trivially copyable types are supported for devices.");
  static_assert(std::is_default_constructible_v<element_type>,
                "Only default constructible types are supported for devices.");

private:

  parsec_data_t* m_data = nullptr;
  parsec_data_copy_t m_data_copy;
  ttg::scope m_scope;

  void create_host_copy(element_type *ptr, std::size_t count) {
    /* TODO: is the construction call necessary? */
    /* TODO: handle the scope */
    PARSEC_OBJ_CONSTRUCT(&m_data_copy, parsec_data_copy_t);
    m_data_copy.device_index    = 0;
    //m_data_copy.original        = &m_data;
    //m_data_copy.older           = NULL;
    m_data_copy.flags           = PARSEC_DATA_FLAG_PARSEC_MANAGED;
    m_data_copy.dtt             = parsec_datatype_int8_t;
    m_data_copy.version         = 1;
    m_data_copy.device_private  = ptr;
    m_data_copy.coherency_state = PARSEC_DATA_COHERENCY_SHARED;

    m_data->nb_elts              = count * sizeof(element_type);
    m_data->owner_device         = 0;
    parsec_data_copy_attach(m_data, &m_data_copy, 0);
  }

  friend parsec_data_t* detail::get_parsec_data<T>(const ttg_parsec::devicescratch<T>&);

public:

  /* Constructing a devicescratch using application-managed memory.
   * The memory pointed to by ptr must be accessible during
   * the life-time of the devicescratch. */
  devicescratch(element_type* ptr, ttg::scope scope = ttg::scope::SyncIn, std::size_t count = 1)
  : m_data(parsec_data_new())
  , m_scope(scope) {
    create_host_copy(ptr, count);
  }

  /* don't allow moving */
  devicescratch(devicescratch&&) = delete;

  /* don't allow copying */
  devicescratch(const devicescratch& db) = delete;

  /* don't allow moving */
  devicescratch& operator=(devicescratch&&) = delete;

  /* don't allow copying */
  devicescratch& operator=(const devicescratch& db) = delete;

  ~devicescratch() {
    PARSEC_OBJ_DESTRUCT(&m_data_copy);
    parsec_data_destroy(m_data);
    m_data = nullptr;
  }

  /* get the current device pointer */
  element_type* device_ptr() {
    assert(is_valid());
    return static_cast<element_type*>(m_data->device_copies[m_data->owner_device]->device_private);
  }

  /* get the current device pointer */
  const element_type* device_ptr() const {
    assert(is_valid());
    return static_cast<element_type*>(m_data->device_copies[m_data->owner_device]->device_private);
  }

  bool is_valid() const {
    // TODO: how to get the current device
    // return (m_data->owner_device == parsec_current_device);
    return true;
  }

  ttg::scope scope() const {
    return m_scope;
  }

  std::size_t size() const {
    return (m_data->nb_elts / sizeof(element_type));
  }

};

template<typename T>
struct is_devicescratch : std::false_type
{ };

template<typename T>
struct is_devicescratch<devicescratch<T>> : std::true_type
{ };

template<typename T>
constexpr static const bool is_devicescratch_v = is_devicescratch<T>::value;

namespace detail {
  template<typename T>
  parsec_data_t* get_parsec_data(const ttg_parsec::devicescratch<T>& scratch) {
    return const_cast<parsec_data_t*>(scratch.m_data);
  }
} // namespace detail

} // namespace ttg_parsec

#endif // TTG_PARSEC_DEVICESCRATCH_H