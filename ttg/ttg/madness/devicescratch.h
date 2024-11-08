#ifndef TTG_MADNESS_DEVICESCRATCH_H
#define TTG_MADNESS_DEVICESCRATCH_H

#include <ttg/devicescope.h>

namespace ttg_madness {

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

  element_type* m_data = nullptr;
  std::size_t m_count = 0;
  ttg::scope m_scope;

public:

  /* Constructing a devicescratch using application-managed memory.
   * The memory pointed to by ptr must be accessible during
   * the life-time of the devicescratch. */
  devicescratch(element_type* ptr, ttg::scope scope = ttg::scope::SyncIn, std::size_t count = 1)
  : m_data(ptr)
  , m_count(count)
  , m_scope(scope)
  { }

  /* don't allow moving */
  devicescratch(devicescratch&&) = delete;

  /* don't allow copying */
  devicescratch(const devicescratch& db) = delete;

  /* don't allow moving */
  devicescratch& operator=(devicescratch&&) = delete;

  /* don't allow copying */
  devicescratch& operator=(const devicescratch& db) = delete;

  ~devicescratch() {
    m_data = nullptr;
    m_count = 0;
  }

  /* get the current device pointer (only host supported) */
  element_type* device_ptr() {
    assert(is_valid());
    return m_data;
  }

  /* get the current device pointer */
  const element_type* device_ptr() const {
    assert(is_valid());
    return m_data;
  }

  bool is_valid() const {
    return true;
  }

  ttg::scope scope() const {
    return m_scope;
  }

  std::size_t size() const {
    return m_count;
  }

};

} // namespace ttg_madness

#endif // TTG_MADNESS_DEVICESCRATCH_H