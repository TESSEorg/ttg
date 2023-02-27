#ifndef TTG_BUFFER_H
#define TTG_BUFFER_H

#include <memory>

namespace ttg {

template<typename T>
using buffer = TTG_IMPL_NS::buffer<T>;

template<typename T>
struct hostbuf {
private:
  T* m_ptr = nullptr;
  std::size_t m_count = 0;

public:
  hostbuf() = default;

  hostbuf(T* ptr, std::size_t count)
  : m_ptr(ptr)
  , m_count(count)
  { }

  T* ptr() const {
    return m_ptr;
  }

  std::size_t count() const {
    return m_count;
  }

  std::size_t size() const {
    return m_count * sizeof(T);
  }
};

/* Applications may override this trait to
 * expose device and host buffer members of
 * data structures. Example:
 * struct data_t {
 *   ttg::buffer<double> db;
 *   double *more_host_data;
 *   int size;
 * };
 * template<>
 * struct container_trait<data_t> {
 *   static auto buffers() {
 *     return std::make_tuple(db);
 *   }
 * };
 */
template<typename T>
struct container_trait;

} // namespace ttg

#endif // TTG_buffer_H