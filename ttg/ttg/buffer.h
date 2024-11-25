#ifndef TTG_BUFFER_H
#define TTG_BUFFER_H

#include <memory>

#include "ttg/fwd.h"
#include "ttg/serialization.h"
#include <memory>

namespace ttg {

template<typename T, typename Allocator = std::allocator<std::decay_t<T>>>
using Buffer = TTG_IMPL_NS::Buffer<T, Allocator>;

namespace meta {

  template<typename T, typename A>
  struct is_buffer<ttg::Buffer<T, A>> : std::true_type
  { };

  template<typename T, typename A>
  struct is_buffer<const ttg::Buffer<T, A>> : std::true_type
  { };

} // namespace meta

namespace detail {
  /**
   * Type traits to check whether we can use serialization
   * to inspect the buffers owned by an object passing
   * through a task graph.
   */
  template<typename T, typename Enabler = void>
  struct has_buffer_apply_helper : std::false_type
  { };

  template<typename T>
  struct has_buffer_apply : has_buffer_apply_helper<T>
  { };

  template<typename T>
  constexpr const bool has_buffer_apply_v = has_buffer_apply<T>::value;

} // namespace detail

} // namespace ttg



#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
#include <madness/world/buffer_archive.h>

namespace madness {
  namespace archive {
    template<typename Fn>
    struct BufferVisitorArchive : public madness::archive::BaseOutputArchive {
    private:
      Fn m_fn;

    public:
      template<typename _Fn>
      BufferVisitorArchive(_Fn&& fn)
      : m_fn(fn)
      { }

      /// Stores (counts) data into the memory buffer.

      /// The function only appears (due to \c enable_if) if \c T is
      /// serializable.
      /// \tparam T Type of the data to be stored (counted).
      /// \param[in] t Pointer to the data to be stored (counted).
      /// \param[in] n Size of data to be stored (counted).
      template <typename T, typename Allocator>
      void store(const ttg::Buffer<T, Allocator>* t, long n) const {
        /* invoke the function on each buffer */
        for (std::size_t i = 0; i < n; ++i) {
          m_fn(t[i]);
        }
      }

      template <typename T>
      void store(const T* t, long n) const {
        /* nothing to be done for other types */
      }

      /// Open a buffer with a specific size.
      void open(std::size_t /*hint*/) {}

      /// Close the archive.
      void close() {}

      /// Flush the archive.
      void flush() {}

      /// Return the amount of data stored (counted) in the buffer.
      /// \return The amount of data stored (counted) in the buffer (zero).
      std::size_t size() const {
          return 0;
      };
    };

    /* deduction guide */
    template<typename Fn>
    BufferVisitorArchive(Fn&&) -> BufferVisitorArchive<Fn>;
  } // namespace archive

  template <typename Fn>
  struct is_archive<archive::BufferVisitorArchive<Fn>> : std::true_type {};

  template <typename Fn>
  struct is_output_archive<archive::BufferVisitorArchive<Fn>> : std::true_type {};

  template <typename Fn, typename T>
  struct is_default_serializable_helper<archive::BufferVisitorArchive<Fn>, T,
                                        std::enable_if_t<is_trivially_serializable<T>::value>>
  : std::true_type {};

  template <typename Fn, typename T, typename Allocator>
  struct is_default_serializable_helper<archive::BufferVisitorArchive<Fn>, ttg::Buffer<T, Allocator>>
  : std::true_type {};
} // namespace madness

namespace ttg::detail {
  template<typename T, typename Fn>
  requires(madness::is_serializable_v<madness::archive::BufferVisitorArchive<Fn>, std::decay<T>>)
  void buffer_apply(T&& t, Fn&& fn) {
    madness::archive::BufferVisitorArchive ar(std::forward<Fn>(fn));
    ar & t;
  }

  /* dummy function type used to check whether buffer_apply is available */
  using buffer_apply_dummy_fn = decltype([]<typename T, typename A>(const ttg::Buffer<T, A>&){});

  template<typename T>
  struct has_buffer_apply_helper<T, std::enable_if_t<madness::is_serializable_v<madness::archive::BufferVisitorArchive<buffer_apply_dummy_fn>, std::decay_t<T>>>>
  : std::true_type
  { };

} // namespace ttg::detail

#else

namespace ttg::detail {
  template<typename T, typename Fn>
  void buffer_apply(T&& t, Fn&& fn) {
    static_assert(ttg::meta::is_void_v<Fn>, "Types using ttg::Buffer must be MADNESS serializable.");
  }

} // namespace ttg::detail

#endif // TTG_SERIALIZATION_SUPPORTS_MADNESS

#endif // TTG_buffer_H