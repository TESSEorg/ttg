#ifndef TTG_MRA_TENSOR_H
#define TTG_MRA_TENSOR_H

#include <algorithm>
#include <numeric>
#include <array>

#include <TiledArray/device/allocators.h>
#if defined(TILEDARRAY_HAS_DEVICE)
#define ALLOCATOR TiledArray::device_pinned_allocator<T>

namespace mra {

  inline void allocator_init(int argc, char **argv) {
    // initialize MADNESS so that TA allocators can be created
  #if defined(TTG_PARSEC_IMPORTED)
    madness::ParsecRuntime::initialize_with_existing_context(ttg::default_execution_context().impl().context());
  #endif // TTG_PARSEC_IMPORTED
    madness::initialize(argc, argv, /* nthread = */ 1, /* quiet = */ true);
  }

  inline void allocator_fini() {
    madness::finalize();
  }
  #else  // TILEDARRAY_HAS_DEVICE
  #define ALLOCATOR std::allocator<T>

  inline void allocator_init(int argc, char **argv) { }

  inline void allocator_fini() { }

  #endif // TILEDARRAY_HAS_DEVICE

  template<typename T, std::size_t NDIM, class Allocator = ALLOCATOR>
  class Tensor : public ttg::TTValue<Tensor<T, NDIM, Allocator>> {
  public:
    using value_type = T;
    static const constexpr std::size_t ndims = NDIM;

  private:
    using ttvalue_type = ttg::TTValue<MatrixTile<T, Allocator>>;
    std::array<std::size_t, ndims> m_dims;
    ttg::buffer<T, Allocator> m_buffer;

    template<std::size_t I, typename... Dims>
    std::size_t offset_impl(std::size_t idx, Dims... idxs) const {
      if constexpr (sizeof...(idxs) == 0) {
        return m_dims[I]*idx;
      } else {
        return m_dims[I]*idx + offset<I+1>(std::forward<Dims...>(idxs...));
      }
    }

    // (Re)allocate the tensor memory
    void realloc() {
      m_buffer.reset(size());
    }

  public:
    template<typename... Dims>
    Tensor(Dims... dims)
    : ttvalue_type()
    , m_dims({dims...})
    , m_buffer(size())
    {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
    }


    Tensor(Tensor<T, NDIM, Allocator>&& other) = default;

    Tensor& operator=(Tensor<T, NDIM, Allocator>&& other) = default;

    /* Deep copy ctor und op are not needed for PO since tiles will never be read
    * and written concurrently. Hence shallow copies are enough, will all
    * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
    * can handle data sharing without excessive copying */
    Tensor(const Tensor<T, NDIM, Allocator>& other)
    : ttvalue_type()
    , ndims(other.ndims)
    , m_dims(other.m_dims)
    , m_buffer(other.size())
    {
      std::copy_n(other.data(), size(), this->data());
    }

    Tensor& operator=(const Tensor<T, NDIM, Allocator>& other) {
      m_dims = other.m_dims;
      this->realloc();
      std::copy_n(other.data(), size(), this->data());
      return *this;
    }

    std::size_t size() const {
      return std::reduce(&m_dims[0], &m_dims[ndims-1]);
    }

    /* return the offset for the provided indexes */
    template<typename... Dims>
    std::size_t offset(Dims... idxs) const {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
      return offset_impl<0>(std::forward<Dims...>(idxs...));
    }

    /* access host-side elements */
    template<typename... Dims>
    value_type& operator()(Dims... idxs) {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
      return m_buffer.get_host_ptr()[offset(std::forward<Dims...>(idxs...))];
    }

    auto get_buffer() {
      return m_buffer;
    }

    value_type* data() {
      return m_buffer.get_host_ptr();
    }

    const value_type* data() const {
      return m_buffer.get_host_ptr();
    }
  };

} // namespace mra

#endif // TTG_MRA_TENSOR_H