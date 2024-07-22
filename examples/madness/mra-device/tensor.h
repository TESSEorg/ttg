#ifndef TTG_MRA_TENSOR_H
#define TTG_MRA_TENSOR_H

#include <algorithm>
#include <numeric>
#include <array>

#include <ttg.h>

#include "tensorview.h"

namespace mra {

  template<typename T, int NDIM, class Allocator = std::allocator<T>>
  class Tensor : public ttg::TTValue<Tensor<T, NDIM, Allocator>> {
  public:
    using value_type = std::decay_t<T>;
    using size_type = std::size_t;
    using allocator_type = Allocator;
    using view_type = TensorView<T, NDIM>;
    using buffer_type = ttg::Buffer<value_type, allocator_type>;

    static constexpr int ndim() { return NDIM; }

    using dims_array_t = std::array<size_type, ndim()>;

  private:
    using ttvalue_type = ttg::TTValue<Tensor<T, NDIM, Allocator>>;
    dims_array_t m_dims;
    buffer_type m_buffer;

    // (Re)allocate the tensor memory
    void realloc() {
      m_buffer.reset(size());
    }

  public:
    template<typename... Dims>
    Tensor(Dims... dims)
    : ttvalue_type()
    , m_dims({static_cast<size_type>(dims)...})
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

    size_type size() const {
      return std::reduce(&m_dims[0], &m_dims[ndim()], 1, std::multiplies<size_type>{});
    }

    auto get_buffer() {
      return m_buffer;
    }

    const auto get_buffer() const {
      return m_buffer;
    }

    /* returns a view for the current memory space */
    view_type current_view() {
      return view_type(m_buffer.current_device_ptr(), m_dims);
    }
  };

} // namespace mra

#endif // TTG_MRA_TENSOR_H