#ifndef TTG_MRA_TENSORVIEW_H
#define TTG_MRA_TENSORVIEW_H

#include <algorithm>
#include <numeric>
#include <array>

#include <ttg.h>

namespace mra {

  template<typename T, int NDIM>
  class TensorView {
  public:
    using value_type = std::decay_t<T>;
    using size_type = std::size_t;
    static const constexpr int ndims = NDIM;
    using dims_array_t = std::array<size_type, ndims>;

  protected:

    template<size_type I, typename... Dims>
    size_type offset_impl(size_type idx, Dims... idxs) const {
      if constexpr (sizeof...(idxs) == 0) {
        return m_dims[I]*idx;
      } else {
        return m_dims[I]*idx + offset_impl<I+1>(std::forward<Dims...>(idxs...));
      }
    }

    void reset(value_type *ptr, dims_array_t dims) {
      m_ptr = ptr;
      m_dims = dims;
    }

  public:
    template<typename... Dims>
    TensorView(value_type *ptr, Dims... dims)
    : m_dims({dims...})
    , m_ptr(ptr)
    {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
    }

    TensorView(value_type *ptr, const dims_array_t& dims)
    : m_dims(dims)
    , m_ptr(ptr)
    { }

    TensorView(TensorView<T, NDIM>&& other) = default;

    TensorView& operator=(TensorView<T, NDIM>&& other) = default;

    /* Deep copy ctor und op are not needed for PO since tiles will never be read
    * and written concurrently. Hence shallow copies are enough, will all
    * receiving tasks sharing tile data. Re-enable this once the PaRSEC backend
    * can handle data sharing without excessive copying */
    TensorView(const TensorView<T, NDIM>& other)
    : m_dims(other.m_dims)
    , m_ptr(other.m_ptr)
    { }

    TensorView& operator=(const TensorView<T, NDIM>& other) {
      m_dims = other.m_dims;
      m_ptr  = other.m_ptr;
      return *this;
    }

    size_type size() const {
      return std::reduce(&m_dims[0], &m_dims[ndims-1], 1, std::multiplies<size_type>{});
    }

    size_type dim(size_type d) const {
      return m_dims[d];
    }

    const dims_array_t& dims() const {
      return m_dims;
    }

    /* return the offset for the provided indexes */
    template<typename... Dims>
    size_type offset(Dims... idxs) const {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
      return offset_impl<0>(std::forward<Dims>(idxs)...);
    }

    /* access host-side elements */
    template<typename... Dims>
    value_type& operator()(Dims... idxs) {
      static_assert(sizeof...(Dims) == NDIM,
                    "Number of arguments does not match number of Dimensions.");
      return m_ptr[offset(std::forward<Dims>(idxs)...)];
    }

    value_type* data() {
      return m_ptr;
    }

    const value_type* data() const {
      return m_ptr;
    }

  private:
    dims_array_t m_dims;
    value_type *m_ptr;
  };

} // namespace mra

#endif // TTG_MRA_TENSORVIEW_H