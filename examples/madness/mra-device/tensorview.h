#ifndef TTG_MRA_TENSORVIEW_H
#define TTG_MRA_TENSORVIEW_H

#include <algorithm>
#include <numeric>
#include <array>

#include <ttg.h>

#include "../../mratypes.h"

namespace mra {

  class Slice {
  public:
    static constexpr long END = std::numeric_limits<long>::max();
    long start;  //< Start of slice (must be signed type)
    long finish; //< Exclusive end of slice (must be signed type)
    long stride;   //< Stride for slice (must be signed type)
    long count;  //< Number of elements in slice (not known until dimension is applied; negative indicates not computed)

    Slice() : start(0), finish(END), stride(1), count(-1) {}; // indicates entire range
    Slice(long start) : start(start), finish(start+1), stride(1) {} // a single element
    Slice(long start, long end, long stride=1) : start(start), finish(end), stride(stride) {};

    /// Once we know the dimension we adjust the start/end/count/finish to match, and do sanity checks
    void apply_dim(long dim) {
        if (start == END) {start = dim-1;}
        else if (start < 0) {start += dim;}

        if (finish == END && stride > 0) {finish = dim;}
        else if (finish == END && stride < 0) {finish = -1;}
        else if (finish < 0) {finish += dim;}

        count = std::max(0l,((finish-start-stride/std::abs(stride))/stride+1));
        assert((count==0) || ((count<=dim) && (start>=0 && start<=dim)));
        finish = start + count*stride; // finish is one past the last element
    }

    struct iterator {
        long value;
        const long stride;
        iterator (long value, long stride) : value(value), stride(stride) {}
        operator long() const {return value;}
        long operator*() const {return value;}
        iterator& operator++ () {value+=stride; return *this;}
        bool operator!=(const iterator&other) {return value != other.value;}
    };

    iterator begin() const {assert(count>=0); return iterator(start,stride); }
    iterator end() const {assert(count>=0); return iterator(finish,stride); }

    Slice& operator=(const Slice& other) {
        if (this != &other) {
            start = other.start;
            finish = other.finish;
            stride = other.stride;
            count = other.count;
        }
        return *this;
    }
  }; // Slice


  template<typename T, Dimension NDIM>
  class TensorView {
  public:
    using value_type = std::decay_t<T>;
    using size_type = std::size_t;
    static constexpr int ndim() { return NDIM; }
    using dims_array_t = std::array<size_type, ndim()>;
    static constexpr bool is_tensor() { return true; }

  protected:

    template<size_type I, typename... Dims>
    size_type offset_impl(size_type idx, Dims... idxs) const {
      if constexpr (sizeof...(idxs) == 0) {
        return idx;
      } else {
        return idx*std::reduce(&m_dims[I+1], &m_dims[ndim()], 1, std::multiplies<size_type>{})
              + offset_impl<I+1>(std::forward<Dims>(idxs)...);
      }
    }

    /* TODO: unused right now, should be used to condense N dimensions down to 3 for devices */
#if 0
    template<typename Fn, typename... Args, std::size_t I, std::size_t... Is>
    void last_level_op_helper(Fn&& fn, std::index_sequence<I, Is...>, Args... args) {
      if constexpr (sizeof...(Is) == 0) {
        fn(args...);
      } else {
        /* iterate over this dimension and recurse down one */
        for (size_type i = 0; i < m_dims[I]; ++i) {
          last_level_op_helper(std::forward<Fn>(fn), std::index_sequence<Is...>{}, args..., i);
        }
      }
    }
#endif // 0

    template<typename Fn>
    void foreach_idx(Fn&& fn) {
      static_assert(ndim() <= 3, "Missing implementation of operator= for NDIM>3");
#ifdef __CUDA_ARCH__
      /* let's start simple: iterate sequentially over all but the fastest dimension and use threads in the last
       *                     dimension to do the assignment in parallel. This should be revisited later.*/
      static_assert(ndim() <= 3, "Missing implementation of operator= for NDIM>3");
      if constexpr(ndim() == 3) {
        for (std::size_t i = threadIdx.z; i < dims(0); i += blockDim.z) {
          for (std::size_t j = threadIdx.y; j < dims(1); j += blockDim.y) {
            for (std::size_t k = threadIdx.x; k < dims(2); k += blockDim.x) {
              fn(i, j, k);
            }
          }
        }
      } else if constexpr (ndim() == 2) {
        for (std::size_t i = threadIdx.z*blockDim.y + threadIdx.y; i < dims(0); i += blockDim.z*blockDim.y) {
          for (std::size_t j = threadIdx.x; j < dims(1); j += blockDim.x) {
            fn(i, j);
          }
        }
      } else if constexpr(ndim() == 1) {
        int tid = threadDim.x * ((threadDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < dim(0); i += blockDim.x*blockDim.y*blockDim.z) {
            fn(i);
        }
      }
      __syncthreads();
#else  // __CUDA_ARCH__
      if constexpr(ndim() ==3) {
        for (int i = 0; i < dim(0); ++i) {
          for (int j = 0; j < dim(1); ++j) {
            for (int k = 0; k < dim(2); ++k) {
              fn(i, j, k);
            }
          }
        }
      } else if constexpr (ndim() ==2) {
        for (int i = 0; i < dim(0); ++i) {
          for (int j = 0; j < dim(1); ++j) {
            fn(i, j);
          }
        }
      } else if constexpr (ndim() ==1) {
        for (int i = 0; i < dim(0); ++i) {
          fn(i);
        }
      }
#endif // __CUDA_ARCH__
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
      static_assert(sizeof...(Dims) == NDIM || sizeof...(Dims) == 1,
                    "Number of arguments does not match number of Dimensions. "
                    "A single argument for all dimensions may be provided.");
      if constexpr (sizeof...(Dims) != m_dims.size()) {
        std::fill(m_dims.begin(), m_dims.end(), dims...);
      }
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
      return std::reduce(&m_dims[0], &m_dims[ndim()-1], 1, std::multiplies<size_type>{});
    }

    size_type dim(size_type d) const {
      return m_dims[d];
    }

    const dims_array_t& dims() const {
      return m_dims;
    }

    size_type stride(size_type d) const {
      std::size_t s = 1;
      for (std::size_t i = 0; i < d; ++i) {
        s *= m_dims[i];
      }
      return s;
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

    std::array<Slice, ndim()> slices() const {
      std::array<Slice, ndim()> res;
      for (std::size_t d = 0; d < ndim(); ++d) {
        res[d] = Slice(0, m_dims[d]);
      }
      return res;
    }

    /// Fill with scalar
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    TensorView& operator=(const value_type& value) {
      foreach_idx([&](auto... args){ this->operator()(args...) = value; });
      return *this;
    }

    /// Scale by scalar
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    TensorView& operator*=(const value_type& value) {
      foreach_idx([&](auto... args){ this->operator()(args...) *= value; });
      return *this;
    }

    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block, synchronizes
    /// Host: assumes this operation is called by a single CPU thread
    template <typename otherT>
    typename std::enable_if<otherT::is_tensor,TensorView&>::type
    operator=(const otherT& other) {
      foreach_idx([&](auto... args){ this->operator()(args...) = other(args...); });
      return *this;
    }

  private:
    dims_array_t m_dims;
    value_type *m_ptr;
  };



  template<typename TensorViewT>
  class TensorSlice {

  public:
    using view_type = TensorViewT;
    using value_type = typename view_type::value_type;

    static constexpr Dimension ndim() { return TensorViewT::ndim(); }

    static constexpr bool is_tensor() { return true; }

  private:
    value_type* m_ptr;
    std::array<Slice, ndim()> m_slices;

    // Computes index in dimension d for underlying tensor using slice info

    template<std::size_t I, std::size_t... Is, typename Arg, typename... Args>
    std::size_t offset_helper(std::index_sequence<I, Is...>, Arg arg, Args... args) const {
      std::size_t idx = (m_slices[I].start + arg)*m_slices[I].stride;
      if constexpr (sizeof...(Args) > 0) {
        idx += offset_helper(std::index_sequence<Is...>{}, std::forward<Args>(args)...);
      }
      return idx;
    }

    template<typename Fn, typename... Args, std::size_t I, std::size_t... Is>
    void last_level_op_helper(Fn&& fn, std::index_sequence<I, Is...>, Args... args) {
      if constexpr (sizeof...(Is) == 0) {
        fn(args...);
      } else {
        /* iterate over this dimension and recurse down one */
        for (std::size_t i = 0; i < m_slices[I].count; ++i) {
          last_level_op_helper(std::forward<Fn>(fn), std::index_sequence<Is...>{}, args..., i);
        }
      }
    }

    /* Same as for TensorView but on different counts.
     * TODO: Can we merge them? */
    template<typename Fn>
    void foreach_idx(Fn&& fn) {
      static_assert(ndim() <= 3, "Missing implementation of operator= for NDIM>3");
#ifdef __CUDA_ARCH__
      /* let's start simple: iterate sequentially over all but the fastest dimension and use threads in the last
       *                     dimension to do the assignment in parallel. This should be revisited later.*/
      static_assert(ndim() <= 3, "Missing implementation of operator= for NDIM>3");
      if constexpr(ndim() == 3) {
        for (std::size_t i = threadIdx.z; i < m_slices[0].count; i += blockDim.z) {
          for (std::size_t j = threadIdx.y; j < m_slices[1].count; j += blockDim.y) {
            for (std::size_t k = threadIdx.x; k < m_slices[2].count; k += blockDim.x) {
              fn(i, j, k);
            }
          }
        }
      } else if constexpr (ndim() == 2) {
        for (std::size_t i = threadIdx.z*blockDim.y + threadIdx.y; i < m_slices[0].count; i += blockDim.z*blockDim.y) {
          for (std::size_t j = threadIdx.x; j < m_slices[1].count; j += blockDim.x) {
            fn(i, j);
          }
        }
      } else if constexpr(ndim() == 1) {
        int tid = threadDim.x * ((threadDim.y*threadIdx.z) + threadIdx.y) + threadIdx.x;
        for (size_t i = tid; i < m_slices[0].count; i += blockDim.x*blockDim.y*blockDim.z) {
            fn(i);
        }
      }
      __syncthreads();
#else  // __CUDA_ARCH__
      if constexpr(ndim() ==3) {
        for (int i = 0; i < m_slices[0].count; ++i) {
          for (int j = 0; j < m_slices[1].count; ++j) {
            for (int k = 0; k < m_slices[2].count; ++k) {
              fn(i, j, k);
            }
          }
        }
      } else if constexpr (ndim() ==2) {
        for (int i = 0; i < m_slices[0].count; ++i) {
          for (int j = 0; j < m_slices[1].count; ++j) {
            fn(i, j);
          }
        }
      } else if constexpr (ndim() ==1) {
        for (int i = 0; i < m_slices[1].count; ++i) {
          fn(i);
        }
      }
#endif // __CUDA_ARCH__
    }


  public:
    TensorSlice() = delete; // slice is useless without a view

    TensorSlice(view_type& view, const std::array<Slice,ndim()>& slices)
    : m_ptr(view.data())
    , m_slices(slices)
    {
      /* adjust the slice dimensions to the tensor */
      auto view_slices = view.slices();
      std::size_t stride = 1;
      for (ssize_t d = ndim()-1; d >= 0; --d) {
        m_slices[d].apply_dim(view.dim(d));
        /* stride stores the stride in the original TensorView */
        m_slices[d].stride *= stride;
        stride *= view.dim(d);
        /* account for the stride of the underlying view */
        m_slices[d].stride *= view_slices[d].stride;
        /* adjust the start relative to the underlying view */
        m_slices[d].start += view_slices[d].start * view_slices[d].stride;
      }
    }

    TensorSlice(TensorSlice&& other) = default;
    TensorSlice(const TensorSlice& other) = default;

    /// Returns the base pointer
    value_type* data() {
      return m_ptr;
    }

    /// Returns the const base pointer
    const value_type* data() const {
      return m_ptr;
    }

    /// Returns number of elements in the tensor at runtime
    size_t size() const {
      std::size_t nelem = 1;
      for (size_t d = 0; d < ndim(); ++d) {
          nelem *= m_slices[d].count;
      }
      return nelem;
    }

    /// Returns size of dimension d at runtime
    std::size_t dim(size_t d) const { return m_slices[d].count; }

    /// Returns array containing size of each dimension at runtime
    std::array<size_t, ndim()> dims() const {
      std::array<size_t, ndim()> dimensions;
      for (size_t d = 0; d < ndim(); ++d) {
        dimensions[d] = m_slices[d].count;
      }
      return dimensions;
    }

    std::array<Slice, ndim()> slices() const {
      return m_slices;
    }

    template <typename...Args>
    auto& operator()(Args...args) {
        static_assert(ndim() == sizeof...(Args), "TensorSlice number of indices must match dimension");
        return m_ptr[offset_helper(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...)];
    }

    template <typename...Args>
    const auto& operator()(Args...args) const {
        static_assert(ndim() == sizeof...(Args), "TensorSlice number of indices must match dimension");
        return m_ptr[offset_helper(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...)];
    }

    /// Fill with scalar
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    template <typename X=TensorSlice<TensorViewT>>
    typename std::enable_if<!std::is_const_v<TensorSlice>,X&>::type
    operator=(const value_type& value) {
      foreach_idx([&](auto... args){ this->operator()(args...) = value; });
      return *this;
    }

    /// Scale by scalar
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    template <typename X=TensorSlice<TensorViewT>>
    typename std::enable_if<!std::is_const_v<TensorSlice>,X&>::type
    operator*=(const value_type& value) {
      foreach_idx([&](auto... args){ this->operator()(args...) *= value; });
      return *this;
    }

    /// Copy into patch
    /// Device: assumes this operation is called by all threads in a block
    /// Host: assumes this operation is called by a single CPU thread
    typename std::enable_if<!std::is_const_v<TensorViewT>,TensorSlice&>::type
    operator=(const TensorSlice& other) {
      foreach_idx([&](auto... args){ this->operator()(args...) = other(args...); });
      return *this;
    }
  };



} // namespace mra

#endif // TTG_MRA_TENSORVIEW_H