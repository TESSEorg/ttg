#ifndef HAVE_DEVICETENSOR_H
#define HAVE_DEVICETENSOR_H

#include <ttg.h>

#if __has_include(<btas/features.h>)
#pragma message("C Preprocessor got here!")
#include <btas/features.h>
#ifdef BTAS_IS_USABLE
#include <btas/btas.h>
#include <btas/optimize/contract.h>
#include <btas/util/mohndle.h>
#include <TiledArray/external/device.h>
#include "../devblas_helper.h"
#include <madness/world/parsec.h>  // need to initialize MADNESS purely for the purposes of TA allocators
#else
#warning "found btas/features.h but Boost.Iterators is missing, hence BTAS is unusable ... add -I/path/to/boost"
#endif
#endif

#if defined(BTAS_IS_USABLE)

namespace detail {
  template<typename T>
  struct mohndle_type;

  template <typename Storage, btas::Handle Handle>
  struct mohndle_type<btas::mohndle<Storage, Handle>> : std::integral_constant<btas::Handle, Handle>
  { };

  template <typename T>
  constexpr btas::Handle mohndle_type_v = mohndle_type<T>::value;
} // namespace detail

/**
 * Derives from btas::Tensor and wraps a ttg::Buffer
 * to enable device support in SPMM. The ttg::Buffer
 * does not own the host memory but mananages the device
 * memory.
 */
template <typename _T, class _Range, class _Storage>
struct DeviceTensor : public ttg::TTValue<DeviceTensor<_T, _Range, _Storage>>
                    , public btas::Tensor<_T, _Range, _Storage> {
  using tensor_type = typename btas::Tensor<_T, _Range, _Storage>;
  using ttvalue_type = typename ttg::TTValue<DeviceTensor<_T, _Range, _Storage>>;
  ttg::Buffer<_T> b; // does not own the host buffer

  using value_type = typename tensor_type::value_type;
  using size_type = typename tensor_type::size_type;
  using storage_type = typename tensor_type::storage_type;
  using range_type = typename tensor_type::range_type;

  static_assert(detail::mohndle_type_v<_Storage> == btas::Handle::shared_ptr,
                "DeviceTensor only supports shared_ptr");


   public:
    DeviceTensor() = default;
    ~DeviceTensor() = default;

    /// constructor with index extent
    template <typename... _args>
    explicit DeviceTensor(const size_type& first, const _args&... rest)
    : ttvalue_type()
    , tensor_type(first, rest...)
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// construct from \c range, allocate data, but not initialized
    template <typename Range>
    explicit DeviceTensor(const Range& range, typename std::enable_if<btas::is_boxrange<Range>::value>::type* = 0)
    : ttvalue_type()
    , tensor_type(range)
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// construct from \c range object, set all elements to \c v
    template <typename Range>
    DeviceTensor(const Range& range, value_type v, typename std::enable_if<btas::is_boxrange<Range>::value>::type* = 0)
    : ttvalue_type()
    , tensor_type(range)
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// construct from \c range object, copy elements from \c vec
    template <typename Range, typename U>
    DeviceTensor(const Range& range, U* vec, typename std::enable_if<btas::is_boxrange<Range>::value>::type* = 0)
    : ttvalue_type()
    , tensor_type(range, vec)
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// construct from \c range and \c storage
    template <typename Range, typename Storage>
    DeviceTensor(const Range& range, const Storage& storage,
           typename std::enable_if<btas::is_boxrange<Range>::value & not std::is_same<Range, range_type>::value &
                                   not std::is_same<Storage, storage_type>::value>::type* = 0)
    : ttvalue_type()
    , tensor_type(range, storage)
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// copy-copy-construct from \c range and \c storage
    DeviceTensor(const range_type& range, const storage_type& storage)
    : ttvalue_type()
    , tensor_type(range, storage)
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// copy-move-construct from \c range and \c storage
    DeviceTensor(const range_type& range, storage_type&& storage)
    : ttvalue_type()
    , tensor_type(range, std::forward<storage_type>(storage))
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// move-construct from \c range and \c storage
    DeviceTensor(range_type&& range, storage_type&& storage)
    : ttvalue_type()
    , tensor_type(std::forward<range_type>(range), std::forward<storage_type>(storage))
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// Construct an evaluated tensor

    /// This constructor will allocate memory for \c range.area() elements. Each element
    /// will be initialized as:
    /// \code
    ///   for(auto&& idx: range)
    ///     (*this)[idx] = op(*(it++));
    /// \endcode
    /// \tparam Range An input Range type.
    /// \tparam InIter An input iterator type.
    /// \tparam Op A unary operation type
    /// \param range the input range type
    /// \param first An input iterator for the argument
    /// \param op The unary operation to be applied to the argument data
    template <typename Range, typename InIter, typename Op>
    DeviceTensor(const Range& range, InIter it, const Op& op,
           typename std::enable_if<btas::is_boxrange<Range>::value>::type* = 0)
    : ttvalue_type()
    , tensor_type(range, it, op)
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    { }

    /// copy constructor
    /// It will accept Tensors and TensorViews
    template <class _Tensor, class = typename std::enable_if<btas::is_boxtensor<_Tensor>::value>::type>
    DeviceTensor(const _Tensor& x) noexcept
    : ttvalue_type()
    , tensor_type(x.clone())
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    {
      //std::cout << "DeviceTensor tensor_type copy ctor" << std::endl;
    }

    /// copy constructor: devicebuf cannot be copied, so deleted
    DeviceTensor(const DeviceTensor& x) noexcept
    : ttvalue_type(x)
    , tensor_type(x.clone())
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    {
      //std::cout << "DeviceTensor copy ctor" << std::endl;
    }

    /// move constructor
    DeviceTensor(tensor_type&& x) noexcept
    : ttvalue_type()
    , tensor_type(std::move(x))
    , b(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size())
    {
      //std::cout << "DeviceTensor tensor_type move ctor" << std::endl;
    }

    DeviceTensor(DeviceTensor&& x) noexcept
    : ttvalue_type(std::move(x))
    , tensor_type(static_cast<tensor_type&&>(x))
    , b(std::move(x.b))
    {
      assert(this->data() == b.host_ptr());
      //std::cout << "DeviceTensor move ctor" << std::endl;
    }

    /// copy assignment operator
    template <class _Tensor, class = typename std::enable_if<
                                 btas::is_boxtensor<_Tensor>::value &&
                                 not std::is_same<typename _Tensor::storage_type, storage_type>::value>::type>
    DeviceTensor& operator=(const _Tensor& x) noexcept {
      tensor_type::operator=(x.clone());
      b.reset(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size());
      //std::cout << "DeviceTensor tensor_type copy operator" << std::endl;
      return *this;
    }

    /// copy assignment operator
    template <class _Tensor, class = typename std::enable_if<btas::is_boxtensor<_Tensor>::value>::type,
              class = typename std::enable_if<
                  std::is_same<typename _Tensor::storage_type, storage_type>::value>::type>
    DeviceTensor& operator=(const _Tensor& x) noexcept {
      tensor_type::operator=(x.clone());
      b.reset(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size());
      //std::cout << "DeviceTensor tensor_type copy operator" << std::endl;
      return *this;
    }

    /// copy assignment: devicebuf cannot be copied, deleted
    DeviceTensor& operator=(const DeviceTensor& x) noexcept {
      ttvalue_type::operator=(x);
      tensor_type::operator=(x.clone());
      b.reset(std::shared_ptr<_T[]>(pointer(), this->size() ? this->data() : nullptr), this->size());
      //std::cout << "DeviceTensor copy operator" << std::endl;
      return *this;
    }

    /// move assignment operator
    DeviceTensor& operator=(DeviceTensor&& x) noexcept {
      ttvalue_type::operator=(std::move(x));
      tensor_type::operator=(static_cast<tensor_type&&>(x));
      b = std::move(x.b);
      //std::cout << "DeviceTensor move ctor" << std::endl;
      return *this;
    }

    auto pointer() {
      return std::get<static_cast<long unsigned int>(detail::mohndle_type_v<storage_type>)>(this->storage().base());
    }

    using tensor_type::begin;
    using tensor_type::cbegin;
    using tensor_type::end;
    using tensor_type::cend;
};

namespace madness {
  namespace archive {
    template <class Archive, typename _T, class _Range, class _Store>
    struct ArchiveLoadImpl<Archive, DeviceTensor<_T, _Range, _Store>> {
      static inline void load(const Archive& ar,
                              DeviceTensor<_T, _Range, _Store>& t) {
        _Range range{};
        ar& range;
        t = DeviceTensor<_T, _Range, _Store>(std::move(range));
        /* pick the buffer out of the archive
         * this should not change the buffer since it's
         * already been allocated above */
        ar & t.b;
      }
    };

    template <class Archive, typename _T, class _Range, class _Store>
    struct ArchiveStoreImpl<Archive, DeviceTensor<_T, _Range, _Store>> {
      static inline void store(const Archive& ar,
                               const DeviceTensor<_T, _Range, _Store>& t) {
        ar& t.range() & t.b;
      }
    };
  }  // namespace archive
}  // namespace madness
#endif // defined(BTAS_IS_USABLE)

#endif // HAVE_DEVICETENSOR_H