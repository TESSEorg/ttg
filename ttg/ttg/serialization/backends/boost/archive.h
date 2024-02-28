//
// Created by Eduard Valeyev on 5/17/21.
//

#ifndef TTG_SERIALIZATION_BACKENDS_BOOST_ARCHIVE_H
#define TTG_SERIALIZATION_BACKENDS_BOOST_ARCHIVE_H

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

// explicitly instantiate for this type of binary stream
#include <boost/archive/impl/basic_binary_iarchive.ipp>
#include <boost/archive/impl/basic_binary_iprimitive.ipp>
#include <boost/archive/impl/basic_binary_oarchive.ipp>
#include <boost/archive/impl/basic_binary_oprimitive.ipp>

namespace ttg::detail {

  // used to serialize data only
  template <typename Archive, typename T>
  void oarchive_save_override_optimized_dispatch(Archive& ar, const T& t) {
    if constexpr (boost::is_array<T>::value) {
      boost::archive::detail::save_array_type<Archive>::invoke(ar, t);
      return;
    } else if constexpr (boost::is_enum<T>::value) {
      boost::archive::detail::save_enum_type<Archive>::invoke(ar, t);
      return;
    } else {
      std::conditional_t<boost::is_pointer<T>::value, T, std::add_pointer_t<const T>> tptr;
      if constexpr (boost::is_pointer<T>::value) {
        static_assert(!std::is_polymorphic_v<T>,
                      "oarchive_save_override does not support serialization of polymorphic types");
        tptr = t;
      } else
        tptr = &t;
      if constexpr (boost::mpl::equal_to<boost::serialization::implementation_level<T>,
                                         boost::mpl::int_<boost::serialization::primitive_type>>::value) {
        boost::archive::detail::save_non_pointer_type<Archive>::save_primitive::invoke(ar, *tptr);
      } else
        boost::archive::detail::save_non_pointer_type<Archive>::save_only::invoke(ar, *tptr);
    }
  }

  // used to serialize data only
  template <typename Archive, typename T>
  void iarchive_load_override_optimized_dispatch(Archive& ar, T& t) {
    if constexpr (boost::is_array<T>::value) {
      boost::archive::detail::load_array_type<Archive>::invoke(ar, t);
      return;
    } else if constexpr (boost::is_enum<T>::value) {
      boost::archive::detail::load_enum_type<Archive>::invoke(ar, t);
      return;
    } else {
      std::conditional_t<boost::is_pointer<T>::value, T, std::add_pointer_t<T>> tptr;
      if constexpr (boost::is_pointer<T>::value) {
        static_assert(!std::is_polymorphic_v<T>,
                      "iarchive_load_override_optimized_dispatch does not support serialization of polymorphic types");
        using Value = std::remove_pointer_t<T>;
        std::allocator<Value> alloc;  // instead use the allocator associated with the archive?
        auto* buf = alloc.allocate(sizeof(Value));
        t = new (buf) Value;
        tptr = t;
      } else
        tptr = &t;
      if constexpr (boost::mpl::equal_to<boost::serialization::implementation_level<T>,
                                         boost::mpl::int_<boost::serialization::primitive_type>>::value) {
        boost::archive::detail::load_non_pointer_type<Archive>::load_primitive::invoke(ar, *tptr);
      } else
        boost::archive::detail::load_non_pointer_type<Archive>::load_only::invoke(ar, *tptr);
    }
  }

  /// optimized data-only serializer

  /// skips metadata (class version, etc.) by providing optimized save_override function that will be called by
  /// `boost::archive::binary_oarchive_impl::save_override`
  ///
  /// \internal Using `boost::archive::binary_oarchive_impl` provides stock implementation for this custom archive. Unfortunately
  /// `boost::archive::binary_oarchive_impl` uses the streambuf object via its std::streambuf base which means that
  /// calls to `xsputn` are not inlined. To work around this problem, this class replaces several functions in
  /// `boost::archive::binary_oarchive_impl` that were provided by `boost::archive::basic_binary_oprimitive`
  /// such as `save`, `save_array` and `save_binary`; the latter calls the streambuf's sputn directly,
  /// not via std::streambuf::sputn . To make sure these "replacements" are called this class must be used directly
  /// rather than cast to `boost::archive::binary_oarchive_impl`, as an argument to
  /// `oarchive_save_override_optimized_dispatch`
  /// if \p StreamOrStreambuf is a streambuf (i.e. derived from std::streambuf).
  ///
  template <typename StreamOrStreambuf>
  class boost_optimized_oarchive
      : private StreamOrStreambuf,
        public boost::archive::binary_oarchive_impl<boost_optimized_oarchive<StreamOrStreambuf>,
                                                    std::ostream::char_type, std::ostream::traits_type> {
   public:
    using pbase_type = StreamOrStreambuf;
    using base_type = boost::archive::binary_oarchive_impl<boost_optimized_oarchive<StreamOrStreambuf>,
                                                           std::ostream::char_type, std::ostream::traits_type>;
    // if pbase_type is derived from std::streambuf can use this information to avoid virtual function calls and inline
    static constexpr bool pbase_derived_from_stdstreambuf = std::is_base_of_v<std::streambuf, pbase_type>;

   private:
    friend class boost::archive::save_access;
    friend class boost::archive::detail::common_oarchive<StreamOrStreambuf>;
    friend base_type;

    const auto& pbase() const { return static_cast<const pbase_type&>(*this); }
    auto& pbase() { return static_cast<pbase_type&>(*this); }
    const auto& base() const { return static_cast<const base_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_optimized_oarchive()
        : pbase_type{}, base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    boost_optimized_oarchive(StreamOrStreambuf sbuf)
        : pbase_type(std::move(sbuf))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    template <typename Arg>
    boost_optimized_oarchive(Arg&& arg)
        : pbase_type(std::forward<Arg>(arg))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    /// these provide optimized implementation that's called by base_type::save_override
    /// @{

    template <class T>
    void save_override(const T& t) {
        oarchive_save_override_optimized_dispatch(*this, t);
    }

    void save_override(const boost::archive::class_id_optional_type& /* t */) {}

    void save_override(const boost::archive::version_type& t) {}
    void save_override(const boost::serialization::item_version_type& t) {}

    void save_override(const boost::archive::class_id_type& t) {}
    void save_override(const boost::archive::class_id_reference_type& t) {}

    /// @}

    void save_object(const void* x, const boost::archive::detail::basic_oserializer& bos) { abort(); }

    /// override default implementations in base_type provided by basic_binary_oprimitive<Archive>
    /// @{

    // default saving of primitives.
    template<class T>
    void save(const T & t)
    {
      save_binary(& t, sizeof(T));
    }

    // trap usage of invalid uninitialized boolean which would
    // otherwise crash on load.
    void save(const bool t){
      BOOST_ASSERT(0 == static_cast<int>(t) || 1 == static_cast<int>(t));
      save_binary(& t, sizeof(t));
    }

   public:

    // the optimized save_array dispatches to save_binary
    template <class ValueType>
    void save_array(boost::serialization::array_wrapper<ValueType> const& a, unsigned int)
    {
      save_binary(a.address(),a.count()*sizeof(ValueType));
    }

    void save_binary(const void *address, std::size_t count) {
      if constexpr (pbase_derived_from_stdstreambuf) {  // if we were given a streambuf use it directly ...
        using Elem = std::ostream::char_type;
        count = (count + sizeof(Elem) - 1) / sizeof(Elem);
        std::streamsize scount = static_cast<StreamOrStreambuf&>(this->pbase())
                                     .sputn(static_cast<const Elem*>(address), static_cast<std::streamsize>(count));
        if (count != static_cast<std::size_t>(scount))
          boost::serialization::throw_exception(
              boost::archive::archive_exception(boost::archive::archive_exception::output_stream_error));
      }
      else {  // ... else let boost::archive::basic_binary_oprimitive handle via std::stringbuf
              // (and associated virtual function calls ... no inlining for you)
        this->base().save_binary(address, count);
      }
    }

    /// @}

    template <class T>
    auto& operator<<(const T& t) {
      this->save_override(t);
      return *this;
    }

    // the & operator
    template <class T>
    auto& operator&(const T& t) {
      return *this << t;
    }

    const auto& streambuf() const {
      if constexpr (pbase_derived_from_stdstreambuf) {
        return static_cast<const StreamOrStreambuf&>(this->pbase());
      }
      else {
        return this->pbase();
      }
    }
    const auto& stream() const { return this->pbase(); }
  };

  /// an archive that counts the size of serialized representation of an object
  using boost_counting_oarchive = boost_optimized_oarchive<counting_streambuf>;

  /// an archive that constructs an IOVEC (= sequence of {pointer,size} pairs) representation of an object
  using boost_iovec_oarchive = boost_optimized_oarchive<iovec_ostreambuf>;

  /// an archive that constructs serialized representation of an object in a memory buffer
  using boost_buffer_oarchive =
      boost_optimized_oarchive<boost::iostreams::stream<boost::iostreams::basic_array_sink<char>>>;

  /// an archive that constructs serialized representation of an object in a memory buffer, in an optimized manner
  using boost_byte_oarchive = boost_optimized_oarchive<byte_ostreambuf>;

  /// constructs a boost_buffer_oarchive object

  /// @param[in] buf pointer to a memory buffer to which serialized representation will be written
  /// @param[in] size the size of the buffer, in bytes
  /// @param[in] buf_offset if non-zero, specifies the first byte of @p buf to which data will be written
  /// @return a boost_buffer_oarchive object referring to @p buf
  inline auto make_boost_buffer_oarchive(void* const buf, std::size_t size, std::size_t buf_offset = 0) {
    assert(buf_offset <= size);
    return ttg::detail::boost_byte_oarchive(ttg::detail::byte_ostreambuf(static_cast<char*>(buf) + buf_offset, size - buf_offset));
  }

  /// constructs a boost_buffer_oarchive object

  /// @tparam N array size
  /// @param[in] buf a buffer to which serialized representation will be written
  /// @param[in] buf_offset if non-zero, specifies the first byte of @p buf to which data will be written
  /// @return a boost_buffer_oarchive object referring to @p buf
  template <std::size_t N>
  inline auto make_boost_buffer_oarchive(char (&buf)[N], std::size_t buf_offset = 0) {
    assert(buf_offset <= N);
    using arrsink_t = boost::iostreams::basic_array_sink<char>;
    return boost_buffer_oarchive(arrsink_t(&(buf[buf_offset]), N - buf_offset));
  }

  /// optimized data-only deserializer for boost_optimized_oarchive
  template <typename StreamOrStreambuf>
  class boost_optimized_iarchive
      : private StreamOrStreambuf,
        public boost::archive::binary_iarchive_impl<boost_optimized_iarchive<StreamOrStreambuf>,
                                                    std::ostream::char_type, std::ostream::traits_type> {
   public:
    using pbase_type = StreamOrStreambuf;
    using base_type = boost::archive::binary_iarchive_impl<boost_optimized_iarchive, std::ostream::char_type,
                                                           std::ostream::traits_type>;

   private:
    friend class boost::archive::save_access;
    friend class boost::archive::detail::common_iarchive<boost_optimized_iarchive>;
    friend base_type;

    const auto& pbase() const { return static_cast<const pbase_type&>(*this); }
    auto& pbase() { return static_cast<pbase_type&>(*this); }
    const auto& base() const { return static_cast<const base_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_optimized_iarchive()
        : pbase_type{}, base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    boost_optimized_iarchive(StreamOrStreambuf sbuf)
        : pbase_type(std::move(sbuf))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    template <typename Arg>
    boost_optimized_iarchive(Arg&& arg)
        : pbase_type(std::forward<Arg>(arg))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    template <class T>
    void load_override(T& t) {
      iarchive_load_override_optimized_dispatch(this->base(), t);
    }

    void load_override(boost::archive::class_id_optional_type& /* t */) {}

    void load_override(boost::archive::version_type& t) {}
    void load_override(boost::serialization::item_version_type& t) {}

    void load_override(boost::archive::class_id_type& t) {}
    void load_override(boost::archive::class_id_reference_type& t) {}

    void load_object(void* x, const boost::archive::detail::basic_oserializer& bos) { abort(); }

    template <class T>
    auto& operator>>(T& t) {
      this->load_override(t);
      return *this;
    }

    // the & operator
    template <class T>
    auto& operator&(T& t) {
      return *this >> t;
    }

    const auto& streambuf() const { return this->pbase(); }
    const auto& stream() const { return this->pbase(); }
  };

  /// the deserializer for boost_iovec_oarchive
  using boost_iovec_iarchive = boost_optimized_iarchive<iovec_istreambuf>;

  /// the deserializer for boost_buffer_oarchive
  using boost_buffer_iarchive =
      boost_optimized_iarchive<boost::iostreams::stream<boost::iostreams::basic_array_source<char>>>;

  /// the deserializer for boost_byte_oarchive
  using boost_byte_iarchive = boost_optimized_iarchive<byte_istreambuf>;

  /// constructs a boost_buffer_iarchive object

  /// @param[in] buf pointer to a memory buffer from which serialized representation will be read
  /// @param[in] size the size of the buffer, in bytes
  /// @param[in] buf_offset if non-zero, specifies the first byte of @p buf from which data will be read
  /// @return a boost_buffer_iarchive object referring to @p buf
  inline auto make_boost_buffer_iarchive(const void* const buf, std::size_t size, std::size_t buf_offset = 0) {
    assert(buf_offset <= size);
    using arrsrc_t = boost::iostreams::basic_array_source<char>;
    return boost_buffer_iarchive(arrsrc_t(static_cast<const char*>(buf) + buf_offset, size - buf_offset));
  }

  /// constructs a boost_buffer_iarchive object

  /// @tparam N array size
  /// @param[in] buf a buffer from which serialized representation will be read
  /// @param[in] buf_offset if non-zero, specifies the first byte of @p buf from which data will be read
  /// @return a boost_buffer_iarchive object referring to @p buf
  template <std::size_t N>
  inline auto make_boost_buffer_iarchive(const char (&buf)[N], std::size_t buf_offset = 0) {
    assert(buf_offset <= N);
    using arrsrc_t = boost::iostreams::basic_array_source<char>;
    return boost_buffer_iarchive(arrsrc_t(&(buf[buf_offset]), N - buf_offset));
  }

}  // namespace ttg::detail

// for some reason need to use array optimization for the base as well ... dispatch to optimized version in
// array_wrapper.hpp:serializer(ar,version) for some reason uses Archive::base_type using apple clang 12.0.5.12050022
#define BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(x) \
  BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(x);                        \
  BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(x::base_type);

BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_counting_oarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(ttg::detail::boost_counting_oarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_iovec_oarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(ttg::detail::boost_iovec_oarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_buffer_oarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(ttg::detail::boost_buffer_oarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_iovec_iarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(ttg::detail::boost_iovec_iarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_buffer_iarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(ttg::detail::boost_buffer_iarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_byte_oarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(ttg::detail::boost_byte_oarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_byte_iarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE(ttg::detail::boost_byte_iarchive);

#undef BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION_FOR_THIS_AND_BASE

#endif  // TTG_SERIALIZATION_BACKENDS_BOOST_ARCHIVE_H
