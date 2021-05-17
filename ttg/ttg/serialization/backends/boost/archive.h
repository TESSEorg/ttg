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

  class boost_counting_oarchive
      : private ttg::detail::counting_streambuf,
        public boost::archive::binary_oarchive_impl<boost_counting_oarchive, std::ostream::char_type,
                                                    std::ostream::traits_type> {
    using pbase_type = ttg::detail::counting_streambuf;
    using base_type = boost::archive::binary_oarchive_impl<boost_counting_oarchive, std::ostream::char_type,
                                                           std::ostream::traits_type>;

    friend class boost::archive::save_access;
    friend class boost::archive::detail::common_oarchive<boost_counting_oarchive>;
    friend base_type;

    auto& pbase() { return static_cast<pbase_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_counting_oarchive()
        : pbase_type{}, base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    using pbase_type::size;

    template <class T>
    void save_override(const T& t) {
      if constexpr (boost::is_array<T>::value) {
        boost::archive::detail::save_array_type<base_type>::invoke(this->base(), t);
        return;
      } else if constexpr (boost::is_enum<T>::value) {
        boost::archive::detail::save_enum_type<base_type>::invoke(this->base(), t);
        return;
      } else {
        std::add_pointer_t<const T> tptr;
        if constexpr (boost::is_pointer<T>::value) {
          static_assert(!std::is_polymorphic_v<T>,
                        "boost_buffer_oarchive does not support serialization of polymorphic types");
          tptr = t;
        } else
          tptr = &t;
        if constexpr (boost::mpl::equal_to<boost::serialization::implementation_level<T>,
                                           boost::mpl::int_<boost::serialization::primitive_type>>::value) {
          boost::archive::detail::save_non_pointer_type<base_type>::save_primitive::invoke(this->base(), *tptr);
        } else
          boost::archive::detail::save_non_pointer_type<base_type>::save_only::invoke(this->base(), *tptr);
      }
      //    base_type::save_override(t);
    }

    void save_override(const boost::archive::class_id_optional_type& /* t */) {}

    void save_override(const boost::archive::version_type& t) {}
    void save_override(const boost::serialization::item_version_type& t) {}

    void save_override(const boost::archive::class_id_type& t) {}
    void save_override(const boost::archive::class_id_reference_type& t) {}

    void save_object(const void* x, const boost::archive::detail::basic_oserializer& bos) { abort(); }

   public:
    BOOST_ARCHIVE_DECL
    void save_binary(const void* address, std::size_t count);
  };

  class boost_buffer_oarchive
      : private boost::iostreams::stream<boost::iostreams::basic_array_sink<char>>,
        public boost::archive::binary_oarchive_impl<boost_buffer_oarchive, std::ostream::char_type,
                                                    std::ostream::traits_type> {
    using Archive = boost_buffer_oarchive;
    using pbase_type = boost::iostreams::stream<boost::iostreams::basic_array_sink<char>>;
    using base_type =
        boost::archive::binary_oarchive_impl<boost_buffer_oarchive, std::ostream::char_type, std::ostream::traits_type>;

    friend class boost::archive::save_access;
    friend class boost::archive::detail::common_oarchive<boost_buffer_oarchive>;
    friend base_type;

    auto& pbase() { return static_cast<pbase_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_buffer_oarchive(void* buf, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_sink<char>(static_cast<char*>(buf), size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    boost_buffer_oarchive(void* buf, std::size_t buf_offset, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_sink<char>(static_cast<char*>(buf) + buf_offset, size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    template <class T>
    void save_override(const T& t) {
      if constexpr (boost::is_array<T>::value) {
        boost::archive::detail::save_array_type<base_type>::invoke(this->base(), t);
        return;
      } else if constexpr (boost::is_enum<T>::value) {
        boost::archive::detail::save_enum_type<base_type>::invoke(this->base(), t);
        return;
      } else {
        std::add_pointer_t<const T> tptr;
        if constexpr (boost::is_pointer<T>::value) {
          static_assert(!std::is_polymorphic_v<T>,
                        "boost_buffer_oarchive does not support serialization of polymorphic types");
          tptr = t;
        } else
          tptr = &t;
        if constexpr (boost::mpl::equal_to<boost::serialization::implementation_level<T>,
                                           boost::mpl::int_<boost::serialization::primitive_type>>::value) {
          boost::archive::detail::save_non_pointer_type<base_type>::save_primitive::invoke(this->base(), *tptr);
        } else
          boost::archive::detail::save_non_pointer_type<base_type>::save_only::invoke(this->base(), *tptr);
      }
    }

    void save_override(const boost::archive::class_id_optional_type& /* t */) {}

    void save_override(const boost::archive::version_type& t) {}
    void save_override(const boost::serialization::item_version_type& t) {}

    void save_override(const boost::archive::class_id_type& t) {}
    void save_override(const boost::archive::class_id_reference_type& t) {}

    void save_object(const void* x, const boost::archive::detail::basic_oserializer& bos) { abort(); }

    template <class T>
    Archive& operator<<(const T& t) {
      this->save_override(t);
      return *this;
    }

    // the & operator
    template <class T>
    Archive& operator&(const T& t) {
      return *this << t;
    }
  };

  class boost_buffer_iarchive
      : private boost::iostreams::stream<boost::iostreams::basic_array_source<char>>,
        public boost::archive::binary_iarchive_impl<boost_buffer_iarchive, std::ostream::char_type,
                                                    std::ostream::traits_type> {
    using Archive = boost_buffer_iarchive;
    using pbase_type = boost::iostreams::stream<boost::iostreams::basic_array_source<char>>;
    using base_type =
        boost::archive::binary_iarchive_impl<boost_buffer_iarchive, std::ostream::char_type, std::ostream::traits_type>;

    friend class boost::archive::save_access;
    friend class boost::archive::detail::common_iarchive<boost_buffer_iarchive>;
    friend base_type;

    auto& pbase() { return static_cast<pbase_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_buffer_iarchive(const void* buf, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_source<char>(static_cast<const char*>(buf), size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    boost_buffer_iarchive(const void* buf, std::size_t buf_offset, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_source<char>(static_cast<const char*>(buf) + buf_offset, size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    template <class T>
    void load_override(T& t) {
      if constexpr (boost::is_array<T>::value) {
        boost::archive::detail::load_array_type<base_type>::invoke(this->base(), t);
        return;
      } else if constexpr (boost::is_enum<T>::value) {
        boost::archive::detail::load_enum_type<base_type>::invoke(this->base(), t);
        return;
      } else {
        std::add_pointer_t<T> tptr;
        if constexpr (boost::is_pointer<T>::value) {
          static_assert(!std::is_polymorphic_v<T>,
                        "boost_buffer_iarchive does not support serialization of polymorphic types");
          tptr = t;
        } else
          tptr = &t;
        if constexpr (boost::mpl::equal_to<boost::serialization::implementation_level<T>,
                                           boost::mpl::int_<boost::serialization::primitive_type>>::value) {
          boost::archive::detail::load_non_pointer_type<base_type>::load_primitive::invoke(this->base(), *tptr);
        } else
          boost::archive::detail::load_non_pointer_type<base_type>::load_only::invoke(this->base(), *tptr);
      }
    }

    void load_override(boost::archive::class_id_optional_type& /* t */) {}

    void load_override(boost::archive::version_type& t) {}
    void load_override(boost::serialization::item_version_type& t) {}

    void load_override(boost::archive::class_id_type& t) {}
    void load_override(boost::archive::class_id_reference_type& t) {}

    void load_object(void* x, const boost::archive::detail::basic_oserializer& bos) { abort(); }

    template <class T>
    Archive& operator>>(T& t) {
      this->load_override(t);
      return *this;
    }

    // the & operator
    template <class T>
    Archive& operator&(T& t) {
      return *this >> t;
    }
  };

}  // namespace ttg::detail

BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_counting_oarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(ttg::detail::boost_counting_oarchive)
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_buffer_oarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(ttg::detail::boost_buffer_oarchive)
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_buffer_iarchive);
BOOST_SERIALIZATION_USE_ARRAY_OPTIMIZATION(ttg::detail::boost_buffer_iarchive)

#endif  // TTG_SERIALIZATION_BACKENDS_BOOST_ARCHIVE_H
