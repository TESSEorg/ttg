//
// Created by Eduard Valeyev on 5/17/21.
//

#ifndef TTG_SERIALIZATION_BACKENDS_BOOST_ARCHIVE_H
#define TTG_SERIALIZATION_BACKENDS_BOOST_ARCHIVE_H

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

namespace ttg::detail {

  class boost_counting_oarchive : private ttg::detail::counting_streambuf, public boost::archive::binary_oarchive {
    using pbase_type = ttg::detail::counting_streambuf;
    using base_type = boost::archive::binary_oarchive;

    friend class boost::archive::save_access;

    auto& pbase() { return static_cast<pbase_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_counting_oarchive()
        : pbase_type{}, base_type(*this, boost::archive::no_header | boost::archive::no_codecvt){};

    using base_type::save_binary;

    using pbase_type::size;

    using base_type::operator<<;
    using base_type::operator&;
  };

  class boost_buffer_oarchive : private boost::iostreams::stream<boost::iostreams::basic_array_sink<char>>,
                                public boost::archive::binary_oarchive {
    using pbase_type = boost::iostreams::stream<boost::iostreams::basic_array_sink<char>>;
    using base_type = boost::archive::binary_oarchive;

    friend class boost::archive::save_access;

    auto& pbase() { return static_cast<pbase_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_buffer_oarchive(void* buf, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_sink<char>(static_cast<char*>(buf), size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    boost_buffer_oarchive(void* buf, std::size_t buf_offset, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_sink<char>(static_cast<char*>(buf) + buf_offset, size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    using base_type::save_binary;

    using base_type::operator<<;
    using base_type::operator&;
  };

  class boost_buffer_iarchive : private boost::iostreams::stream<boost::iostreams::basic_array_source<char>>,
                                public boost::archive::binary_iarchive {
    using pbase_type = boost::iostreams::stream<boost::iostreams::basic_array_source<char>>;
    using base_type = boost::archive::binary_iarchive;

    friend class boost::archive::save_access;

    auto& pbase() { return static_cast<pbase_type&>(*this); }
    auto& base() { return static_cast<base_type&>(*this); }

   public:
    boost_buffer_iarchive(const void* buf, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_source<char>(static_cast<const char*>(buf), size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    boost_buffer_iarchive(const void* buf, std::size_t buf_offset, std::size_t size)
        : pbase_type(boost::iostreams::basic_array_source<char>(static_cast<const char*>(buf) + buf_offset, size))
        , base_type(this->pbase(), boost::archive::no_header | boost::archive::no_codecvt){};

    using base_type::operator>>;
    using base_type::operator&;
  };

}  // namespace ttg::detail

BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_counting_oarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_buffer_oarchive);
BOOST_SERIALIZATION_REGISTER_ARCHIVE(ttg::detail::boost_buffer_iarchive);

#endif  // TTG_SERIALIZATION_BACKENDS_BOOST_ARCHIVE_H
