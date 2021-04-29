#include <array>

#include "ttg.h"

namespace madness {
  namespace archive {

    template <typename Archive, typename T, typename Enabler = void>
    inline constexpr bool is_serializable_v = false;

    template <typename Archive, typename T>
    inline constexpr bool
        is_serializable_v<Archive, T, std::void_t<decltype(std::declval<Archive&>() & std::declval<T&>())>> = true;

  }  // namespace archive
}  // namespace madness

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/serialization.hpp>

namespace boost {
  namespace serialization {

    template <typename Archive, typename T, typename Enabler = void>
    inline constexpr bool is_serializable_v = false;

    template <typename Archive, typename T>
    inline constexpr bool
        is_serializable_v<Archive, T, std::void_t<decltype(std::declval<Archive&>() & std::declval<T&>())>> = true;

  }  // namespace serialization
}  // namespace boost

class Fred {
  int value;

 public:
  Fred() = default;
  Fred(int value) : value(value) {}

  int get() const { return value; }
};

std::ostream& operator<<(std::ostream& s, const Fred& f) {
  s << "Fred(" << f.get() << ")";
  return s;
}

template <std::size_t N>
std::ostream& operator<<(std::ostream& s, const std::array<Fred, N>& freds) {
  s << "{ ";
  for (auto& f : freds) s << " Fred(" << f.get() << ") ";
  s << " }";
  return s;
}

template <std::size_t N>
std::ostream& operator<<(std::ostream& s, Fred (&freds)[N]) {
  s << "{ ";
  for (auto& f : freds) s << " Fred(" << f.get() << ") ";
  s << " }";
  return s;
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& vec) {
  s << "{ ";
  for (auto& v : vec) s << " " << v << " ";
  s << " }";
  return s;
}

#include "ttg/util/serialization.h"

#include <catch2/catch.hpp>

TEST_CASE("MADNESS Serialization", "[serialization]") {
  // Test code written as if calling from C
  auto test = [](const auto& t) {
    using T = std::decay_t<decltype(t)>;

    CHECK(madness::archive::is_serializable_v<madness::archive::BufferOutputArchive, T>);
    CHECK(madness::archive::is_serializable_v<madness::archive::BufferInputArchive, T>);

    // This line has to be in a piece of C++ that knows the type T
    const ttg_data_descriptor* d = ttg::get_data_descriptor<T>();

    // The rest could be in C ... deliberately use printf below rather than C++ streamio
    void* vt = (void*)&t;
    // printf("%s header_size=%llu, payload_size=%llu\n", d->name, d->header_size(vt), d->payload_size(vt));

    // Serialize into a buffer
    char buf[256];
    void* buf_ptr = (void*)buf;
    uint64_t pos = 0;
    CHECK_NOTHROW(pos = d->pack_payload(vt, sizeof(T), pos, buf_ptr));
    printf("serialized ");
    d->print(vt);

    T g_obj;
    void* g = (void*)&g_obj;
    CHECK_NOTHROW(d->unpack_payload(g, sizeof(T), 0, (const void*)buf));
    printf("deserialized ");
    d->print(g);
  };

  test(99);
  test(Fred(33));
  test(99.0);
  test(std::array<Fred, 3>{{Fred(55), Fred(66), Fred(77)}});
  int a[4] = {1, 2, 3, 4};
  test(a);
  Fred b[4] = {Fred(1), Fred(2), Fred(3), Fred(4)};
  test(b);
  test(std::vector<int>{1, 2, 3});
}

TEST_CASE("Boost Serialization", "[serialization]") {
  auto test = [](const auto& t) {
    using T = std::decay_t<decltype(t)>;
    CHECK(boost::serialization::is_serializable_v<boost::archive::binary_iarchive, T>);
    CHECK(boost::serialization::is_serializable_v<boost::archive::binary_oarchive, T>);
  };

  test(99);
  test(Fred(33));
  test(99.0);
  test(std::array<Fred, 3>{{Fred(55), Fred(66), Fred(77)}});
  int a[4] = {1, 2, 3, 4};
  test(a);
  Fred b[4] = {Fred(1), Fred(2), Fred(3), Fred(4)};
  test(b);
  test(std::vector<int>{1, 2, 3});
}
