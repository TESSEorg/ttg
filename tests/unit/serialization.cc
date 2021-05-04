#include "ttg/serialization.h"

class POD {
  int value;

 public:
  POD() = default;
  POD(int value) : value(value) {}

  int get() const { return value; }
};
static_assert(std::is_trivially_copyable_v<POD>);

std::ostream& operator<<(std::ostream& s, const POD& f) {
  s << "POD(" << f.get() << ")";
  return s;
}

template <std::size_t N>
std::ostream& operator<<(std::ostream& s, const std::array<POD, N>& freds) {
  s << "{ ";
  for (auto& f : freds) s << " " << f << " ";
  s << " }";
  return s;
}

template <std::size_t N>
std::ostream& operator<<(std::ostream& s, POD (&freds)[N]) {
  s << "{ ";
  for (auto& f : freds) s << " " << f << " ";
  s << " }";
  return s;
}

class NonPOD {
  int value;

 public:
  NonPOD() = default;
  NonPOD(int value) : value(value) {}
  NonPOD(const NonPOD& other) : value(other.value) {}

  int get() const { return value; }
};
static_assert(!std::is_trivially_copyable_v<NonPOD>);

#include <vector>

#include "ttg/serialization/data_descriptor.h"

#include <catch2/catch.hpp>

template <typename T>
struct type_printer;

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
TEST_CASE("MADNESS Serialization", "[serialization]") {
  // Test code written as if calling from C
  auto test = [](const auto& t) {
    using T = std::decay_t<decltype(t)>;

    CHECK(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, T>);
    CHECK(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, T>);

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
  test(POD(33));
  test(99.0);
  test(std::array<POD, 3>{{POD(55), POD(66), POD(77)}});
  int a[4] = {1, 2, 3, 4};
  test(a);
  POD b[4] = {POD(1), POD(2), POD(3), POD(4)};
  test(b);
  test(std::vector<int>{1, 2, 3});

  //  static_assert(!ttg::detail::is_madness_input_serializable_v<madness::archive::BufferInputArchive, NonPOD>);
  //  static_assert(!ttg::detail::is_madness_output_serializable_v<madness::archive::BufferOutputArchive, NonPOD>);
}
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
TEST_CASE("Boost Serialization", "[serialization]") {
  auto test = [](const auto& t) {
    using T = std::remove_reference_t<decltype(t)>;
    CHECK(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, T>);
    using Tnc = std::remove_const_t<T>;
    CHECK(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, Tnc>);
  };

  test(99);
  test(POD(33));
  test(99.0);
  test(std::array<POD, 3>{{POD(55), POD(66), POD(77)}});
  int a[4] = {1, 2, 3, 4};
  test(a);
  POD b[4] = {POD(1), POD(2), POD(3), POD(4)};
  test(b);
  test(std::vector<int>{1, 2, 3});
}
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
TEST_CASE("Cereal Serialization", "[serialization]") {
  auto test = [](const auto& t) {
    using T = std::remove_reference_t<decltype(t)>;
    CHECK(ttg::detail::is_cereal_serializable_v<cereal::BinaryOutputArchive, T>);
    using Tnc = std::remove_const_t<T>;
    CHECK(ttg::detail::is_cereal_serializable_v<cereal::BinaryInputArchive, Tnc>);
  };

  test(99);
  test(POD(33));
  test(99.0);
  test(std::array<POD, 3>{{POD(55), POD(66), POD(77)}});
  int a[4] = {1, 2, 3, 4};
  test(a);
  POD b[4] = {POD(1), POD(2), POD(3), POD(4)};
  test(b);
  test(std::vector<int>{1, 2, 3});
}
#endif  // TTG_SERIALIZATION_SUPPORTS_CEREAL
