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

namespace intrusive::symmetric::any {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }

    template <typename Archive>
    void serialize(Archive& ar) {
      ar& value;
    }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);
}  // namespace intrusive::symmetric::any

namespace intrusive::symmetric::bc_v {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }

    // boost uses `unsigned int` for version, cereal uses `std::uint32_t`
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar& value;
    }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);
}  // namespace intrusive::symmetric::bc_v

namespace intrusive::symmetric::c {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }

    // versioned
    template <class Archive>
    void serialize(Archive& ar) {
      ar(value);
    }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);
}  // namespace intrusive::symmetric::c

namespace intrusive::symmetric::c_v {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }

    // versioned
    template <class Archive>
    void serialize(Archive& ar, std::uint32_t const version) {
      ar(value);
    }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);

}  // namespace intrusive::symmetric::c_v

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
CEREAL_CLASS_VERSION(intrusive::symmetric::c_v::NonPOD, 17);
#endif

namespace intrusive::asymmetric::bc {}
namespace intrusive::asymmetric::c {}

namespace nonintrusive::symmetric::m {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);

}  // namespace nonintrusive::symmetric::m

namespace madness::archive {
  template <class Archive>
  struct ArchiveSerializeImpl<Archive, nonintrusive::symmetric::m::NonPOD> {
    static inline void serialize(const Archive& ar, nonintrusive::symmetric::m::NonPOD& obj) { ar& obj.get(); };
  };
}  // namespace madness::archive

namespace nonintrusive::asymmetric::m {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }
    int& get() { return value; }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);

}  // namespace nonintrusive::asymmetric::m

namespace madness::archive {
  template <class Archive>
  struct ArchiveLoadImpl<Archive, nonintrusive::asymmetric::m::NonPOD> {
    static inline void load(const Archive& ar, nonintrusive::asymmetric::m::NonPOD& obj) { ar >> obj.get(); };
  };
  template <class Archive>
  struct ArchiveStoreImpl<Archive, nonintrusive::asymmetric::m::NonPOD> {
    static inline void store(const Archive& ar, const nonintrusive::asymmetric::m::NonPOD& obj) { ar << obj.get(); };
  };
}  // namespace madness::archive

namespace freestanding::symmetric::bc {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);

  template <typename Archive>
  void serialize(Archive&, NonPOD& obj) {
    abort();
  }

}  // namespace freestanding::symmetric::bc

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
    std::size_t obj_size;
    CHECK_NOTHROW(obj_size = d->payload_size(vt));
    auto buf = std::make_unique<char[]>(obj_size);
    uint64_t pos = 0;
    CHECK_NOTHROW(pos = d->pack_payload(vt, obj_size, pos, buf.get()));
    printf("serialized ");
    d->print(vt);

    T g_obj;
    void* g = (void*)&g_obj;
    CHECK_NOTHROW(d->unpack_payload(g, obj_size, 0, buf.get()));
    printf("deserialized ");
    d->print(g);
  };

  test(99);
  test(99.0);
  int a[4] = {1, 2, 3, 4};
  test(a);
  test(std::array{1, 2, 3, 4});
  test(std::vector<int>{1, 2, 3});

  test(POD(33));
  test(std::array{POD(55), POD(66), POD(77)});
  POD b[4] = {POD(1), POD(2), POD(3), POD(4)};
  test(b);

  // these should pass, but don't due to MADNESS not properly disabling operator&(Archive,Obj) (?)
  // static_assert(!ttg::detail::is_madness_input_serializable_v<madness::archive::BufferInputArchive, NonPOD>);
  // static_assert(!ttg::detail::is_madness_output_serializable_v<madness::archive::BufferOutputArchive, NonPOD>);

  auto test_nonpod = [&test](const auto& t) {
    using T = std::decay_t<decltype(t)>;
    static_assert(ttg::detail::is_madness_input_serializable_v<madness::archive::BufferInputArchive, T>);
    static_assert(ttg::detail::is_madness_output_serializable_v<madness::archive::BufferOutputArchive, T>);
    test(T(33));
    test(std::array{T(55), T(66), T(77)});
    T b[4] = {T(1), T(2), T(3), T(4)};
    test(b);
  };

  test_nonpod(intrusive::symmetric::any::NonPOD{});
  test_nonpod(intrusive::symmetric::bc_v::NonPOD{});
  test_nonpod(nonintrusive::symmetric::m::NonPOD{});
  test_nonpod(nonintrusive::asymmetric::m::NonPOD{});
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
