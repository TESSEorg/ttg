#include "ttg/serialization.h"

#include "ttg/util/meta.h"

#include "ttg/serialization/std/array.h"
#include "ttg/serialization/std/vector.h"

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/array.hpp>
#endif

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

namespace intrusive::symmetric::mc {

  // POD with serialization specified explicitly, to ensure that user-provided serialization takes precedence over
  // bitcopy
  class POD {
    int value;

   public:
    POD() = default;
    POD(int value) : value(value) {}

    int get() const { return value; }

    template <typename Archive>
    void serialize(Archive& ar) {
      std::int64_t junk = 17;
      ar& value& junk;
    }
  };
  static_assert(std::is_trivially_copyable_v<POD>);

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
}  // namespace intrusive::symmetric::mc

static_assert(madness::has_member_serialize_v<intrusive::symmetric::mc::POD, madness::archive::BufferOutputArchive>);
static_assert(madness::has_member_serialize_v<intrusive::symmetric::mc::POD, madness::archive::BufferInputArchive>);
static_assert(madness::is_user_serializable_v<madness::archive::BufferOutputArchive, intrusive::symmetric::mc::POD>);
static_assert(madness::is_user_serializable_v<madness::archive::BufferInputArchive, intrusive::symmetric::mc::POD>);
static_assert(madness::is_output_archive_v<madness::archive::BufferOutputArchive>);

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

    bool operator==(const NonPOD& other) const { return value == other.value; }
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

#ifdef TTG_SERIALIZATION_SUPPORTS_CEREAL
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

namespace freestanding::symmetric::bc_v {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get_value() const { return value; }

    bool operator==(const NonPOD& other) const { return value == other.value; }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);

  template <typename Archive>
  void serialize(Archive& ar, NonPOD& obj, const unsigned int) {
    if constexpr (ttg::detail::is_output_archive_v<Archive>)
      ar& obj.get_value();
    else {
      int v;
      ar& v;
      obj = NonPOD(v);
    }
  }

}  // namespace freestanding::symmetric::bc_v

#include <vector>

#include "ttg/serialization/data_descriptor.h"

#include <catch2/catch.hpp>

static_assert(ttg::detail::is_madness_buffer_serializable_v<int>);
static_assert(!ttg::detail::is_madness_user_buffer_serializable_v<int>);
static_assert(!ttg::detail::is_boost_user_buffer_serializable_v<int>);
static_assert(!ttg::detail::is_cereal_user_buffer_serializable_v<int>);
static_assert(!ttg::detail::is_user_buffer_serializable_v<int>);
static_assert(ttg::detail::is_madness_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_madness_user_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_boost_user_buffer_serializable_v<int[4]>);
// static_assert(!ttg::detail::is_cereal_user_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_user_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_user_buffer_serializable_v<std::array<int, 4>>);

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
TEST_CASE("MADNESS Serialization", "[serialization]") {
  // Test code written as if calling from C
  auto test = [](const auto& t) {
    using T = ttg::meta::remove_cvr_t<decltype(t)>;

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

    T g_obj;
    void* g = (void*)&g_obj;
    CHECK_NOTHROW(d->unpack_payload(g, obj_size, 0, buf.get()));
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

  static_assert(!ttg::detail::is_madness_input_serializable_v<madness::archive::BufferInputArchive, NonPOD>);
  static_assert(!ttg::detail::is_madness_output_serializable_v<madness::archive::BufferOutputArchive, NonPOD>);

  auto test_nonpod = [&test](const auto& t) {
    using T = ttg::meta::remove_cvr_t<decltype(t)>;
    static_assert(ttg::detail::is_madness_input_serializable_v<madness::archive::BufferInputArchive, T>);
    static_assert(ttg::detail::is_madness_output_serializable_v<madness::archive::BufferOutputArchive, T>);
    test(T(33));
    test(std::array{T(55), T(66), T(77)});
    T b[4] = {T(1), T(2), T(3), T(4)};
    test(b);
  };

  test_nonpod(intrusive::symmetric::mc::NonPOD{});
  test_nonpod(intrusive::symmetric::bc_v::NonPOD{});
  test_nonpod(nonintrusive::symmetric::m::NonPOD{});
  test_nonpod(nonintrusive::asymmetric::m::NonPOD{});

  {  // test that user-provided serialization overrides the default
    using nonpod_t = intrusive::symmetric::mc::NonPOD;
    static_assert(!std::is_trivially_copyable_v<nonpod_t> && ttg::detail::is_user_buffer_serializable_v<nonpod_t>);
    const ttg_data_descriptor* d_nonpod = ttg::get_data_descriptor<nonpod_t>();

    using pod_t = intrusive::symmetric::mc::POD;
    // ! user-defined serialization method overrides the default methods
    static_assert(std::is_trivially_copyable_v<pod_t> && ttg::detail::is_user_buffer_serializable_v<pod_t>);
    const ttg_data_descriptor* d_pod = ttg::get_data_descriptor<pod_t>();

    nonpod_t nonpod;
    pod_t pod;
    CHECK_NOTHROW(d_pod->payload_size(&pod));
    CHECK(d_nonpod->payload_size(&nonpod) + sizeof(std::int64_t) == d_pod->payload_size(&pod));
    // MADNESS buffer archives do not use cookies, so should be as efficient as direct copy
    CHECK(d_nonpod->payload_size(&nonpod) == sizeof(nonpod));
  }
}
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST

static_assert(std::is_same_v<typename boost::serialization::implementation_level<int>::type,
                             boost::mpl::int_<boost::serialization::level_type::primitive_type>>);
static_assert(std::is_same_v<typename boost::serialization::implementation_level<int[4]>::type,
                             boost::mpl::int_<boost::serialization::level_type::object_serializable>>);
static_assert(std::is_same_v<typename boost::serialization::implementation_level<std::array<int, 4>>::type,
                             boost::mpl::int_<boost::serialization::level_type::object_class_info>>);
static_assert(std::is_same_v<typename boost::serialization::implementation_level<POD>::type,
                             boost::mpl::int_<boost::serialization::level_type::object_class_info>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, int>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, int>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, int[4]>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, int[4]>);
static_assert(ttg::detail::is_stlcontainer_boost_serializable_v<boost::archive::binary_oarchive, std::array<int, 4>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::array<int, 4>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, const std::array<int, 4>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::array<int, 4>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, const std::array<int, 4>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::vector<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, const std::vector<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::vector<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, const std::vector<int>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, POD>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, POD>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, POD[4]>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, POD[4]>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::array<POD, 4>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::array<POD, 4>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::vector<POD>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::vector<POD>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, NonPOD>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, NonPOD>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, NonPOD[4]>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, NonPOD[4]>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::array<NonPOD, 4>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::array<NonPOD, 4>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::vector<NonPOD>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::vector<NonPOD>>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, intrusive::symmetric::mc::POD>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, intrusive::symmetric::mc::POD>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, intrusive::symmetric::mc::NonPOD>);
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, intrusive::symmetric::mc::NonPOD>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, intrusive::symmetric::bc_v::NonPOD>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, intrusive::symmetric::bc_v::NonPOD>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, intrusive::symmetric::bc_v::NonPOD[4]>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, intrusive::symmetric::bc_v::NonPOD[4]>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive,
                                                   std::array<intrusive::symmetric::bc_v::NonPOD, 4>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive,
                                                   std::array<intrusive::symmetric::bc_v::NonPOD, 4>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive,
                                                   std::vector<intrusive::symmetric::bc_v::NonPOD>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive,
                                                   std::vector<intrusive::symmetric::bc_v::NonPOD>>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, freestanding::symmetric::bc_v::NonPOD>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, freestanding::symmetric::bc_v::NonPOD>);

TEST_CASE("Boost Serialization", "[serialization]") {
  auto test = [](const auto& t) {
    using T = std::remove_reference_t<decltype(t)>;
    using Tnc = std::remove_cv_t<T>;
    if constexpr (ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, T> &&
                  ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, Tnc>) {
      CHECK(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, T>);
      CHECK(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, Tnc>);

      constexpr std::size_t buffer_size = 4096;
      char buffer[buffer_size];
      boost::iostreams::basic_array_sink<char> oabuf(buffer, buffer_size);
      boost::iostreams::stream<boost::iostreams::basic_array_sink<char>> sink(oabuf);
      boost::archive::binary_oarchive oa(sink);
      oa << t;

      boost::iostreams::basic_array_source<char> iabuf(buffer, buffer_size);
      boost::iostreams::stream<boost::iostreams::basic_array_source<char>> source(iabuf);
      boost::archive::binary_iarchive ia(source);
      std::remove_cv_t<std::remove_reference_t<decltype(t)>> t_copy;
      ia >> t_copy;

      if constexpr (!boost::is_array<T>::value) {  // why Catch2 fails to compare plain arrays correctly?
        CHECK(t == t_copy);
      } else {
        constexpr auto n = std::extent_v<T>;
        for (size_t i = 0; i != n; ++i) CHECK(t[i] == t_copy[i]);
      }
    }
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
  test(intrusive::symmetric::bc_v::NonPOD{17});
  test(freestanding::symmetric::bc_v::NonPOD{18});
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
