#include "ttg/serialization.h"

#include "ttg/util/meta.h"

#include "ttg/serialization/std/array.h"
#include "ttg/serialization/std/tuple.h"
#include "ttg/serialization/std/vector.h"

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#ifdef TTG_HAS_BTAS
#include <btas/serialization.h>
#include <btas/tensor.h>
#include <btas/util/mohndle.h>
using tensor_t =
    btas::Tensor<double, btas::DEFAULT::range, btas::mohndle<btas::varray<double>, btas::Handle::shared_ptr>>;
namespace ttg::detail {
  // BTAS defines all of its Boost serializers in boost::serialization namespace ... as explained in
  // ttg/serialization/boost.h such functions are not detectable via SFINAE, so must explicitly define serialization
  // traits here
  template <typename Archive>
  inline static constexpr bool is_boost_serializable_v<Archive, tensor_t> = is_boost_archive_v<Archive>;
  template <typename Archive>
  inline static constexpr bool is_boost_serializable_v<Archive, const tensor_t> = is_boost_archive_v<Archive>;
}  // namespace ttg::detail
#endif

class POD {
  int value;

 public:
  POD() = default;
  POD(int value) : value(value) {}

  int get() const { return value; }

  bool operator==(const POD& other) const { return value == other.value; }
};
static_assert(std::is_trivially_copyable_v<POD>);
static_assert(ttg::detail::is_memcpyable_v<POD>);

static_assert(!ttg::detail::is_madness_user_buffer_serializable_v<POD>);
#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
static_assert(!ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, POD>);
static_assert(ttg::detail::has_freestanding_boost_serialize_with_version_v<POD, boost::archive::binary_oarchive>);
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST
static_assert(!ttg::detail::is_boost_user_buffer_serializable_v<POD>);
static_assert(!ttg::detail::is_madness_user_buffer_serializable_v<std::array<POD, 3>>);
static_assert(!ttg::detail::is_boost_user_buffer_serializable_v<std::array<POD, 3>>);

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
// non-default ctor breaks trivial copyability
static_assert(!std::is_trivially_copyable_v<NonPOD>);
static_assert(!ttg::detail::is_memcpyable_v<NonPOD>);

// but can allow use of std::memcpy on type
class MemcpyableNonPOD : public NonPOD {};
namespace ttg::detail {
  template<> inline constexpr bool is_memcpyable_override_v<MemcpyableNonPOD> = true;
}  // namespace ttg::detail
static_assert(!std::is_trivially_copyable_v<MemcpyableNonPOD>);
static_assert(ttg::detail::is_memcpyable_v<MemcpyableNonPOD>);

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

    // boost uses `unsigned int` for version
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar& value;
    }

    bool operator==(const NonPOD& other) const { return value == other.value; }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);
}  // namespace intrusive::symmetric::bc_v

namespace intrusive_private::symmetric::bc_v {

  class NonPOD {
    int value;

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
    friend class boost::serialization::access;
#endif

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar& value;
    }

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }

    bool operator==(const NonPOD& other) const { return value == other.value; }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);
}  // namespace intrusive_private::symmetric::bc_v

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST

// boost serialization, with version (and object) tracking
namespace intrusive::asymmetric::b_v {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }

    template <typename Archive>
    void save(Archive& ar, const unsigned int version) const {
      ar& value;
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int version) {
      ar& value;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    bool operator==(const NonPOD& other) const { return value == other.value; }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);
}  // namespace intrusive::asymmetric::b_v

// boost serialization, without version (and object) tracking
namespace intrusive::asymmetric::b {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    int get() const { return value; }

    template <typename Archive>
    void save(Archive& ar, const unsigned int version) const {
      ar& value;
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int version) {
      ar& value;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    bool operator==(const NonPOD& other) const { return value == other.value; }
  };
  static_assert(!std::is_trivially_copyable_v<NonPOD>);
}  // namespace intrusive::asymmetric::b

// turns off saving class info, including version
BOOST_CLASS_IMPLEMENTATION(::intrusive::asymmetric::b::NonPOD, boost::serialization::object_serializable)
// turns off tracking
BOOST_CLASS_TRACKING(::intrusive::asymmetric::b::NonPOD, boost::serialization::track_never)

#if 0  // uncomment to debug internals of serialization dispatch in boost.ser (see oserializer.hpp)
template <typename Archive, typename T>
using typex =
typename boost::mpl::eval_if<
    // if its primitive
    boost::mpl::equal_to<
    boost::serialization::implementation_level< T >,
    boost::mpl::int_<boost::serialization::primitive_type>
>,
boost::mpl::identity<typename boost::archive::detail::save_non_pointer_type<Archive>::save_primitive>,
// else
typename boost::mpl::eval_if<
    // class info / version
    boost::mpl::greater_equal<
    boost::serialization::implementation_level< T >,
    boost::mpl::int_<boost::serialization::object_class_info>
>,
// do standard save
boost::mpl::identity<typename boost::archive::detail::save_non_pointer_type<Archive>::save_standard>,
// else
typename boost::mpl::eval_if<
    // no tracking
    boost::mpl::equal_to<
    boost::serialization::tracking_level< T >,
    boost::mpl::int_<boost::serialization::track_never>
>,
// do a fast save
boost::mpl::identity<typename boost::archive::detail::save_non_pointer_type<Archive>::save_only>,
// else
// do a fast save only tracking is turned off
boost::mpl::identity<typename boost::archive::detail::save_non_pointer_type<Archive>::save_conditional>
> > >::type;

static_assert(std::is_same_v<typex<boost::archive::binary_oarchive, intrusive::asymmetric::b_v::NonPOD>,
                             typename boost::archive::detail::save_non_pointer_type<boost::archive::binary_oarchive>::save_standard>);

static_assert(std::is_same_v<typex<boost::archive::binary_oarchive, intrusive::asymmetric::b::NonPOD>,
                             typename boost::archive::detail::save_non_pointer_type<boost::archive::binary_oarchive>::save_only>);
#endif

#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

namespace nonintrusive::symmetric::m {

  class NonPOD {
    int value;

   public:
    NonPOD() = default;
    NonPOD(int value) : value(value) {}
    NonPOD(const NonPOD& other) : value(other.value) {}

    // N.B. MUST provide ref for iovec serialization
    const int& get() const { return value; }
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

    const int& get() const { return value; }
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

    // N.B. MUST provide ref for iovec serialization
    int& get_value() { return value; }
    const int& get_value() const { return value; }

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

#include <catch2/catch_all.hpp>

static_assert(ttg::detail::is_madness_buffer_serializable_v<int>);
static_assert(!ttg::detail::is_madness_user_buffer_serializable_v<int>);
static_assert(!ttg::detail::is_boost_user_buffer_serializable_v<int>);
static_assert(!ttg::detail::is_user_buffer_serializable_v<int>);
static_assert(ttg::detail::is_madness_buffer_serializable_v<const int>);
static_assert(!ttg::detail::is_madness_user_buffer_serializable_v<const int>);
static_assert(!ttg::detail::is_boost_user_buffer_serializable_v<const int>);
static_assert(!ttg::detail::is_user_buffer_serializable_v<const int>);
static_assert(ttg::detail::is_madness_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_madness_user_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_boost_user_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_user_buffer_serializable_v<int[4]>);
static_assert(!ttg::detail::is_user_buffer_serializable_v<std::array<int, 4>>);

// default_data_descriptor<std::pair<int,int>> should be defined but std::is_trivially_copyable_v<std::pair<int,int>> is false
// is_memcpyable_v<std::pair<int,int>> is true
static_assert(!std::is_trivially_copyable_v<std::pair<int,int>>);
static_assert(ttg::detail::is_memcpyable_v<std::pair<int,int>>);

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS

static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, int>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, int>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, int[4]>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, int[4]>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, std::array<int, 4>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, const std::array<int, 4>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, std::array<int, 4>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, const std::array<int, 4>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, std::vector<int>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, std::vector<int>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, std::tuple<int, double>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, std::tuple<int, double>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, POD>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, POD>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, POD[4]>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, POD[4]>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, std::array<POD, 4>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, std::array<POD, 4>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, std::vector<POD>>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, std::vector<POD>>);
static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, NonPOD>);
static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, NonPOD>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, NonPOD[4]>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, NonPOD[4]>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, std::array<NonPOD, 4>>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, std::array<NonPOD, 4>>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, std::vector<NonPOD>>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, std::vector<NonPOD>>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive,
// intrusive::symmetric::mc::POD>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive,
// intrusive::symmetric::mc::POD>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive,
// intrusive::symmetric::mc::NonPOD>);
// static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive,
// intrusive::symmetric::mc::NonPOD>);

static_assert(
    !ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, intrusive::symmetric::bc_v::NonPOD>);
static_assert(
    !ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, intrusive::symmetric::bc_v::NonPOD>);

static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive,
                                                      freestanding::symmetric::bc_v::NonPOD>);
static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive,
                                                      freestanding::symmetric::bc_v::NonPOD>);

// private serialize method is not accessible to MADNESS
static_assert(
    !ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, intrusive::symmetric::bc_v::NonPOD>);
static_assert(
    !ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, intrusive::symmetric::bc_v::NonPOD>);

static_assert(
    ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, nonintrusive::symmetric::m::NonPOD>);
static_assert(ttg::detail::is_madness_user_buffer_serializable_v<nonintrusive::symmetric::m::NonPOD>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive,
                                                     const nonintrusive::symmetric::m::NonPOD>);
static_assert(ttg::detail::is_madness_user_buffer_serializable_v<const nonintrusive::symmetric::m::NonPOD>);
static_assert(
    ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, nonintrusive::asymmetric::m::NonPOD>);
static_assert(ttg::detail::is_madness_user_buffer_serializable_v<nonintrusive::asymmetric::m::NonPOD>);
static_assert(ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive,
                                                     const nonintrusive::asymmetric::m::NonPOD>);
static_assert(ttg::detail::is_madness_user_buffer_serializable_v<const nonintrusive::asymmetric::m::NonPOD>);

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
static_assert(
    !ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive, intrusive::asymmetric::b_v::NonPOD>);
static_assert(
    !ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive, intrusive::asymmetric::b_v::NonPOD>);
static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferOutputArchive,
                                                      std::array<intrusive::asymmetric::b_v::NonPOD, 3>>);
static_assert(!ttg::detail::is_madness_serializable_v<madness::archive::BufferInputArchive,
                                                      std::array<intrusive::asymmetric::b_v::NonPOD, 3>>);
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

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
  test(std::make_tuple(1, 2, 3));

  test(POD(33));
  test(std::array{POD(55), POD(66), POD(77)});
  POD b[4] = {POD(1), POD(2), POD(3), POD(4)};
  test(b);

#ifdef TTG_HAS_BTAS
  {
    tensor_t t(2, 3, 4);
    test(t);
  }
#endif

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
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::allocator<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::vector<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, const std::vector<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::vector<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, const std::vector<int>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::vector<std::vector<int>>>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, const std::vector<std::vector<int>>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::vector<std::vector<int>>>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, const std::vector<std::vector<int>>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, std::tuple<int, double>>);
static_assert(ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, std::tuple<int, double>>);

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
    ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, intrusive::symmetric::bc_v::NonPOD>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, intrusive::symmetric::bc_v::NonPOD>);

static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_oarchive, intrusive::asymmetric::b_v::NonPOD>);
static_assert(
    ttg::detail::is_boost_serializable_v<boost::archive::binary_iarchive, intrusive::asymmetric::b_v::NonPOD>);

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
  test(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}});
  test(std::make_tuple(1, 2, 3));
  test(intrusive::symmetric::bc_v::NonPOD{17});
  test(freestanding::symmetric::bc_v::NonPOD{18});

  // serialize pointers of various kinds
  auto test_ptr = [&](const auto& t) {
    using T = std::remove_reference_t<decltype(t)>;
    const auto* t_ptr = new T(t);

    constexpr std::size_t buffer_size = 4096;
    char buffer[buffer_size];
    boost::iostreams::basic_array_sink<char> oabuf(buffer, buffer_size);
    boost::iostreams::stream<boost::iostreams::basic_array_sink<char>> sink(oabuf);
    boost::archive::binary_oarchive oa(sink);
    oa << t_ptr;

    boost::iostreams::basic_array_source<char> iabuf(buffer, buffer_size);
    boost::iostreams::stream<boost::iostreams::basic_array_source<char>> source(iabuf);
    boost::archive::binary_iarchive ia(source);
    std::remove_cv_t<std::remove_reference_t<decltype(t_ptr)>> t_ptr_copy;
    ia >> t_ptr_copy;

    CHECK(t_ptr != t_ptr_copy);
    CHECK(*t_ptr == *t_ptr_copy);

    delete t_ptr;
  };
  // test_ptr(POD(33));  // N.B. POD is not boost serializable
  test_ptr(intrusive::symmetric::bc_v::NonPOD{17});
  test_ptr(freestanding::symmetric::bc_v::NonPOD{18});

#ifdef TTG_HAS_BTAS
  {
    tensor_t t(2, 3, 4);
    test(t);
  }
#endif
}
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#if defined(TTG_SERIALIZATION_SUPPORTS_MADNESS) && defined(TTG_SERIALIZATION_SUPPORTS_BOOST)
TEST_CASE("TTG Serialization", "[serialization]") {
  // Test code written as if calling from C
  auto test = [](const auto& t) {
    using T = ttg::meta::remove_cvr_t<decltype(t)>;

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
  test(std::vector<std::vector<int>>{{1, 2}, {3, 4, 5}, {6, 7, 8, 9}});

  auto test_struct = [&test](const auto& t) {
    using T = ttg::meta::remove_cvr_t<decltype(t)>;
    test(T(33));
    test(std::array{T(55), T(66), T(77)});
    T b[4] = {T(1), T(2), T(3), T(4)};
    test(b);
  };

  test_struct(POD{15});  // default
  // test_struct(NonPOD{16});
#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  test_struct(intrusive::symmetric::mc::POD{17});        // MADNESS
  test_struct(intrusive::symmetric::mc::NonPOD{17});     // MADNESS
  test_struct(nonintrusive::symmetric::m::NonPOD{18});   // MADNESS
  test_struct(nonintrusive::asymmetric::m::NonPOD{19});  // MADNESS
#endif
#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  test_struct(intrusive::asymmetric::b_v::NonPOD{21});     // Boost
  test_struct(intrusive::asymmetric::b::NonPOD{21});       // Boost
  test_struct(intrusive::symmetric::bc_v::NonPOD{20});     // Boost
  test_struct(freestanding::symmetric::bc_v::NonPOD{21});  // Boost
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  // verify that turning off version and object tracking for Boost produces same archive since the TTG boost archives
  // already strip out all the extras
  {
    using tracked_t = intrusive::asymmetric::b_v::NonPOD;
    const ttg_data_descriptor* d_tracked = ttg::get_data_descriptor<tracked_t>();

    using untracked_t = intrusive::asymmetric::b::NonPOD;
    const ttg_data_descriptor* d_untracked = ttg::get_data_descriptor<untracked_t>();

    tracked_t obj_tracked;
    untracked_t obj_untracked;
    CHECK_NOTHROW(d_tracked->payload_size(&obj_tracked));
    CHECK_NOTHROW(d_untracked->payload_size(&obj_untracked));
    // with tracking ON pack version + other metadata
    CHECK(d_tracked->payload_size(&obj_tracked) == d_untracked->payload_size(&obj_untracked));
  }
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  // try iovec archives
  {
    ttg::detail::boost_iovec_oarchive oar;
    std::tuple<intrusive::asymmetric::b_v::NonPOD, intrusive::asymmetric::b::NonPOD, intrusive::symmetric::bc_v::NonPOD,
               freestanding::symmetric::bc_v::NonPOD>
        t;
    oar << t;
    const auto& iovec = oar.streambuf().iovec();
    CHECK(iovec.size() == 4);
    CHECK(iovec[0].first == static_cast<const void*>(&(std::get<0>(t))));
    CHECK(iovec[0].second == sizeof(std::get<0>(t)));
    CHECK(iovec[1].first == static_cast<const void*>(&(std::get<1>(t))));
    CHECK(iovec[1].second == sizeof(std::get<1>(t)));
    CHECK(iovec[2].first == static_cast<const void*>(&(std::get<2>(t))));
    CHECK(iovec[2].second == sizeof(std::get<2>(t)));
    CHECK(iovec[3].first == static_cast<const void*>(&(std::get<3>(t))));
    CHECK(iovec[3].second == sizeof(std::get<3>(t)));
  }
#endif  // TTG_SERIALIZATION_SUPPORTS_BOOST
}

#endif
