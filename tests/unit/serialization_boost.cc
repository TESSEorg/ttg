//
// Created by Eduard Valeyev on 2/27/24.
//

#include "ttg/serialization.h"

#include "ttg/util/meta.h"

#include "ttg/serialization/data_descriptor.h"

struct pod {
  double a;
  int b;
  float c[3];
  friend bool operator==(const pod& lhs, const pod& rhs) {
    return lhs.a == rhs.a && lhs.b == rhs.b && lhs.c[0] == rhs.c[0] && lhs.c[1] == rhs.c[1] && lhs.c[2] == rhs.c[2];
  }
};

BOOST_CLASS_IMPLEMENTATION(pod, primitive_type)
BOOST_IS_BITWISE_SERIALIZABLE(pod)

#include "ttg/serialization/std/vector.h"
#include "ttg/serialization/std/array.h"

static_assert(ttg::detail::is_memcpyable_v<pod>);
static_assert(ttg::detail::is_boost_buffer_serializable_v<std::vector<pod>>);

template <typename T>
void save_to_buffer(const T& t, char* buffer, std::size_t buffer_size) {
  ttg::detail::byte_ostreambuf oabuf(buffer, buffer_size);
  ttg::detail::boost_byte_oarchive oa(oabuf);
  oa << t;
}

int main() {

  std::array<char, 32768> buf;

  constexpr auto N = 10;
  pod x{1., 2, {3., 4., 5.}};
  std::vector<pod> vx(N,x);
  std::array<pod, 5> ax{{x, x, x, x, x}};

//  const ttg_data_descriptor* pod_dd = ttg::get_data_descriptor<pod>();
//  auto x_size = pod_dd->payload_size(&x);

  auto vx_size = ttg::default_data_descriptor<decltype(vx)>::pack_payload(&vx, size(buf), 0, data(buf));
  auto ax_size = ttg::default_data_descriptor<decltype(ax)>::pack_payload(&ax, size(buf)-vx_size, vx_size, data(buf));

  decltype(vx) vx_copy;
  decltype(ax) ax_copy;
  auto vx_copy_size = ttg::default_data_descriptor<decltype(vx)>::unpack_payload(&vx_copy, size(buf), 0, data(buf));
  assert(vx_copy == vx);
  ttg::default_data_descriptor<decltype(ax)>::unpack_payload(&ax_copy, size(buf)-vx_copy_size, vx_copy_size, data(buf));
  assert(ax_copy == ax);

//  constexpr std::size_t buffer_size = 4096;
//  char buffer[buffer_size];
//  save_to_buffer(vx, buffer, buffer_size);
//  save_to_buffer(ax, buffer, buffer_size);

  return 0;
}