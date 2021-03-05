#include <array>

#include "ttg.h"

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

// Test code written as if calling from C
template <typename T>
void test_serialization(const T& t) {
  // This line has to be in a piece of C++ that knows the type T
  const ttg_data_descriptor* d = ttg::get_data_descriptor<T>();

  // The rest could be in C ... deliberately use printf below rather than C++ streamio
  void* vt = (void*)&t;
  // printf("%s header_size=%llu, payload_size=%llu\n", d->name, d->header_size(vt), d->payload_size(vt));

  // Serialize into a buffer
  char buf[256];
  void* buf_ptr = (void*)buf;
  uint64_t pos = 0;
  pos = d->pack_payload(vt, sizeof(T), pos, buf_ptr);
  printf("serialized ");
  d->print(vt);

  T g_obj;
  void* g = (void*)&g_obj;
  d->unpack_payload(g, sizeof(T), 0, (const void*)buf);
  printf("deserialized ");
  d->print(g);
}

int main(int argc, char** argv) {
  test_serialization(99);
  test_serialization(Fred(33));
  test_serialization(99.0);
  test_serialization(std::array<Fred, 3>{{Fred(55), Fred(66), Fred(77)}});
  int a[4] = {1, 2, 3, 4};
  test_serialization(a);
  Fred b[4] = {Fred(1), Fred(2), Fred(3), Fred(4)};
  test_serialization(b);
  test_serialization(std::vector<int>{1, 2, 3});

  return 0;
}
