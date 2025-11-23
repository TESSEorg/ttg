// SPDX-License-Identifier: BSD-3-Clause
#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/MADworld.h>
#include <madness/world/buffer_archive.h>

#include <cstdio>

using namespace madness;
using namespace std;

namespace c_interface {

  // Have to use a struct here since cannot have partial specialization of functions
  // and below use partial specialization to make templates for arrays.
  template <typename T>
  struct ugh {
    static int size(const void* t) {
      madness::archive::BufferOutputArchive ar;
      ar&(*(T*)t);
      return int(ar.size());
    }

    static void serialize(const void* t, void* ptr, int nbyte) {
      madness::archive::BufferOutputArchive ar(ptr, nbyte);
      ar&(*(T*)t);
    }

    static void* allocate_and_deserialize(const void* ptr, int nbyte) {
      T* t = new T;
      madness::archive::BufferInputArchive ar(ptr, nbyte);
      ar&(*t);
      return (void*)t;
    }

    static void deserialize(void* t, const void* ptr, int nbyte) {
      madness::archive::BufferInputArchive ar(ptr, nbyte);
      ar&(*(T*)t);
    }

    static void* allocate() { return (void*)(new T); }

    static void deallocate(void* ptr) { delete (T*)ptr; }

    static void print(const void* ptr) { std::cout << *(T*)ptr << std::endl; }
  };

  // Have to use a struct here since cannot have partial specialization of functions
  template <typename T, int N>
  struct ugh<T[N]> {
    static int size(const void* t) {
      madness::archive::BufferOutputArchive ar;
      ar& archive::wrap((T*)t, N);
      return int(ar.size());
    }

    static void serialize(const void* t, void* ptr, int nbyte) {
      madness::archive::BufferOutputArchive ar(ptr, nbyte);
      ar& archive::wrap((T*)t, N);
    }

    static void* allocate_and_deserialize(const void* ptr, int nbyte) {
      T* t = new T[N];
      madness::archive::BufferInputArchive ar(ptr, nbyte);
      ar& archive::wrap(t, N);
      return (void*)t;
    }

    static void deserialize(void* t, const void* ptr, int nbyte) {
      madness::archive::BufferInputArchive ar(ptr, nbyte);
      ar& archive::wrap((T*)t, N);
    }

    static void* allocate() { return (void*)(new T[N]); }

    static void deallocate(void* ptr) { delete[](T*) ptr; }

    static void print(const void* ptr) {
      std::cout << "[";
      for (int i = 0; i < N; i++) {
        std::cout << ((T*)ptr)[i];
        if (i == (N - 1))
          std::cout << "]" << std::endl;
        else
          std::cout << ", ";
      }
    }
  };

  // This struct and functions within it should be accessible from C
  struct data_descriptor {
    const char* name;
    void (*deallocate)(void*);
    void* (*allocate)();
    void* (*allocate_and_deserialize)(const void*, int);
    void (*deserialize)(void*, const void*, int);
    void (*serialize)(const void*, void*, int);
    int (*size)(const void*);
    void (*print)(const void*);  // prints item to C++ cout which is usually same as C stdout.
  };

  // Returns a pointer to a constant static instance initialized
  // once at run time.  Call this from a piece of C++ code (see
  // example below) and return the pointer to C.
  template <typename T>
  const data_descriptor* get_data_descriptor() {
    static const data_descriptor d = {
        typeid(T).name(),     &ugh<T>::deallocate, &ugh<T>::allocate, &ugh<T>::allocate_and_deserialize,
        &ugh<T>::deserialize, &ugh<T>::serialize,  &ugh<T>::size,     &ugh<T>::print};
    return &d;
  }
}  // namespace c_interface

class Fred {
  int value;

 public:
  Fred() : value(-1) {}

  Fred(int value) : value(value) {}

  int get() const { return value; }

  template <typename Archive>
  void serialize(const Archive& ar) {
    ar& value;
  }
};

std::ostream& operator<<(std::ostream& s, const Fred& f) {
  s << "Fred(" << f.get() << ")";
  return s;
}

// Test code written as if calling from C
template <typename T>
void test(const T& t) {
  // This line has to be in a piece of C++ that knows the type T
  const c_interface::data_descriptor* d = c_interface::get_data_descriptor<T>();

  // The rest could be in C ... deliberately use printf below rather than C++ streamio
  void* vt = (void*)&t;
  printf("%s %d\n", d->name, d->size(vt));

  // Serialize into a buffer
  char buf[256];
  d->serialize(vt, (void*)buf, sizeof(buf));

  void* g = d->allocate();
  d->deserialize(g, (void*)buf, sizeof(buf));
  printf("deserialize ");
  d->print(g);
  d->deallocate((void*)g);

  void* f = d->allocate_and_deserialize((void*)buf, sizeof(buf));
  printf("allocate and deserialize ");
  d->print(f);
  d->deallocate(f);
}

int main(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);
  try {
    test(99);
    test(Fred(33));
    test(99.0);
    test(std::vector<Fred>(4, Fred(55)));
    int a[4] = {1, 2, 3, 4};
    test(a);
    Fred b[4] = {Fred(1), Fred(2), Fred(3), Fred(4)};
    test(b);

  } catch (SafeMPI::Exception e) {
    error("caught an MPI exception");
  } catch (madness::MadnessException e) {
    print("XXX", e);
    error("caught a MADNESS exception");
  } catch (const char* s) {
    print(s);
    error("caught a string exception");
  } catch (...) {
    error("caught unhandled exception");
  }

  finalize();
  return 0;
}
