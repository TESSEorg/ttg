// SPDX-License-Identifier: BSD-3-Clause

#ifndef TTG_DEMANGLE_H_INCLUDED
#define TTG_DEMANGLE_H_INCLUDED

#include <cxxabi.h>
#include <string>
#include <typeinfo>
#include <memory>
#define HAVE_CXA_DEMANGLE
#ifdef HAVE_CXA_DEMANGLE

namespace ttg {
  namespace detail {
    template <typename T>
    static std::string demangled_type_name(T *x = nullptr) {
      const char *name = nullptr;
      if constexpr (std::is_void_v<T>)
        name = "void";
      else
        name = (x != nullptr) ? typeid(*x).name() :  // this works for polymorphic types
                   typeid(T).name();
      static size_t buf_size = 1024;
      static std::unique_ptr<char, decltype(std::free) *> buf{reinterpret_cast<char *>(malloc(sizeof(char) * buf_size)),
                                                              std::free};
      int status;
      char *res = abi::__cxa_demangle(name, buf.get(), &buf_size, &status);
      if (status != 0 || res == nullptr) {
        return name;
      } else {
        if (res != buf.get()) {
          buf.release();
          buf = std::unique_ptr<char, decltype(std::free) *>{res, std::free};
        }
        return res;
      }
    }
#else
template <typename T>
std::string demangled_type_name(T* x = nullptr) {
  const char* name = (x != nullptr) ? typeid(*x).name() :  // this works for polymorphic types
                         typeid(T).name();
  return std::string(name);
}
#endif
  }  // namespace detail
}  // namespace ttg

#endif  // TTG_DEMANGLE_H_INCLUDED
