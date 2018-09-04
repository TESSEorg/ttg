
#ifndef TTG_DEMANGLE_H_INCLUDED
#define TTG_DEMANGLE_H_INCLUDED

#include <cxxabi.h>
#include <string>
#include <typeinfo>
#define HAVE_CXA_DEMANGLE
#ifdef HAVE_CXA_DEMANGLE

namespace ttg {
  namespace detail {
    template <typename T>
    static std::string demangled_type_name(T *x = nullptr) {
      const char *name;
      if constexpr (std::is_void_v<T>)
        name = "void";
      else
        name = (x != nullptr) ? typeid(*x).name() :  // this works for polymorphic types
                                typeid(T).name();
      static char buf[10240];  // should really be allocated with malloc
      size_t size = 10240;
      int status;
      char *res = abi::__cxa_demangle(name, buf, &size, &status);
      if (res)
        return res;
      else
        return name;
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