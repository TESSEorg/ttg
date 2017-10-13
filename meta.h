#ifndef CXXAPI_META_H
#define CXXAPI_META_H

#include <type_traits>

namespace ttg {

namespace meta {

#if __cplusplus >= 201703L
using std::void_t;
#else
template<class...> using void_t = void;
#endif

}  // namespace meta

}  // namespace ttg

#endif //CXXAPI_SERIALIZATION_H_H
