#include <typeinfo>
#include <cxxabi.h>
#include <string>
#define HAVE_CXA_DEMANGLE
#ifdef HAVE_CXA_DEMANGLE
template <typename T>
static
std::string demangled_type_name()
{
    const char* name = typeid(T).name();
    static char buf[10240]; // should really be allocated with malloc
    size_t size=10240;
    int status;
    char* res = abi::__cxa_demangle (name, buf, &size, &status);
    if (res) return res;
    else return name;
}
#else
template <typename T>
std::string demangled_type_name()
{
    return std::string(typeid(T).name());
}
#endif  
