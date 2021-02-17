#ifndef TTG_UTIL_MACRO_H
#define TTG_UTIL_MACRO_H


/* Used to suppres compiler warnings on unused variables */
#define TTGUNUSED(x) ((void)(x))


// pattern from https://www.fluentcpp.com/2017/10/27/function-aliases-cpp/
#define TTG_UTIL_ALIAS_TEMPLATE_FUNCTION(aliasname,funcname)\
template<typename... Args> \
inline auto aliasname(Args&&... args) \
{ \
    return funcname(std::forward<Args>(args)...); \
}

#endif // TTG_UTIL_MACRO_H
