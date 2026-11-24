#ifndef TTG_UTIL_GENERATOR_H
#define TTG_UTIL_GENERATOR_H

#include <version>

/**
 * If the standard library provides std::generator, use it. Otherwise, use
 * the generator implementation adapted from Lewis Baker's generator
 * reference implementation.
 */

#if !defined(__cpp_lib_generator)

#include "ttg/3rd-party/generator/generator.h"

#else // __cpp_lib_generator

#include <generator>

#endif // __cpp_lib_generator

namespace ttg {

template<typename T>
using generator = std::generator<T>;

} // namespace ttg


#endif // TTG_UTIL_GENERATOR_H