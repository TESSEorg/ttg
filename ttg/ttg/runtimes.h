//
// Created by Eduard Valeyev on 8/28/18.
//

#ifndef TTG_TRAITS_H
#define TTG_TRAITS_H

#include <cstddef>
#include <cstdint>

#include "ttg/execution.h"

namespace ttg {

  enum class Runtime { PaRSEC, MADWorld };

  template <Runtime R>
  struct runtime_traits;

  template <>
  struct runtime_traits<Runtime::PaRSEC> {
    static constexpr const bool supports_streaming_terminal = true;
    static constexpr const bool supports_async_reduction = false;
    using hash_t = unsigned long;  // must be same as parsec_key_t
    constexpr static ExecutionSpace execution_spaces[] = {ExecutionSpace::CUDA, ExecutionSpace::Host};
    constexpr static std::size_t num_execution_spaces = sizeof(execution_spaces) / sizeof(ExecutionSpace);
  };

  template <>
  struct runtime_traits<Runtime::MADWorld> {
    static constexpr const bool supports_streaming_terminal = true;
    static constexpr const bool supports_async_reduction = true;
    using hash_t = uint64_t;
    constexpr static ExecutionSpace execution_spaces[] = {ExecutionSpace::Host};
    constexpr static std::size_t num_execution_spaces = sizeof(execution_spaces) / sizeof(ExecutionSpace);
  };

}  // namespace ttg

#endif  // TTG_TRAITS_H
