//
// Created by Eduard Valeyev on 8/28/18.
//

#ifndef TTG_EXECUTION_H
#define TTG_EXECUTION_H

namespace ttg {

/// denotes task execution policy
enum class Execution {
  Inline, // calls on the caller's thread
  Async   // calls asynchronously, e.g. by firing off a task
};

/// denotes task execution space
enum class ExecutionSpace {
  Host,   // a CPU
  CUDA,   // an NVIDIA CUDA device
  HIP,    // an AMD HIP device
  L0,     // an Intel L0 device
  Invalid
};

namespace detail {
  inline const char *execution_space_name(ExecutionSpace space) noexcept {
    switch (space) {
      case ExecutionSpace::Host: return "Host";
      case ExecutionSpace::CUDA: return "CUDA";
      case ExecutionSpace::HIP: return "HIP";
      case ExecutionSpace::Invalid: return "INVALID";
      default: return "UNKNOWN";
    }
  }
} // namespace detail

};

#endif //TTG_EXECUTION_H
