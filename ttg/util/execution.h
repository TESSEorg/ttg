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
  Invalid
};

};

#endif //TTG_EXECUTION_H
