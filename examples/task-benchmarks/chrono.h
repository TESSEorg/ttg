//
// Created by Eduard Valeyev on 10/24/21.
//

#ifndef TEST_BENCHMARKS_CHRONO_H
#define TEST_BENCHMARKS_CHRONO_H

#include <chrono>

using time_point = std::chrono::high_resolution_clock::time_point;

inline time_point now() { return std::chrono::high_resolution_clock::now(); }

inline std::chrono::system_clock::time_point system_now() {
  return std::chrono::system_clock::now();
}

inline int64_t duration_in_mus(time_point const &t0, time_point const &t1) {
  return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

#endif // TEST_BENCHMARKS_CHRONO_H
