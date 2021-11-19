//
// Created by Eduard Valeyev on 11/6/21.
//

#include "ttg/util/env.h"

#include <thread>
#include <stdexcept>

#include <cstdlib>

namespace ttg {
  namespace detail {

    int num_threads() {
      std::size_t result = 0;
      const char* ttg_num_threads_cstr = std::getenv("TTG_NUM_THREADS");
      if (ttg_num_threads_cstr) {
        const auto result_long = std::atol(ttg_num_threads_cstr);
        if (result_long >= 1)
          result = static_cast<std::size_t>(result_long);
        else
          throw std::runtime_error("ttg: invalid value of environment variable TTG_NUM_THREADS");
      } else {
        result = std::thread::hardware_concurrency();
      }
      if (result > std::numeric_limits<int>::max())
        throw std::runtime_error("ttg: number of threads exceeds the maximum limit");

      return static_cast<int>(result);
    }

  }  // namespace detail
}  // namespace ttg
