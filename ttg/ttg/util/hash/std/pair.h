#ifndef TTG_UTIL_HASH_STD_PAIR_H
#define TTG_UTIL_HASH_STD_PAIR_H

#include "ttg/util/hash.h"

#include <utility>

namespace ttg::overload {

  template <typename T1, typename T2>
  struct hash<std::pair<T1, T2>,
              std::enable_if_t<meta::has_ttg_hash_specialization_v<T1> && meta::has_ttg_hash_specialization_v<T2>>> {
    auto operator()(const std::pair<T1, T2>& t) const {
      std::size_t seed = 0;
      hash_combine(seed, t.first);
      hash_combine(seed, t.second);
      return seed;
    }
  };

}  // namespace ttg::overload

#endif  // TTG_UTIL_HASH_PAIR_H
