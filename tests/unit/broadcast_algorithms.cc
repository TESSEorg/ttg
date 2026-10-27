// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include "ttg.h"

#ifdef TTG_USE_PARSEC

#include <deque>
#include <map>
#include <set>

#include "ttg/parsec/broadcast.h"

namespace {

  /**
   * Simulates a broadcast from root to the processes in procset by letting each
   * process that holds the data forward it as the broadcast algorithm prescribes.
   * Returns the number of times each process received the data.
   */
  template <template <class> class Bcast>
  std::map<int, int> simulate(int root, const std::set<int> &procset, bool &sent_without_peers) {
    using Iter = std::set<int>::const_iterator;
    std::map<int, int> received;
    std::deque<int> active{root};
    sent_without_peers = false;
    while (!active.empty()) {
      int me = active.front();
      active.pop_front();
      Bcast<Iter> bcast(root, me, procset.begin(), procset.end());
      bool sent = false;
      bcast([&](int p) {
        sent = true;
        /* only forward on first receipt so that a cycle cannot loop forever */
        if (p != me && ++received[p] == 1) active.push_back(p);
      });
      /* has_peers() decides whether a message is prepared, it must match whether one is sent */
      if (sent != bcast.has_peers()) sent_without_peers = true;
    }
    return received;
  }

  template <template <class> class Bcast>
  void check_all_procsets(int max_procs) {
    for (int nprocs = 1; nprocs <= max_procs; ++nprocs) {
      for (int root = 0; root < nprocs; ++root) {
        for (int mask = 0; mask < (1 << nprocs); ++mask) {
          std::set<int> procset;
          for (int p = 0; p < nprocs; ++p) {
            if (mask & (1 << p)) procset.insert(p);
          }
          CAPTURE(nprocs, root, mask);
          bool sent_without_peers;
          auto received = simulate<Bcast>(root, procset, sent_without_peers);
          CHECK(!sent_without_peers);
          /* every process in the set except the root receives exactly once ... */
          for (int p : procset) {
            if (p != root) CHECK(received[p] == 1);
          }
          /* ... and no other process receives anything */
          for (auto [p, n] : received) {
            CHECK((p != root && procset.contains(p)));
          }
        }
      }
    }
  }

}  // namespace

TEST_CASE("Broadcast-Algorithms", "[broadcast]") {
  SECTION("star") { check_all_procsets<ttg_parsec::BroadcastStar>(7); }
  SECTION("pipe") { check_all_procsets<ttg_parsec::BroadcastPipe>(7); }
}

#endif  // TTG_USE_PARSEC
