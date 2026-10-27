// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

#include "ttg.h"

#ifdef TTG_USE_PARSEC

#include <deque>
#include <map>
#include <set>
#include <utility>

#include "ttg/parsec/broadcast.h"

namespace {

  /**
   * Simulates a broadcast from root to the processes in procset by letting each
   * process that holds the data forward it as the broadcast algorithm prescribes.
   * Returns the number of times each process received the data and the
   * largest number of hops from the root to any process.
   */
  template <template <class> class Bcast>
  std::map<int, int> simulate(int root, const std::set<int> &procset, bool &sent_without_peers, int &max_hops) {
    using Iter = std::set<int>::const_iterator;
    std::map<int, int> received;
    std::deque<std::pair<int, int>> active{{root, 0}};  // process and the number of hops it took to reach it
    sent_without_peers = false;
    max_hops = 0;
    while (!active.empty()) {
      auto [me, hops] = active.front();
      active.pop_front();
      max_hops = std::max(max_hops, hops);
      Bcast<Iter> bcast(root, me, procset.begin(), procset.end());
      bool sent = false;
      bcast([&](int p) {
        sent = true;
        /* only forward on first receipt so that a cycle cannot loop forever */
        if (p != me && ++received[p] == 1) active.push_back({p, hops + 1});
      });
      /* has_peers() decides whether a message is prepared, it must match whether one is sent */
      if (sent != bcast.has_peers()) sent_without_peers = true;
    }
    return received;
  }

  /* floor(log2(n)) */
  int ilog2(int n) {
    int l = 0;
    while (n > 1) {
      n /= 2;
      ++l;
    }
    return l;
  }

  /* max_hops returns the largest number of hops allowed for a given number of participants, including the root */
  template <template <class> class Bcast, typename MaxHops>
  void check_all_procsets(int max_procs, MaxHops &&max_hops_allowed) {
    for (int nprocs = 1; nprocs <= max_procs; ++nprocs) {
      for (int root = 0; root < nprocs; ++root) {
        for (int mask = 0; mask < (1 << nprocs); ++mask) {
          std::set<int> procset;
          for (int p = 0; p < nprocs; ++p) {
            if (mask & (1 << p)) procset.insert(p);
          }
          CAPTURE(nprocs, root, mask);
          bool sent_without_peers;
          int max_hops;
          auto received = simulate<Bcast>(root, procset, sent_without_peers, max_hops);
          CHECK(!sent_without_peers);
          const int num_participants = procset.size() + (procset.contains(root) ? 0 : 1);
          CHECK(max_hops <= max_hops_allowed(num_participants));
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
  SECTION("star") {
    check_all_procsets<ttg_parsec::BroadcastStar>(7, [](int n) { return 1; });
  }
  SECTION("pipe") {
    check_all_procsets<ttg_parsec::BroadcastPipe>(7, [](int n) { return n - 1; });
  }
  SECTION("binary tree") {
    check_all_procsets<ttg_parsec::BroadcastBinaryTree>(7, [](int n) { return ilog2(n); });
  }
}

#endif  // TTG_USE_PARSEC
