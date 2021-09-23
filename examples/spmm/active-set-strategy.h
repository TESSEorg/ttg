//
// Created by herault on 9/22/21.
//

#ifndef TTG_ACTIVE_SET_STRATEGY_H
#define TTG_ACTIVE_SET_STRATEGY_H

#include <set>
#include <tuple>

class ActiveSetStrategy {
  using bcastset_t = std::set<std::tuple<long, long>>;

 public:
  ActiveSetStrategy() = delete;
  explicit ActiveSetStrategy(const std::vector<std::vector<long>> &a_rowidx_to_colidx,
                             const std::vector<std::vector<long>> &a_colidx_to_rowidx,
                             const std::vector<std::vector<long>> &b_rowidx_to_colidx,
                             const std::vector<std::vector<long>> &b_colidx_to_rowidx, const std::vector<long> &mTiles,
                             const std::vector<long> &nTiles, const std::vector<long> &kTiles, size_t memory)
      : a_rowidx_to_colidx_(a_rowidx_to_colidx)
      , a_colidx_to_rowidx_(a_colidx_to_rowidx)
      , b_rowidx_to_colidx_(b_rowidx_to_colidx)
      , b_colidx_to_rowidx_(b_colidx_to_rowidx)
      , mTiles_(mTiles)
      , nTiles_(nTiles)
      , kTiles_(kTiles)
      , memory_(memory)
      , mt(0)
      , nt(0)
      , kt(0) {
    mt = a_rowidx_to_colidx_.size();
    nt = b_colidx_to_rowidx_.size();
    kt = a_colidx_to_rowidx_.size();
    long tmp = b_rowidx_to_colidx_.size();
    if (tmp > kt) kt = tmp;
  }

  size_t memory_used(long m, long n, long k, bcastset_t &a, bcastset_t &b, bcastset_t &c) const {
    size_t res = 0;
    if (a.find({m, k}) != a.end()) res += mTiles_[m] * kTiles_[k] * 8;
    if (b.find({k, n}) != b.end()) res += kTiles_[k] * nTiles_[n] * 8;
    if (c.find({m, n}) != c.end()) res += mTiles_[m] * nTiles_[n] * 8;
    return res;
  }

  std::pair<long, size_t> best_cube_dim_and_size(long m0, long n0, long k0) const {
    bcastset_t a, b, c;
    long res = 1;
    size_t used = 0, prev_used;

    assert(m0 < mt);
    assert(n0 < nt);
    assert(k0 < kt);

    if (!a_rowidx_to_colidx_[m0].empty() && !b_colidx_to_rowidx_[n0].empty()) {
      if (std::find(a_rowidx_to_colidx_[m0].begin(), a_rowidx_to_colidx_[m0].end(), k0) !=
              a_rowidx_to_colidx_[m0].end() &&
          std::find(b_colidx_to_rowidx_[n0].begin(), b_colidx_to_rowidx_[n0].end(), k0) !=
              b_colidx_to_rowidx_[n0].end()) {
        used += memory_used(m0, n0, k0, a, b, c);
        a.insert({m0, k0});
        b.insert({k0, n0});
        c.insert({m0, n0});
      }
    }

    prev_used = used;

    int walls = 0;
    long mbound, nbound, kbound;
    while (walls < 3 && used < memory_) {
      walls = 0;
      mbound = m0 + res;
      if (mbound > mt) {
        mbound = mt;
        walls++;
      }
      nbound = n0 + res;
      if (nbound > nt) {
        nbound = nt;
        walls++;
      }
      kbound = k0 + res;
      if (kbound > kt) {
        kbound = kt;
        walls++;
      }

      prev_used = used;

      // Try to extend the cube in 'm'
      if (m0 + res < mt) {
        if (!a_rowidx_to_colidx_[m0 + res].empty()) {
          for (long n = n0; n < nbound; n++) {
            if (b_colidx_to_rowidx_[n].empty()) continue;
            auto ait = a_rowidx_to_colidx_[m0 + res].begin();
            auto bit = b_colidx_to_rowidx_[n].begin();
            while (ait != a_rowidx_to_colidx_[m0 + res].end() && *ait <= k0 + res &&
                   bit != b_colidx_to_rowidx_[n].end() && *bit <= k0 + res) {
              if (*ait == *bit) {
                if (*ait >= k0) {
                  used += memory_used(m0 + res, n, *ait, a, b, c);
                  a.insert({m0 + res, *ait});
                  b.insert({*ait, n});
                  c.insert({m0 + res, n});
                }
                ait++;
                bit++;
              } else if (*ait < *bit) {
                ait++;
              } else {
                bit++;
              }
            }
          }
        }
      }

      // Try to extend the cube in 'n'
      if (n0 + res < nt) {
        if (!b_colidx_to_rowidx_[n0 + res].empty()) {
          for (long m = m0; m < mbound; m++) {
            if (a_rowidx_to_colidx_[m].empty()) continue;
            auto bit = b_colidx_to_rowidx_[n0 + res].begin();
            auto ait = a_rowidx_to_colidx_[m].begin();
            while (ait != a_rowidx_to_colidx_[m].end() && *ait <= k0 + res &&
                   bit != b_colidx_to_rowidx_[n0 + res].end() && *bit <= k0 + res) {
              if (*ait == *bit) {
                if (*ait >= k0) {
                  used += memory_used(m, n0 + res, *ait, a, b, c);
                  a.insert({m, *ait});
                  b.insert({*ait, n0 + res});
                  c.insert({m0 + res, n0 + res});
                }
                ait++;
                bit++;
              } else if (*ait < *bit) {
                ait++;
              } else {
                bit++;
              }
            }
          }
        }
      }

      // And try to extend in 'k'
      if (k0 + res < kt) {
        for (long m = m0; m < mbound; m++) {
          for (long n = n0; n < nbound; n++) {
            if (a_rowidx_to_colidx_[m].empty()) continue;
            if (b_colidx_to_rowidx_[n].empty()) continue;
            auto ait = a_rowidx_to_colidx_[m].begin();
            auto bit = b_colidx_to_rowidx_[n].begin();
            while (ait != a_rowidx_to_colidx_[m].end() && *ait <= k0 + res &&
                   bit != b_colidx_to_rowidx_[n0 + res].end() && *bit <= k0 + res) {
              if (*ait == *bit) {
                if (*ait == k0 + res) {
                  used += memory_used(m, n, *ait, a, b, c);
                  a.insert({m, *ait});
                  b.insert({*ait, n});
                  c.insert({m, n});
                }
                ait++;
                bit++;
              } else if (*ait < *bit) {
                ait++;
              } else {
                bit++;
              }
            }
          }
        }
      }

      if (used <= memory_) res++;
    }

    if (res == 1) return std::make_pair(1, used);
    return std::make_pair(res - 1, prev_used);
  }

  long best_cube_dim(long m0, long n0, long k0) const {
    std::pair<long, size_t> res = best_cube_dim_and_size(m0, n0, k0);
    return std::get<0>(res);
  }

  long mt;
  long nt;
  long kt;

 private:
  const std::vector<std::vector<long>> &a_rowidx_to_colidx_;
  const std::vector<std::vector<long>> &a_colidx_to_rowidx_;
  const std::vector<std::vector<long>> &b_rowidx_to_colidx_;
  const std::vector<std::vector<long>> &b_colidx_to_rowidx_;
  const std::vector<long> &mTiles_;
  const std::vector<long> &nTiles_;
  const std::vector<long> &kTiles_;
  size_t memory_;
};

#endif  // TTG_ACTIVE_SET_STRATEGY_H
