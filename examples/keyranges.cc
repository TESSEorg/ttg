#include <vector>
#include <chrono>
#include <unordered_map>
#include "ttg.h"

/* 1M keys */
#define NUM_KEYS (1024*1024)
#define NUM_POP_REPS 100
#define NUM_ITER_REPS 100
#define NUM_SORT_REPS 100

/* Key taken from FW, 3x int */
struct Key {
  int I = 0, J = 0, K = 0;

  bool operator==(const Key& b) const {
    if (this == &b) return true;
    return I == b.I &&
           J == b.J &&
           K == b.K;
  }

  Key& operator+=(const Key& b) {
    I += b.I;
    J += b.J;
    K += b.K;
    //rehash();
    return *this;
  }

  Key operator-(const Key& b) {
    Key key;
    I += b.I;
    J += b.J;
    K += b.K;
    //rehash();
    return *this;
  }

  bool operator!=(const Key& b) const { return !((*this) == b); }

  mutable size_t hash_val;

  Key() { rehash(); }
  //Key(const std::pair<std::pair<int, int>, int>& e) : execution_info(e) { rehash(); }
  Key(int e_f_f, int e_f_s, int e_s) : I(e_f_f), J(e_f_s), K(e_s) { rehash(); }

  size_t hash() const { rehash(); return hash_val; }
private:
  void rehash() const {
    std::hash<int> int_hasher;
    hash_val = int_hasher(I) * 2654435769 + int_hasher(J) * 40503 +
               int_hasher(K);
  }

public:
#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
  template <typename Archive>
  void serialize(Archive& ar) {
    ar& madness::archive::wrap((unsigned char*)this, sizeof(*this));
  }
#endif

#ifdef TTG_SERIALIZATION_SUPPORTS_BOOST
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& I;
    ar& J;
    ar& K;
    if constexpr (ttg::detail::is_boost_input_archive_v<Archive>) rehash();
  }
#endif

  friend std::ostream& operator<<(std::ostream& out, Key const& k) {
    out << "Key(" << k.I << "," << k.J << ","
        << k.K << ")";
    return out;
  }
};


Key operator-(const Key& a, const Key& b) {
  Key key = a;
  key.I -= b.I;
  key.J -= b.J;
  key.K -= b.K;
  //rehash();
  return key;
}

auto populate_vector() {
  /* vector with enough elements reserved */
  std::vector<Key> vec;
  vec.reserve(NUM_KEYS);
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    vec.emplace_back(i, i, i);
  }
  return vec;
}

void sort_vector(std::vector<Key>& vec) {
  std::sort(vec.begin(), vec.end(),
            [](const Key& a, const Key& b){ return a.I < b.I; });
}

template<typename Range, typename Func>
uint64_t iterate(Range&& vec, Func&& fn, const char* msg) {
  std::chrono::high_resolution_clock::time_point t1, t2;
  uint64_t duration;
  uint64_t res = 0;
  std::cout << msg;
  t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_ITER_REPS; ++i) {
    for (const auto& v : vec) {
      res += fn(v);
    }
  }
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << duration/NUM_ITER_REPS << "us" << std::endl;

  /* some side effect */
  if (0 == res) {
    std::cout << "Ouch!" << std::endl;
  }
  return res;
}

struct stride_info {
  Key last_point;
  Key stride;
  bool has_stride = false;

  stride_info(Key last_point) : last_point(last_point)
  { }
};

template<typename Range, typename Func>
auto find_stride_map(Range&& range, Func&& fn, const char *msg) {
  std::chrono::high_resolution_clock::time_point t1, t2;
  uint64_t duration;
  std::cout << msg;
  t1 = std::chrono::high_resolution_clock::now();
  std::unordered_map<int, stride_info> info;
  for (int i = 0; i < NUM_ITER_REPS; ++i) {
    info.clear();
    for (const auto& v : range) {
      int proc = fn(v);
      auto iter = info.find(proc);
      if (iter != info.end()) {
        auto stride = v - iter->second.last_point;
        if (!iter->second.has_stride) {
          iter->second.stride = stride;
          iter->second.has_stride = true;
        } else {
          /* check if stride is correct */
          if (iter->second.stride != stride) {
            std::cout << "STRIDE MISMATCH: " << iter->second.stride << " vs " << stride << " for v " << v << std::endl;
            return std::unordered_map<int, stride_info>();
          }
        }
        iter->second.last_point = v;
      } else {
        /* insert a new element */
        info.insert(std::make_pair(proc, stride_info(v)));
      }
    }
  }
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << duration/NUM_ITER_REPS << "us" << std::endl;
  return info;
}


template<typename Range, typename Func>
auto find_stride_vec(Range&& range, Func&& keymap, const char *msg) {
  std::chrono::high_resolution_clock::time_point t1, t2;
  uint64_t duration;
  Key stride;
  std::cout << msg;
  t1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < NUM_ITER_REPS; ++k) {
    std::vector<int> peers;
    peers.push_back(keymap(*range.begin())); // mark the first element
    int i = 0;
    stride = *range.begin();
    bool have_stride = false;
    for (const auto& v : range) {
      int proc = keymap(v);
      if (peers[0] == proc) {
        // we found the beginning of the cycle
        i = 0;
        if (!have_stride && i > 0) {
          stride = v - stride;
          have_stride = true;
        }
      } else if (i > peers.size()-1) {
        // we found a new peer
        peers.push_back(proc);
      } else if (peers[i] != proc) {
        // STRIDE ERROR?
        std::cout << "STRIDE_ERROR at " << i << ": expected proc " << peers[i] << " but found " << proc << " for key " << v << std::endl;
        return false;
      } else {
        // All is good, we found the peer we expect
      }
      ++i;
    }
  }
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << duration/NUM_ITER_REPS << "us" << std::endl;

  std::cout << "Detected stride: " << stride << std::endl;

  return true;
}

int main(int argc, char **argv)
{
  std::chrono::high_resolution_clock::time_point t1, t2;
  uint64_t duration;
  uint64_t res;
  ttg::initialize(argc, argv, 1);

  auto fn = [](const Key& key){ return key.I; };
  auto stdfn = std::function(fn);
  auto range = ttg::make_keyrange(Key(0, 0, 0),
                                  Key(NUM_KEYS, NUM_KEYS, NUM_KEYS),
                                  Key(1, 1, 1));

  std::cout << "Using vector/range with " << NUM_KEYS << " keys" << std::endl;

  std::cout << "Populating vector... ";
  t1 = std::chrono::high_resolution_clock::now();
  std::vector<Key> vec;
  for (int i = 0; i < NUM_POP_REPS; ++i) {
    vec = populate_vector();
  }
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << duration/NUM_POP_REPS << "us" << std::endl;

  std::cout << "Sorting vector... ";
  t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_SORT_REPS; ++i) {
    sort_vector(vec);
  }
  t2 = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << duration/NUM_POP_REPS << "us" << std::endl;

  iterate(vec, fn, "Iterating vector (inlined)... ");

  iterate(vec, stdfn, "Iterating vector (not inlined)... ");

  iterate(range, fn, "Iterating keyrange (inlined)... ");

  iterate(range, stdfn, "Iterating keyrange (not inlined)... ");

  /* find the stride, assuming we have 1024 processes */
  find_stride_map(range, [](const Key& key){ return key.I % 1024; }, "Finding stride with map... ");

  find_stride_vec(range, [](const Key& key){ return key.I % 1024; }, "Finding stride with vector... ");

  ttg::finalize();

  return 0;
}
