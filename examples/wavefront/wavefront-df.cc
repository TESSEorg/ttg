#include <algorithm>  // for std::max
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include "../blockmatrix.h"
#include "ttg.h"

#include "ttg/serialization.h"
#include "ttg/serialization/std/pair.h"
#include "ttg/serialization/std/vector.h"

/* TODO: Get rid of using statement */
using namespace ttg;
#include <madness/world/world.h>

/*!
        \file wavefront_df.impl.h
        \brief Wavefront computation on distributed memory
        \defgroup
        ingroup examples

        \par Points of interest
        - dynamic recursive DAG.
*/

using Key = std::pair<int, int>;
namespace std {
  std::ostream& operator<<(std::ostream& out, Key const& k) {
    out << "Key(" << k.first << ", " << k.second << ")";
    return out;
  }
}  // namespace std

// An empty class used for pure control flows
struct Control {};

template <typename T>
inline BlockMatrix<T> stencil_computation(int i, int j, int M, int N,
                                          BlockMatrix<T> bm,
                                          BlockMatrix<T> left,
                                          BlockMatrix<T> top,
                                          BlockMatrix<T> right,
                                          BlockMatrix<T> bottom) {
  // i==0 -> no top block
  // j==0 -> no left block
  // i==M-1 -> no bottom block
  // j==N-1 -> no right block
  int MB = bm.rows();
  int NB = bm.cols();
  BlockMatrix<T> current = bm;
  for (int ii = 0; ii < MB; ++ii) {
    for (int jj = 0; jj < NB; ++jj) {
      current(ii, jj) = (current(ii, jj) + ((ii == 0) ? (i > 0 ? top(MB - 1, jj) : 0.0) : current(ii - 1, jj)));
      current(ii, jj) = (current(ii, jj) + ((ii == MB - 1) ? (i < M - 1 ? bottom(0, jj) : 0.0) : current(ii + 1, jj)));
      current(ii, jj) = (current(ii, jj) + ((jj == 0) ? (j > 0 ? left(ii, NB - 1) : 0.0) : current(ii, jj - 1)));
      current(ii, jj) = (current(ii, jj) + ((jj == NB - 1) ? (j < N - 1 ? right(ii, 0) : 0.0) : current(ii, jj + 1)));
      current(ii, jj) = current(ii, jj) * 0.25;
    }
  }
  return current;
}

// serial implementation of wavefront computation.
template <typename T>
void wavefront_serial(Matrix<T>* m, Matrix<T>* result, int n_brows, int n_bcols) {
  for (int i = 0; i < n_brows; i++) {
    for (int j = 0; j < n_bcols; j++) {
      BlockMatrix<T> left, top, right, bottom;
      if (i < n_brows - 1) bottom = ((*m)(i + 1, j));
      if (j < n_bcols - 1) right = ((*m)(i, j + 1));
      if (j > 0) left = ((*m)(i, j - 1));
      if (i > 0) top = ((*m)(i - 1, j));

      (*result)(i, j) = stencil_computation(i, j, n_brows, n_bcols, ((*m)(i, j)), (left), (top), (right), (bottom));
    }
  }
}

namespace madness {
  namespace archive {

    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, std::tuple<std::pair<T, T>, std::pair<T, T>>> {
      static inline void store(const Archive& ar, const std::tuple<std::pair<T, T>, std::pair<T, T>>& p) {
        std::cout << "Storing...\n";
        ar & std::get<0>(p).first & std::get<0>(p).second;
        ar & std::get<1>(p).first & std::get<1>(p).second;
        std::cout << "Stored...\n";
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, std::tuple<std::pair<T, T>, std::pair<T, T>>> {
      static inline void load(const Archive& ar, std::tuple<std::pair<T, T>, std::pair<T, T>>& p) {
        std::cout << "Loading...\n";
        T i, j;
        std::pair<T, T> p1, p2;
        ar & i & j;
        p1 = std::make_pair(i, j);
        ar & i & j;
        p2 = std::make_pair(i, j);
        p = std::make_tuple(p1, p2);
        std::cout << "Loaded...\n";
      }
    };
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, std::unordered_map<std::pair<int, int>, BlockMatrix<T>>> {
      static inline void store(const Archive& ar,
                               const std::unordered_map<std::pair<int, int>, BlockMatrix<T>>& m) {
        std::cout << "Storing map...\n";
        ar << m.size();
        for (auto const& p: m) {
          ar << m.first << m.second;
        }
        std::cout << "Stored map...\n";
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, std::unordered_map<std::pair<int, int>, BlockMatrix<T>>> {
      static inline void load(const Archive& ar,
                              std::unordered_map<std::pair<int, int>, BlockMatrix<T>>& m) {
        std::cout << "Loading map...\n";
        int size;
        ar >> size;
        for (size_t i = 0; i != size; ++i) {
          std::pair<int, int> key;
          BlockMatrix<T> value;
          ar >> key >> value;
          m[key] = value;
        }
        ar >> wrap(m, m.size());
        std::cout << "Loaded map...\n";
      }
    };
  }
}

template <typename T>
auto initiator(int MB, int NB,
               std::unordered_map<Key, BlockMatrix<T>, pair_hash>& m,
               Edge<Key, BlockMatrix<T>>& out0,
               Edge<Key, BlockMatrix<T>>& out1BR,
               Edge<Key, BlockMatrix<T>>& out2BR,
               Edge<Key, BlockMatrix<T>>& out1R,
               Edge<Key, BlockMatrix<T>>& out2R,
               Edge<Key, BlockMatrix<T>>& out1B,
               Edge<Key, BlockMatrix<T>>& out2B,
               Edge<Key, BlockMatrix<T>>& out2L) {
  auto f = [&m, MB, NB](const Key& key, std::tuple<Out<Key, BlockMatrix<T>>,
                Out<Key, BlockMatrix<T>>,
                Out<Key, BlockMatrix<T>>,
                Out<Key, BlockMatrix<T>>,
                Out<Key, BlockMatrix<T>>,
                Out<Key, BlockMatrix<T>>,
                Out<Key, BlockMatrix<T>>,
                Out<Key, BlockMatrix<T>>>& out)
  {
    if (ttg_default_execution_context().size() == 1) {
      for (int i = 0; i < MB; i++) {
        for (int j = 0; j < NB; j++) {
          if (i == 0 && j == 0) {
            std::cout << "send 0 : " << i << " " << j << std::endl;
            send<0>(Key(i,j), m[std::make_pair(i,j)], out);
          }
          else if ((i == 0 && j > 0 && j < NB-1) ||
                   (i > 0 && i < MB-1 && j == 0)) {
            std::cout << "send 1 : " << i << " " << j << std::endl;
            send<1>(Key(i,j), m[std::make_pair(i,j)], out);
          }
          else if (i > 0 && i < MB-1 && j > 0 && j < NB-1) {
            std::cout << "send 2 : " << i << " " << j << std::endl;
            send<2>(Key(i,j), m[std::make_pair(i,j)], out);
          }
          else if (i == MB-1 && j == 0) {
            std::cout << "send 3 : " << i << " " << j << std::endl;
            send<3>(Key(i,j), m[std::make_pair(i,j)], out);
          }
          else if (j == MB-1 && i == 0) {
            std::cout << "send 5 : " << i << " " << j << std::endl;
            send<5>(Key(i,j), m[std::make_pair(i,j)], out);
          }
          else if (i == MB-1 && j > 0 && j < NB-1) {
            std::cout << "send 4 : " << i << " " << j << std::endl;
            send<4>(Key(i,j), m[std::make_pair(i,j)], out);
          }
          else if (j == NB-1 && i > 0 && i < MB-1) {
            std::cout << "send 6 : " << i << " " << j << std::endl;
            send<6>(Key(i,j), m[std::make_pair(i,j)], out);
          }
          else if (i == MB-1 && j == NB-1) {
            std::cout << "send 7 : " << i << " " << j << std::endl;
            send<7>(Key(i,j), m[std::make_pair(i,j)], out);
          }
        }
      }
    }
    else {
      int i = ttg_default_execution_context().rank();
      for (int j = 0; j < m.size(); j++) {
        if (i == 0 && j == 0) {
          std::cout << "send 0 : " << i << " " << j << std::endl;
          send<0>(Key(i,j), m[std::make_pair(i,j)], out);
        }
        else if ((i == 0 && j > 0 && j < NB-1) ||
                 (i > 0 && i < MB-1 && j == 0)) {
          std::cout << "send 1 : " << i << " " << j << std::endl;
          send<1>(Key(i,j), m[std::make_pair(i,j)], out);
        }
        else if (i > 0 && i < MB-1 && j > 0 && j < NB-1) {
          std::cout << "send 2 : " << i << " " << j << std::endl;
          send<2>(Key(i,j), m[std::make_pair(i,j)], out);
        }
        else if (i == MB-1 && j == 0) {
          std::cout << "send 3 : " << i << " " << j << std::endl;
          send<3>(Key(i,j), m[std::make_pair(i,j)], out);
        }
        else if (j == MB-1 && i == 0) {
          std::cout << "send 5 : " << i << " " << j << std::endl;
          send<5>(Key(i,j), m[std::make_pair(i,j)], out);
        }
        else if (i == MB-1 && j > 0 && j < NB-1) {
          std::cout << "send 4 : " << i << " " << j << std::endl;
          send<4>(Key(i,j), m[std::make_pair(i,j)], out);
        }
        else if (j == NB-1 && i > 0 && i < MB-1) {
          std::cout << "send 6 : " << i << " " << j << std::endl;
          send<6>(Key(i,j), m[std::make_pair(i,j)], out);
        }
        else if (i == MB-1 && j == NB-1) {
          std::cout << "send 7 : " << i << " " << j << std::endl;
          send<7>(Key(i,j), m[std::make_pair(i,j)], out);
        }
      }
    }
  };

  return wrap<Key>(f, edges(), edges(out0, out1BR, out2BR, out1R, out2R, out1B,
                                     out2B, out2L),
                   "initiator", {}, {"out0", "out1BR", "out2BR", "out1R",
                    "out2R", "out1B", "out2B", "out2L"});
}

std::vector<Key>  get_bottomindex(const Key& key, int MB, int NB) {
  //std::tuple<std::pair<int, int>> t;
  auto [i,j] = key;
  std::cout << "called get_bottomindex with " << i << " " << j << std::endl;
  std::vector<std::pair<int, int>> v;
    if (i == 0 && j == 0) {
      v.push_back(std::make_pair(i+1,j));
      return v;
    }
    if ((i == 0 && j > 0) || (i > 0 && j == 0)) {
      if (i < MB - 1) {
        v.push_back(std::make_pair(i+1,j));
        return v;
      }
    }
    if (i > 0 && j > 0) {
      if (i < MB - 1) {
        v.push_back(std::make_pair(i+1,j));
        return v;
      }
    }
    return v;
}

std::vector<Key>  get_rightindex(const Key& key, int MB, int NB) {
  //std::tuple<std::pair<int, int>> t;
  auto [i,j] = key;
  std::cout << "called get_rightindex with " << i << " " << j << std::endl;
  std::vector<std::pair<int, int>> v;
    if (i == 0 && j == 0) {
      v.push_back(std::make_pair(i,j+1));
      return v;
    }
    if ((i == 0 && j > 0) || (i > 0 && j == 0)) {
      if (j < NB - 1) {
        v.push_back(std::make_pair(i,j+1));
        return v;
      }
    }
    if (i > 0 && j > 0) {
      if (j < NB - 1) {
        v.push_back(std::make_pair(i,j+1));
        return v;
      }
    }
    return v;
}

template <typename T>
auto make_get_bottomblock(std::unordered_map<Key, BlockMatrix<T>, pair_hash>& m,
                  Edge<Key, BlockMatrix<T>>& bottom) {
  auto f = [m](const Key& key) -> BlockMatrix<T> {
    //auto [key, o] = keys;
    auto [i,j] = key;
    return m.at(Key(i,j));
  };

  return wrap<Key>(f, edges(bottom), "get_bottomblock", {"get_bottomblock"});
}

template <typename T>
auto make_get_rightblock(std::unordered_map<Key, BlockMatrix<T>, pair_hash>& m,
                  Edge<Key, BlockMatrix<T>>& right) {
  auto f = [m](const Key& key) -> BlockMatrix<T> {
    //auto [key, o] = keys;
    auto [i,j] = key;
    std::cout << "get right block: " << i << " " << j << " for " << i << ":" << j << std::endl;

    return m.at(Key(i,j));
  };

  return wrap<Key>(f, edges(right), "get_rightblock", {"get_rightblock"});
}

template <typename funcT, typename T>
auto make_wavefront0(const funcT& func, int MB, int NB,
                     Edge<Key, BlockMatrix<T>>& input,
                     Edge<Key, BlockMatrix<T>>& toporleft,
                     Edge<Key, BlockMatrix<T>>& toporleftLR,
                     Edge<Key, BlockMatrix<T>>& toporleftLB,
                     Edge<Key, BlockMatrix<T>>& bottom0,
                     Edge<Key, BlockMatrix<T>>& right0,
                     Edge<Key, BlockMatrix<T>>& result) {
  auto f = [func, MB, NB](const Key& key, BlockMatrix<T>& input,
                          BlockMatrix<T>& bottom0, BlockMatrix<T>& right0,
                          std::tuple<Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i,j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    std::cout << "wf0 " << i << " " << j << std::endl;
    BlockMatrix<T> res = func(i,j,MB,NB,input,input,input,right0,bottom0);

    if(next_j == NB - 1)
      send<2>(Key(i,next_j), res, out);
    else
      send<0>(Key(i, next_j), res, out);
    if(next_i == MB - 1)
      send<1>(Key(next_i,j), res, out);
    else
      send<0>(Key(next_i,j), res, out);

    send<3>(Key(i,j), res, out);
  };

  return wrap(f, edges(input, bottom0, right0), edges(toporleft, toporleftLR,
                                                      toporleftLB, result),
              "wavefront0", {"block_input", "bottom0", "right0"},
              {"toporleft", "toporleftLR", "toporleftLB", "result"});

}

// Method to generate wavefront task with single input.
template <typename funcT, typename T>
auto make_wavefront1(const funcT& func, int MB, int NB,
                     Edge<Key, BlockMatrix<T>>& input,
                     Edge<Key, BlockMatrix<T>>& toporleft,
                     Edge<Key, BlockMatrix<T>>& bottom0,
                     Edge<Key, BlockMatrix<T>>& right0,
                     Edge<Key, BlockMatrix<T>>& output1,
                     Edge<Key, BlockMatrix<T>>& output2,
                     Edge<Key, BlockMatrix<T>>& output1R,
                     Edge<Key, BlockMatrix<T>>& output1B,
                     Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input,
                          BlockMatrix<T>&& previous,
                          BlockMatrix<T>&& bottom0,
                          BlockMatrix<T>&& right0,
                          std::tuple<Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    std::cout << "wf1 " << i << " " << j << std::endl;
    BlockMatrix<T> res;
    res = func(i, j, MB, NB, input, previous, previous, right0, bottom0);

    //func(i, j, MB, NB, input, previous, previous, bottom_right[0], bottom_right[1]);
    //Processing finished for this block, so send it to output
    send<5>(Key(i,j), res, out);

    if (next_i < MB) {
      if (j == 0 && next_i < MB - 1)  {
        // Single predecessor, no left block
        send<0>(Key(next_i, j), res, out); //send top block
      }
      else if (j == 0 && next_i == MB - 1) {
        send<3>(Key(next_i, j), res, out);
      }
      else {
        // Two predecessors
        send<2>(Key(next_i, j), res, out); //send top block
      }
    }
    if (next_j < NB) {
      if (i == 0 && next_j < NB - 1) {
        // Single predecessor, no top block
        send<0>(Key(i, next_j), res, out); //send left block
      }
      else if (i == 0 && next_j == NB - 1) {
        send<4>(Key(i, next_j), res, out);
      }
      else {
        // Two predecessors
        send<1>(Key(i, next_j), res, out);  // send left block
      }
    }
  };

  return wrap(f, edges(input, toporleft, bottom0, right0),
              edges(toporleft, output1, output2, output1R, output1B, result),
              "wavefront1",
              {"block_input", "toporleft", "bottom0", "right0"},
              {"recur", "output1", "output2", "output1R", "output1B",
               "result"});
}

template <typename funcT, typename T>
auto make_wavefront1R(const funcT& func, int MB, int NB,
                      Edge<Key, BlockMatrix<T>>& input,
                      Edge<Key, BlockMatrix<T>>& toporleft,
                      Edge<Key, BlockMatrix<T>>& output1R,
                      Edge<Key, BlockMatrix<T>>& output12R,
                      Edge<Key, BlockMatrix<T>>& outputLR,
                      Edge<Key, BlockMatrix<T>>& right0,
                      Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input,
                          BlockMatrix<T>&& top,
                          BlockMatrix<T>&& right0,
                          std::tuple<Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_j = j + 1;
    if (i != MB - 1)
      std::cout << "Error in 1R, should only be triggered for last row\n";

    std::cout << "wf1R " << i << " " <<  std::endl;
    BlockMatrix<T> res;
    res = func(i, j, MB, NB, input, top, top, right0, right0);

    //Processing finished for this block, so send it to output
    send<2>(Key(i,j), res, out);

    if (next_j == NB - 1 && i == MB - 1)
      send<1>(Key(i, next_j), res, out); //send left block
    else
      // Single predecessor, no top block
      send<0>(Key(i, next_j), res, out); //send left block
  };

  return wrap(f, edges(input, fuse(toporleft, output1R), right0),
              edges(output12R, outputLR, result),
              "wavefront1R",
              {"block_input", "output1R", "right0"},
              {"output12R", "outputLR", "result"});

}

template <typename funcT, typename T>
auto make_wavefront1B(const funcT& func, int MB, int NB,
                      Edge<Key, BlockMatrix<T>>& input,
                      Edge<Key, BlockMatrix<T>>& toporleft,
                      Edge<Key, BlockMatrix<T>>& output1B,
                      Edge<Key, BlockMatrix<T>>& output12B,
                      Edge<Key, BlockMatrix<T>>& outputLB,
                      Edge<Key, BlockMatrix<T>>& bottom0,
                      Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input,
                          BlockMatrix<T>&& left,
                          BlockMatrix<T>&& bottom0,
                          std::tuple<Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    if (j != NB - 1)
      std::cout << "Error in 1B, should only be triggered for last column\n";

    std::cout << "wf1B " << i << " " << j << std::endl;
    BlockMatrix<T> res;
    res = func(i, j, MB, NB, input, left, left, bottom0, bottom0);

    //Processing finished for this block, so send it to output
    send<2>(Key(i,j), res, out);

    if (next_i == MB - 1 && j == NB - 1)
      send<1>(Key(next_i, j), res, out); //send left block to last block
    else
      // Single predecessor, no top block
      send<0>(Key(next_i, j), res, out); //send left block to next row
  };

  return wrap(f, edges(input, fuse(toporleft, output1B), bottom0),
              edges(output12B, outputLB, result),
              "wavefront1B",
              {"block_input", "output1B", "bottom0"},
              {"output12B", "outputLB", "result"});

}

// Method to generate  wavefront tasks with two inputs.
template <typename funcT, typename T>
auto make_wavefront2BR(const funcT& func, int MB, int NB,
                       Edge<Key, BlockMatrix<T>>& input,
                       Edge<Key, BlockMatrix<T>>& left,
                       Edge<Key, BlockMatrix<T>>& top,
                       Edge<Key, BlockMatrix<T>>& bottom0,
                       Edge<Key, BlockMatrix<T>>& right0,
                       Edge<Key, BlockMatrix<T>>& output2R,
                       Edge<Key, BlockMatrix<T>>& output2B,
                       Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input,
                          BlockMatrix<T>&& left, BlockMatrix<T>&& top,
                          BlockMatrix<T>&& bottom0, BlockMatrix<T>&& right0,
                          std::tuple<Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    std::cout << "wf2BR " << i << " " << j <<  std::endl;
    BlockMatrix<T> res;
    //    if (i == MB - 1 && j == NB - 1)
    //  res = func(i, j, MB, NB, input, left, top, input, input);
    res = func(i, j, MB, NB, input, left, top, right0, bottom0);

    //Processing finished for this block, so send it to output Op
    send<4>(Key(i,j), res, out);

    if (next_i < MB - 1) {
      send<1>(Key(next_i, j), res, out);
    }
    else
      send<2>(Key(next_i, j), res, out);

    if (next_j < NB - 1) {
      send<0>(Key(i, next_j), res, out);
    }
    else
      send<3>(Key(i, next_j), res, out);
  };

  return wrap(f, edges(input, left, top, bottom0, right0),
              edges(left, top, output2R, output2B, result), "wavefront2BR",
              {"block_input", "left", "top", "bottom0", "right0"},
              {"left", "top", "output2R", "output2B", "result"});
}

template <typename funcT, typename T>
auto make_wavefront2R(const funcT& func, int MB, int NB,
                      Edge<Key, BlockMatrix<T>>& input,
                      Edge<Key, BlockMatrix<T>>& output12R,
                      Edge<Key, BlockMatrix<T>>& output2R,
                      Edge<Key, BlockMatrix<T>>& right0,
                      Edge<Key, BlockMatrix<T>>& output2LL,
                      Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input,
                          BlockMatrix<T>&& left,
                          BlockMatrix<T>&& top,
                          BlockMatrix<T>&& right0,
                          std::tuple<Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_j = j + 1;
    if (i != MB - 1)
      std::cout << "Error in 2R, should only be triggered for last row\n";

    std::cout << "wf2R " << i << " " << j <<  std::endl;
    BlockMatrix<T> res;
    res = func(i, j, MB, NB, input, top, left, right0, right0);

    //Processing finished for this block, so send it to output
    send<2>(Key(i,j), res, out);

    if (next_j == NB - 1)
      // Single predecessor, no top block
      send<1>(Key(i, next_j), res, out); //send top block
    else
      send<0>(Key(i, next_j), res, out);
  };

  return wrap(f, edges(input, output12R, output2R, right0),
              edges(output12R, output2LL, result),
              "wavefront2R",
              {"block_input", "output12R", "output2R", "right0"},
              {"recur", "output2LL", "result"});

}

template <typename funcT, typename T>
auto make_wavefront2B(const funcT& func, int MB, int NB,
                      Edge<Key, BlockMatrix<T>>& input2B,
                      Edge<Key, BlockMatrix<T>>& output12B,
                      Edge<Key, BlockMatrix<T>>& output2B,
                      Edge<Key, BlockMatrix<T>>& bottom0,
                      Edge<Key, BlockMatrix<T>>& output2TL,
                      Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input,
                          BlockMatrix<T>&& left,
                          BlockMatrix<T>&& top,
                          BlockMatrix<T>&& bottom0,
                          std::tuple<Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>,
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    if (j != NB - 1)
      std::cout << "Error in 2B, should only be triggered for last column\n";

    std::cout << "wf2B " << i << " " << j <<  std::endl;
    BlockMatrix<T> res;
    res = func(i, j, MB, NB, input, top, left, bottom0, bottom0);

    //Processing finished for this block, so send it to output
    send<2>(Key(i,j), res, out);

    if (next_i == MB - 1)
      // Single predecessor, no top block
      send<1>(Key(next_i, j), res, out); //send top block
    else
      send<0>(Key(next_i, j), res, out);
  };

  return wrap(f, edges(input2B, output12B, output2B, bottom0),
              edges(output12B, output2TL, result),
              "wavefront2B",
              {"block_input", "output12B", "output2B", "bottom0"},
              {"recur", "output2TL", "result"});

}

template <typename funcT, typename T>
auto make_wavefront2L(const funcT& func, int MB, int NB,
                      Edge<Key, BlockMatrix<T>>& input2L,
                      Edge<Key, BlockMatrix<T>>& output12R,
                      Edge<Key, BlockMatrix<T>>& output12B,
                      Edge<Key, BlockMatrix<T>>& output2TL,
                      Edge<Key, BlockMatrix<T>>& output2LL,
                      Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input,
                          BlockMatrix<T>&& left,
                          BlockMatrix<T>&& top,
                          std::tuple<Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    if (i != MB - 1 && j != NB - 1)
      std::cout << "Error in 2L, should only be triggered for last corner block\n";

    std::cout << "wf2L " << i << " " << j <<  std::endl;
    BlockMatrix<T> res;
    res = func(i, j, MB, NB, input, top, left, input, input);

    //Processing finished for this block, so send it to output
    send<0>(Key(i,j), res, out);

  };

  return wrap(f, edges(input2L, fuse(output12B, output2TL),
                       fuse(output12R, output2LL)),
              edges(result),
              "wavefront2L",
              {"block_input", "output2TL", "output2LL"},
              {"result"});

}

template <typename T>
auto make_result(Matrix<T>* r, const Edge<Key, BlockMatrix<T>>& result) {
  auto f = [r](const Key& key, BlockMatrix<T>&& bm, std::tuple<>& out) {
    auto [i, j] = key;
    if (bm(i, j) != (*r)(i, j)) {
      std::cout << "ERROR in block (" << i << "," << j << ")\n";
    }
    else std::cout << "NO ERRORS!\n";
  };

  return wrap(f, edges(result), edges(), "Final Output", {"result"}, {});
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Usage: ./wavefront-df-mad <matrix size> <block size> " <<
      "<verify 1/0>" << std::endl;
    exit(-1);
  }
  //Initialize TTG
  ttg_initialize(argc, argv, -1);

  int n_rows, n_cols, B;
  int n_brows, n_bcols;

  n_rows = n_cols = atoi(argv[1]);//8192;
  B = atoi(argv[2]);//128;
  bool verify = atoi(argv[3]);

  n_brows = (n_rows / B) + (n_rows % B > 0);
  n_bcols = (n_cols / B) + (n_cols % B > 0);

  //Get the number of processes in this run
  int world_size = ttg_default_execution_context().size();
  int my_rank = ttg_default_execution_context().rank();

  //How to partition the data among processes?
  //Lets say P should be a power of two and same for matrix and block sizes.
  //If n_brows * n_bcols % P != 0, we cannot divide equally.
  //For the simplest case, let's assume #blocks is divisible by P.
  if (world_size > 1 && (n_brows * n_bcols) % world_size != 0) {
    std::cout << "Number of block rows * Number of block cols should be " <<
      "equal to number of processes" << std::endl;
    exit(-1);
  }

  //Make an unordered_map with key as the block indices and value as the block.
  //int local_block_count = n_brows * n_bcols / world_size;

  //Define the matrix blocks.
  std::unordered_map<Key, BlockMatrix<double>, pair_hash> m;

  //How to map indices based on process ID?
  if (world_size > 1) {
    //Complete row belongs to a single process here.
    for (int cols = 0; cols < n_bcols; cols++) {
      BlockMatrix<double> bm(B,B);
      bm.fill();
      m[std::make_pair(my_rank, cols)] = std::move(bm);
    }
  }
  else {
    for (int rows = 0; rows < n_brows; rows++) {
      for (int cols = 0; cols < n_bcols; cols++) {
        BlockMatrix<double> bm(B,B);
        bm.fill();
        m[std::make_pair(rows, cols)] = std::move(bm);
      }
    }
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  Matrix<double>* r2 = new Matrix<double>(n_brows, n_bcols, B, B);

  //Matrix<double>* m = new Matrix<double>(n_brows, n_bcols, B, B);
  if (verify) {
    Matrix<double>* m2 = new Matrix<double>(n_brows, n_bcols, B, B);
    //m->fill();
    m2->fill();

    std::cout << "Computing using serial version....";
    beg = std::chrono::high_resolution_clock::now();
    wavefront_serial(m2, r2, n_brows, n_bcols);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "....done!" << std::endl;
    std::cout << "Serial Execution Time (milliseconds) : "
            << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1e3 << std::endl;
    delete m2;
  }

  std::function<std::vector<std::pair<int, int>>(const Key&)> get_bottomindex_func =
                 [n_brows,n_bcols](const Key& key) {
                   return get_bottomindex(key, n_brows, n_bcols);
                 };

  std::function<std::vector<std::pair<int, int>>(const Key&)> get_rightindex_func =
                 [n_brows,n_bcols](const Key& key) {
                   return get_rightindex(key, n_brows, n_bcols);
                 };


  Edge<Key, BlockMatrix<double>> input0("input0"), input1BR("input1BR"),
    input2BR("input2BR"), input1R("input1R"), input2R("input2R"),
    input1B("input1B"), input2B("input2B"), input2L("input2L"),
    toporleft("toporleft"), toporleftLR("toporleftLR"),
    toporleftLB("toporleftLB"), output1("output1"),
    output2("output2"), output1R("output1R"), output1B("output1B"),
    output12R("output12R"), output12B("output12B"), outputLB("outputLB"),
    outputLR("outputLR"), output2R("output2R"), output2B("output2B"), output2TL("output2TL"),
    output2LL("output2LL"), result("result");
  Edge<Key, BlockMatrix<double>> bottom0("bottom0", true, m, get_bottomindex_func);
  Edge<Key, BlockMatrix<double>> right0("right0", true, m, get_rightindex_func);

  //OpBase::set_trace_all(true);
  //OpBase::set_lazy_pull(true);

  auto i = initiator(n_brows, n_bcols, m, input0, input1BR, input2BR, input1R,
                     input2R, input1B, input2B, input2L);
  auto b = make_get_bottomblock(m, bottom0);
  auto r = make_get_rightblock(m, right0);

  auto s0 = make_wavefront0(stencil_computation<double>, n_brows, n_bcols,
                            input0, toporleft, toporleftLR, toporleftLB,
                            bottom0, right0, result);
  auto s1 = make_wavefront1(stencil_computation<double>, n_brows, n_bcols,
                            input1BR, toporleft, bottom0, right0, output1,
                            output2, output1R, output1B, result);
  auto s1R = make_wavefront1R(stencil_computation<double>, n_brows, n_bcols,
                              input1R, toporleftLR, output1R, output12R, outputLR,
                              right0, result);
  auto s1B = make_wavefront1B(stencil_computation<double>, n_brows, n_bcols,
                              input1B, toporleftLB, output1B, output12B, outputLB,
                              bottom0, result);
  auto s2 = make_wavefront2BR(stencil_computation<double>, n_brows, n_bcols,
                            input2BR, output1, output2, bottom0, right0, output2R,
                            output2B, result);
  auto s2R = make_wavefront2R(stencil_computation<double>, n_brows, n_bcols,
                              input2R, output12R, output2R, right0,
                              output2LL, result);
  auto s2B = make_wavefront2B(stencil_computation<double>, n_brows, n_bcols,
                              input2B, output12B, output2B, bottom0,
                              output2TL, result);
  auto s2L = make_wavefront2L(stencil_computation<double>, n_brows, n_bcols,
                              input2L, outputLR, outputLB, output2TL, output2LL, result);
  auto res = make_result(r2, result);

  if (world_size > 1) {
    auto keymap = [](const Key& key) { return key.first; };
    b->set_keymap(keymap);
    r->set_keymap(keymap);
    i->set_keymap(keymap);
    s0->set_keymap(keymap);
    s1->set_keymap(keymap);
    s1R->set_keymap(keymap);
    s1B->set_keymap(keymap);
    s2->set_keymap(keymap);
    s2R->set_keymap(keymap);
    s2B->set_keymap(keymap);
    s2L->set_keymap(keymap);
  }

  std::cout << "Checking for graph connectivity...\n";
  auto connected = make_graph_executable(i.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected.\n";

  if (ttg_default_execution_context().rank() == 0) {
    //std::cout << "==== begin dot ====\n";
    //std::cout << Dot()(i.get()) << std::endl;
    //std::cout << "==== end dot ====\n";
    beg = std::chrono::high_resolution_clock::now();
    i->invoke(Key(0, 0));
    // i->in<0>()->send(Key(0, 0), Control());
    // This doesn't work!
    // s->send<0>(Key(0,0), Control());
  }

  //Trigger the initiator on all other processes as well
  if (world_size > 1 && ttg_default_execution_context().rank() != 0) {
    i->invoke(Key(my_rank,0));
  }

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());

  if (ttg_default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;
  }

  ttg_finalize();

  /*m->print();
  std::cout << std::endl << std::endl;
  r->print();
  std::cout << std::endl << std::endl;
  m2->print();
  std::cout << std::endl << std::endl;
  r2->print();*/
  /*r->print();
  std::cout << std::endl << std::endl;
  m2->print();*/

  //delete m;
  //delete m2;
  delete r2;
  return 0;
}
