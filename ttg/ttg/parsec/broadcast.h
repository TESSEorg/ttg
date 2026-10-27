#ifndef TTG_PARSEC_BROADCAST_H
#define TTG_PARSEC_BROADCAST_H

#include <ttg/util/span.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>


namespace ttg_parsec {

  enum class BroadcastType {
    Star,
    Pipe,
    BinaryTree
  };

  inline BroadcastType get_broadcast_type() {
    static std::once_flag init_flag;
    static BroadcastType bcast_type = BroadcastType::BinaryTree;
    std::call_once(init_flag, [&](){
      const char *bcast_type_env = std::getenv("TTG_BCAST_TYPE");
      if (bcast_type_env) {
        if (std::strcmp(bcast_type_env, "star") == 0) {
          bcast_type = BroadcastType::Star;
        } else if (std::strcmp(bcast_type_env, "pipe") == 0) {
          bcast_type = BroadcastType::Pipe;
        } else if (std::strcmp(bcast_type_env, "tree") == 0) {
          bcast_type = BroadcastType::BinaryTree;
        }
      }
    });
    return bcast_type;
  }

  template<typename Iter>
  class BroadcastStar {

    int m_root;
    int m_me;
    Iter m_procs_begin;
    Iter m_procs_end;

  public:

    BroadcastStar(int root, int me, Iter procs_begin, Iter procs_end)
    : m_root(root), m_me(me), m_procs_begin(procs_begin), m_procs_end(procs_end)
    {}

    bool has_peers() const {
      return m_me == m_root && std::any_of(m_procs_begin, m_procs_end, [&](int p){ return p != m_root; });
    }

    template<typename SendFn>
    void operator()(SendFn&& send_fn) {
      if (has_peers()) {
        for (auto it = m_procs_begin; it != m_procs_end; ++it) {
          int p = *it;
          if (p != m_root && p != m_me) {
            send_fn(p);
          }
        }
      }
    }
  };

  template<typename Iter>
  class BroadcastPipe {

    int m_root;
    int m_me;
    Iter m_procs_begin;
    Iter m_procs_end;

    /* position in the pipe: processes following the root in ascending order, then wrapping around */
    std::pair<bool, int> pipe_pos(int p) const {
      return {p < m_root, p};
    }

    /* the process following this process in the pipe, or -1 if this process is the last */
    int next_peer() const {
      auto me_pos = pipe_pos(m_me);
      int next = -1;
      for (auto it = m_procs_begin; it != m_procs_end; ++it) {
        int p = *it;
        if (p == m_root || pipe_pos(p) <= me_pos) continue;
        if (next < 0 || pipe_pos(p) < pipe_pos(next)) next = p;
      }
      return next;
    }

  public:

    /* the root does not have to be in the set of processes */
    BroadcastPipe(int root, int me, Iter procs_begin, Iter procs_end)
    : m_root(root), m_me(me), m_procs_begin(procs_begin), m_procs_end(procs_end)
    { }

    bool has_peers() const {
      return next_peer() >= 0;
    }

    template<typename SendFn>
    void operator()(SendFn&& send_fn) {
      int next = next_peer();
      if (next >= 0) {
        send_fn(next);
      }
    }
  };

  template<typename Iter>
  class BroadcastBinaryTree {

    /* the root followed by the other processes in pipe order (following the root, then wrapping around);
     * all processes see the same set of processes so they all build the same tree */
    std::vector<int> m_procs;
    /* position of this process in m_procs, or m_procs.size() if it does not participate */
    std::size_t m_me_idx;

  public:

    /* the root does not have to be in the set of processes */
    BroadcastBinaryTree(int root, int me, Iter procs_begin, Iter procs_end)
    {
      m_procs.push_back(root);
      for (auto it = procs_begin; it != procs_end; ++it) {
        if (*it > root) m_procs.push_back(*it);
      }
      for (auto it = procs_begin; it != procs_end; ++it) {
        if (*it < root) m_procs.push_back(*it);
      }
      m_me_idx = std::find(m_procs.begin(), m_procs.end(), me) - m_procs.begin();
    }

    bool has_peers() const {
      return 2*m_me_idx + 1 < m_procs.size();
    }

    template<typename SendFn>
    void operator()(SendFn&& send_fn) {
      /* the process at position i sends to the processes at positions 2i+1 and 2i+2 */
      for (std::size_t child = 2*m_me_idx + 1; child <= 2*m_me_idx + 2 && child < m_procs.size(); ++child) {
        send_fn(m_procs[child]);
      }
    }
  };

} // namespace ttg_parsec

#endif // TTG_PARSEC_BROADCAST_H