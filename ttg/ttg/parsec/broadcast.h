#ifndef TTG_PARSEC_BROADCAST_H
#define TTG_PARSEC_BROADCAST_H

#include <ttg/util/span.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>


namespace ttg_parsec {

  enum class BroadcastType {
    Star,
    Pipe
  };

  inline BroadcastType get_broadcast_type() {
    static std::once_flag init_flag;
    static BroadcastType bcast_type = BroadcastType::Star;
    std::call_once(init_flag, [&](){
      const char *bcast_type_env = std::getenv("TTG_BCAST_TYPE");
      if (bcast_type_env) {
        if (std::strcmp(bcast_type_env, "star") == 0) {
          bcast_type = BroadcastType::Star;
        } else if (std::strcmp(bcast_type_env, "pipe") == 0) {
          bcast_type = BroadcastType::Pipe;
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

} // namespace ttg_parsec

#endif // TTG_PARSEC_BROADCAST_H