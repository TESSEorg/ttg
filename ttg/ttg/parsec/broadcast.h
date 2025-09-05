#ifndef TTG_PARSEC_BROADCAST_H
#define TTG_PARSEC_BROADCAST_H

#include <ttg/util/span.h>
#include <cstdlib>
#include <mutex>


namespace ttg_parsec {

  enum class BroadcastType {
    Star,
    Pipe
  };

  BroadcastType get_broadcast_type() {
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
      return m_me == m_root && m_procs_begin != m_procs_end;
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

  public:

    BroadcastPipe(int root, int me, Iter procs_begin, Iter procs_end)
    : m_root(root), m_me(me), m_procs_begin(procs_begin), m_procs_end(procs_end)
    {
      assert(std::is_sorted(m_procs_begin, m_procs_end));
    }

    bool has_peers() const {
      auto me_iter = std::find(m_procs_begin, m_procs_end, m_me);
      return me_iter != m_procs_end;
    }

    template<typename SendFn>
    void operator()(SendFn&& send_fn) {
      auto iter = std::find(m_procs_begin, m_procs_end, m_me);
      assert(iter != m_procs_end);
      // wrap around if we reached the end
      if ((++iter) == m_procs_end) iter = m_procs_begin;
      // if we reached root we reached the end of the pipe
      if (*iter != m_root) {
        //while (next != m_procs_end && (*next == m_me || *next == m_root)) ++next;
        send_fn(*iter);
      }
    }
  };

} // namespace ttg_parsec

#endif // TTG_PARSEC_BROADCAST_H