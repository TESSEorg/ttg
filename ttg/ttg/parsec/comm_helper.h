#ifndef TTG_PARSEC_COMM_HELPER_H
#define TTG_PARSEC_COMM_HELPER_H

#include <parsec.h>

#include <map>
#include <mutex>
#include <functional>

#include "ttg/impl_selector.h"

#include "ttg/util/trace.h"
#include "ttg/base/tt.h"
#include "ttg/world.h"

#include "ttg/parsec/world.h"
#include "ttg/parsec/ttg_data_copy.h"

namespace ttg_parsec {
  namespace detail {

    typedef void (*static_set_arg_fct_type)(void *, size_t, ttg::TTBase *);
    typedef std::pair<static_set_arg_fct_type, ttg::TTBase *> static_set_arg_fct_call_t;
    inline std::map<uint64_t, static_set_arg_fct_call_t> static_id_to_op_map;
    inline std::mutex static_map_mutex;
    typedef std::tuple<int, void *, size_t> static_set_arg_fct_arg_t;
    inline std::multimap<uint64_t, static_set_arg_fct_arg_t> delayed_unpack_actions;

    struct msg_header_t {
      typedef enum {
        MSG_SET_ARG = 0,
        MSG_SET_ARGSTREAM_SIZE = 1,
        MSG_FINALIZE_ARGSTREAM_SIZE = 2,
        MSG_GET_FROM_PULL =3 } fn_id_t;
      uint32_t taskpool_id;
      uint64_t op_id;
      fn_id_t fn_id;
      int32_t param_id;
      int num_keys;
    };

    struct msg_t {
      msg_header_t tt_id;
      unsigned char bytes[WorldImpl::PARSEC_TTG_MAX_AM_SIZE - sizeof(msg_header_t)];

      msg_t() = default;
      msg_t(uint64_t tt_id, uint32_t taskpool_id, msg_header_t::fn_id_t fn_id, int32_t param_id, int num_keys = 1)
          : tt_id{taskpool_id, tt_id, fn_id, param_id, num_keys} {}
    };

    inline int static_unpack_msg(parsec_comm_engine_t *ce, uint64_t tag, void *data, long unsigned int size,
                                 int src_rank, void *obj) {
      static_set_arg_fct_type static_set_arg_fct;
      parsec_taskpool_t *tp = NULL;
      msg_header_t *msg = static_cast<msg_header_t *>(data);
      uint64_t op_id = msg->op_id;
      tp = parsec_taskpool_lookup(msg->taskpool_id);
      assert(NULL != tp);
      static_map_mutex.lock();
      try {
        auto op_pair = static_id_to_op_map.at(op_id);
        static_map_mutex.unlock();
        tp->tdm.module->incoming_message_start(tp, src_rank, NULL, NULL, 0, NULL);
        static_set_arg_fct = op_pair.first;
        static_set_arg_fct(data, size, op_pair.second);
        tp->tdm.module->incoming_message_end(tp, NULL);
        return 0;
      } catch (const std::out_of_range &e) {
        void *data_cpy = malloc(size);
        assert(data_cpy != 0);
        memcpy(data_cpy, data, size);
        ttg::trace("ttg_parsec(", ttg_default_execution_context().rank(), ") Delaying delivery of message (", src_rank,
                   ", ", op_id, ", ", data_cpy, ", ", size, ")");
        delayed_unpack_actions.insert(std::make_pair(op_id, std::make_tuple(src_rank, data_cpy, size)));
        static_map_mutex.unlock();
        return 1;
      }
    }

    template <typename ActivationT>
    inline int get_complete_cb(parsec_comm_engine_t *comm_engine, parsec_ce_mem_reg_handle_t lreg, ptrdiff_t ldispl,
                               parsec_ce_mem_reg_handle_t rreg, ptrdiff_t rdispl, size_t size, int remote,
                               void *cb_data) {
      parsec_ce.mem_unregister(&lreg);
      ActivationT *activation = static_cast<ActivationT *>(cb_data);
      if (activation->complete_transfer()) {
        delete activation;
      }
      return PARSEC_SUCCESS;
    }

    inline int get_remote_complete_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                      int src, void *cb_data) {
      std::intptr_t *fn_ptr = static_cast<std::intptr_t *>(msg);
      std::function<void(void)> *fn = reinterpret_cast<std::function<void(void)> *>(*fn_ptr);
      (*fn)();
      delete fn;
      return PARSEC_SUCCESS;
    }

    template <typename FuncT>
    inline int invoke_get_remote_complete_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                             int src, void *cb_data) {
      std::intptr_t *iptr = static_cast<std::intptr_t *>(msg);
      FuncT *fn_ptr = reinterpret_cast<FuncT *>(*iptr);
      (*fn_ptr)();
      delete fn_ptr;
      return PARSEC_SUCCESS;
    }

    template <typename KeyT, typename ActivationCallbackT>
    class rma_delayed_activate {
      std::vector<KeyT> _keylist;
      std::atomic<int> _outstanding_transfers;
      ActivationCallbackT _cb;
      detail::ttg_data_copy_t *_copy;

     public:
      rma_delayed_activate(std::vector<KeyT> &&key, detail::ttg_data_copy_t *copy, int num_transfers, ActivationCallbackT cb)
          : _keylist(std::move(key)), _outstanding_transfers(num_transfers), _cb(cb), _copy(copy) {}

      bool complete_transfer(void) {
        int left = --_outstanding_transfers;
        if (0 == left) {
          _cb(std::move(_keylist), _copy);
          return true;
        }
        return false;
      }
    };


  }  // namespace detail

} // namespace ttg_parsec

#endif // TTG_PARSEC_COMM_HELPER_H
