#ifndef TTG_PARSEC_FWD_H
#define TTG_PARSEC_FWD_H

#include "ttg/fwd.h"
#include "ttg/util/typelist.h"

#include <future>
#include <parsec.h>

namespace ttg_parsec {


  namespace detail {
    inline int static_unpack_msg(parsec_comm_engine_t *ce, uint64_t tag, void *data, long unsigned int size,
                                 int src_rank, void *obj);

    inline int get_remote_complete_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                      int src, void *cb_data);
  }


  template <typename keyT, typename output_terminalsT, typename derivedT, typename input_valueTs = ttg::typelist<>>
  class TT;

  /// \internal the OG name
  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  using Op [[deprecated("use TT instead")]] = TT<keyT, output_terminalsT, derivedT, ttg::typelist<input_valueTs...>>;
  /// \internal the name in the ESPM2 paper
  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  using TemplateTask = TT<keyT, output_terminalsT, derivedT, ttg::typelist<input_valueTs...>>;

  class WorldImpl;

  template <typename... RestOfArgs>
  inline void ttg_initialize(int argc, char **argv, int num_threads = -1, RestOfArgs &&...);

  inline void ttg_finalize();

  static inline void ttg_abort();

  inline ttg::World ttg_default_execution_context();

  inline void ttg_execute(ttg::World world);

  inline void ttg_fence(ttg::World world);

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr);

  inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr);

  template <typename Callback>
  inline void ttg_register_callback(ttg::World world, Callback &&callback);

  inline ttg::Edge<> &ttg_ctl_edge(ttg::World world);

  inline void ttg_sum(ttg::World world, double &value);

  /// broadcast
  /// @tparam T a serializable type
  template <typename T>
  static void ttg_broadcast(ttg::World world, T &data, int source_rank);

}  // namespace ttg_parsec

#endif  // TTG_PARSEC_FWD_H
