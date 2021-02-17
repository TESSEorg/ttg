#ifndef TTG_MADNESS_FWD_H
#define TTG_MADNESS_FWD_H

#include "ttg/fwd.h"

#include <future>

namespace ttg_madness {

  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  class Op;

  class WorldImpl;

  template <typename... RestOfArgs>
  static inline void ttg_initialize(int argc, char **argv, RestOfArgs &&...);

  static inline void ttg_finalize();

  static inline void ttg_abort();

  static inline ttg::World ttg_default_execution_context();

  static inline void ttg_execute(ttg::World world);

  static inline void ttg_fence(ttg::World world);

  template <typename T>
  static inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr);

  static inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr);

  static inline ttg::Edge<> &ttg_ctl_edge(ttg::World world);

  template <typename T>
  static inline void ttg_sum(ttg::World world, T &value);

  template <typename T>
  static inline void ttg_broadcast(ttg::World world, T &data, int source_rank);

}  // namespace ttg_madness

#endif  // TTG_MADNESS_FWD_H
