#ifndef TTG_PARSEC_FUNCS_H
#define TTG_PARSEC_FUNCS_H

#include "ttg/impl_selector.h"

#include <parsec.h>

#include "ttg/base/tt.h"
#include "ttg/util/trace.h"
#include "ttg/util/env.h"
#include "ttg/serialization.h"

#include "ttg/parsec/fwd.h"
#include "ttg/parsec/vars.h"
#include "ttg/parsec/world.h"

namespace ttg_parsec {

  template <typename... RestOfArgs>
  inline void ttg_initialize(int argc, char **argv, int num_threads, RestOfArgs &&...) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (num_threads < 1) num_threads = ttg::detail::num_threads();
    auto world_ptr = new ttg_parsec::WorldImpl{&argc, &argv, num_threads};
    std::shared_ptr<ttg::base::WorldImplBase> world_sptr{static_cast<ttg::base::WorldImplBase *>(world_ptr)};
    ttg::World world{std::move(world_sptr)};
    ttg::detail::set_default_world(std::move(world));
  }
  inline void ttg_finalize() {
    ttg::detail::set_default_world(ttg::World{});  // reset the default world
    ttg::detail::destroy_worlds<ttg_parsec::WorldImpl>();
    MPI_Finalize();
  }
  inline ttg::World ttg_default_execution_context() { return ttg::get_default_world(); }
  inline void ttg_abort() { MPI_Abort(ttg_default_execution_context().impl().comm(), 1); }
  inline void ttg_execute(ttg::World world) { world.impl().execute(); }
  inline void ttg_fence(ttg::World world) { world.impl().fence(); }

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr) {
    world.impl().register_ptr(ptr);
  }

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, std::unique_ptr<T> &&ptr) {
    world.impl().register_ptr(std::move(ptr));
  }

  inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr) {
    world.impl().register_status(status_ptr);
  }

  template <typename Callback>
  inline void ttg_register_callback(ttg::World world, Callback &&callback) {
    world.impl().register_callback(std::forward<Callback>(callback));
  }

  inline ttg::Edge<> &ttg_ctl_edge(ttg::World world) { return world.impl().ctl_edge(); }

  inline void ttg_sum(ttg::World world, double &value) {
    double result = 0.0;
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, world.impl().comm());
    value = result;
  }

  /// broadcast
  /// @tparam T a serializable type
  template <typename T>
  void ttg_broadcast(::ttg::World world, T &data, int source_rank) {
    int64_t BUFLEN;
    if (world.rank() == source_rank) {
      BUFLEN = ttg::default_data_descriptor<T>::payload_size(&data);
    }
    MPI_Bcast(&BUFLEN, 1, MPI_INT64_T, source_rank, world.impl().comm());

    unsigned char *buf = new unsigned char[BUFLEN];
    if (world.rank() == source_rank) {
      ttg::default_data_descriptor<T>::pack_payload(&data, BUFLEN, 0, buf);
    }
    MPI_Bcast(buf, BUFLEN, MPI_UNSIGNED_CHAR, source_rank, world.impl().comm());
    if (world.rank() != source_rank) {
      ttg::default_data_descriptor<T>::unpack_payload(&data, BUFLEN, 0, buf);
    }
    delete[] buf;
  }


} // namespace ttg_parsec

#endif // TTG_PARSEC_FUNCS_H
