#ifndef TTG_PARSEC_FWD_H
#define TTG_PARSEC_FWD_H

#include "ttg/impl_selector.h"
#include "ttg/fwd.h"

namespace ttg {

  TTG_IMPL_PARSEC_INLINE_NS namespace ttg_parsec {

    template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
    class Op;

    class WorldImpl;

    template <typename... RestOfArgs>
    inline void ttg_initialize(int argc, char **argv, int taskpool_size = -1, RestOfArgs &&...);

    static
    inline void ttg_finalize();

    static
    inline void ttg_abort();

    static
    inline ::ttg::World ttg_default_execution_context();

    static
    inline void ttg_execute(::ttg::World world);

    static
    inline void ttg_fence(::ttg::World world);

    template <typename T>
    static
    inline void ttg_register_ptr(::ttg::World world, const std::shared_ptr<T>& ptr);

    static
    inline void ttg_register_status(::ttg::World world, const std::shared_ptr<std::promise<void>>& status_ptr);

    static
    inline ::ttg::Edge<>& ttg_ctl_edge(::ttg::World world);

    static
    inline void ttg_sum(::ttg::World world, double &value);

    /// broadcast
    /// @tparam T a serializable type
    template <typename T>
    static
    void ttg_broadcast(::ttg::World world, T &data, int source_rank);


  } // namespace ttg_parsec
} // namespace ttg

#endif // TTG_PARSEC_FWD_H
