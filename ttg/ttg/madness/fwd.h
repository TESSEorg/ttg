#ifndef TTG_MADNESS_FWD_H
#define TTG_MADNESS_FWD_H

#include "ttg/fwd.h"
#include "ttg/util/typelist.h"
#include "ttg/util/span.h"

#include <future>

namespace ttg_madness {

  template <typename keyT, typename output_terminalsT, typename derivedT,
            typename input_valueTs = ttg::typelist<>,
            ttg::ExecutionSpace Space = ttg::ExecutionSpace::Host>
  class TT;

  /// \internal the OG name
  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  using Op [[deprecated("use TT instead")]] = TT<keyT, output_terminalsT, derivedT, ttg::typelist<input_valueTs...>>;
  /// \internal the name in the ESPM2 paper
  template <typename keyT, typename output_terminalsT, typename derivedT, typename... input_valueTs>
  using TemplateTask = TT<keyT, output_terminalsT, derivedT, ttg::typelist<input_valueTs...>>;

  class WorldImpl;

  inline void make_executable_hook(ttg::World&);

  inline void ttg_initialize(int argc, char **argv, int num_threads = -1);

  inline void ttg_finalize();

  [[noreturn]]
  inline void ttg_abort();

  inline ttg::World ttg_default_execution_context();

  inline void ttg_execute(ttg::World world);

  inline void ttg_fence(ttg::World world);

  template <typename T>
  inline void ttg_register_ptr(ttg::World world, const std::shared_ptr<T> &ptr);

  inline void ttg_register_status(ttg::World world, const std::shared_ptr<std::promise<void>> &status_ptr);

  inline ttg::Edge<> &ttg_ctl_edge(ttg::World world);

  template <typename T>
  inline void ttg_sum(ttg::World world, T &value);

  template <typename T>
  inline void ttg_broadcast(ttg::World world, T &data, int source_rank);


  /* device definitions, not currently provided by this impl */
  template<typename T, typename Allocator = std::allocator<T>>
  struct Buffer;

  template<typename T>
  struct Ptr;

  template<typename T>
  struct devicescratch;

  template<typename T>
  struct TTValue;

  template<typename T, typename... Args>
  Ptr<T> make_ptr(Args&&... args);

  template<typename T>
  auto get_ptr(T&& obj);

  template<typename... Views>
  inline bool register_device_memory(std::tuple<Views&...> &views);

  template<typename T, std::size_t N>
  inline bool register_device_memory(const ttg::span<T, N>& span);

  template<typename... Buffer>
  inline void post_device_out(std::tuple<Buffer&...> &b);

  template<typename... Buffer>
  inline void mark_device_out(std::tuple<Buffer&...> &b);

  inline int num_devices();

}  // namespace ttg_madness

#endif  // TTG_MADNESS_FWD_H
