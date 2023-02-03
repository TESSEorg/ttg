#ifndef TTG_FWD_H
#define TTG_FWD_H

/**
 * @file ttg/fwd.h
 * Includes forward declarations for the entire TTG codebase
 */

// namespaces first ////////////////////////////////////////////////////////////////////////////////////////////////////

/// top-level TTG namespace contains runtime-neutral functionality
namespace ttg {}
// predeclare runtime-specific namespaces
// note that these are top-level (but differ from the namespaces reserved by the runtime itself to avoid ambiguities)
/// this contains MADNESS-based TTG functionality
namespace ttg_madness {}
/// this contains PaRSEC-based TTG functionality
namespace ttg_parsec {}

// classes + functions /////////////////////////////////////////////////////////////////////////////////////////////////

namespace ttg {

  class TTBase;

  /// \internal OG name
  using OpBase [[deprecated("use TTBase instead")]] = TTBase;
  /// \internal the name used in the ESPM2 paper
  using TemplateTaskBase = TTBase;

  template <typename keyT = void, typename valueT = void>
  class Edge;

  template <typename input_terminalsT, typename output_terminalsT>
  class TTG;

  /// \internal the name consistent with the API defined in the ESPM2 paper
  template <typename input_terminalsT, typename output_terminalsT>
  using TemplateTaskGraph = TTG<input_terminalsT, output_terminalsT>;

  /// \internal the OG name
  template <typename input_terminalsT, typename output_terminalsT>
  using CompositeOp [[deprecated("use TTG instead")]] = TTG<input_terminalsT, output_terminalsT>;

  class World;

  template <typename... RestOfArgs>
  void initialize(int argc, char **argv, int num_threads = -1, RestOfArgs &&...);
  void finalize();
  [[noreturn]]
  void abort();
  World default_execution_context();
  void execute(ttg::World world);
  void fence(ttg::World world);

}  // namespace ttg

#include "ttg/impl_selector.h"
#if TTG_USE_PARSEC
#include "ttg/parsec/fwd.h"
#endif
#if TTG_USE_MADNESS
#include "ttg/madness/fwd.h"
#endif

#endif  // TTG_FWD_H
