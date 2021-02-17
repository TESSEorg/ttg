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

  class OpBase;

  template <typename keyT = void, typename valueT = void>
  class Edge;

  class World;

}  // namespace ttg

#include "ttg/impl_selector.h"
#if TTG_USE_PARSEC
#include "ttg/parsec/fwd.h"
#endif
#if TTG_USE_MADNESS
#include "ttg/madness/fwd.h"
#endif

#endif  // TTG_FWD_H
