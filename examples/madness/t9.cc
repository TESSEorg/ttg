#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/worldmutex.h>

#define TTG_RUNTIME_H "madness/ttg.h"
#if 0
#define IMPORT_TTG_RUNTIME_NS   \
  using namespace madness;      \
  using namespace madness::ttg; \
  using namespace ::ttg;        \
  constexpr const ::ttg::Runtime ttg_runtime = ::ttg::Runtime::MADWorld;
#endif

#define IMPORT_TTG_RUNTIME_NS   \
  using namespace ttg;
#include "../t9.impl.h"
