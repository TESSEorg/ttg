#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#define TTG_RUNTIME_H "madness/ttg.h"
#define IMPORT_TTG_RUNTIME_NS   \
  using namespace madness;      \
  using namespace madness::ttg; \
  using namespace ::ttg;        \
  constexpr const ttg::Runtime ttg_runtime = ttg::Runtime::MADWorld;

#include "../randomaccess.impl.h"
