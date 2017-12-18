#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#define TTG_RUNTIME_H "madness/ttg.h"
#define IMPORT_TTG_RUNTIME_NS   \
  using namespace madness;      \
  using namespace madness::ttg; \
  using namespace ::ttg;

#include "../spmm.impl.h"
