#define TTG_RUNTIME_H "parsec/ttg.h"
#define IMPORT_TTG_RUNTIME_NS  \
  using namespace parsec;      \
  using namespace parsec::ttg; \
  using namespace ::ttg;       \
  constexpr const ::ttg::Runtime ttg_runtime = ::ttg::Runtime::PaRSEC;

#include "../spmm.impl.h"
