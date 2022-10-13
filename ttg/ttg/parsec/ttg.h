#ifndef PARSEC_TTG_H_INCLUDED
#define PARSEC_TTG_H_INCLUDED

/* set up env if this header was included directly */
#if !defined(TTG_IMPL_NAME)
#define TTG_USE_PARSEC 1
#endif  // !defined(TTG_IMPL_NAME)

#include "ttg/impl_selector.h"

/* include ttg header to make symbols available in case this header is included directly */
#include "../../ttg.h"


/* header files exposed to application */
#include "ttg/parsec/tt.h"
#include "ttg/parsec/copy_handler.h"
#include "ttg/parsec/funcs.h"

/* needed for make_tt */
#include "ttg/util/meta.h"
#include "ttg/util/meta/callable.h"

namespace ttg_parsec {


#include "ttg/make_tt.h"



} // namespace ttg_parsec

#endif  // PARSEC_TTG_H_INCLUDED
