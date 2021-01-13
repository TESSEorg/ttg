#ifndef TTG_H_INCLUDED
#define TTG_H_INCLUDED

#include "util/impl_selector.h"

#include "util/demangle.h"
#include "util/meta.h"
#include "util/runtimes.h"
#include "util/hash.h"
#include "util/void.h"
#include "util/trace.h"
#include "util/print.h"

#include "base/op.h"
#include "base/terminal.h"
#include "base/world.h"
#include "base/keymap.h"
#include "util/world.h"
#include "util/dot.h"
#include "util/traverse.h"
#include "util/op.h"
#include "util/data_descriptor.h"
#include "util/print.h"
#include "util/func.h"
#include "util/macro.h"
#include "util/broadcast.h"
#include "util/reduce.h"

#include "edge.h"

#if defined(TTG_USE_MADNESS)
#include "madness/ttg.h"
#elif defined(TTG_USE_PARSEC)
#include "parsec/ttg.h"
#endif // TTG_USE_MADNESS|PARSEC

#endif  // TTG_H_INCLUDED
