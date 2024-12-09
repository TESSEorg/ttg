#ifndef TTG_H_INCLUDED
#define TTG_H_INCLUDED

#include "ttg/config.h"
#include "ttg/fwd.h"

#if defined(TTG_USE_PARSEC)
#include "ttg/parsec/ttg.h"
#elif defined(TTG_USE_MADNESS)
#include "ttg/madness/ttg.h"
#endif  // TTG_USE_{PARSEC|MADNESS}

#include "ttg/runtimes.h"
#include "ttg/util/demangle.h"
#include "ttg/util/hash.h"
#include "ttg/util/meta.h"
#include "ttg/util/print.h"
#include "ttg/util/trace.h"
#include "ttg/util/void.h"
#include "ttg/util/typelist.h"

#include "ttg/base/keymap.h"
#include "ttg/base/terminal.h"
#include "ttg/base/world.h"
#include "ttg/broadcast.h"
#include "ttg/func.h"
#include "ttg/reduce.h"
#include "ttg/traverse.h"
#include "ttg/tt.h"
#include "ttg/util/dot.h"
#include "ttg/util/macro.h"
#include "ttg/util/print.h"
#include "ttg/world.h"

#include "ttg/constraint.h"
#include "ttg/edge.h"

#include "ttg/ptr.h"
#include "ttg/buffer.h"
#include "ttg/devicescratch.h"
#include "ttg/ttvalue.h"
#include "ttg/devicescope.h"
#include "ttg/device/device.h"
#include "ttg/device/task.h"

// these headers use the default backend
#include "ttg/run.h"

#endif  // TTG_H_INCLUDED
