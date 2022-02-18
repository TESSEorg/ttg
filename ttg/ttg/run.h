//
// Created by Eduard Valeyev on 11/5/21.
//

#ifndef TTG_RUN_H
#define TTG_RUN_H

#include "ttg/fwd.h"

#include "ttg/util/env.h"
#include "ttg/util/bug.h"

namespace ttg {

  /// Initializes the TTG runtime with the default backend

  /// @note Dispatches to the default backend's `ttg_initialize`.
  /// @note This is a collective operation with respect to the default backend's default execution context
  /// @internal ENABLE_WHEN_TTG_CAN_MULTIBACKEND To initialize the TTG runtime with multiple
  /// backends must call the corresponding `ttg_initialize` functions explicitly.
  /// @param argc the argument count; this is typically the value received by `main`
  /// @param argv the vector of arguments; this is typically the value received by `main`
  /// @param num_threads the number of compute threads to initialize. If less than 1 then
  ///        ttg::detail::num_threads will be used to determine the default number of compute threads
  template <typename... RestOfArgs>
  inline void initialize(int argc, char **argv, int num_threads, RestOfArgs &&...args) {
    // if requested by user, create a Debugger object
    if (auto debugger_cstr = std::getenv("TTG_DEBUGGER")) {
      using mpqc::Debugger;
      auto debugger = std::make_shared<Debugger>();
      Debugger::set_default_debugger(debugger);
      debugger->set_exec(argv[0]);
      debugger->set_cmd(debugger_cstr);
    }

    if (num_threads < 1) num_threads = detail::num_threads();
    TTG_IMPL_NS::ttg_initialize(argc, argv, num_threads, std::forward<RestOfArgs>(args)...);

    // finish setting up the Debugger, if needed
    if (mpqc::Debugger::default_debugger())
      mpqc::Debugger::default_debugger()->set_prefix(ttg::default_execution_context().rank());
  }

  /// Finalizes the TTG runtime

  /// This will possibly try to release as many resources as possible (some resources may only be released at
  /// the conclusion of the program). Execution of TTG code is not possible after calling this.
  /// @note Dispatches to the default backend's `ttg_finalize`.
  /// @note This is a collective operation with respect to the default execution context used by the matching
  /// `initialize` call
  /// @internal ENABLE_WHEN_TTG_CAN_MULTIBACKEND To finalize the TTG runtime with multiple backends must call the
  /// corresponding `ttg_finalize` functions explicitly.
  inline void finalize() { TTG_IMPL_NS::ttg_finalize(); }

  /// Aborts the TTG program using the default backend's `ttg_abort` method
  inline void abort() { TTG_IMPL_NS::ttg_abort(); }

  /// Accesses the default backend's default execution context

  /// @note Dispatches to the `ttg_default_execution_context` method of the default backend
  /// @return the default backend's default execution context
  inline World default_execution_context() { return TTG_IMPL_NS::ttg_default_execution_context(); }

  /// Starts the execution in the given execution context

  /// @param world an execution context associated with the default backend
  /// @note Dispatches to the `ttg_execute` method of the default backend
  inline void execute(World world = default_execution_context()) { TTG_IMPL_NS::ttg_execute(world); }

  /// Returns when all tasks associated with the given execution context have finished on all ranks.

  /// @param world  an execution context associated with the default backend
  /// @note Dispatches to the `ttg_fence` method of the default backend
  /// @note This is a collective operation with respect to @p world
  inline void fence(World world = default_execution_context()) { TTG_IMPL_NS::ttg_fence(world); }

}  // namespace ttg

#endif  // TTG_RUN_H
