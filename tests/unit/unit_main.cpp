#define CATCH_CONFIG_RUNNER

#include <catch2/catch.hpp>

#include <clocale>
#include <iostream>

#ifdef TTG_EXECUTABLE
#include <ttg.h>
#endif

#include "ttg/util/bug.h"

int main(int argc, char** argv) {
  Catch::Session session;

  // global setup...
  std::setlocale(LC_ALL, "en_US.UTF-8");
  std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cerr.precision(std::numeric_limits<double>::max_digits10);
  std::wcout.precision(std::numeric_limits<double>::max_digits10);
  std::wcerr.precision(std::numeric_limits<double>::max_digits10);
  std::wcout.sync_with_stdio(false);
  std::wcerr.sync_with_stdio(false);
  std::wcout.imbue(std::locale("en_US.UTF-8"));
  std::wcerr.imbue(std::locale("en_US.UTF-8"));
  std::wcout.sync_with_stdio(true);
  std::wcerr.sync_with_stdio(true);

#ifdef TTG_EXECUTABLE
  ttg::initialize(argc, argv);
  ttg::diagnose_off();  // turn off diagnostics

  const auto nranks = ttg::default_execution_context().size();
  std::cout << "ready to run TTG unit tests with " << nranks << " rank" << (nranks > 1 ? "s" : "") << std::endl;

  if (const auto* ttg_debugger_str = std::getenv("TTG_DEBUGGER")) {
    using mpqc::Debugger;
    auto debugger = std::make_shared<Debugger>();
    Debugger::set_default_debugger(debugger);
    debugger->set_exec(argv[0]);
    debugger->set_prefix(ttg::default_execution_context().rank());
    debugger->set_cmd("gdb_xterm");
  }
  ttg::execute();
#endif

  int result = session.run(argc, argv);

  // global clean-up...
#ifdef TTG_EXECUTABLE
  ttg::fence();
  ttg::finalize();
#endif

  return result;
}
