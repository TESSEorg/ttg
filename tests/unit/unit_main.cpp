#define CATCH_CONFIG_RUNNER

#include <catch2/catch_all.hpp>

#include <clocale>
#include <iostream>

#ifdef TTG_EXECUTABLE
#include <ttg.h>
#include <ttg/util/bug.h>
#include <ttg/env.h>
#endif

int main(int argc, char** argv) {
  Catch::Session session;

  // global setup...
  std::cout.precision(std::numeric_limits<double>::max_digits10);
  std::cerr.precision(std::numeric_limits<double>::max_digits10);
  std::wcout.precision(std::numeric_limits<double>::max_digits10);
  std::wcerr.precision(std::numeric_limits<double>::max_digits10);
  std::wcout.sync_with_stdio(false);
  std::wcerr.sync_with_stdio(false);
  std::wcout.sync_with_stdio(true);
  std::wcerr.sync_with_stdio(true);

#ifdef TTG_EXECUTABLE
  // ttg::launch_lldb();
  // ttg::launch_gdb();
  ttg::initialize(argc, argv);
  ttg::diagnose_off();  // turn off diagnostics

  const auto nranks = ttg::default_execution_context().size();
  if (session.config().verbosity() != Catch::Verbosity::Quiet)
    std::cout << "ready to run TTG unit tests with " << nranks << " rank" << (nranks > 1 ? "s" : "") << std::endl;

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
