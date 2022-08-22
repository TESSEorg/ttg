//
// Created by Eduard Valeyev on 3/27/22.
//

#include <ttg.h>

int main(int argc, char *argv[]) {
  ttg::initialize(argc, argv);

  auto tt = ttg::make_tt([]() { std::cout << "Hello, World!"; });

  ttg::make_graph_executable(tt);
  ttg::execute();
  if (ttg::get_default_world().rank() == 0) tt->invoke();
  ttg::fence();

  ttg::finalize();
  return 0;
}
