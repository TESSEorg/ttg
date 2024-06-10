#include <ttg.h>

using namespace ttg;

  int main(int argc, char *argv[]) {
    ttg::initialize(argc, argv);

    auto tt = ttg::make_tt([]() { std::cout << "Hello, World!\n"; });

    ttg::make_graph_executable(tt);
    ttg::execute();
    if (ttg::get_default_world().rank() == 0) tt->invoke();
    ttg::fence();

    ttg::finalize();
    return 0;
}
