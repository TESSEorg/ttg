#include <ttg.h>
#include <ttg/serialization/std/pair.h>

const double threshold = 100.0;
using Key2 = std::pair<int, int>;

namespace std {
  std::ostream &operator<<(std::ostream &os, const Key2 &key) {
    os << "{" << std::get<0>(key) << ", " << std::get<1>(key) << "}";
    return os;
  }
}  // namespace std

static void b(const Key2 &key, const double &input, std::tuple<ttg::Out<int, double>> &out) {
  ttg::print("Called task B(", key, ") on rank", ttg::ttg_default_execution_context().rank(), "with input data ", input);
  /**! \link ttg::send() */  ttg::send/**! \endlink */<0>(std::get<0>(key), input + 1.0, out);
}

static void c(const int &k, const double &sum, std::tuple<ttg::Out<int, double>> &out) {
  ttg::print("Called task C(", k, ") on rank", ttg::ttg_default_execution_context().rank(), "with input ", sum);
  if (sum < threshold) {
    ttg::print("  ", sum, "<", threshold, " so continuing to iterate");
    /**! \link ttg::send() */
    ttg::send/**! \endlink */<0>(k + 1, sum, out);
  } else {
    ttg::print("  ", sum, ">=", threshold, " so stopping the iterations");
  }
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv, -1);

  ttg::Edge<Key2, double> A_B("A(k)->B(k, i)");
  ttg::Edge<int, double> B_C("B(k, i)->C(k)");
  ttg::Edge<int, double> C_A("C(k)->A(k)");

  auto wc(ttg::make_tt(c, ttg::edges(B_C), ttg::edges(C_A), "C", {"From B"}, {"to A"}));
  
  /**! \link TT::set_input_reducer() */wc->set_input_reducer/**! \endlink */<0>([](double &a, const double &b) { a += b; });

  auto wa(ttg::make_tt([&](const int &k, const double &input, std::tuple<ttg::Out<Key2, double>> &out) {
      ttg::print("Called task A(", k, ") on rank", ttg::ttg_default_execution_context().rank());
      wc->set_argstream_size<0>(k, k+1);
      for(int i = 0; i < k+1; i++) {
        /**! \link ttg::send() */
          ttg::send/**! \endlink */<0>(Key2{k, i}, 1.0 + k + input, out);
      }
    }, ttg::edges(C_A), ttg::edges(A_B), "A", {"from C"}, {"to B"}));

  auto wb(ttg::make_tt(b, ttg::edges(A_B), ttg::edges(B_C), "B", {"from A"}, {"to C"}));

  wa->set_keymap([&](const int &k) { return 0; });
  wb->set_keymap([&](const Key2 &k) { return std::get<1>(k) % wb->get_world().size(); });
  wc->set_keymap([&](const int &k) { return 0; });

  ttg::make_graph_executable(wa);

  if (wa->get_world().rank() == 0) wa->invoke(0, 0.0);

  ttg::execute();
  ttg::fence(ttg::get_default_world());

  ttg::finalize();
  return EXIT_SUCCESS;
}

/**
 * \example distributed.cc
 * This is the iterative diamond DAG with variable number of inputs using the reducing
 * terminals of Template Task Graph, adapted to run in distributed: iteratively, a reducing 
 * diamond of data-dependent width is run, until the amount of data gathered at the bottom 
 * of the diamond exceeds a given threshold. First and last tasks of each diamond are run
 * on rank 0, while the tasks inside the diamond are distributed between the ranks in a
 * round-robin fashion.
 */
