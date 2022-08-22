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

static void a(const int &k, const double &input, std::tuple<ttg::Out<Key2, double>> &out) {
  ttg::print("Called task A(", k, ")");
  /**! \link ttg::send() */  ttg::send/**! \endlink */<0>(Key2{k, 0}, 1.0 + input, out);
  /**! \link ttg::send() */  ttg::send/**! \endlink */<0>(Key2{k, 1}, 2.0 + input, out);
}

static void b(const Key2 &key, const double &input, std::tuple<ttg::Out<int, double>, ttg::Out<int, double>> &out) {
  ttg::print("Called task B(", key, ") with input data ", input);
  if (std::get<1>(key) == 0)
    /**! \link ttg::send() */
    ttg::send/**! \endlink */<0>(std::get<0>(key), input + 1.0, out);
  else
    /**! \link ttg::send() */
    ttg::send/**! \endlink */<1>(std::get<0>(key), input + 1.0, out);
}

static void c(const int &k, const double &b0, const double &b1, std::tuple<ttg::Out<int, double>> &out) {
  ttg::print("Called task C(", k, ") with inputs ", b0, " from B(", k, " 0) and ", b1, " from B(", k, " 1)");
  if (b0 + b1 < threshold) {
    ttg::print("  ", b0, "+", b1, "<", threshold, " so continuing to iterate");
    /**! \link ttg::send() */
    ttg::send/**! \endlink */<0>(k + 1, b0 + b1, out);
  } else {
    ttg::print("  ", b0, "+", b1, ">=", threshold, " so stopping the iterations");
  }
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv, -1);

  ttg::Edge<Key2, double> A_B("A(k)->B(k)");
  ttg::Edge<int, double> B_C0("B(k)->C0(k)");
  ttg::Edge<int, double> B_C1("B(k)->C1(k)");
  ttg::Edge<int, double> C_A("C(k)->A(k)");

  auto wa(ttg::make_tt(a, ttg::edges(C_A), ttg::edges(A_B), "A", {"from C"}, {"to B"}));
  auto wb(ttg::make_tt(b, ttg::edges(A_B), ttg::edges(B_C0, B_C1), "B", {"from A"}, {"to 1st input of C", "to 2nd input of C"}));
  auto wc(ttg::make_tt(c, ttg::edges(B_C0, B_C1), ttg::edges(C_A), "C", {"From B", "From B"}, {"to A"}));

  ttg::make_graph_executable(wa);

  if (wa->get_world().rank() == 0) wa->invoke(0, 0.0);

  ttg::execute();
  ttg::fence(ttg::get_default_world());

  ttg::finalize();
  return EXIT_SUCCESS;
}

/**
 * \example iterative.cc
 * This is the iterative diamond DAG using Template Task Graph: iteratively, a simple diamond is run,
 * until the amount of data gathered at the bottom of the diamond exceeds a given threshold.
 */
