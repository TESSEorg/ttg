#include <ttg.h>

static void a(std::tuple<ttg::Out<int, double>> &out) {
  ttg::print("Called task A ");

  /**! \link ttg::send() */  ttg::send/**! \endlink */<0>(0, 1.0, out);

  /**! \link ttg::send() */  ttg::send/**! \endlink */<0>(1, 2.0, out);
}

static void b(const int &key, const double &input, std::tuple<ttg::Out<void, double>, ttg::Out<void, double>> &out) {
  ttg::print("Called task B(", key, ") with input data ", input);
  if (key == 0)
    /**! \link ttg::sendv() */  ttg::sendv/**! \endlink */<0>(input + 1.0, out);
  else
    /**! \link ttg::sendv() */  ttg::sendv/**! \endlink */<1>(input + 1.0, out);
}

static void c(const double &b0, const double &b1, std::tuple<> &out) {
  ttg::print("Called task C with inputs ", b0, " from B(0) and ", b1, " from B(1)");
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv, -1);

  ttg::Edge<int, double> A_B("A->B");
  ttg::Edge<void, double> B_C0("B->C0");
  ttg::Edge<void, double> B_C1("B->C1");

  auto wa(ttg::make_tt<void>(a, ttg::edges(), ttg::edges(A_B), "A", {}, {"to B"}));
  auto wb(ttg::make_tt(b, ttg::edges(A_B), ttg::edges(B_C0, B_C1), "B", {"from A"}, {"to 1st input of C", "to 2nd input of C"}));
  auto wc(ttg::make_tt(c, ttg::edges(B_C0, B_C1), ttg::edges(), "C", {"From B", "From B"}, {}));

  ttg::make_graph_executable(wa);

  if (wa->get_world().rank() == 0) wa->invoke();

  ttg::execute();
  ttg::fence(ttg::get_default_world());

  ttg::finalize();
  return EXIT_SUCCESS;
}

/**
 * \example simple.cc
 * This is the first example of a simple diamond DAG using Template Task Graph.
 */
