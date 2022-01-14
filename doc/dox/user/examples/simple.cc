//
// Created by herault on 1/14/22.
//
#include <ttg.h>

using namespace ttg;

static void a(std::tuple<Out<int, double>> &out) {
  ttg::print("Called task A ");
  send<0>(0, 1.0, out);
  send<0>(1, 2.0, out);
}

static void b(const int &key, const double &input, std::tuple<Out<void, double>, Out<void, double>> &out) {
  ttg::print("Called task B(", key, ") with input data ", input);
  if (key == 0)
    sendv<0>(input + 1.0, out);
  else
    sendv<1>(input + 1.0, out);
}

static void c(const double &b0, const double &b1, std::tuple<> &out) {
  ttg::print("Called task C with inputs ", b0, " from B(0) and ", b1, " from B(1)");
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv, -1);

  {
    Edge<int, double> A_B("A->B");
    Edge<void, double> B_C0("B->C0");
    Edge<void, double> B_C1("B->C1");

    auto wa(make_tt<void>(a, edges(), edges(A_B), "A", {}, {"to B"}));
    auto wb(make_tt(b, edges(A_B), edges(B_C0, B_C1), "B", {"from A"}, {"to 1st input of C", "to 2nd input of C"}));
    auto wc(make_tt(c, edges(B_C0, B_C1), edges(), "C", {"From B", "From B"}, {}));

    wa->make_executable();
    wb->make_executable();
    wc->make_executable();

    if (wa->get_world().rank() == 0) wa->invoke();

    ttg::execute();
    ttg::fence(ttg::get_default_world());
  }

  ttg::finalize();
  return EXIT_SUCCESS;
}
