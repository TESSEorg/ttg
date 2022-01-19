#include <cstdio>
#include <iostream>
#include <cassert>
#include "ttg.h"

using namespace ttg;

using keyT = int;

struct Control {
  template <typename Archive>
  void serialize(Archive& ar) {}
};

int generator(const keyT& k) {
  if (k < 100)
    return k+1;
}

//Needed just to kickstart the computation
auto make_initiator(Edge<keyT, Control>& ctl) {
  auto f = [](const keyT& key, std::tuple<Out<keyT, Control>>& out) {
             send<0>(key, Control(), out);
           };

  return make_tt<keyT>(f, edges(), edges(ctl), "initiator", {}, {"control"});
}

//Input is the pull edge which will pull data from the generator function.
//Caveat - Ops cannot have only pull edges, hence we need a Control edge here.
auto make_func(Edge<keyT, Control>& ctl, Edge<keyT, int>& input, Edge<keyT, int>& output) {
  auto f = [](const keyT &key, Control dummy, int value, std::tuple<Out<keyT, Control>, Out<keyT, int>>&out) {
             std::cout << "Pulled : " << value << std::endl;
             send<1>(key, value, out);
             if (value < 100)
               send<0>(key+1, Control(), out);
           };

  return make_tt(f, edges(ctl, input), edges(ctl, output), "func",
                 {"control","input"}, {"recur","output"});
}

auto make_output(Edge<keyT, int>& input) {
  auto f = [](const keyT &key, int value, std::tuple<>& out) {
             std::cout << "Consumed : " << value << std::endl;
           };

  return make_tt(f, edges(input), edges(), "consumer",
                 {"input"}, {});
}

int main(int argc, char** argv) {

  ttg_initialize(argc, argv, -1);

  int world_size = ttg_default_execution_context().size();
  auto keymap = [world_size](const keyT &key) {
                  return key % world_size;
                };

  Edge<keyT, Control> ctl("control");
  Edge<keyT, int> input("input",true,generator,
                        [](const keyT& key) { return key; },
                        keymap);
  Edge<keyT, int> output("output");

  auto init = make_initiator(ctl);
  auto func = make_func(ctl, input, output);
  auto out = make_output(output);

  init->set_keymap(keymap);
  func->set_keymap(keymap);
  out->set_keymap(keymap);

  auto connected = make_graph_executable(init.get());

  assert(connected);
  TTGUNUSED(connected);

  if (ttg::default_execution_context().rank() == 0) {
    std::cout << Dot()(init.get()) << std::endl;
    init->invoke(0);
  }

  execute();
  fence();

  ttg_finalize();
  return 0;
}
