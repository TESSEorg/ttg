#include <cstdio>
#include <iostream>
#include <cassert>
#include "ttg.h"

using namespace ttg;

using keyT = int;

int generator(const keyT& k) {
  if (k < 100)
    return k+1;

  return -1;
}

//Needed just to kickstart the computation
auto make_initiator(Edge<keyT, void>& ctl) {
  auto f = [](const keyT& key, std::tuple<Out<keyT, void>>& out) {
             sendk<0>(key, out);
           };

  return make_tt<keyT>(f, edges(), edges(ctl), "initiator", {}, {"control"});
}

//Input is the pull edge which will pull data from the generator function.
//Caveat - Ops cannot have only pull edges, hence we need a Control edge here.
auto make_func(Edge<keyT, void>& ctl, Edge<keyT, int>& input, Edge<keyT, int>& output) {
  auto f = [](const keyT &key, int value, std::tuple<Out<keyT, int>, Out<keyT, void>>&out) {
             std::cout << "Pulled : " << value << std::endl;
             send<0>(key, value, out);
             if (value < 100)
               sendk<1>(key+1, out);
           };

  return make_tt(f, edges(input, ctl), edges(output, ctl), "func",
                 {"input","control"}, {"output","recur"});
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

  Edge<keyT, void> ctl("control");
  Edge<keyT, int> input("input", true, {generator,
                        [](const keyT& key)-> const keyT { return key; },
                                       keymap});
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
