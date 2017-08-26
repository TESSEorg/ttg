#include <iostream>
#include <string>
#include <tuple>
#include "flow.h"

void h(int index, const std::tuple<std::string>& args, Flow<double, std::string>& out) {
  out.send<0>(double(index), std::get<0>(args) + "hello");
}

void w(double index, const std::tuple<std::string>& args, Flows<>& out) {
  std::cout << std::get<0>(args) << " world!" << std::endl;
}

void h2(int index, const std::string& args, Flow<double, std::string>& out) {
  out.send<0>(double(index), args + "hello");
}

void w2(double index, const std::string& args, Flows<>& out) { std::cout << args << " world2!" << std::endl; }

int main() {
  BaseOp::set_trace(true);
  Flow<int, std::string> control;
  Flow<double, std::string> pipe, pipe2;
  Flows<> nothing;

  auto hello = make_optuple_wrapper(&h, control, pipe, "hello");
  auto world = make_optuple_wrapper(&w, pipe, nothing, "world");

  auto hello2 = make_op_wrapper(&h2, control, pipe2, "hello");
  auto world2 = make_op_wrapper(&w2, pipe, nothing, "world");

  control.send(0, "");

  return 0;
}
