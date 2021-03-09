//
// Created by Eduard Valeyev on 3/5/21.
//

#ifndef TTG_COMMON_CTL_H
#define TTG_COMMON_CTL_H

class Control : public Op<void, std::tuple<ttg::Out<>>, Control> {
  using baseT = Op<void, std::tuple<ttg::Out<>>, Control>;

 public:
  Control(ttg::Edge<> &ctl) : baseT(ttg::edges(), edges(ctl), "Control", {}, {"ctl"}) {}

  void op(std::tuple<ttg::Out<>> &out) { ::ttg::send<0>(out); }

  void start() { invoke(); }
};

#endif  // TTG_COMMON_CTL_H
