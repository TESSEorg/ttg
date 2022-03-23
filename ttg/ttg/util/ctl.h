//
// Created by Eduard Valeyev on 3/5/21.
//

#ifndef TTG_UTIL_CTL_H
#define TTG_UTIL_CTL_H

class Control : public TT<void, std::tuple<ttg::Out<>>, Control> {
  using baseT = TT<void, std::tuple<ttg::Out<>>, Control>;

 public:
  Control(ttg::Edge<> &ctl) : baseT(ttg::edges(), edges(ctl), "Control", {}, {"ctl"}) {}

  void op(std::tuple<ttg::Out<>> &out) { ::ttg::send<0>(out); }

  void start() { invoke(); }
};

#endif  // TTG_UTIL_CTL_H
