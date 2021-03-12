//
// Created by Eduard Valeyev on 3/5/21.
//

#ifndef TTG_COMMON_CTL_H
#define TTG_COMMON_CTL_H

/**
 * Ctl provides a control input for a TTG
 */
class Ctl : public Op<void, std::tuple<ttg::Out<>>, Ctl> {
  using baseT = Op<void, std::tuple<ttg::Out<>>, Ctl>;

 public:
  Ctl(ttg::Edge<> &ctl) : baseT(ttg::edges(), edges(ctl), "Ctl", {}, {"ctl"}) {}

  void op(std::tuple<ttg::Out<>> &out) { ::ttg::send<0>(out); }

  void start() { invoke(); }
};

#endif  // TTG_COMMON_CTL_H
