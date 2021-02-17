#ifndef TTG_UTIL_OP_H
#define TTG_UTIL_OP_H

#include <vector>
#include <memory>

#include "ttg/fwd.h"

#include "ttg/base/op.h"
#include "ttg/edge.h"

namespace ttg {

  template <typename input_terminalsT, typename output_terminalsT>
  class CompositeOp : public OpBase {
   public:
    static constexpr int numins = std::tuple_size<input_terminalsT>::value;    // number of input arguments
    static constexpr int numouts = std::tuple_size<output_terminalsT>::value;  // number of outputs or results

    using input_terminals_type = input_terminalsT;    // should be a tuple of pointers to input terminals
    using output_terminals_type = output_terminalsT;  // should be a tuple of pointers to output terminals

   private:
    std::vector<std::unique_ptr<OpBase>> ops;
    input_terminals_type ins;
    output_terminals_type outs;

    CompositeOp(const CompositeOp &) = delete;
    CompositeOp &operator=(const CompositeOp &) = delete;
    CompositeOp(const CompositeOp &&) = delete;  // Move should be OK

   public:
    template <typename opsT>
    CompositeOp(opsT &&ops_take_ownership,
                const input_terminals_type &ins,    // tuple of pointers to input terminals
                const output_terminals_type &outs,  // tuple of pointers to output terminals
                const std::string &name = "compositeop")
        : OpBase(name, numins, numouts), ops(std::forward<opsT>(ops_take_ownership)), ins(ins), outs(outs) {
      if (ops.size() == 0) throw name + ":CompositeOp: need to wrap at least one op";  // see fence

      set_is_composite(true);
      for (auto &op : ops) op->set_is_within_composite(true, this);
      set_terminals(ins, &CompositeOp<input_terminalsT, output_terminalsT>::set_input);
      set_terminals(outs, &CompositeOp<input_terminalsT, output_terminalsT>::set_output);

      // traversal is still broken ... need to add checking for composite
    }

    /// Return a pointer to i'th input terminal
    template <std::size_t i>
    typename std::tuple_element<i, input_terminals_type>::type in() {
      return std::get<i>(ins);
    }

    /// Return a pointer to i'th output terminal
    template <std::size_t i>
    typename std::tuple_element<i, output_terminalsT>::type out() {
      return std::get<i>(outs);
    }

    OpBase *get_op(std::size_t i) { return ops.at(i).get(); }

    void fence() { ops[0]->fence(); }

    void make_executable() {
      for (auto &op : ops) op->make_executable();
    }
  };

  template <typename opsT, typename input_terminalsT, typename output_terminalsT>
  std::unique_ptr<CompositeOp<input_terminalsT, output_terminalsT>> make_composite_op(
      opsT &&ops, const input_terminalsT &ins, const output_terminalsT &outs, const std::string &name = "compositeop") {
    return std::make_unique<CompositeOp<input_terminalsT, output_terminalsT>>(std::forward<opsT>(ops), ins, outs, name);
  }


  /// A data sink for one input
  template <typename keyT, typename input_valueT>
  class OpSink : public OpBase {
    static constexpr int numins = 1;
    static constexpr int numouts = 0;

    using input_terminals_type = std::tuple<ttg::In<keyT, input_valueT>>;
    using input_edges_type = std::tuple<ttg::Edge<keyT, std::decay_t<input_valueT>>>;
    using output_terminals_type = std::tuple<>;

   private:
    input_terminals_type input_terminals;
    output_terminals_type output_terminals;

    OpSink(const OpSink &other) = delete;
    OpSink &operator=(const OpSink &other) = delete;
    OpSink(OpSink &&other) = delete;
    OpSink &operator=(OpSink &&other) = delete;

    template <typename terminalT>
    void register_input_callback(terminalT &input) {
      using valueT = std::decay_t<typename terminalT::value_type>;
      auto move_callback = [](const keyT &key, valueT &&value) {};
      auto send_callback = [](const keyT &key, const valueT &value) {};
      auto setsize_callback = [](const keyT &key, std::size_t size) {};
      auto finalize_callback = [](const keyT &key) {};

      input.set_callback(send_callback, move_callback, setsize_callback, finalize_callback);
    }

  public:
    OpSink(const std::string& inname="junk") : OpBase("sink", numins, numouts) {
      register_input_terminals(input_terminals, std::vector<std::string>{inname});
      register_input_callback(std::get<0>(input_terminals));
    }

    OpSink(const input_edges_type &inedges, const std::string& inname="junk") : OpBase("sink", numins, numouts) {
      register_input_terminals(input_terminals, std::vector<std::string>{inname});
      register_input_callback(std::get<0>(input_terminals));
      std::get<0>(inedges).set_out(&std::get<0>(input_terminals));
    }

    virtual ~OpSink() {}

    void fence() {}

    void make_executable() {
        OpBase::make_executable();
    }

    /// Returns pointer to input terminal i to facilitate connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    typename std::tuple_element<i, input_terminals_type>::type *in() {
      static_assert(i==0);
      return &std::get<i>(input_terminals);
    }
  };


} // namespace ttg

#endif // TTG_UTIL_OP_H
