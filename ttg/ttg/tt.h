#ifndef TTG_TT_H
#define TTG_TT_H

#include "ttg/config.h"
#include "ttg/fwd.h"

#include "ttg/base/tt.h"
#include "ttg/edge.h"

#ifdef TTG_HAVE_COROUTINE
#include "ttg/coroutine.h"
#endif

#include <cassert>
#include <memory>
#include <vector>

namespace ttg {

  // TODO describe TT concept (preferably as a C++20 concept)
  // N.B. TT::op returns void or ttg::coroutine_handle<>
  // see TTG_PROCESS_TT_OP_RETURN below

  /// @brief a template task graph implementation

  /// It contains (owns) one or more TT objects. Since it can also be viewed as a TT object itself,
  /// it is a TTBase and can be for recursive composition of TTG objects.
  /// @tparam input_terminalsT a tuple of pointers to input terminals
  /// @tparam output_terminalsT a tuple of pointers to output terminals
  template <typename input_terminalsT, typename output_terminalsT>
  class TTG : public TTBase {
   public:
    static constexpr int numins = std::tuple_size_v<input_terminalsT>;    // number of input arguments
    static constexpr int numouts = std::tuple_size_v<output_terminalsT>;  // number of outputs or results

    using input_terminals_type = input_terminalsT;
    using output_terminals_type = output_terminalsT;

   private:
    std::vector<std::unique_ptr<TTBase>> tts;
    input_terminals_type ins;
    output_terminals_type outs;

    // not copyable
    TTG(const TTG &) = delete;
    TTG &operator=(const TTG &) = delete;
    // movable
    TTG(TTG &&other)
        : TTBase(static_cast<TTBase &&>(other))
        , tts(other.tts)
        , ins(std::move(other.ins))
        , outs(std::move(other.outs)) {
      is_ttg_ = true;
      own_my_tts();
    }
    TTG &operator=(TTG &&other) {
      static_cast<TTBase &>(*this) = static_cast<TTBase &&>(other);
      is_ttg_ = true;
      tts = std::move(other.tts);
      ins = std::move(other.ins);
      outs = std::move(other.outs);
      own_my_tts();
      return *this;
    };

   public:
    /// @tparam ttseqT a sequence of std::unique_ptr<TTBase>
    template <typename ttseqT>
    TTG(ttseqT &&tts,
        const input_terminals_type &ins,    // tuple of pointers to input terminals
        const output_terminals_type &outs,  // tuple of pointers to output terminals
        const std::string &name = "ttg")
        : TTBase(name, numins, numouts), tts(std::forward<ttseqT>(tts)), ins(ins), outs(outs) {
      if (this->tts.size() == 0) throw name + ":TTG: need to wrap at least one TT";  // see fence

      set_terminals(ins, &TTG<input_terminalsT, output_terminalsT>::set_input);
      set_terminals(outs, &TTG<input_terminalsT, output_terminalsT>::set_output);
      is_ttg_ = true;
      own_my_tts();

      // traversal is still broken ... need to add checking for composite
    }

    /// Return a pointer to i'th input terminal
    template <std::size_t i>
    auto in() {
      return std::get<i>(ins);
    }

    /// Return a pointer to i'th output terminal
    template <std::size_t i>
    auto out() {
      return std::get<i>(outs);
    }

    TTBase *get_op(std::size_t i) { return tts.at(i).get(); }

    ttg::World get_world() const override final { return tts[0]->get_world(); }

    void fence() override { tts[0]->fence(); }

    void make_executable() override {
      for (auto &op : tts) op->make_executable();
    }

   private:
    void own_my_tts() const {
      for (auto &op : tts) op->owning_ttg = this;
    }
  };

  template <typename ttseqT, typename input_terminalsT, typename output_terminalsT>
  auto make_ttg(ttseqT &&tts, const input_terminalsT &ins, const output_terminalsT &outs,
                const std::string &name = "ttg") {
    return std::make_unique<TTG<input_terminalsT, output_terminalsT>>(std::forward<ttseqT>(tts), ins, outs, name);
  }

  /// A data sink for one input
  template <typename keyT, typename input_valueT>
  class SinkTT : public TTBase {
    static constexpr int numins = 1;
    static constexpr int numouts = 0;

    using input_terminals_type = std::tuple<ttg::In<keyT, input_valueT>>;
    using input_edges_type = std::tuple<ttg::Edge<keyT, std::decay_t<input_valueT>>>;
    using output_terminals_type = std::tuple<>;

   private:
    input_terminals_type input_terminals;
    output_terminals_type output_terminals;

    SinkTT(const SinkTT &other) = delete;
    SinkTT &operator=(const SinkTT &other) = delete;
    SinkTT(SinkTT &&other) = delete;
    SinkTT &operator=(SinkTT &&other) = delete;

    template <typename terminalT>
    void register_input_callback(terminalT &input) {
      using valueT = std::decay_t<typename terminalT::value_type>;
      auto move_callback = [](const keyT &key, valueT &&value) {};
      auto send_callback = [](const keyT &key, const valueT &value) {};
      auto broadcast_callback = [](const ttg::span<const keyT> &key, const valueT &value) {};
      auto setsize_callback = [](const keyT &key, std::size_t size) {};
      auto finalize_callback = [](const keyT &key) {};

      input.set_callback(send_callback, move_callback, broadcast_callback, setsize_callback, finalize_callback);
    }

   public:
    SinkTT(const std::string &inname = "junk") : TTBase("sink", numins, numouts) {
      register_input_terminals(input_terminals, std::vector<std::string>{inname});
      register_input_callback(std::get<0>(input_terminals));
    }

    SinkTT(const input_edges_type &inedges, const std::string &inname = "junk") : TTBase("sink", numins, numouts) {
      register_input_terminals(input_terminals, std::vector<std::string>{inname});
      register_input_callback(std::get<0>(input_terminals));
      std::get<0>(inedges).set_out(&std::get<0>(input_terminals));
    }

    virtual ~SinkTT() {}

    void fence() override final {}

    void make_executable() override final { TTBase::make_executable(); }

    World get_world() const override final { return get_default_world(); }

    /// Returns pointer to input terminal i to facilitate connection --- terminal cannot be copied, moved or assigned
    template <std::size_t i>
    std::tuple_element_t<i, input_terminals_type> *in() {
      static_assert(i == 0);
      return &std::get<i>(input_terminals);
    }
  };

}  // namespace ttg

#ifndef TTG_PROCESS_TT_OP_RETURN
#ifdef TTG_HAVE_COROUTINE
#define TTG_PROCESS_TT_OP_RETURN(Space, result, id, invoke)                                                           \
  {                                                                                                                   \
    using return_type = decltype(invoke);                                                                             \
    if constexpr (std::is_same_v<return_type, void>) {                                                                \
      invoke;                                                                                                         \
      id = ttg::TaskCoroutineID::Invalid;                                                                             \
    } else {                                                                                                          \
      auto coro_return = invoke;                                                                                      \
      static_assert(std::is_same_v<return_type, void> ||                                                              \
                    std::is_base_of_v<ttg::coroutine_handle<ttg::resumable_task_state>, decltype(coro_return)>||      \
                    std::is_base_of_v<ttg::coroutine_handle<ttg::device::detail::device_task_promise_type<Space>>,    \
                                      decltype(coro_return)>);                                                        \
      if constexpr (std::is_base_of_v<ttg::coroutine_handle<ttg::resumable_task_state>, decltype(coro_return)>)       \
        id = ttg::TaskCoroutineID::ResumableTask;                                                                     \
      else if constexpr (std::is_base_of_v<                                                                           \
                                        ttg::coroutine_handle<ttg::device::detail::device_task_promise_type<Space>>,  \
                                        decltype(coro_return)>)                                                       \
        id = ttg::TaskCoroutineID::DeviceTask;                                                                        \
      else                                                                                                            \
        std::abort();                                                                                                 \
      result = coro_return.address();                                                                                 \
    }                                                                                                                 \
  }
#else
#define TTG_PROCESS_TT_OP_RETURN(Space, result, id, invoke) invoke
#endif
#else
#error "TTG_PROCESS_TT_OP_RETURN already defined in ttg/tt.h, check your header guards"
#endif  // !defined(TTG_PROCESS_TT_OP_RETURN)

#endif  // TTG_TT_H
