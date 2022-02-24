#include <catch2/catch.hpp>

#include "ttg.h"

#include <memory>

// {task_id,data} = {void, void}
namespace tt_v_v {

  class tt : public ttg::TT<void, std::tuple<>, tt, ttg::typelist<void>> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"void"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    void op(baseT::output_terminals_type &outs) {}

    ~tt() {}
  };
}  // namespace tt_v_v

// {task_id,data} = {int, void}
namespace tt_i_v {

  class tt : public ttg::TT<int, std::tuple<>, tt, ttg::typelist<void>> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"void"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    void op(const int &task_id, baseT::output_terminals_type &outs) {}

    ~tt() {}
  };
}  // namespace tt_i_v

// {task_id,data} = {void, int}
namespace tt_v_i {

  class tt : public ttg::TT<void, std::tuple<>, tt, ttg::typelist<const int>> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"int"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    void op(const baseT::input_refs_tuple_type &data, baseT::output_terminals_type &outs) {}

    ~tt() {}
  };

}  // namespace tt_v_i

// {task_id,data} = {void, int, void}
namespace tt_v_iv {

  class tt : public ttg::TT<void, std::tuple<>, tt, ttg::typelist<const int, void>> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"int", "void"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    void op(const baseT::input_refs_tuple_type &data, baseT::output_terminals_type &outs) {}

    ~tt() {}
  };

}  // namespace tt_v_iv

// {task_id,data} = {int, int}
namespace tt_i_i {

  class tt : public ttg::TT<int, std::tuple<>, tt, ttg::typelist<const int>> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"int"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    void op(const int &key, const baseT::input_refs_tuple_type &data, baseT::output_terminals_type &outs) {}

    ~tt() {}
  };
}  // namespace tt_i_i

// {task_id,data} = {int, T}; templated op
namespace tt_i_t {

  class tt : public ttg::TT<int, std::tuple<>, tt, ttg::typelist<const int>> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"int"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    template<typename InputTupleT, typename OutputTupleT>
    void op(const int &key, const InputTupleT &data, OutputTupleT &outs) {
      static_assert(std::is_const_v<std::remove_reference_t<std::tuple_element_t<0, InputTupleT>>>, "Const input type must be const!");
    }

    ~tt() {}
  };
}  // namespace tt_i_i

// {task_id,data} = {int, int, void}
namespace tt_i_iv {

  class tt : public ttg::TT<int, std::tuple<>, tt, ttg::typelist<const int, void>> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"int", "void"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    void op(const int &key, const baseT::input_refs_tuple_type &data, baseT::output_edges_type &outs) {}

    ~tt() {}
  };
}  // namespace tt_i_iv


TEST_CASE("TemplateTask", "[core]") {
  SECTION("constructors") {
    {  // void task id, void data
      ttg::Edge<void, void> in;
      CHECK_NOTHROW(std::make_unique<tt_v_v::tt>(ttg::edges(in), ttg::edges(), ""));

      // compilation error: can't deduce task_id
      // CHECK_NOTHROW(ttg::make_tt([](std::tuple<>& outs) {}, ttg::edges(in), ttg::edges()));
      // compilation error: can't deal with generic lambdas
      // CHECK_NOTHROW(ttg::make_tt([](auto& outs) {}, ttg::edges(in), ttg::edges()));
      // compilation error: outs must be passed by nonconst ref
      // CHECK_NOTHROW(ttg::make_tt([](const std::tuple<>& outs) {}, ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(ttg::make_tt<void>([](std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
    }
    {  // nonvoid task id, void data
      ttg::Edge<int, void> in;
      CHECK_NOTHROW(std::make_unique<tt_i_v::tt>(ttg::edges(in), ttg::edges(), ""));

      // compilation error: must pass task_id by const lvalue ref
      // CHECK_NOTHROW(ttg::make_tt([](int task_id, std::tuple<>& outs) {}, ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(ttg::make_tt([](const int &task_id, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
    }
    {  // void task id, nonvoid data
      ttg::Edge<void, int> in;
      CHECK_NOTHROW(std::make_unique<tt_v_i::tt>(ttg::edges(in), ttg::edges(), ""));
      CHECK_NOTHROW(ttg::make_tt<void>([](const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));

      ttg::Edge<void, void> in2;
      CHECK_NOTHROW(std::make_unique<tt_v_iv::tt>(ttg::edges(in, in2), ttg::edges(), ""));
      CHECK_NOTHROW(ttg::make_tt<void>([](const int &datum, std::tuple<> &outs) {}, ttg::edges(in, in2), ttg::edges()));
    }
    {  // nonvoid task id, nonvoid data
      ttg::Edge<int, int> in;
      CHECK_NOTHROW(std::make_unique<tt_i_i::tt>(ttg::edges(in), ttg::edges(), ""));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
    }
    {  // nonvoid task id, nonvoid data, generic operator
      ttg::Edge<int, int> in;
      CHECK_NOTHROW(std::make_unique<tt_i_t::tt>(ttg::edges(in), ttg::edges(), ""));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, auto &datum, auto &outs) {
            static_assert(std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Const input edge type expected to be const!");
          }, ttg::edges(ttg::make_const(in)), ttg::edges()));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, const auto &datum, auto &outs) {
            static_assert(std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Const input edge type expected to be const!");
          }, ttg::edges(ttg::make_const(in)), ttg::edges()));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, auto &&datum, auto &outs) {
            static_assert(std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Const input edge type expected to be const!");
          }, ttg::edges(ttg::make_const(in)), ttg::edges()));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, auto &datum, auto &outs) {
            static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Nonconst datum expected");
          }, ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, const auto &datum, auto &outs) {
            static_assert(std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Const datum expected");
          }, ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, auto &&datum, auto &outs) {
            static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Nonconst datum expected");
          }, ttg::edges(in), ttg::edges()));
    }
  }
}
