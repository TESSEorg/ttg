#include <catch2/catch.hpp>

#include "ttg.h"

#include <memory>

#include "ttg/util/meta/callable.h"

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

    template <typename InputTupleT, typename OutputTupleT>
    void op(const int &key, const InputTupleT &data, OutputTupleT &outs) {
      static_assert(std::is_const_v<std::remove_reference_t<std::tuple_element_t<0, InputTupleT>>>,
                    "Const input type must be const!");
    }

    ~tt() {}
  };
}  // namespace tt_i_t

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

// {task_id,data} = {int, aggregator}
namespace tt_i_i_a {

  class tt : public ttg::TT<int, std::tuple<>, tt, ttg::typelist<ttg::Aggregator<int>>> {
    using baseT = typename TT::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"aggregator<int>"}, {}) {}

    static constexpr const bool have_cuda_op = false;

    void op(const int &key, const baseT::input_refs_tuple_type &data, baseT::output_terminals_type &outs) {
      static_assert(ttg::detail::is_aggregator_v<std::decay_t<std::tuple_element_t<0, baseT::input_refs_tuple_type>>>);
    }

    ~tt() {}
  };
}  // namespace tt_i_i_a

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
      ttg::Edge<int, int> out;
      CHECK_NOTHROW(std::make_unique<tt_i_t::tt>(ttg::edges(in), ttg::edges(), ""));

      // same, but using generic lambdas

      // testing generic lambda introspection
      auto func0 = [](auto key, auto &&datum0, auto &datum1, const auto &datum2, auto datum3, auto &outs) {};
      static_assert(std::is_invocable<decltype(func0), int, const float &, const float &, const float &, const float &,
                                      std::tuple<> &>::value);
      // error: auto& does not bind to T&&
      // static_assert(std::is_invocable<decltype(func0), int, float&&, float&&, float&&, float&&, std::tuple<>
      // &>::value);
      static_assert(ttg::meta::is_invocable_typelist_v<
                    decltype(func0), ttg::typelist<int, float &&, const float &, float &&, float &&, std::tuple<> &>>);
      static_assert(std::is_same_v<
                    decltype(compute_arg_binding_types(
                        func0, ttg::typelist<ttg::typelist<int>, ttg::typelist<float &&, const float &>,
                                             ttg::typelist<float &&>, ttg::typelist<float &&, const float &>,
                                             ttg::typelist<float &&, const float &>, ttg::typelist<std::tuple<> &>>{})),
                    ttg::typelist<>>);
      static_assert(
          std::is_same_v<
              decltype(compute_arg_binding_types(
                  func0, ttg::typelist<ttg::typelist<int>, ttg::typelist<float &&, const float &>,
                                       ttg::typelist<float &&, const float &>, ttg::typelist<float &&, const float &>,
                                       ttg::typelist<float &&, const float &>, ttg::typelist<std::tuple<> &>>{})),
              ttg::typelist<int, float &&, const float &, float &&, float &&, std::tuple<> &>>);
      // voids are skipped
      static_assert(
          std::is_same_v<
              decltype(compute_arg_binding_types(
                  func0, ttg::typelist<ttg::typelist<int>, ttg::typelist<void>, ttg::typelist<float &&, const float &>,
                                       ttg::typelist<float &&, const float &>, ttg::typelist<float &&, const float &>,
                                       ttg::typelist<float &&, const float &>, ttg::typelist<void, std::tuple<> &>,
                                       ttg::typelist<void>>{})),
              ttg::typelist<int, void, float &&, const float &, float &&, float &&, std::tuple<> &, void>>);

      CHECK_NOTHROW(ttg::make_tt(
          [](const int &key, auto &datum, auto &outs) {
            static_assert(std::is_lvalue_reference_v<decltype(datum)>, "Lvalue datum expected");
            static_assert(std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Const datum expected");
          },
          ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(ttg::make_tt(
          [](const int &key, const auto &datum, auto &outs) {
            static_assert(std::is_lvalue_reference_v<decltype(datum)>, "Lvalue datum expected");
            static_assert(std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Const datum expected");
          },
          ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(ttg::make_tt(
          [](const int &key, auto &&datum, auto &outs) {
            static_assert(std::is_rvalue_reference_v<decltype(datum)>, "Rvalue datum expected");
            static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Nonconst datum expected");
          },
          ttg::edges(in), ttg::edges()));

      // and without an output terminal
      CHECK_NOTHROW(ttg::make_tt(
          [](const int &key, auto &datum) {
            static_assert(std::is_lvalue_reference_v<decltype(datum)>, "Lvalue datum expected");
            static_assert(std::is_const_v<std::remove_reference_t<decltype(datum)>>, "Const datum expected");
            ttg::send(0, std::decay_t<decltype(key)>{}, std::decay_t<decltype(datum)>{});
          },
          ttg::edges(in), ttg::edges()));
    }
    {  // nonvoid task id, aggregator input
      ttg::Edge<int, int> in;
      size_t count = 16;
      CHECK_NOTHROW(std::make_unique<tt_i_i_a::tt>(ttg::edges(ttg::make_aggregator(in)), ttg::edges(), ""));
      CHECK_NOTHROW(
          ttg::make_tt(
            [](const int &key, const ttg::Aggregator<int> &datum, std::tuple<> &outs) {
              for (auto&& v : datum)
              { }

              for (const auto& v : datum)
              { }
            }, ttg::edges(ttg::make_aggregator(in)), ttg::edges()));
      CHECK_NOTHROW(
          ttg::make_tt(
            [](const int &key, ttg::Aggregator<int> &&datum, std::tuple<> &outs) {
              for (auto&& v : datum)
              { }

              for (const auto& v : datum)
              { }
            }, ttg::edges(ttg::make_aggregator(in)), ttg::edges()));
    }
  }
}
