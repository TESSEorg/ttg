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

// {task_id,data} = {int, int, void}
namespace tt_i_i_p {

  struct Policy : public ttg::TTPolicyBase<int> {

    Policy() : TTPolicyBase() {}

    int procmap(const int&) const { return 0; }

  };

  class tt : public ttg::TT<int, std::tuple<>, tt, ttg::typelist<const int>, Policy> {
    using baseT = typename tt::ttT;

   public:
    tt(const typename baseT::input_edges_type &inedges, const typename baseT::output_edges_type &outedges,
       const std::string &name)
        : baseT(inedges, outedges, name, {"int"}, {}, Policy()) {}

    static constexpr const bool have_cuda_op = false;

    void op(const int &key, const baseT::input_refs_tuple_type &data, baseT::output_terminals_type &outs) {}

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
    {  // nonvoid task id, nonvoid data, w/ policies
      ttg::Edge<int, int> in;
      CHECK_NOTHROW(std::make_unique<tt_i_i_p::tt>(ttg::edges(in), ttg::edges(), ""));
      CHECK_NOTHROW(
          ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges(),
                       tt_i_i_p::Policy()));

          auto tt = ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges(),
                                 tt_i_i_p::Policy());
          auto policy = tt->get_policy();
          auto procmap = tt->get_procmap();
    }
  }

  SECTION("policies") {
    { // default policy
      ttg::Edge<int, int> in;
      auto tt = ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges());
      auto policy = tt->get_policy();
      CHECK(tt->procmap(0) == 0);
      CHECK(tt->get_procmap()(0) == 0);
      CHECK(policy.procmap(0) == 0);
    }
    { // custom procmap
      ttg::Edge<int, int> in;
      auto tt = ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges(),
                             ttg::make_policy([](const int& key){ return key*1000; }));
      auto policy = tt->get_policy();
      CHECK(tt->procmap(1) == 1000);
      CHECK(tt->get_procmap()(2) == 2000);
      CHECK(policy.procmap(3) == 3000);
      tt->set_priomap([](const int&){ return 0; });
    }
    { // custom all maps
      ttg::Edge<int, int> in;
      auto tt = ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges(),
                             ttg::make_policy([](const int& key){ return key*1000; },
                                              [](const int& key){ return key*1000; },
                                              [](const int& key){ return key*1000; }));
      auto policy = tt->get_policy();
      CHECK(policy.procmap(1) == 1000);
      CHECK(tt->procmap(2) == 2000);
      CHECK(tt->get_procmap()(3) == 3000);

      CHECK(tt->priomap(1) == 1000);
      CHECK(tt->get_priomap()(2) == 2000);

      CHECK(tt->inlinemap(1) == 1000);
      CHECK(tt->get_inlinemap()(2) == 2000);
    }
    { // custom all maps static
      ttg::Edge<int, int> in;
      auto tt = ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges(),
                             ttg::make_static_policy([](const int& key){ return key*1000; },
                                                     [](const int& key){ return key*1000; },
                                                     [](const int& key){ return key*1000; }));
      auto policy = tt->get_policy();
      CHECK(policy.procmap(1) == 1000);
      CHECK(tt->procmap(2) == 2000);
      CHECK(tt->get_procmap()(3) == 3000);

      CHECK(tt->priomap(1) == 1000);
      CHECK(tt->get_priomap()(2) == 2000);

      CHECK(tt->inlinemap(1) == 1000);
      CHECK(tt->get_inlinemap()(2) == 2000);
    }
    { // custom all maps static, void key
      ttg::Edge<void, int> in;
      auto tt = ttg::make_tt([](const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges(),
                             ttg::make_static_policy([](){ return 1000; },
                                                     [](){ return 2000; },
                                                     [](){ return 3000; }));
      auto policy = tt->get_policy();
      CHECK(policy.procmap() == 1000);
      CHECK(tt->procmap() == 1000);
      CHECK(tt->get_procmap()() == 1000);

      CHECK(tt->priomap() == 2000);
      CHECK(tt->get_priomap()() == 2000);

      CHECK(tt->inlinemap() == 3000);
      CHECK(tt->get_inlinemap()() == 3000);
    }
  }
}
