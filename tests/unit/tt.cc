// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_all.hpp>

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

  template <typename K, typename D1, typename D2>
  void func0(K &key, D1 &datum1, D2 &&datum2) {
    ttg::abort();
  }
}  // namespace tt_i_iv

namespace args_pmf {
  struct X {
    auto f(int &i) { i = 1; }
    template <typename T>
    auto g(T &i) {
      i = 1;
    }
    int i_m;
  };
}  // namespace args_pmf

TEST_CASE("TemplateTask", "[core]") {
  SECTION("constructors") {
    {  // void task id, void data
      ttg::Edge<void, void> in;

      // write TT as a class
      CHECK_NOTHROW(std::make_unique<tt_v_v::tt>(ttg::edges(in), ttg::edges(), ""));

      // ... or wrap a lambda via make_tt

      // N.B. no need anymore to explicitly specify void task id
      CHECK_NOTHROW(ttg::make_tt([](std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
      // generic lambdas OK too
      CHECK_NOTHROW(ttg::make_tt([](auto &outs) {}, ttg::edges(in), ttg::edges()));
      // OK: output terminals passed by nonconst ref
      CHECK_NOTHROW(ttg::make_tt([](std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
      // OK: output terminals are optional
      CHECK_NOTHROW(ttg::make_tt([]() {}, ttg::edges(in), ttg::edges()));

      // compilation error: outs must be passed by nonconst ref
      // CHECK_NOTHROW(ttg::make_tt([](const std::tuple<>& outs) {}, ttg::edges(in), ttg::edges()));
    }
    {  // nonvoid task id, void data
      ttg::Edge<int, void> in;

      // write TT as a class
      CHECK_NOTHROW(std::make_unique<tt_i_v::tt>(ttg::edges(in), ttg::edges(), ""));

      // ... or wrap a lambda via make_tt

      // OK
      CHECK_NOTHROW(ttg::make_tt([](const int &task_id, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
      // compilation error: must pass task_id by const lvalue ref
      // CHECK_NOTHROW(ttg::make_tt([](int task_id, std::tuple<>& outs) {}, ttg::edges(in), ttg::edges()));
      // OK: out-terminals are optional
      CHECK_NOTHROW(ttg::make_tt([](const int &task_id) {}, ttg::edges(in), ttg::edges()));
      // generic lambdas OK too
      CHECK_NOTHROW(ttg::make_tt([](auto &task_id) {}, ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(ttg::make_tt([](auto &task_id, auto &&outs) {}, ttg::edges(in), ttg::edges()));
      // WARNING: mixed (some generic, some not) lambdas can't work, the generic mechanism deduces the last argument to
      // be std::tuple<>&
      CHECK_NOTHROW(ttg::make_tt([](auto &task_id, const std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));

      // compilation error: must pass task_id by const lvalue ref
      CHECK_NOTHROW(ttg::make_tt([](auto &task_id, const std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
    }
    {  // void task id, nonvoid data
      ttg::Edge<void, int> in;
      ttg::Edge<void, void> in2;

      // write TT as a class
      CHECK_NOTHROW(std::make_unique<tt_v_i::tt>(ttg::edges(in), ttg::edges(), ""));
      CHECK_NOTHROW(std::make_unique<tt_v_iv::tt>(ttg::edges(in, in2), ttg::edges(), ""));

      // ... or wrap a lambda via make_tt

      // nongeneric
      CHECK_NOTHROW(ttg::make_tt([](const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(ttg::make_tt([](const int &datum, std::tuple<> &outs) {}, ttg::edges(in, in2), ttg::edges()));
      // generic
      CHECK_NOTHROW(ttg::make_tt([](auto &datum, auto &&outs) {}, ttg::edges(in), ttg::edges()));
      CHECK_NOTHROW(ttg::make_tt([](auto &datum, auto &&outs) {}, ttg::edges(in, in2), ttg::edges()));
      // OK: outterm tuple is optional
      CHECK_NOTHROW(ttg::make_tt([](auto &datum) {}, ttg::edges(in, in2), ttg::edges()));
    }
    {  // nonvoid task id, nonvoid data

      // case 1: data arguments have concrete types
      {
        ttg::Edge<int, int> in;
        // write TT as a class
        CHECK_NOTHROW(std::make_unique<tt_i_i::tt>(ttg::edges(in), ttg::edges(), ""));

        // ... or wrap a lambda via make_tt
        CHECK_NOTHROW(
            ttg::make_tt([](const int &key, const int &datum, std::tuple<> &outs) {}, ttg::edges(in), ttg::edges()));

        // test introspection of nongeneric arguments by the runtime (i.e. contents of TT::input_args_type)
        {
          // N.B. DO NOT mix generic and non-generic data arguments (OK to use either for ID/key and terminals),

          // case A: explicit out-terminals
          {
            auto tt = ttg::make_tt(
                [](const int &key, const int &datum1, int &&datum2, std::tuple<> &outs) {
                  // datum1 is lvalue ref to const, was detected to be READ-ONLY
                  static_assert(std::is_const_v<std::remove_reference_t<decltype(datum1)>>, "Const datum expected");
                  static_assert(std::is_lvalue_reference_v<decltype(datum1)>, "Lvalue datum expected");

                  // datum2 is rvalue ref to nonconst, was detected as CONSUMABLE
                  static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum2)>>, "Nonconst datum expected");
                  static_assert(std::is_rvalue_reference_v<decltype(datum2)>, "Rvalue datum expected");
                },
                ttg::edges(in, in), ttg::edges());
            using tt_t = typename std::remove_reference_t<decltype(*tt)>;
            // `const int&` means READ-ONLY
            static_assert(std::is_const_v<std::tuple_element_t<0, tt_t::input_args_type>>);
            // `int&&` means CONSUMABLE
            static_assert(!std::is_const_v<std::tuple_element_t<1, tt_t::input_args_type>>);
          }
          // case B: implicit out-terminals
          {
            auto tt = ttg::make_tt(
                [](const int &key, const int &datum1, int &&datum2) {
                  // datum1 is lvalue ref to const, was detected to be READ-ONLY
                  static_assert(std::is_const_v<std::remove_reference_t<decltype(datum1)>>, "Const datum expected");
                  static_assert(std::is_lvalue_reference_v<decltype(datum1)>, "Lvalue datum expected");

                  // datum2 is rvalue ref to nonconst, was detected as CONSUMABLE
                  static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum2)>>, "Nonconst datum expected");
                  static_assert(std::is_rvalue_reference_v<decltype(datum2)>, "Rvalue datum expected");
                },
                ttg::edges(in, in), ttg::edges());
            using tt_t = typename std::remove_reference_t<decltype(*tt)>;
            // `const int&` means READ-ONLY
            static_assert(std::is_const_v<std::tuple_element_t<0, tt_t::input_args_type>>);
            // `int&&` means CONSUMABLE
            static_assert(!std::is_const_v<std::tuple_element_t<1, tt_t::input_args_type>>);
          }
        }
      }  // nongeneric data args

      // case 2: data arguments have generic arguments
      {
        ttg::Edge<int, int> in;
        ttg::Edge<int, int> out;

        // write TT as a class
        CHECK_NOTHROW(std::make_unique<tt_i_t::tt>(ttg::edges(in), ttg::edges(), ""));

        // ... or wrap a lambda via make_tt

        // testing generic lambda introspection
        auto func0 = [](auto key, auto &&datum0, auto &datum1, const auto &datum2, auto datum3, auto &outs) {};
        // OK: all of {auto&&, auto&, const auto&} bind to const T&
        static_assert(std::is_invocable<decltype(func0), int, const float &, const float &, const float &,
                                        const float &, std::tuple<> &>::value);
        static_assert(std::is_same_v<std::invoke_result_t<decltype(func0), int, const float &, const float &,
                                                          const float &, const float &, std::tuple<> &>,
                                     void>);
        // OK: ditto
        static_assert(std::is_void_v<decltype(tt_i_iv::func0(std::declval<const int &>(), std::declval<const float &>(),
                                                             std::declval<const float &>()))>);
        // compile error: auto& does not bind to T&&, while {auto&&, const auto&} do
        // static_assert(
        //  std::is_invocable<decltype(func0), int, float &&, float &&, float &&, float &&, std::tuple<> &>::value);
        // compile error: D1& does not bind to T&&, D2&& does
        // static_assert(std::is_void_v<decltype(tt_i_iv::func0(std::declval<const int &>(), std::declval<float &&>(),
        //                                                      std::declval<float &&>()))>);
        // OK: all of {auto&&, auto&, const auto&} bind to T&
        static_assert(
            std::is_invocable<decltype(func0), int, float &, float &, float &, float &, std::tuple<> &>::value);
        // OK: ditto
        static_assert(std::is_void_v<decltype(tt_i_iv::func0(std::declval<const int &>(), std::declval<float &>(),
                                                             std::declval<float &>()))>);

        static_assert(
            ttg::meta::is_invocable_typelist_v<
                decltype(func0), ttg::typelist<int, float &&, const float &, float &&, float &&, std::tuple<> &>>);
        static_assert(
            std::is_same_v<decltype(compute_arg_binding_types(
                               func0,
                               ttg::typelist<ttg::typelist<int>, ttg::typelist<float &&, const float &>,
                                             ttg::typelist<float &&>, ttg::typelist<float &&, const float &>,
                                             ttg::typelist<float &&, const float &>, ttg::typelist<std::tuple<> &>>{})),
                           ttg::typelist<ttg::typelist<>, ttg::typelist<>>>);
        static_assert(
            std::is_same_v<
                decltype(compute_arg_binding_types(
                    func0, ttg::typelist<ttg::typelist<int>, ttg::typelist<float &&, const float &>,
                                         ttg::typelist<float &&, const float &>, ttg::typelist<float &&, const float &>,
                                         ttg::typelist<float &&, const float &>, ttg::typelist<std::tuple<> &>>{})),
                ttg::typelist<ttg::typelist<void>,
                              ttg::typelist<int, float &&, const float &, float &&, float &&, std::tuple<> &>>>);
        // voids are skipped
        static_assert(
            std::is_same_v<
                decltype(compute_arg_binding_types(
                    func0, ttg::typelist<ttg::typelist<int>, ttg::typelist<void>,
                                         ttg::typelist<float &&, const float &>, ttg::typelist<float &&, const float &>,
                                         ttg::typelist<float &&, const float &>, ttg::typelist<float &&, const float &>,
                                         ttg::typelist<void, std::tuple<> &>, ttg::typelist<void>>{})),
                ttg::typelist<ttg::typelist<void>, ttg::typelist<int, void, float &&, const float &, float &&, float &&,
                                                                 std::tuple<> &, void>>>);

        // test introspection of generic arguments by the runtime (i.e. contents of TT::input_args_type) and
        // the deduced types inside the function body

        // case A: explicit out-terminals
        {
          // N.B. NEVER USE const auto& like datum3 here
          // N.N.B. DO NOT mix generic and non-generic data arguments (OK to use either for ID/key and terminals),
          //        here we illustrate the issues that happen
          auto tt = ttg::make_tt(
              [](const int &key, auto &datum1, auto &&datum2, const auto &datum3, const int &datum4, int &&datum5,
                 auto &outs) {
                // `auto&` means READ-ONLY: datum1 is an lvalue ref to const
                static_assert(std::is_const_v<std::remove_reference_t<decltype(datum1)>>, "Const datum expected");
                static_assert(std::is_lvalue_reference_v<decltype(datum1)>, "Lvalue datum expected");

                // `auto&&` means CONSUMABLE: datum2 is rvalue ref to nonconst
                // the runtime will treat it as CONSUMABLE, hence it is safe to move from it
                // N.B. this is due to
                static_assert(std::is_rvalue_reference_v<decltype(datum2)>, "Rvalue datum expected");
                static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum2)>>, "Nonconst datum expected");

                // `const auto&` means CONSUMABLE ...
                // although it LOOKS READ-ONLY from the inside the function body (lvalue ref to const)
                // the runtime will treat it as CONSUMABLE hence it is safe to move from it
                static_assert(std::is_const_v<std::remove_reference_t<decltype(datum3)>>, "Const datum expected");
                static_assert(std::is_lvalue_reference_v<decltype(datum3)>, "Lvalue datum expected");

                // WARNING: DO NOT mix non-generic and generic arguments as it's not possible to detect
                //          binding of non-generic argument correctly!

                // NOT OK: datum4 is lvalue ref to const but it was detected as CONSUMABLE
                static_assert(std::is_const_v<std::remove_reference_t<decltype(datum4)>>, "Const datum expected");
                static_assert(std::is_lvalue_reference_v<decltype(datum4)>, "Lvalue datum expected");

                // OK: datum5 is rvalue ref to nonconst, was detected as CONSUMABLE
                static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum5)>>, "Nonconst datum expected");
                static_assert(std::is_rvalue_reference_v<decltype(datum5)>, "Rvalue datum expected");
              },
              ttg::edges(in, in, in, in, in, in), ttg::edges());
          using tt_t = typename std::remove_reference_t<decltype(*tt)>;
          // `auto&` means READ-ONLY
          static_assert(std::is_const_v<std::tuple_element_t<0, tt_t::input_args_type>>);
          // `auto&&` means CONSUMABLE
          static_assert(!std::is_const_v<std::tuple_element_t<1, tt_t::input_args_type>>);
          // `const auto&` means CONSUMABLE (binds like `auto&&`, not like `auto&`)
          static_assert(!std::is_const_v<std::tuple_element_t<2, tt_t::input_args_type>>);

          // WARNING: DO NOT mix non-generic and generic arguments as it's not possible to detect
          //          binding of non-generic argument correctly!

          // NOT OK: `const int&` SHOULD mean READ-ONLY BUT is detected to be CONSUMABLE
          static_assert(!std::is_const_v<std::tuple_element_t<3, tt_t::input_args_type>>);
          // OK: `int&&` means CONSUMABLE
          static_assert(!std::is_const_v<std::tuple_element_t<4, tt_t::input_args_type>>);
        }

        // case B: implicit out-terminals
        {
          auto tt = ttg::make_tt(
              [](const int &key, auto &datum1, auto &&datum2) {
                static_assert(std::is_lvalue_reference_v<decltype(datum1)>, "Lvalue datum expected");
                static_assert(std::is_const_v<std::remove_reference_t<decltype(datum1)>>, "Const datum expected");
                static_assert(std::is_rvalue_reference_v<decltype(datum2)>, "Rvalue datum expected");
                static_assert(!std::is_const_v<std::remove_reference_t<decltype(datum2)>>, "Nonconst datum expected");
              },
              ttg::edges(in, in), ttg::edges());
          using tt_t = typename std::remove_reference_t<decltype(*tt)>;
          // `auto&` means READ-ONLY
          static_assert(std::is_const_v<std::tuple_element_t<0, tt_t::input_args_type>>);
          // `auto&&` means CONSUMABLE
          static_assert(!std::is_const_v<std::tuple_element_t<1, tt_t::input_args_type>>);
        }
      }
    }
  }  // SECTION("constuctors")

  SECTION("args_t") {
    int i = 0;
    auto f = [&i](int &j) { j = i; };
    auto g = [&i](auto &j) { j = i; };
    static_assert(!ttg::meta::is_generic_callable_v<decltype(f)>);
    static_assert(ttg::meta::is_generic_callable_v<decltype(g)>);
    auto [f_is_generic, f_args_t_v] = ttg::meta::callable_args<decltype(f)>;
    CHECK(!f_is_generic);
    static_assert(std::is_same_v<decltype(f_args_t_v), std::pair<ttg::typelist<void>, ttg::typelist<int &>>>);
    auto [g_is_generic, g_args_t_v] = ttg::meta::callable_args<decltype(g)>;
    CHECK(g_is_generic);
    static_assert(std::is_same_v<decltype(g_args_t_v), std::pair<ttg::typelist<>, ttg::typelist<>>>);

    {
      static_assert(!ttg::meta::is_generic_callable_v<decltype(&args_pmf::X::f)>);
      static_assert(std::is_same_v<boost::callable_traits::args_t<decltype(&args_pmf::X::f), ttg::typelist>,
                                   ttg::typelist<args_pmf::X &, int &>>);
      static_assert(!ttg::meta::is_generic_callable_v<decltype(&args_pmf::X::g<int>)>);
    }
  }
}
