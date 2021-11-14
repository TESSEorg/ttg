#include <catch2/catch.hpp>

#include "ttg.h"

#include <memory>

template <typename Iterator>
auto range2flow(Iterator b, Iterator e, ttg::World world = ttg::default_execution_context()) {
  ttg::Edge<std::size_t, typename std::iterator_traits<Iterator>::value_type> out;
  ttg::Edge<> ctl;
  auto range_reader = ttg::make_tt<void>(
      [b, e](std::tuple<ttg::Out<std::size_t, typename std::iterator_traits<Iterator>::value_type>> &outs) {
        std::size_t idx = 0;
        for (auto it = b; it != e; ++it, ++idx) ttg::send<0>(idx, *it, outs);
      },
      ttg::edges(ctl), ttg::edges(out));
  range_reader->make_executable();
  ttg_register_ptr(world, std::move(range_reader));
  return std::make_tuple(out, ctl);
}

template <typename Value, typename Iterator>
auto flow2range(const ttg::Edge<std::size_t, Value> &in, Iterator b, Iterator e,
                ttg::World world = ttg::default_execution_context()) {
  auto range_writer = ttg::make_tt(
      [b, e](const std::size_t &idx, Value &&value, std::tuple<> &outs) {
        assert(idx < std::distance(b, e));
        auto it = std::advance(b, idx);
        *it = std::move(value);
      },
      ttg::edges(in), ttg::edges());
  range_writer->make_executable();
  ttg_register_ptr(world, std::move(range_writer));
}

namespace ttg {
  template <typename Iterator, typename Op>
  auto for_each(Iterator b, Iterator e, Op op, ttg::World world = ttg::default_execution_context()) {
    auto [in, ctl] = range2flow(b, e, world);
    auto foreach_op = ttg::make_tt([&op](const std::size_t &idx, const int &datum, std::tuple<> &outs) { op(datum); },
                                   ttg::edges(in), ttg::edges());
    foreach_op->make_executable();
    ttg::ttg_register_ptr(world, std::move(foreach_op));
    ctl.fire();
    ttg::ttg_fence(world);
    return std::move(op);
  }
}  // namespace ttg

TEST_CASE("Ranges", "[core][ranges]") {
  SECTION("range_flow") {
    std::vector<int> v;
    CHECK_NOTHROW(range2flow(v.begin(), v.end()));
  }
  SECTION("foreach") {
    std::vector<int> v{1, 2, 3};
    ttg::for_each(v.begin(), v.end(), [](const auto &value) { std::cout << value << std::endl; });
  }
}
