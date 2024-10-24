#include <catch2/catch_all.hpp>
#include <ctime>

#include "ttg.h"

#include "ttg/serialization/std/pair.h"
#include "ttg/util/hash/std/pair.h"
#include "ttg/util/multiindex.h"

using Key = ttg::MultiIndex<2>;

TEST_CASE("constraints", "") {

  SECTION("sequenced") {
    ttg::Edge<Key, int> e;
    auto world = ttg::default_execution_context();
    std::atomic<int> last_ord = world.rank();
    auto tt = ttg::make_tt([&](const Key& key, const int& value){
      int check_ord = last_ord;
      CHECK(((key[1] == check_ord) || (key[1] == check_ord+1)));
      last_ord = key[1];
    }, ttg::edges(e), ttg::edges());
    // every process executes all tasks
    tt->set_keymap([&](const Key&){ return world.rank(); });
    auto constraint = ttg::make_shared_constraint<ttg::SequencedKeysConstraint>([](const Key& k){ return k[1]; });
    tt->add_constraint(constraint);
    constraint->stop();

    auto bcast = ttg::make_tt([&](){
      std::vector<Key> keys;
      // loop iteration order intentionally reversed
      for (int i = 10; i > 0; --i) {
        for (int j = 10; j > world.rank(); --j) {
          keys.push_back(Key{i, j});
        }
      }
      ttg::broadcast<0>(std::move(keys), 0);

      // explicit start here to ensure absolute order
      constraint->start();
    }, ttg::edges(), ttg::edges(e));
    bcast->set_keymap([&](){ return world.rank(); });

    /**
     * Constraints are currently only implemented in the PaRSEC backend.
     * Codes using constraints will still compile but they will not
     * affect the execution order in other backends.
     */
#ifdef TTG_USE_PARSEC
    make_graph_executable(bcast);
    ttg::execute(ttg::default_execution_context());
    bcast->invoke();

    ttg::ttg_fence(ttg::default_execution_context());
#endif // TTG_USE_PARSEC

  }
}  // TEST_CASE("streams")
