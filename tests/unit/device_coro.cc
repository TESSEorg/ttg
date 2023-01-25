#include <catch2/catch.hpp>

#include "ttg.h"
#include "ttg/ttg/view.h"


TEST_CASE("Device", "coro") {
  SECTION("device_task") {
    ttg::Edge<int, double> edge;
    auto fn = [](const int& key, double&& val) -> ttg::device_task {
      ttg::View<double, double> view = ttg::make_view(val);
      /* wait for the view to be available on the device */
      co_yield view;
      /* once we're back here the data has been transferred */
      CHECK(view.get_device_ptr() != nullptr);
      CHECK(view.size() == sizeof(val));
      CHECK(&view.get_host_object() == &val);

      /* here we suspend to wait for a kernel to complete */
      co_yield device_op_wait_kernel{};

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        ttg::send<0>(key+1, val);
      }
    };
    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence();
  }
}
