#include <catch2/catch.hpp>

#include "ttg.h"
#include "ttg/view.h"


TEST_CASE("Device", "coro") {
  SECTION("device_task") {
    ttg::Edge<int, double> edge;
    auto fn = [&](const int& key, double&& val) -> ttg::device_task {
      ttg::View<double, double> view = ttg::make_view(val, ttg::ViewScope::SyncInOut);
      ttg::print("device_task key ", key, ", value ", val);
      /* wait for the view to be available on the device */
      co_yield view;
      // co_yield std::tie(view1, view2);
      // TTG_WAIT_VIEW(view);
      /* once we're back here the data has been transferred */
      CHECK(view.get_device_ptr<0>()  != nullptr);
      CHECK(view.get_device_size<0>() == sizeof(val));
      CHECK(&view.get_host_object() == &val);

      ttg::print("device_task key ", key, ", device pointer ", view.get_device_ptr<0>());

      /* here we suspend to wait for a kernel to complete */
      co_yield ttg::device_op_wait_kernel{};
      // TTG_WAIT_KERNEL();

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        ttg::send<0>(key+1, val);
      }
    };
    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence(ttg::default_execution_context());
  }

#if 0
  SECTION("device_task") {
    ttg::Edge<int, double> edge;
    auto fn = [&](const int& key, double&& val) -> ttg::device_task {
      // will schedule the view for transfer to and from the device
      ttg::View<double> view = ttg::make_view(val, ttg::ViewScope::SyncInOut);
      ttg::print("device_task key ", key, ", value ", val);
      /* wait for the view to be available on the device */
      co_await ttg::device_task_wait_views{};

      /* once we're back here the data has been transferred */
      CHECK(view.get_device_ptr<0>()  != nullptr);
      CHECK(view.get_device_size<0>() == sizeof(val));
      CHECK(&view.get_host_object() == &val);

      ttg::print("device_task key ", key, ", device pointer ", view.get_device_ptr<0>());

      while (val < 10.0) {

        view.set_scope(ttg::ViewScope::SyncOut);

        /* <submit kernel here> */

        /* here we suspend to wait for a kernel to complete */
        co_await ttg::device_task_wait_kernel{};

        // TTG_WAIT_KERNEL();
      }

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        ttg::send<0>(key+1, val);
      }
    };
    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence(ttg::default_execution_context());
  }

  struct A {
    double norm;
    std::vector<double> d;
  };

  SECTION("device_task") {
    ttg::Edge<int, double> edge;
    auto fn = [&](const int& key, ttg::ptr<A>&& a) -> ttg::device_task {
      // will schedule the view for transfer to and from the device
      View<double> norm_view = a.to_host(&A::norm);

      View<A::norm> norm_view{a};
      co_await ttg::device::wait_transfer{};




      if (val)
      val += 1.0;
      ptr.sync_to_device();
      /* wait for the view to be available on the device */
      co_await ttg::device_task_wait_views{};

      /* once we're back here the data has been transferred */
      CHECK(view.get_device_ptr<0>()  != nullptr);
      CHECK(view.get_device_size<0>() == sizeof(val));
      CHECK(&view.get_host_object() == &val);

      ttg::print("device_task key ", key, ", device pointer ", view.get_device_ptr<0>());

      while (val < 10.0) {

        view.set_scope(ttg::ViewScope::SyncOut);

        /* <submit kernel here> */

        /* here we suspend to wait for a kernel to complete */
        co_await ttg::device_task_wait_kernel{};

        // TTG_WAIT_KERNEL();
      }

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        ttg::send<0>(key+1, val);
      }
    };
    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence(ttg::default_execution_context());
  }

  struct A {
    int a[10];
    double b[10];
  };


  SECTION("device_task_struct") {
    ttg::Edge<int, double> edge;
    auto fn = [](const int& key, A&& val) -> ttg::device_task {
      auto view = ttg::make_view(val, {val.a, 10, ttg::ViewScope::SyncIn},
                                      {val.b, 10, ttg::ViewScope::SyncIn});
      /* wait for the view to be available on the device */
      co_yield view;
      // co_yield std::tie(view1, view2);
      // TTG_WAIT_VIEW(view);
      /* once we're back here the data has been transferred */
      CHECK(view.get_device_ptr<0>() != nullptr);
      CHECK(view.size<0>() == 10*sizeof(int));
      CHECK(&view.get_host_object() == &val);

      // <submit kernel here>

      /* here we suspend to wait for a kernel to complete */
      co_yield ttg::device_op_wait_kernel{};
      // TTG_WAIT_KERNEL();

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        ttg::send<0>(key+1, val);
      }
    };
    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence(ttg::default_execution_context());
  }



  SECTION("device_task_struct_multi") {
    ttg::Edge<int, double> edge;
    auto fn = [](const int& key, A&& val) -> ttg::device_task {
      auto view = ttg::make_view(val, {val.a, 10},
                                      {val.b, 10, ttg::ViewScope::SyncOut});
      /* wait for the view to be available on the device */
      co_yield view;
      // co_yield std::tie(view1, view2);
      // TTG_WAIT_VIEW(view);
      /* once we're back here the data has been transferred */
      CHECK(view.get_device_ptr<0>() != nullptr);
      CHECK(view.size<0>() == 10*sizeof(int));
      CHECK(&view.get_host_object() == &val);

      // <submit kernel here>

      /* here we suspend to wait for a kernel to complete */
      co_yield ttg::device_op_wait_kernel{};
      // TTG_WAIT_KERNEL();

      if (val.b[0] < 100.0) {
        // <submit another kernel here>
        view.sync_out<0>();
        // anything submitted before this point must finish before the data is returned
        co_yield view;
      }

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        ttg::send<0>(key+1, val);
      }
    };
    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence(ttg::default_execution_context());
  }
#endif // 0
}
