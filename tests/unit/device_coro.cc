#include <catch2/catch.hpp>

#include "ttg.h"
#include "ttg/view.h"

struct value_t {
  ttg::buffer<double> db; // TODO: rename
  int quark;

  template<typename Archive>
  void ttg_serialize(Archive& ar) {
    ar& quark;
    ar& db; // input:
  }
};

/* devicebuf is non-POD so provide serialization
 * information for members not a devicebuf */
namespace madness::archive {
  template <class Archive>
  struct ArchiveSerializeImpl<Archive, value_t> {
    static inline void serialize(const Archive& ar, value_t& obj) { ar& obj.quark; };
  };
}  // namespace madness::archive

/* announce that this type contains a device buffer */
template<>
struct ttg::container_trait<value_t> {
  static auto devicebuf_members(value_t&& v) {
    return std::tie(v.db);
  }
};



TEST_CASE("Device", "coro") {
  SECTION("devicebuf") {

    ttg::Edge<int, value_t> edge;
    ttg::ptr<value_t> ptr;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device_task {
      double scratch = 1.0;
      ttg::devicescratch<double> scratch_view = ttg::make_scratch(&scratch, ttg::scope::SyncOut);
      ttg::print("device_task key ", key);
      /* wait for the view to be available on the device */
      co_await ttg::to_device(scratch_view, val.db);
      // co_yield std::tie(view1, view2);
      // TTG_WAIT_VIEW(view);
      /* once we're back here the data has been transferred */
      //CHECK(view.get_rw_device_ptr<0>()  != nullptr);
      CHECK(scratch_view.device_ptr()  != nullptr);

      ttg::print("device_task key ", key, ", device pointer ", scratch_view.device_ptr());

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::device_op_wait_kernel{};
      // TTG_WAIT_KERNEL();

      /* force the data back to the host
       * useful if the buffer tracks a non-owned pointer
       * and we want to send on that pointer
       * TODO: needs implementing, how does this interact with scratch?
       */
      //co_await ttg::to_host(val.db);

      /* we're back, the kernel executed and we can send */
      if (key < 1 || scratch < 0.0) {
        /* TODO: should we move the view in here if we want to get the device side data */
        ttg::send<0>(key+1, std::move(val));
      } else {
        /* exfiltrate the value */
        ptr = ttg::get_ptr(val);
      }
    };

    //ptr.get_view<ttg::ExecutionSpace::CUDA>(device_id);

    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{});
    ttg::ttg_fence(ttg::default_execution_context());
    CHECK(ptr.is_valid());
    ptr.reset();
  }










#if 0

  SECTION("device_task") {
    ttg::Edge<int, double> edge;
    auto fn = [&](const int& key, double&& val) -> ttg::device_task {
      ttg::View<double> view = ttg::make_view(val, ttg::ViewScope::SyncInOut);
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
        ttg::send<0>(key+1, std::move(val));
      }
    };
    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("get_ptr") {
    ttg::Edge<int, double> edge;
    ttg::ptr<double> ptr;
    auto fn = [&](const int& key, double&& val) -> ttg::device_task {
      ttg::View<double> view = ttg::make_view(val, ttg::ViewScope::SyncInOut);
      ttg::print("device_task key ", key, ", value ", val);
      /* wait for the view to be available on the device */
      co_yield view;
      // co_yield std::tie(view1, view2);
      // TTG_WAIT_VIEW(view);
      /* once we're back here the data has been transferred */
      //CHECK(view.get_rw_device_ptr<0>()  != nullptr);
      CHECK(view.get_device_ptr<0>()  != nullptr);
      CHECK(view.get_device_size<0>() == sizeof(val));
      CHECK(&view.get_host_object() == &val);

      ttg::print("device_task key ", key, ", device pointer ", view.get_device_ptr<0>());

      /* here we suspend to wait for a kernel to complete */
      co_yield ttg::device_op_wait_kernel{};
      // TTG_WAIT_KERNEL();

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        ttg::send<0>(key+1, std::move(val));
      } else {
        /* exfiltrate the value */
        ptr = ttg::get_ptr(val);
      }
    };

    //ptr.get_view<ttg::ExecutionSpace::CUDA>(device_id);

    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, 0.0);
    ttg::ttg_fence(ttg::default_execution_context());

#if 0
    /* feed the host-side value back into the graph */
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, pview.get_ptr());
    ttg::ttg_fence(ttg::default_execution_context());

    /* feed the device-side value back into the graph */
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, pview);
    ttg::ttg_fence(ttg::default_execution_context());
#endif // 0
  }




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
