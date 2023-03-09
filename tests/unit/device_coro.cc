#include <catch2/catch.hpp>

#include "ttg.h"
#include "ttg/view.h"

#include "cuda_kernel.h"

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
    static inline void serialize(const Archive& ar, value_t& obj) { ar& obj.quark & obj.db; };
  };
}  // namespace madness::archive


TEST_CASE("Device", "coro") {

#if 0
  SECTION("devicebuf") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device_task {
      ttg::print("device_task key ", key);
      /* wait for the view to be available on the device */
      co_await ttg::to_device(val.db);
      /* once we're back here the data has been transferred */
      CHECK(val.db.current_device_ptr() != nullptr);

      /* NO KERNEL */

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::wait_kernel();

      /* we're back, the kernel executed and we can send */
      if (key < 1) {
        /* TODO: should we move the view in here if we want to get the device side data */
        ttg::send<0>(key+1, std::move(val));
      }
    };

    //ptr.get_view<ttg::ExecutionSpace::CUDA>(device_id);

    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{});
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("scratch") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device_task {
      double scratch = 0.0;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::SyncOut);

      /* wait for the view to be available on the device */
      co_await ttg::to_device(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* call a kernel */
#ifdef TTG_HAVE_CUDA
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());
#endif // TTG_HAVE_CUDA

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::wait_kernel();

#ifdef TTG_HAVE_CUDA
      /* buffer is increment once per task, so it should be the same as key */
      CHECK((static_cast<int>(scratch)-1) == key);
#endif // 0

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        ttg::send<0>(key+1, std::move(val));
      }
    };

    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{});
    ttg::ttg_fence(ttg::default_execution_context());
  }
#endif // 0

  SECTION("ptr") {

    ttg::Edge<int, value_t> edge;
    ttg::Ptr<value_t> ptr;
    int last_key = 0;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device_task {
      double scratch = 1.0;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::SyncOut);

      /* wait for the view to be available on the device */
      co_await ttg::to_device(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* KERNEL */
#ifdef TTG_HAVE_CUDA
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());
#endif // TTG_HAVE_CUDA

      /* here we suspend to wait for a kernel and the out-transfer to complete */
      co_await ttg::wait_kernel_out(val.db);

#ifdef TTG_HAVE_CUDA
      /* buffer is increment once per task, so it should be the same as key */
      CHECK(static_cast<int>(scratch) == key+1);
      CHECK(static_cast<int>(*val.db.host_ptr()) == key+1);
#endif // TTG_HAVE_CUDA

      /* we're back, the kernel executed and we can send */
      if (key < 10 || scratch < 0.0) {
        ttg::send<0>(key+1, std::move(val));
      } else {
        /* exfiltrate the value */
        /* TODO: what consistency do we expect from get_ptr? */
        ptr = ttg::get_ptr(val);
        last_key = key;
      }
    };

    //ptr.get_view<ttg::ExecutionSpace::CUDA>(device_id);

    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{});
    ttg::ttg_fence(ttg::default_execution_context());
    CHECK(ptr.is_valid());

    /* feed the ptr back into a graph */
    if (ttg::default_execution_context().rank() == 0) tt->invoke(last_key+1, ptr);
    ttg::ttg_fence(ttg::default_execution_context());

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
