#include <catch2/catch.hpp>

#include "ttg.h"
#include "ttg/view.h"

#include "ttg/serialization.h"

#include "cuda_kernel.h"

struct value_t {
  ttg::buffer<double> db; // TODO: rename
  int quark;

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& quark;
    ar& db; // input:
  }
};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
/* devicebuf is non-POD so provide serialization
 * information for members not a devicebuf */
namespace madness::archive {
  template <class Archive>
  struct ArchiveSerializeImpl<Archive, value_t> {
    static inline void serialize(const Archive& ar, value_t& obj) { ar& obj.quark & obj.db; };
  };
}  // namespace madness::archive
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

TEST_CASE("Device", "coro") {

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
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        //ttg::send<0>(key+1, std::move(val));
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
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
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::Allocate);

      /* wait for the view to be available on the device */
      co_await ttg::to_device(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* call a kernel */
#ifdef TTG_HAVE_CUDA
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());
#endif // TTG_HAVE_CUDA

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::wait_kernel(ds);

#ifdef TTG_HAVE_CUDA
      /* buffer is increment once per task, so it should be the same as key */
      CHECK((static_cast<int>(scratch)-1) == key);
#endif // 0

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        //ttg::send<0>(key+1, std::move(val));
        /* NOTE: we use co_await here instead of co_return because co_return destroys all local variables first;
         *       we will not return from this co_await!*/
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
      }
    };

    auto tt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(fn, ttg::edges(edge), ttg::edges(edge),
                                                      "device_task", {"edge_in"}, {"edge_out"});
    make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{});
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("ptr") {

    ttg::Edge<int, value_t> edge;
    ttg::Ptr<value_t> ptr;
    int last_key = 0;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device_task {
      double scratch = 1.0;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::SyncIn);

      /* wait for the view to be available on the device */
      co_await ttg::to_device(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* KERNEL */
#ifdef TTG_HAVE_CUDA
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());
#endif // TTG_HAVE_CUDA

      /* here we suspend to wait for a kernel and the out-transfer to complete */
      co_await ttg::wait_kernel(val.db, ds);

#ifdef TTG_HAVE_CUDA
      /* buffer is increment once per task, so it should be the same as key */
      CHECK(static_cast<int>(scratch) == key+1);
      CHECK(static_cast<int>(*val.db.host_ptr()) == key+1);
#endif // TTG_HAVE_CUDA

      /* we're back, the kernel executed and we can send */
      if (key < 10 || scratch < 0.0) {
        //ttg::send<0>(key+1, std::move(val));
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
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
  /* TODO: enabel this test once we control the PaRSEC state machine! */
  SECTION("device_host_tasks") {

    ttg::Edge<int, value_t> h2d, d2h;

    auto host_fn = [&](const int& key, value_t&& val) {
      /* check that the data has been synced back */
      CHECK(static_cast<int>(*val.db.host_ptr()) == key);

      /* modify the data */
      *val.db.host_ptr() += 1.0;
      CHECK(static_cast<int>(*val.db.host_ptr()) == key+1);

      /* send back to the device */
      ttg::send<0>(key+1, std::move(val));
    };
    auto htt = ttg::make_tt(host_fn, ttg::edges(d2h), ttg::edges(h2d),
                            "host_task", {"d2h"}, {"h2d"});

    auto device_fn = [&](const int& key, value_t&& val) -> ttg::device_task {
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

    auto dtt = ttg::make_tt<ttg::ExecutionSpace::CUDA>(device_fn, ttg::edges(h2d), ttg::edges(d2h),
                                                      "device_task", {"h2d"}, {"d2h"});
    make_graph_executable(dtt);
    if (ttg::default_execution_context().rank() == 0) htt->invoke(0, value_t{});
    ttg::ttg_fence(ttg::default_execution_context());
  }


#endif // 0

}