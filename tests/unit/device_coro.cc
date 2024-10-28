#include <catch2/catch_all.hpp>

#include "ttg.h"

#include "ttg/serialization.h"

#include "cuda_kernel.h"

#if defined(TTG_HAVE_CUDA)
#define SPACE ttg::ExecutionSpace::CUDA
#else
#define SPACE ttg::ExecutionSpace::Host
#endif // 0

struct value_t {
  ttg::Buffer<double> db; // TODO: rename
  int quark;

  value_t()
  {}

  value_t(std::size_t c)
  : db(c)
  , quark(c)
  { }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& quark;
    ar& db; // input:
  }
};

static void increment_buffer(
  double* buffer, std::size_t buffer_size,
  double* scratch, std::size_t scratch_size)
{
#if defined(TTG_HAVE_CUDA)
  increment_buffer(buffer, buffer_size, scratch, scratch_size);
#else  // TTG_HAVE_CUDA
  for (std::size_t i = 0 ; i < buffer_size; ++i) {
    buffer[i] += 1.0;
  }
  if (scratch != nullptr) {
    *scratch += 1.0;
  }
#endif // TTG_HAVE_CUDA
}

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

#if defined(TTG_IMPL_DEVICE_SUPPORT)

TEST_CASE("Device", "coro") {

  SECTION("devicebuf") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device::Task<SPACE> {
      //ttg::print("device_task key ", key);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(val.db);
      /* once we're back here the data has been transferred */
      CHECK(val.db.current_device_ptr() != nullptr);

      /* NO KERNEL */

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::device::wait();

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        //ttg::send<0>(key+1, std::move(val));
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
      }
    };

    //ptr.get_view<ttg::ExecutionSpace::CUDA>(device_id);

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{1});
    std::cout << "Entering fence" << std::endl;
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("devicebuf-inc") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device::Task<SPACE> {
      //ttg::print("device_task key ", key);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(val.db);
      /* once we're back here the data has been transferred */
      CHECK(val.db.current_device_ptr() != nullptr);

      std::cout << "KEY " << key << " VAL IN DEV " << *val.db.current_device_ptr() << " VAL IN HOST " << *val.db.host_ptr() << std::endl;

      /* call a kernel */
      increment_buffer(val.db.current_device_ptr(), val.db.size(), nullptr, 0);

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::device::wait(val.db);

      std::cout << "KEY " << key << " VAL OUT DEV " << *val.db.current_device_ptr() << " VAL OUT HOST " << *val.db.host_ptr() << std::endl;

      /* buffer is increment once per task, so it should be the same as key */
      CHECK(static_cast<int>(*val.db.host_ptr()) == key+1);

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        //ttg::send<0>(key+1, std::move(val));
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
      }
    };

    //ptr.get_view<ttg::ExecutionSpace::CUDA>(device_id);

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    value_t v{1};
    *v.db.host_ptr() = 2.0; // start from non-zero value
    if (ttg::default_execution_context().rank() == 0) tt->invoke(2, std::move(v));
    std::cout << "Entering fence" << std::endl;
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("scratch") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device::Task<SPACE> {
      double scratch = 0.0;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::Allocate);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* call a kernel */
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::device::wait(ds);

      /* the scratch is allocated but no data is transferred in; it's incremented once */
      CHECK((static_cast<int>(scratch)-1) == 0);

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        //ttg::send<0>(key+1, std::move(val));
        /* NOTE: we use co_await here instead of co_return because co_return destroys all local variables first;
         *       we will not return from this co_await!*/
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
      }
    };

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{1});
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("scratch-syncin") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device::Task<SPACE> {
      double scratch = key;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::SyncIn);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* call a kernel */
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::device::wait(ds);

      /* scratch is increment once per task, so it should be the same as key */
      CHECK((static_cast<int>(scratch))-1 == key);

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        //ttg::send<0>(key+1, std::move(val));
        /* NOTE: we use co_await here instead of co_return because co_return destroys all local variables first;
         *       we will not return from this co_await!*/
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
      }
    };

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{1});
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("scratch-value-out") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device::Task<SPACE> {
      double scratch = 0.0;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::Allocate);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* call a kernel */
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());

      /* here we suspend to wait for a kernel to complete */
      co_await ttg::device::wait(ds, val.db);

      /* buffer is increment once per task, so it should be 1 */
      CHECK((static_cast<int>(scratch)-1) == 0);

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        //ttg::send<0>(key+1, std::move(val));
        /* NOTE: we use co_await here instead of co_return because co_return destroys all local variables first;
         *       we will not return from this co_await!*/
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
      }
    };

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{1});
    ttg::ttg_fence(ttg::default_execution_context());
  }


#if 0
  /* TODO: Ptr seems broken atm, fix or remove! */
  SECTION("ptr") {

    ttg::Edge<int, value_t> edge;
    ttg::Ptr<value_t> ptr;
    int last_key = 0;
    constexpr const int num_iter = 10;
    auto fn = [&](const int& key, value_t&& val) -> ttg::device::Task<SPACE> {
      double scratch = key;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::SyncIn);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      /* KERNEL */
      increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());

      /* here we suspend to wait for a kernel and the out-transfer to complete */
      co_await ttg::device::wait(val.db, ds);

      /* buffer is increment once per task, so it should be the same as key */
      CHECK(static_cast<int>(scratch) == key+1);
      CHECK(static_cast<int>(*val.db.host_ptr()) == key+1);

      /* we're back, the kernel executed and we can send */
      if (key < num_iter) {
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

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{1});
    ttg::ttg_fence(ttg::default_execution_context());
    if (num_iter == last_key) {
      CHECK(ptr.is_valid());
      assert(ptr.is_valid());
    }

    /* feed the ptr back into a graph */
    if (ttg::default_execution_context().rank() == 0) tt->invoke(last_key+1, ptr);
    ttg::ttg_fence(ttg::default_execution_context());

    ptr.reset();
  }

#endif // 0

  /* TODO: enabel this test once we control the PaRSEC state machine! */
  SECTION("device-host-tasks") {

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

    auto device_fn = [&](const int& key, value_t&& val) -> ttg::device::Task<SPACE> {

      /* wait for the view to be available on the device */
      co_await ttg::device::select(val.db);

      /* call a kernel */
      increment_buffer(val.db.current_device_ptr(), val.db.size(), nullptr, 0);

      /* here we suspend to wait for a kernel to complete */
      //co_await ttg::device::wait(val.db);
      co_await ttg::device::wait();

      /* we're back, the kernel executed and we can send */
      if (key < 10) {
        /* TODO: should we move the view in here if we want to get the device side data */
        std::cout << "Sending to host key " << key+1 <<std::endl;
        co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(val)));
      }
    };

    auto dtt = ttg::make_tt(device_fn, ttg::edges(h2d), ttg::edges(d2h),
                            "device_task", {"h2d"}, {"d2h"});
    ttg::make_graph_executable(dtt);
    if (ttg::default_execution_context().rank() == 0) htt->invoke(0, value_t{1});
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("loop") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](int key, value_t&& val) -> ttg::device::Task<SPACE> {
      double scratch = 1.0;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::Allocate);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      for (int i = 0; i < 10; ++i) {

        CHECK(ds.device_ptr() != nullptr);
        CHECK(val.db.current_device_ptr() != nullptr);

        /* KERNEL */
        increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());
        //increment_buffer(val.db.current_device_ptr(), val.db.size(), 0, 0);

        /* here we suspend to wait for a kernel and the out-transfer to complete */
        co_await ttg::device::wait(val.db);

        /* buffer is increment once per task, so it should be the same as key */
        //CHECK(static_cast<int>(scratch) == i);
        CHECK(static_cast<int>(*val.db.host_ptr()) == i+1);
      }
    };

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{1});
    ttg::ttg_fence(ttg::default_execution_context());
  }

  SECTION("loop-scratchout") {

    ttg::Edge<int, value_t> edge;
    auto fn = [&](int key, value_t&& val) -> ttg::device::Task<SPACE> {
      double scratch = -10.0;
      ttg::devicescratch<double> ds = ttg::make_scratch(&scratch, ttg::scope::SyncIn);

      /* wait for the view to be available on the device */
      co_await ttg::device::select(ds, val.db);
      /* once we're back here the data has been transferred */
      CHECK(ds.device_ptr()  != nullptr);

      for (int i = 0; i < 10; ++i) {

        CHECK(ds.device_ptr() != nullptr);
        CHECK(val.db.current_device_ptr() != nullptr);

        /* KERNEL */
        increment_buffer(val.db.current_device_ptr(), val.db.size(), ds.device_ptr(), ds.size());
        //increment_buffer(val.db.current_device_ptr(), val.db.size(), 0, 0);

        /* here we suspend to wait for a kernel and the out-transfer to complete */
        co_await ttg::device::wait(val.db, ds);

        /* buffer is increment once per task, so it should be the same as key */
        CHECK(static_cast<int>(scratch) == (-10+i+1));
        CHECK(static_cast<int>(*val.db.host_ptr()) == i+1);
      }
    };

    auto tt = ttg::make_tt(fn, ttg::edges(edge), ttg::edges(edge),
                           "device_task", {"edge_in"}, {"edge_out"});
    ttg::make_graph_executable(tt);
    if (ttg::default_execution_context().rank() == 0) tt->invoke(0, value_t{1});
    ttg::ttg_fence(ttg::default_execution_context());
  }
}

#endif // TTG_IMPL_DEVICE_SUPPORT
