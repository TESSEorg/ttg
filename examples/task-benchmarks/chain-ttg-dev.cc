//#define TTG_USE_USER_TERMDET 1
#include "ttg.h"

#include "chrono.h"

#if defined(TTG_HAVE_CUDA)
#define ES ttg::ExecutionSpace::CUDA
#elif defined(TTG_HAVE_HIP)
#define ES ttg::ExecutionSpace::HIP
#else
#error "Either CUDA OR HIP is required to build this test!"
#endif // 0

#define NUM_TASKS 100000

using namespace ttg;

std::atomic<int> task_counter = 0;

struct A : public ttg::TTValue<A> {
  ttg::Buffer<int> b;
  A() : b(1) { }

  A(A&& a) = default;

  template <typename Archive>
  void serialize(Archive& ar) {
    ttg_abort();
  }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ttg_abort();
  }

};

template <int num_flows>
auto make_ttg();

// flows task ids via values
template <>
auto make_ttg<1>() {
  Edge<int, A> I2N, N2N;
  Edge<void, A> N2S;

  auto init = make_tt<void>(
    []() {
      ++task_counter;
      std::cout << "init 1 " << std::endl;
      send<0>(0, A{});
    }, edges(), edges(I2N));

  auto next = make_tt<ES, int>([=](const int &key, auto&& value) -> ttg::device::Task {
    //++task_counter;
    co_await ttg::device::select(value.b);
    if (key < NUM_TASKS) {
      co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(value)));
    } else {
    }
  } , edges(fuse(I2N, N2N)), edges(N2N));

  return std::make_tuple(std::move(init), std::move(next));
}

template <>
auto make_ttg<2>() {
  Edge<int, A> I2N1, I2N2;
  Edge<int, A> N2N1, N2N2;
  Edge<void, A> N2S1, N2S2;

  auto init = make_tt<void>([]() {
    send<0>(0, A{});
    send<1>(0, A{});
  }, edges(), edges(I2N1, I2N2));

  auto next = make_tt<ES, int>([=](const int &key, A&& v1, A&& v2) -> ttg::device::Task {
    co_await ttg::device::select(v1.b, v2.b);
    if (key < NUM_TASKS) {
      co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(v1)),
                                    ttg::device::send<1>(key+1, std::move(v2)));
    }
  } , edges(fuse(I2N1, N2N1), fuse(I2N2, N2N2)), edges(N2N1, N2N2));

  return std::make_tuple(std::move(init), std::move(next));
}

template <>
auto make_ttg<4>() {
  Edge<int, A> I2N1, I2N2, I2N3, I2N4;
  Edge<int, A> N2N1, N2N2, N2N3, N2N4;
  Edge<void, A> N2S1, N2S2, N2S3, N2S4;

  auto init = make_tt<void>(
    []() {
      send<0>(0, A{});
      send<1>(0, A{});
      send<2>(0, A{});
      send<3>(0, A{});
    }, edges(), edges(I2N1, I2N2, I2N3, I2N4));

  auto next = make_tt<ES, int>([=](const int &key, A&& v1, A&& v2, A&& v3, A&& v4) -> ttg::device::Task {
    co_await ttg::device::select(v1.b, v2.b, v3.b, v4.b);
    if (key < NUM_TASKS) {
      co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(v1)),
                                    ttg::device::send<1>(key+1, std::move(v2)),
                                    ttg::device::send<2>(key+1, std::move(v3)),
                                    ttg::device::send<3>(key+1, std::move(v4)));
    }
  }, edges(fuse(I2N1, N2N1), fuse(I2N2, N2N2),
           fuse(I2N3, N2N3), fuse(I2N4, N2N4)),
     edges(N2N1, N2N2, N2N3, N2N4));

  return std::make_tuple(std::move(init), std::move(next));
}

template <>
auto make_ttg<8>() {
  Edge<int, A> I2N1, I2N2, I2N3, I2N4, I2N5, I2N6, I2N7, I2N8;
  Edge<int, A> N2N1, N2N2, N2N3, N2N4, N2N5, N2N6, N2N7, N2N8;
  Edge<void, A> N2S1, N2S2, N2S3, N2S4, N2S5, N2S6, N2S7, N2S8;

  auto init = make_tt<void>(
    []() {
      send<0>(0, A{});
      send<1>(0, A{});
      send<2>(0, A{});
      send<3>(0, A{});
      send<4>(0, A{});
      send<5>(0, A{});
      send<6>(0, A{});
      send<7>(0, A{});
    }, edges(), edges(I2N1, I2N2, I2N3, I2N4, I2N5, I2N6, I2N7, I2N8));

  auto next = make_tt<ES, int>([=](const int &key, auto&& v1, auto&& v2, auto&& v3, auto&& v4, auto&& v5, auto&& v6, auto&& v7, auto&& v8) -> ttg::device::Task {
    co_await ttg::device::select(v1.b, v2.b, v3.b, v4.b, v5.b, v6.b, v7.b, v8.b);
    if (key < NUM_TASKS) {
      co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(v1)),
                                    ttg::device::send<1>(key+1, std::move(v2)),
                                    ttg::device::send<2>(key+1, std::move(v3)),
                                    ttg::device::send<3>(key+1, std::move(v4)),
                                    ttg::device::send<4>(key+1, std::move(v5)),
                                    ttg::device::send<5>(key+1, std::move(v6)),
                                    ttg::device::send<6>(key+1, std::move(v7)),
                                    ttg::device::send<7>(key+1, std::move(v8)));
    }
  }, edges(fuse(I2N1, N2N1), fuse(I2N2, N2N2), fuse(I2N3, N2N3), fuse(I2N4, N2N4), fuse(I2N5, N2N5), fuse(I2N6, N2N6), fuse(I2N7, N2N7), fuse(I2N8, N2N8)),
     edges(N2N1, N2N2, N2N3, N2N4, N2N5, N2N6, N2N7, N2N8));

  return std::make_tuple(std::move(init), std::move(next));
}

// flows task ids via keys
template <>
auto make_ttg<0>() {
  Edge<int, void> I2N, N2N;
  Edge<void, int> N2S;

  auto init = make_tt<void>([](std::tuple<Out<int, void>> &outs) { sendk<0>(0, outs); }, edges(), edges(I2N));

  auto next = make_tt<ES>([](const int& key) -> ttg::device::Task {
    co_await ttg::device::select();
    if (key < NUM_TASKS) {
      co_await ttg::device::forward(ttg::device::sendk<0>(key+1));
    }
  }, edges(fuse(I2N, N2N)), edges(N2N));

  return std::make_tuple(std::move(init), std::move(next));
}

template<int num_flows>
void run_bench()
{
  auto [init, next] = make_ttg<num_flows>();

  auto connected = make_graph_executable(init.get());
  assert(connected);
  std::cout << "Graph " << num_flows << " is connected.\n";

  if (ttg::default_execution_context().rank() == 0) init->invoke();

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());

  auto t0 = now();
  if (ttg::default_execution_context().rank() == 0) init->invoke();

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());
  auto t1 = now();

  std::cout << "# of tasks = " << NUM_TASKS << std::endl;
  std::cout << "time elapsed (microseconds) = " << duration_in_mus(t0, t1) << ", avg " << duration_in_mus(t0, t1) / (double)NUM_TASKS << std::endl;
}

int main(int argc, char* argv[]) {

  int num_flows = 0;
  ttg_initialize(argc, argv, -1);

  if (argc > 1) {
    num_flows = std::atoi(argv[1]);
  }

  switch(num_flows) {
  case 0: run_bench<0>(); break;
  case 1: run_bench<1>(); break;
  case 2: run_bench<2>(); break;
  case 4: run_bench<4>(); break;
  case 8: run_bench<8>(); break;
  default: std::cout << "Unsupported number of flows: " << NUM_TASKS << std::endl;
  }

  ttg_finalize();
  return 0;
}

