//#define TTG_USE_USER_TERMDET 1
#include "ttg.h"

#include "chrono.h"

#define NUM_TASKS 100000

using namespace ttg;

template <int num_flows>
auto make_ttg(bool do_move);

// flows task ids via values
template <>
auto make_ttg<1>(bool do_move) {
  Edge<int, int> I2N, N2N;
  Edge<void, int> N2S;

  auto init = make_tt<void>([]() { send<0>(0, 0); }, edges(), edges(I2N));

  auto next = make_tt<int>([=](const int &key, auto&& value) {
    if (key < NUM_TASKS) {
    //std::cout << &value << " -> " << value << std::endl;
    //if (key < 10) {
      //value++;
      if (do_move) {
        send<0>(key+1, std::move(value));
        //send<0>(key+1, value);
      } else {
        send<0>(key+1, value);
      }
    }
    else {
      sendv<1>(std::move(value));
    }
  } , edges(fuse(I2N, N2N)), edges(N2N, N2S));

  auto stop = make_tt<void>([](const int& v) {
     //std::cout << "last task received v=" << v << std::endl;
     ttg::default_execution_context().impl().final_task();
  }, edges(N2S), edges());

  return std::make_tuple(std::move(init), std::move(next), std::move(stop));
}

template <>
auto make_ttg<2>(bool do_move) {
  Edge<int, int> I2N1, I2N2;
  Edge<int, int> N2N1, N2N2;
  Edge<void, int> N2S1, N2S2;

  auto init = make_tt<void>([]() {
    send<0>(0, 0);
    send<1>(0, 0);
  }, edges(), edges(I2N1, I2N2));

  auto next = make_tt<int>([=](const int &key, int&& v1, int&& v2) {
    if (key < NUM_TASKS) {
      v1++; v2++;
      if (do_move) {
        send<0>(key+1, std::move(v1));
        send<1>(key+1, std::move(v2));
      } else {
        send<0>(key+1, v1);
        send<1>(key+1, v2);
      }
    }
    else {
      sendv<2>(std::move(v1));
      sendv<3>(std::move(v2));
    }
  } , edges(fuse(I2N1, N2N1), fuse(I2N2, N2N2)), edges(N2N1, N2N2, N2S1, N2S2));

  auto stop = make_tt<void>([](const int &v1, const int &v2) {
     //std::cout << "last task received v=" << v1 << std::endl;
     ttg::default_execution_context().impl().final_task();
  }, edges(N2S1, N2S2), edges());

  return std::make_tuple(std::move(init), std::move(next), std::move(stop));
}

template <>
auto make_ttg<4>(bool do_move) {
  Edge<int, int> I2N1, I2N2, I2N3, I2N4;
  Edge<int, int> N2N1, N2N2, N2N3, N2N4;
  Edge<void, int> N2S1, N2S2, N2S3, N2S4;

  auto init = make_tt<void>([]() {
    send<0>(0, 0);
    send<1>(0, 0);
    send<2>(0, 0);
    send<3>(0, 0);
  }, edges(), edges(I2N1, I2N2, I2N3, I2N4));

  auto next = make_tt<int>([=](const int &key, int&& v1, int&& v2, int&& v3, int&& v4) {
    if (key < NUM_TASKS) {
      v1++; v2++;
      v3++; v4++;
      if (do_move) {
        send<0>(key+1, std::move(v1));
        send<1>(key+1, std::move(v2));
        send<2>(key+1, std::move(v3));
        send<3>(key+1, std::move(v4));
      } else {
        send<0>(key+1, v1);
        send<1>(key+1, v2);
        send<2>(key+1, v3);
        send<3>(key+1, v4);
      }
    }
    else {
      sendv<4>(std::move(v1));
      sendv<5>(std::move(v2));
      sendv<6>(std::move(v3));
      sendv<7>(std::move(v4));
    }
  }, edges(fuse(I2N1, N2N1), fuse(I2N2, N2N2),
           fuse(I2N3, N2N3), fuse(I2N4, N2N4)),
     edges(N2N1, N2N2, N2N3, N2N4, N2S1, N2S2, N2S3, N2S4));

  auto stop = make_tt<void>([](const int& v1, const int& v2, const int& v3, const int& v4){
     //std::cout << "last task received v=" << v1 << std::endl;
     ttg::default_execution_context().impl().final_task();
  }, edges(N2S1, N2S2, N2S3, N2S4), edges());

  return std::make_tuple(std::move(init), std::move(next), std::move(stop));
}

template <>
auto make_ttg<8>(bool do_move) {
  Edge<int, int> I2N1, I2N2, I2N3, I2N4, I2N5, I2N6, I2N7, I2N8;
  Edge<int, int> N2N1, N2N2, N2N3, N2N4, N2N5, N2N6, N2N7, N2N8;
  Edge<void, int> N2S1, N2S2, N2S3, N2S4, N2S5, N2S6, N2S7, N2S8;

  auto init = make_tt<void>([]() {
    send<0>(0, 0);
    send<1>(0, 0);
    send<2>(0, 0);
    send<3>(0, 0);
    send<4>(0, 0);
    send<5>(0, 0);
    send<6>(0, 0);
    send<7>(0, 0);
  }, edges(), edges(I2N1, I2N2, I2N3, I2N4, I2N5, I2N6, I2N7, I2N8));

  auto next = make_tt<int>([=](const int &key, auto&& v1, auto&& v2, auto&& v3,
                               auto&& v4, auto&& v5, auto&& v6, auto&& v7, auto&& v8) {
    if (key < NUM_TASKS) {
    //if (key < 1000) {
      v1++; v2++;
      v3++; v4++;
      v5++; v6++;
      v6++; v8++;
      if (do_move) {
        send<0>(key+1, std::move(v1));
        send<1>(key+1, std::move(v2));
        send<2>(key+1, std::move(v3));
        send<3>(key+1, std::move(v4));
        send<4>(key+1, std::move(v5));
        send<5>(key+1, std::move(v6));
        send<6>(key+1, std::move(v7));
        send<7>(key+1, std::move(v8));
      } else {
        send<0>(key+1, v1);
        send<1>(key+1, v2);
        send<2>(key+1, v3);
        send<3>(key+1, v4);
        send<4>(key+1, v5);
        send<5>(key+1, v6);
        send<6>(key+1, v7);
        send<7>(key+1, v8);
      }
    }
    else {
      sendv<8>(std::move(v1));
      sendv<9>(std::move(v2));
      sendv<10>(std::move(v3));
      sendv<11>(std::move(v4));
      sendv<12>(std::move(v5));
      sendv<13>(std::move(v6));
      sendv<14>(std::move(v7));
      sendv<15>(std::move(v8));
    }
  }, edges(fuse(I2N1, N2N1), fuse(I2N2, N2N2), fuse(I2N3, N2N3), fuse(I2N4, N2N4), fuse(I2N5, N2N5), fuse(I2N6, N2N6), fuse(I2N7, N2N7), fuse(I2N8, N2N8)),
     edges(N2N1, N2N2, N2N3, N2N4, N2N5, N2N6, N2N7, N2N8, N2S1, N2S2, N2S3, N2S4, N2S5, N2S6, N2S7, N2S8));

  auto stop = make_tt<void>([](const int &v1, const int &v2, const int &v3, const int &v4, const int &v5, const int &v6, const int &v7, const int &v8) {
     //std::cout << "last task received v=" << v1 << std::endl;
     ttg::default_execution_context().impl().final_task();
  }, edges(N2S1, N2S2, N2S3, N2S4, N2S5, N2S6, N2S7, N2S8), edges());

  return std::make_tuple(std::move(init), std::move(next), std::move(stop));
}

// flows task ids via keys
template <>
auto make_ttg<0>(bool do_move) {
  Edge<int, void> I2N, N2N;
  Edge<void, int> N2S;

  auto init = make_tt<void>([]() { sendk<0>(0); }, edges(), edges(I2N));

  auto next = make_tt([](const int& key) {
    if (key < NUM_TASKS) {
      ::sendk<0>(key+1);
    }
    else {
      ::sendv<1>(key);
    }
  }, edges(fuse(I2N, N2N)), edges(N2N, N2S));

  auto stop = make_tt<void>([](const int &v) {
     //std::cout << "last task received v=" << v << std::endl;
     ttg::default_execution_context().impl().final_task();
  }, edges(N2S), edges());

  return std::make_tuple(std::move(init), std::move(next), std::move(stop));
}

template<int num_flows>
void run_bench(bool do_move)
{
  auto [init, next, stop] = make_ttg<num_flows>(do_move);

  auto connected = make_graph_executable(init.get());
  assert(connected);
  std::cout << "Graph is connected.\n";

  auto t0 = now();
  if (ttg::default_execution_context().rank() == 0) init->invoke();

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());
  auto t1 = now();

  std::cout << "# of tasks = " << NUM_TASKS << std::endl;
  std::cout << "time elapsed (microseconds) = " << duration_in_mus(t0, t1)
            << ", avg " << duration_in_mus(t0, t1) / (double)NUM_TASKS << std::endl;
}

int main(int argc, char* argv[]) {

  int num_flows = 0;
  int do_move = 1;
  ttg_initialize(argc, argv, -1);

  if (argc > 1) {
    num_flows = std::atoi(argv[1]);
  }

  if (argc > 2) {
    do_move = std::atoi(argv[2]);
  }

  switch(num_flows) {
  case 0: run_bench<0>(do_move); break;
  case 1: run_bench<1>(do_move); break;
  case 2: run_bench<2>(do_move); break;
  case 4: run_bench<4>(do_move); break;
  case 8: run_bench<8>(do_move); break;
  default: std::cout << "Unsupported number of flows: " << num_flows << std::endl;
  }

  ttg_finalize();
  return 0;
}

