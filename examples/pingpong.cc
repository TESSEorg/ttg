
#include <ttg.h>
#include <ttg/serialization.h>
#include <cassert>

#define NITER 10000

#define MAX_SIZE 1024*1024*1024

struct Message : public ttg::TTValue<Message> {
  ttg::Buffer<std::byte> buf;

  Message(std::size_t size)
  : buf(size)
  { }

  Message(const Message&) = delete;
  Message(Message) = default;

  Message& operator=(const Message&) = delete;
  Message& operator=(Message&&) = default;

  std::size_t size() const {
    return buf.size();
  }

};

#ifdef TTG_SERIALIZATION_SUPPORTS_MADNESS
namespace madness {
  namespace archive {
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, BlockMatrix<T>> {
      static inline void store(const Archive& ar, const BlockMatrix<T>& bm) {
        ar << msg.buf.size();
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, BlockMatrix<T>> {
      static inline void load(const Archive& ar, Message& msg) {
        std::size_t size;
        ar >> size;
        msg = Message(size);
      }
    };
  }  // namespace archive
}  // namespace madness

static_assert(madness::is_serializable_v<madness::archive::BufferOutputArchive, Message>);
#endif  // TTG_SERIALIZATION_SUPPORTS_MADNESS

static void run_inter()
{
  auto world = ttg::default_execution_context();
  int comm_rank = world.rank();
  int comm_size = world.size();
  ttg::Edge<int, Message> edge;
  auto ping = ttg::make_tt([&](int key, Message&& msg) -> ttg::device::Task {
                              co_await ttg::device::select(msg.buf);
                              if (NITER > key) {
                                /* go for another ride */
                                co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(msg)));
                              }
                            }, ttg::edges(edge), ttg::edges(edge), "ping");

  ping->set_keymap([](int key){ return key % comm_size; });

  ttg::make_graph_executable(ping);

  for (std::size_t size = 1; size < MAX_SIZE; size *= 2) {
    /* mark executable */
    ttg::execute();
    auto start_ts = std::chrono::steady_clock::now();
    /* kick off run */
    if (rank == 0) {
      ping->invoke(0, Message(size));
    }
    /* wait for completion */
    ttg::fence();
    auto end_ts = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end_ts - start_ts);
    std::cout << "INTER " << size <<  " B : " << duration << " s : " << size / duration.count() / 1E6 " MB/s" << std::endl;
  }
}

static void run_intra()
{
  auto world = ttg::default_execution_context();
  int comm_rank = world.rank();
  int comm_size = world.size();
  ttg::Edge<int, Message> edge;
  auto ping = ttg::make_tt([&](int key, Message&& msg) -> ttg::device::Task {
                              co_await ttg::device::select(msg.buf);
                              if (NITER > key) {
                                /* go for another ride */
                                co_await ttg::device::forward(ttg::device::send<0>(key+1, std::move(msg)));
                              }
                            }, ttg::edges(edge), ttg::edges(edge), "ping");

  ping->set_keymap([](int key){ return rank; /* stay within a process */ });
  ping->set_devicemap([](int key){ return key % ttg::device::num_devices()});

  ttg::make_graph_executable(ping);

  for (std::size_t size = 1; size < MAX_SIZE; size *= 2) {
    /* mark executable */
    ttg::execute();
    auto start_ts = std::chrono::steady_clock::now();
    /* kick off run */
    if (rank == 0) {
      ping->invoke(0, Message(size));
    }
    /* wait for completion */
    ttg::fence();
    auto end_ts = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end_ts - start_ts);
    std::cout << "INTRA " << size <<  " B : " << duration << " s : " << size / duration.count() / 1E6 " MB/s" << std::endl;
  }
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv);
  // inter-node benchmark
  run_inter();
  ttg::finalize();

  return 0;
}