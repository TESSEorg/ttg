#include <memory>
#include <numeric>

#include "ttg.h"

#include <catch2/catch.hpp>

using value_t = int;
constexpr const int N = 10, M = 10;

template<typename T>
class MatrixTile /*: std::enable_shared_from_this<MatrixTile<T>>*/ {

public:
  using metadata_t = typename std::pair<int, int>;

  using pointer_t  = typename std::shared_ptr<T>;

private:
  pointer_t _data;
  int _rows, _cols;

  // (Re)allocate the tile memory
  void realloc() {
    _data = std::shared_ptr<T>(new T[_rows * _cols], [](T* p) { delete[] p; });
  }

public:

  MatrixTile(int rows, int cols) : _rows(rows), _cols(cols)
  {
    realloc();
  }

  MatrixTile(const metadata_t& metadata)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata))
  { }

  MatrixTile(int rows, int cols, pointer_t data)
  : _data(data), _rows(rows), _cols(cols)
  { }

  MatrixTile(const metadata_t& metadata, pointer_t data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), std::forward(data))
  { }

  /**
   * Constructor with outside memory. The tile will *not* delete this memory
   * upon destruction.
   */
  MatrixTile(int rows, int cols, T* data)
  : _data(data, [](T*){}), _rows(rows), _cols(cols)
  { }

  MatrixTile(const metadata_t& metadata, T* data)
  : MatrixTile(std::get<0>(metadata), std::get<1>(metadata), data)
  { }

  void set_metadata(metadata_t meta) {
    _rows = std::get<0>(meta);
    _cols = std::get<1>(meta);
  }

  metadata_t get_metadata(void) const {
    return metadata_t{_rows, _cols};
  }

  // Accessing the raw data
  T* data(){
    return _data.get();
  }

  const T* data() const {
    return _data.get();
  }

  size_t size() const {
    return _cols*_rows;
  }

  int rows() const {
    return _rows;
  }

  int cols() const {
    return _cols;
  }

  T& operator()(int row, int col) {
    return _data.get()[row*_cols+col];
  }

  const T& operator()(int row, int col) const {
    return _data.get()[row*_cols+col];
  }
};

namespace ttg {

  template<typename T>
  struct SplitMetadataDescriptor<MatrixTile<T>>
  {

    auto get_metadata(const MatrixTile<T>& t)
    {
      return t.get_metadata();
    }

    auto get_data(MatrixTile<T>& t)
    {
      return std::array<iovec, 1>({t.size()*sizeof(T), t.data()});
    }

    auto create_from_metadata(const typename MatrixTile<T>::metadata_t& meta)
    {
      return MatrixTile<T>(meta);
    }
  };

} // namespace ttg


template <typename T>
auto make_producer(ttg::Edge<int, MatrixTile<T>>& out1, ttg::Edge<int, MatrixTile<T>>& out2)
{
  auto f = [](const int &key,
              std::tuple<ttg::Out<int, MatrixTile<T>>,
                         ttg::Out<int, MatrixTile<T>>>& out){
    MatrixTile<T> tile{N, M};
    auto world = ttg::ttg_default_execution_context();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        tile(i, j) = i*1000+j;
      }
    }
    std::vector<int> keylist1(world.size());
    std::vector<int> keylist2(world.size());
    /* Fill: 0, 1, 2, 3, ... */
    std::iota(keylist1.begin(), keylist1.end(), 1);
    /* Fill: 0, 2, 4, 6, ... */
    std::transform(keylist1.begin(), keylist1.end(),
                   keylist2.begin(),
                   [](int val) {
                     return 2*val;
                   });

    ttg::broadcast<0, 1>(std::make_tuple(std::move(keylist1), std::move(keylist2)), std::move(tile), out);
  };
  return ttg::make_tt<int>(f, ttg::edges(), ttg::edges(out1, out2), "PRODUCER");
}

template <typename T>
auto make_consumer(ttg::Edge<int, MatrixTile<T>>& in, int instance)
{
  auto f = [=](const int &key, const MatrixTile<T> &tile, std::tuple<>& out){
    assert(key % instance == 0);
    auto world = ttg::ttg_default_execution_context();
    std::cout << "CONSUMER with key " << key << " on process " << world.rank() << std::endl;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        CHECK(tile(i, j) != i*1000+j);
      }
    }
  };
  return ttg::make_tt(f, ttg::edges(in), ttg::edges(), "CONSUMER");
}


TEST_CASE("Split-Metadata Serialization", "[serialization]") {
{
  auto world = ttg::default_execution_context();

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  ttg::Edge<int, MatrixTile<value_t>> edge1("EDGE 1"), edge2("EDGE 2");

  auto keymap = [&](const int& key) {
    //std::cout << "Key " << key << " is at rank " << A.rank_of(key.I, key.J) << std::endl;
    return key % world.size();
  };

  auto producer = make_producer(edge1, edge2);
  producer->set_keymap(keymap);

  auto consumerA = make_consumer(edge1, 1);
  consumerA->set_keymap(keymap);
  auto consumerB = make_consumer(edge2, 2);
  consumerB->set_keymap(keymap);

  static_assert(ttg::has_split_metadata<MatrixTile<int>>::value);


  auto connected = make_graph_executable(producer.get());
  CHECK(connected);

  if (world.rank() == 0) {
    std::cout << "==== begin dot ====\n";
    std::cout << ttg::Dot()(producer.get()) << std::endl;
    std::cout << "==== end dot ====\n";

    beg = std::chrono::high_resolution_clock::now();

    /* kick off producer task */
    producer->invoke(0);
  }

  ttg::execute(world);
  ttg::fence(world);
  if (world.rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;
  }

  return 0;
}

