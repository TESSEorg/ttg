#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdlib.h> // std::atoi, std::rand()
#include <iomanip>
#include <string>
#include <memory>

#define BLOCK_SIZE 1024

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

using fileKey = std::string;
template<typename T>
using Key = std::pair<std::string, T>; //<filename, chunkID>J
template<typename T>
using MapKey = std::multimap<std::string, T>;

namespace madness {
  namespace archive {
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, MapKey<T>> {
      static inline void store(const Archive& ar, const MapKey<T>& mk) {
        ar & mk; //BlockMatrix<T>(bm.rows(), bm.cols());
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, MapKey<T>> {
      static inline void load(const Archive& ar, MapKey<T>& mk) {
        int size;
        ar & size;
        for (const auto &mki : mk) ar & mki;
      }
    };
  }
}

template<typename T>
auto make_reader(Edge<Key<T>, std::string>& mapEdge)
{
  auto f = [](const fileKey& filename, std::tuple<Out<Key<T>,std::string>>& out) {
    //auto [filename, id] = key; 
    //check if file exists
    std::ifstream fin(filename);
    if (!fin) {
      std::cout << "File not found : " << fin << std::endl;
      ttg_abort();
    }

    //Read the file in chunks and send it to a mapper.
    std::string buffer; //reads only the first BLOCK_SIZE bytes
    buffer.resize(BLOCK_SIZE);
    int first = 0;
    int chunkID = 0;

    while(!fin.eof()) {
      char * b = const_cast< char * >( buffer.c_str() );
      fin.read(b + first, BLOCK_SIZE );
      //fin.read(buffer.data(), buffer.size());
      std::streamsize s = first + fin.gcount();
      buffer.resize(s);
      //Special handling to avoid splitting words between chunks.
      if (s > 0) {
        auto last = buffer.find_last_of(" \t\n");
        first = s - last - 1;
        std::string tmp;
        if (fin) {
          tmp.resize(BLOCK_SIZE + first);
          if (first > 0) tmp.replace(0, first, buffer, last + 1, first);
        }
        buffer.resize(last);
        std::cout << buffer << std::endl;
        send<0>(Key<T>(filename, chunkID), buffer, out);
        buffer = tmp;
        chunkID++;
      }
    }
  };   

  return wrap<fileKey>(f, edges(), edges(mapEdge), "reader", {}, {"mapEdge"});
}

template<typename T>
void mapper(std::string chunk, MapKey<T>& resultMap) {
  //Prepare the string by removing all punctuation marks
  chunk.erase(std::remove_if(chunk.begin(), chunk.end(),
          []( auto const& c ) -> bool { return ispunct(c); } ), chunk.end());
  std::istringstream ss(chunk);
  std::string word;
  while (ss >> word)
  {
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    //std::cout << "Mapped " << word << std::endl;
    resultMap.insert(std::make_pair(word, 1));
  }
}

template <typename funcT, typename T>
auto make_mapper(const funcT& func, Edge<Key<T>, std::string>& mapEdge, Edge<std::pair<Key<T>, T>, MapKey<T>>& reduceEdge) 
{
  auto f = [func](const Key<T>& key, std::string&& chunk, std::tuple<Out<std::pair<Key<T>, T>, MapKey<T>>>& out)
  {
    MapKey<T> resultMap;
    //Call the mapper function
    func(chunk, resultMap);
    send<0>(std::make_pair(key, 0), resultMap, out);
    //for (MapKey<T>::iterator it = resultMap.begin(); it != resultMap.end(); it++)
    //  std::cout << it->first << ":" << it->second << std::endl;
  };
  
  return wrap(f, edges(mapEdge), edges(reduceEdge), "mapper", {"mapEdge"}, {"reduceEdge"}); 
}

template <typename funcT, typename T>
auto make_reducer(const funcT& func, Edge<std::pair<Key<T>, T>, MapKey<T>>& reduceEdge, 
                Edge<std::pair<Key<T>, T>, std::pair<std::string, T>>& writerEdge)
{
  auto f = [func](const std::pair<Key<T>, T>& key, MapKey<T>&& inputMap, 
                std::tuple<Out<std::pair<Key<T>, T>, MapKey<T>>,
                Out<std::pair<Key<T>, T>, std::pair<std::string, T>>>& out)
  {  
    typename MapKey<T>::iterator iter;
    int value = 0;
    //Need a tokenID to make keys unique for recurrence
    int tokenID = key.second + 1;
  
    iter = inputMap.begin();
    int count = inputMap.count(iter->first);
    if (count > 1) {
      for (iter; iter != inputMap.end() && count > 0; iter++) 
      {
        value = func(value, iter->second); 
        count--;
        if (count == 0) {
          send<1>(std::make_pair(key.first, tokenID), std::make_pair(iter->first, value), out);
          //std::cout << "Sent token " << tokenID << " <" << iter->first << " " << value << ">" << std::endl;
          tokenID++;
          value = 0;
        }
        inputMap.erase(iter);
      }
      if (iter != inputMap.end()) {
        send<0>(std::make_pair(key.first, tokenID), inputMap, out);
        //std::cout << "Recurring token " << tokenID << std::endl;
        tokenID++;
      }
    }
    else {
      send<1>(std::make_pair(key.first, tokenID), std::make_pair(iter->first, iter->second), out);
      //std::cout << "Sent token " << tokenID << " <" << iter->first << " " << iter->second << ">" << std::endl;
      tokenID++;
      inputMap.erase(iter);
      iter++;
      if (iter != inputMap.end()) {
        send<0>(std::make_pair(key.first, tokenID), inputMap, out);
        //std::cout << "Recurring token " << tokenID << std::endl;
        tokenID++;
      }
    }
  };
    
  return wrap(f, edges(reduceEdge), edges(reduceEdge, writerEdge), "reducer", {"reduceEdge"}, 
            {"recurReduceEdge","writerEdge"});
}

template<typename T>
auto make_writer(Edge<std::pair<Key<T>, T>, std::pair<std::string, T>>& writerEdge)
{
  auto f = [](const std::pair<Key<T>, T>& key, std::pair<std::string, T> &&value, std::tuple<>& out) {
    std::cout << value.first << " " << value.second << std::endl;
  };

  return wrap(f, edges(writerEdge), edges(), "writer", {"writerEdge"}, {});
}

int main(int argc, char* argv[]) {
  if (argc < 2)
  {
    std::cout << "Usage: ./mapreduce file1 [file2, ...]\n";
    exit(-1);
  }

  
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  ttg_initialize(argc, argv, -1);
  //OpBase::set_trace_all(true);

  Edge<Key<int>, std::string> mapEdge;
  Edge<std::pair<Key<int>, int>, MapKey<int>> reduceEdge;
  Edge<std::pair<Key<int>, int>, std::pair<std::string, int>> writerEdge;  
  
  auto rd = make_reader(mapEdge);
  auto m = make_mapper(mapper<int>, mapEdge, reduceEdge);
  auto r = make_reducer(std::plus<int>(), reduceEdge, writerEdge);
  auto w = make_writer(writerEdge);
  
  auto connected = make_graph_executable(rd.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected.\n";

  if (ttg_default_execution_context().rank() == 0) {
    //std::cout << "==== begin dot ====\n";
    //std::cout << Dot()(rd.get()) << std::endl;
    //std::cout << "==== end dot ====\n";
  
    beg = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < argc; i++) {
      std::string s(argv[i]);
      rd->invoke(s);
      //rd->in<0>()->send(argv[i]);
    }
  }

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());

  end = std::chrono::high_resolution_clock::now();
  
  std::cout << "Mapreduce took " << 
      (std::chrono::duration_cast<std::chrono::seconds>(end - beg).count()) << 
      " seconds" << std::endl;
  
  ttg_finalize();
  return 0;
}
