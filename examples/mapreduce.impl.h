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
using Key = std::pair<std::string, long long int>; //<filename, chunkID>J
using MapType = std::pair<std::string, int>;

auto make_reader(Edge<Key, std::string>& mapEdge)
{
  auto f = [](const fileKey& filename, std::tuple<Out<Key,std::string>>& out) {
    //auto [filename, id] = key; 
    //check if file exists
    std::ifstream fin(filename);
    if (!fin) {
      std::cout << "File not found : " << fin << std::endl;
    }

    //Read the file in chunks and send it to a mapper.
    std::string buffer; //reads only the first 1024 bytes
    buffer.resize(BLOCK_SIZE);
    int first = 0;
    long long int chunkID = 0;

    while(!fin.eof()) {
      char * b = const_cast< char * >( buffer.c_str() );
      fin.read(b + first, BLOCK_SIZE );
      //fin.read(buffer.data(), buffer.size());
      std::streamsize s = fin.gcount();
      //Special handling to avoid splitting words between chunks.
      if (s > 0) {
        auto last = buffer.find_last_of(" \t\n");
        first = s - last - 1;
        std::string tmp;
        if (fin) {
          tmp.resize(1024 + first);
          if (first > 0) tmp.replace(0, first, buffer, last + 1, first);
        }
        buffer.resize(last);
        send<0>(Key(filename, chunkID), buffer, out);
      }
    }
  };   

  return wrap<fileKey>(f, edges(), edges(mapEdge), "reader", {}, {"mapEdge"});
}

std::map<std::string, int> mapper(std::string chunk) {
  std::istringstream ss(chunk);
  std::string word;
  std::map<std::string, int> resultMap;
  while (ss >> word)
  {
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    resultMap.insert(std::make_pair(word, 1));
  }

  return resultMap;
}

template <typename funcT>
auto make_mapper(const funcT& func, Edge<Key, std::string>& mapEdge, Edge<std::string, int>& reduceEdge) 
{
  auto f = [func](const Key& key, std::string&& chunk, std::tuple<Out<std::string, int>>& out)
  {
    std::map<std::string, int> resultMap;
    //Call the mapper function
    resultMap = func(chunk);
    std::map<std::string, int>::iterator iter;
    for (iter = resultMap.begin(); iter != resultMap.end(); ++iter) {
      //std::cout << "Mapper sent " << iter->first << " " << iter->second << std::endl;
      send<0>(iter->first, iter->second, out); 
    }
  };
  
  return wrap(f, edges(mapEdge), edges(reduceEdge), "mapper", {"mapEdge"}, {"reduceEdge"}); 
}

template <typename funcT>
auto make_reducer(const funcT& func, Edge<std::string, int>& reduceEdge)
{
  auto f = [func](const std::string& key, int value, std::tuple<>& out) {
    std::cout << "<" << key << "," << value << ">" << std::endl;
  };
    
  return wrap(f, edges(reduceEdge), edges(), "reducer", {"reduceEdge"}, {});
}

int main(int argc, char* argv[]) {
  if (argc < 2)
  {
    std::cout << "Usage: ./mapreduce file1 [file2, ...]\n";
    exit(-1);
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;

  ttg_initialize(argc, argv, -1);

  Edge<Key, std::string> mapEdge;
  Edge<std::string, int> reduceEdge;
  
  auto rd = make_reader(mapEdge);
  auto m = make_mapper(mapper, mapEdge, reduceEdge);
  auto r = make_reducer(std::plus<int>(), reduceEdge);
  
  //r->set_input_reducer<0>([](int &&a, int &&b) { return a + b; });

  auto connected = make_graph_executable(rd.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected.\n";

  if (ttg_default_execution_context().rank() == 0) {
    std::cout << "==== begin dot ====\n";
    std::cout << Dot()(rd.get()) << std::endl;
    std::cout << "==== end dot ====\n";
  
    beg = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < argc; i++) {
      std::string s(argv[i]);
      rd->invoke(s);
      //rd->in<0>()->send(argv[i]);
    }
  }

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());
  //rd->finalize<0>->send(string::empty);
  //ttg_fence(ttg_defailt_execution_context());

  end = std::chrono::high_resolution_clock::now();
  
  std::cout << "Mapreduce took " << 
      (std::chrono::duration_cast<std::chrono::seconds>(end - beg).count()) << 
      " seconds" << std::endl;
  
  ttg_finalize();
  return 0;
}
