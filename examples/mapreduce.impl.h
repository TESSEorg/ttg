#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdlib.h> // std::atoi, std::rand()
#include <iomanip>
#include <string>
#include <memory>

#define BLOCK_SIZE 16

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS
#include "../ttg/util/reduce.h"

template<typename T>
using Key = std::pair<std::pair<std::string, T>, T>;
template<typename T>
using MapKey = std::multimap<std::string, T>;

namespace madness {
  namespace archive {
    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, MapKey<T>> {
      static inline void store(const Archive& ar, const MapKey<T>& mk) {
        int size = mk.size();;
        ar & size;
        typename MapKey<T>::const_iterator it = mk.begin();
        //std::cout << "Storing ..." << mk.size() << std::endl;
        //for (typename MapKey<T>::const_iterator it = mk.begin(); it != mk.end(); it++)
        while (size--) {
          ar & it->first;
          //std::cout << "Sent : " << it->first << std::endl;
          ar & it->second;
          it++;
        }
      }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, MapKey<T>> {
      static inline void load(const Archive& ar, MapKey<T>& mk) {
        int size;
        ar & size;
        while (size--) {
          std::string s;
          T v;
          ar & s;
          ar & v;
          //std::cout << "Loading..." << s << " " << v << std::endl;
          mk.insert(std::make_pair(s, v));
        }
      }
    };
  }
}

template <typename T>
std::ostream& operator<<(std::ostream& s, const Key<T>& key) {
  s << "Key((" << key.first.first << "," << key.first.second << "), " << key.second << ")";
  return s;
}

template<typename T>
auto make_reader(Edge<Key<T>, std::string>& mapEdge)
{
  auto f = [](const Key<T>& filename, std::tuple<Out<Key<T>,std::string>>& out) {
    //auto [filename, id] = key; 
    //check if file exists
    std::ifstream fin(filename.first.first);
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
        //std::cout << buffer << std::endl;
        send<0>(std::make_pair(std::make_pair(filename.first.first, chunkID), 0), buffer, out);
        buffer = tmp;
        chunkID++;
      }
    }
    //This marks the end of the file and is needed for combining the output of all reducers
    //send<0>(std::make_pair(std::make_pair(filename.first.first, chunkID), 0), std::string(), out);
  };   

  return wrap<Key<T>>(f, edges(), edges(mapEdge), "reader", {}, {"mapEdge"});
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
auto make_mapper(const funcT& func, Edge<Key<T>, std::string>& mapEdge, Edge<Key<T>, MapKey<T>>& reduceEdge) 
{
  auto f = [func](const Key<T>& key, std::string& chunk, std::tuple<Out<Key<T>, MapKey<T>>>& out)
  {
    MapKey<T> resultMap;
    //If buffer is empty, Key contains all chunks for the input file, which is required for reducing reducers
    if (chunk == std::string())
    {
      //Send the chunk count to reducers
      send<0>(key, resultMap, out); 
    }
    else {
      //Call the mapper function
      func(chunk, resultMap);
      send<0>(key, resultMap, out);
    }
  };
  
  return wrap(f, edges(mapEdge), edges(reduceEdge), "mapper", {"mapEdge"}, {"reduceEdge"}); 
}

template <typename funcT, typename T>
auto make_reducer(const funcT& func, Edge<Key<T>, MapKey<T>>& reduceEdge, 
                Edge<void, std::pair<std::string, T>>& writerEdge)
{
  auto f = [func](const Key<T>& key, MapKey<T>& inputMap, 
                std::tuple<Out<Key<T>, MapKey<T>>,
                Out<void, std::pair<std::string, T>>>& out)
  { 
    typename MapKey<T>::iterator iter;
    int value = 0;
    //Need a tokenID to make keys unique for recurrence
    int tokenID = key.second + 1;

    /*if (inputMap.begin() == inputMap.end()) {
      //std::cout << "empty...\n";
      finalize<1>(std::make_pair(key.first, 0), out);
    }*/

    iter = inputMap.begin();
    //std::pair <typename MapKey<T>::iterator, typename MapKey<T>::iterator> range = inputMap.equal_range(iter->first);
    //value = std::reduce(range.first, range.second, 0, func);

    //Count of elements with same key
    int count = inputMap.count(iter->first);
    if (count > 1) {
      for (iter; iter != inputMap.end() && count > 0; iter++) 
      {
        value = func(value, iter->second); 
        count--;
        if (count == 0) {
          sendv<1>(std::make_pair(iter->first, value), out);
          //std::get<1>(out).send(std::make_pair(key.first, tokenID), std::make_pair(iter->first, value));
          //std::cout << "Sent token " << tokenID << " <" << iter->first << " " << value << ">" << std::endl;
          value = 0;
        }
        inputMap.erase(iter);
      }
      if (iter != inputMap.end()) {
        send<0>(std::make_pair(key.first, tokenID), inputMap, out);
        //std::cout << "Recurring token " << tokenID << std::endl;
      }
    }
    else {
      sendv<1>(std::make_pair(iter->first, iter->second), out);
      //std::get<1>(out).send(std::make_pair(key.first, tokenID), std::make_pair(iter->first, iter->second)); 
      //std::cout << "Sent token " << tokenID << " <" << iter->first << " " << iter->second << ">" << std::endl;
      inputMap.erase(iter);
      iter++;
      if (iter != inputMap.end()) {
        send<0>(std::make_pair(key.first, tokenID), inputMap, out);
        //std::cout << "Recurring token " << tokenID << std::endl;
      }
    }
  };
    
  return wrap(f, edges(reduceEdge), edges(reduceEdge, writerEdge), "reducer", {"reduceEdge"}, 
            {"recurReduceEdge","writerEdge"});
}

template <typename T>
struct Node {
  T data;
  Node* left;
  Node* right;
  Node* parent;
};

template<typename T>
auto make_writer(std::map<std::string, T>& resultMap, Edge<void, std::pair<std::string, T>>& writerEdge)
{
  auto f = [&resultMap](std::pair<std::string, T> &value, std::tuple<>& out) {
    //Lock is required since multiple threads can stream the tokens to this unique task.
    auto it = resultMap.find(value.first);
    if (it != resultMap.end())
      resultMap[value.first] += value.second;
    else
      resultMap.insert(value);
    //std::cout << "Write Rank : " << ttg_default_execution_context().rank() << std::endl;
    //std::cout << key.second << ":" <<  value.first << " " << resultMap[value.first] << std::endl;
  };

  //auto w = make_op(f, edges(writerEdge), edges(), "writer");//, {"writerEdge"}, {});
  //w->set_input_reducer<0>([](std::pair<std::string, int> &&a, std::pair<std::string, int> &&b)
  //                      {
  //                        return a.second + b.second;
  //                      });
  //return w;
  //return std::unique_ptr(wrap<void>(f, edges(writerEdge), edges(), "writer", {"writerEdge"}, {}));
  return wrap<void>(f, edges(writerEdge), edges(), "writer", {"writerEdge"}, {}); 
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

  //madness::redirectio(madness::World::get_default(), false);
  
  Edge<Key<int>, std::string> mapEdge;
  Edge<Key<int>, MapKey<int>> reduceEdge;
  Edge<void, std::pair<std::string, int>> writerEdge;  
  
  auto rd = make_reader(mapEdge);
  auto m = make_mapper(mapper<int>, mapEdge, reduceEdge);
  auto r = make_reducer(std::plus<int>(), reduceEdge, writerEdge);
  
  std::map<std::string, int> result;
  auto w = make_writer(result, writerEdge);

  /*w->set_input_reducer<0>([](std::pair<std::string, int> &&a, std::pair<std::string, int> &&b)
                        {
                          std::cout << "Reduced : " << a.first << " " << b.first << " " << a.second + b.second << std::endl;
                          return std::pair(a.first, a.second + b.second);
                          
                       });
  */

  auto connected = make_graph_executable(rd.get());
  assert(connected);
  TTGUNUSED(connected);
  //std::cout << "Graph is connected.\n";

  if (ttg_default_execution_context().rank() == 0) {
    //std::cout << "==== begin dot ====\n";
    //std::cout << Dot()(rd.get()) << std::endl;
    //std::cout << "==== end dot ====\n";
  
    beg = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < argc; i++) {
      std::string s(argv[i]);
      rd->invoke(std::make_pair(std::make_pair(s,0),0));
      //rd->in<0>()->send(argv[i]);
    }
  }

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());

  if (ttg_default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();

    /*std::cout << "==================== RESULT ===================\n"; 
    for(auto it : result) {
      std::cout << it.first << " " << it.second << std::endl;   
    } */
    std::cout << "Mapreduce took " << 
        (std::chrono::duration_cast<std::chrono::seconds>(end - beg).count()) << 
        " seconds" << std::endl;
    for(auto it : result) {
      std::cout << it.first << " " << it.second << std::endl;
    }
  }
  
  /*std::cout << "==================== RESULT from rank " << ttg_default_execution_context().rank() << " ===================\n";
    for(auto it : result) {
      std::cout << it.first << " " << it.second << std::endl;
    }*/
  ttg_finalize();
  return 0;
}
