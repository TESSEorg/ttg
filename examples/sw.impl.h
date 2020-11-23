#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdlib.h> // std::atoi, std::rand()
#include <iomanip>
#include <string>
#include <memory>
#include "blockmatrix.h"
//#include <omp.h>

// #include "util.h" --> the code was copied here

/*
 * How to Compile? 
 *    1) source modules
 *    2) g++ -std=c++11 -fopenmp SW_OpenMP.cpp -o exec_OpenMP
 *
 * How to Run?
 *    the argument 256 will make two string of size 256 and pass it to the functions
 *    1) ./exec_OpenMP -n 256 -r 2 -b 16
 *
 */

static int M[5][5] = {{-8, -2, -2, -2, -2},
                      {-4,  5,  2,  2,  2},
                      {-4,  2,  5,  2,  2},
                      {-4,  2,  2,  5,  2},
                      {-4,  2,  2,  2,  5}};

static inline int charToIdx(char c) {
   switch (c) {
      case '_':
      case 0:
         return 0;
      case 'A':
      case 'a':
         return 1;
      case 'C':
      case 'c':
         return 2;
      case 'G':
      case 'g':
         return 3;
      case 'T':
      case 't':
         return 4;
      default:
         return -1;
   }
}

static inline int get_score(char a, char b) {
   return M[charToIdx(a)][charToIdx(b)];
}

int SW_serial(const std::string &a, const std::string &b);
int SW_OpenMP(const std::string &a, const std::string &b,
              int r, int base_size);
//void SW_OpenMP(int *X, int block_size, int i_lb, int j_lb, int  r,
//              int base_size, const std::string &a, const std::string &b,
//              int problem_size); 

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

using Key = std::pair<int, int>; //I, J

template <typename T>
BlockMatrix<T> sw_iterative(int I, int J, BlockMatrix<T> X, BlockMatrix<T> left, BlockMatrix<T> top, 
                  BlockMatrix<T> diag, int block_size, const std::string &a, 
                  const std::string &b, int problem_size) {
  //std::cout << "Executing " << I << " " << J << "-------" << std::endl;
  for(int i = 0; i < block_size; ++i) {
    int abs_i = I * block_size + i;
    for(int j = 0; j < block_size; ++j) {
      int abs_j = J * block_size + j;

      int left_value = ((j == 0) ? (J > 0 ? left(i, block_size - 1) : 
                      (abs_i+1)*get_score(a[abs_i],'_')) : X(i, j-1)) + 
                      get_score('_', b[abs_j]);

      int top_value = ((i == 0) ? (I > 0 ? top(block_size - 1, j) : 
                      (abs_j+1)*get_score('_', b[abs_j])) : X(i-1, j)) + 
                      get_score(a[abs_i], '_');

      int diag_value = get_score(a[abs_i], b[abs_j]);

      if (abs_i > 0 && abs_j > 0) { 
        if (i > 0 && j > 0)
          diag_value += X(i-1, j-1);
        else if (i > 0)
          diag_value += left(i-1, block_size-1);
        else if (j > 0)
          diag_value += top(block_size-1, j-1);
        else
          diag_value += diag(block_size-1, block_size-1);
      }
      else if (abs_i > 0) { 
        diag_value += (abs_i)*get_score(a[abs_i],'_'); 
      }
      else if (abs_j > 0) { 
        diag_value += (abs_j)*get_score('_', b[abs_j]); 
      }
    
      X(i,j) = std::max({left_value, top_value, diag_value});
      //std::cout << left_value << " " << top_value << " " << diag_value << "-->" << X(i,j) << " ";
    }
    //std::cout << std::endl;
  }
  return X;
}

template <typename T>
auto make_result(bool verify, T expected, Edge<Key, T> result)
{
  auto f = [verify, expected](const Key& key, const T& r, std::tuple<>& out) {
    if (verify) {
      if (r != expected) 
        std::cout << "FAILED! " << r << " != " << expected << std::endl;
      else
        std::cout << "SUCCESS!\n";
    }
  };
    
  return wrap(f, edges(result), edges(), "Final Output", {"result"}, {}); 
}

template <typename funcT, typename T>
auto make_sw2(const funcT& func, int block_size, const std::string &a, const std::string &b, 
            int problem_size, Edge<Key, BlockMatrix<T>>& leftedge, Edge<Key, BlockMatrix<T>>& topedge,
            Edge<Key, BlockMatrix<T>>& diagedge, Edge<Key, T>& resultedge) {
  auto f = [block_size, problem_size, a, b, func](const Key& key, BlockMatrix<T>&& left,
              BlockMatrix<T>&& top, BlockMatrix<T>&& diag, 
              std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, 
              Out<Key, BlockMatrix<T>>, Out<Key, T>>& out) { 
    // Getting the block coordinates
    auto[i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;
    int num_blocks = problem_size / block_size;

    BlockMatrix<T> X(block_size, block_size);
    X = sw_iterative(i, j, X, left, top, diag, block_size, a, b, problem_size);
 
    //std::cout << X << std::endl; 
    if (next_i < num_blocks) {
      send<1>(Key(next_i, j), X, out);
    }
    if (next_j < num_blocks) {
      send<0>(Key(i, next_j), X, out);
    }
    if (next_i < num_blocks && next_j < num_blocks) {
      send<2>(Key(next_i, next_j), X, out); //send diagonal block for next block computation
    }
  
    if (i == num_blocks - 1 && j == num_blocks - 1)
      send<3>(Key(i,j), X(block_size-1, block_size-1), out);
  };

  return wrap(f, edges(leftedge, topedge, diagedge), edges(leftedge, topedge, diagedge, resultedge), 
            "sw2", {"leftedge", "topedge", "diagedge"}, {"leftedge", "topedge", "diagedge", "result"});
}

template <typename funcT, typename T>
auto make_sw1(const funcT& func, int block_size, const std::string &a, const std::string &b,
            int problem_size, Edge<Key, BlockMatrix<T>>& leftedge, Edge<Key, BlockMatrix<T>>& topedge,
            Edge<Key, BlockMatrix<T>>& diagedge, Edge<Key, T>& resultedge) {
  auto f = [block_size, problem_size, a, b, func](const Key& key, BlockMatrix<T>&& toporleft,
              std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
              Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, T>>& out) {
    // Getting the block coordinates
    auto[i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;
    int num_blocks = problem_size / block_size;

    BlockMatrix<T> X(block_size, block_size);
    if (i == 0 && j == 0) {
      //No top, left or diagonal blocks
      X = sw_iterative(i, j, X, X, X, X, block_size, a, b, problem_size);
    }
    else if (i == 0) {
      //Only left block, single dependency
      X = sw_iterative(i, j, X, toporleft, X, X, block_size, a, b, problem_size);
    }
    else if (j == 0) {
      //Only top block, single dependency
      X = sw_iterative(i, j, X, X, toporleft, X, block_size, a, b, problem_size);
    }

    //std::cout << X << std::endl;
    if (next_i < num_blocks) {
      //std::cout << "left " << next_i << " " << j << std::endl;
      if (j == 0)  // send top block for next block computation
        send<0>(Key(next_i, j), X, out);
      else  // send top block for next block computation
        send<2>(Key(next_i, j), X, out);
    }
    if (next_j < num_blocks) {
      if (i == 0)  // send left block for next block computation
        send<0>(Key(i, next_j), X, out);
      else  // // send left block for next block computation
        send<1>(Key(i, next_j), X, out);
    }
    if (next_i < num_blocks && next_j < num_blocks) {
      send<3>(Key(next_i, next_j), X, out); //send diagonal block for next block computation
    }
  
    if (i == num_blocks - 1 && j == num_blocks - 1)
      send<4>(Key(i,j), X(block_size-1, block_size-1), out);
  };

  Edge<Key, BlockMatrix<T>> recur("recur");
  return wrap(f, edges(recur), edges(recur, leftedge, topedge, diagedge, resultedge), "sw1", {"recur"},
              {"recur", "leftedge", "topedge", "diagedge", "resultedge"});
}

int main(int argc, char* argv[]) {
  if (argc < 6)
  {
    std::cout << "Usage: ./sw -n <String length> -b <block_size> <1/0 - verify>\n";
    exit(-1);
  }

  int problem_size = std::atoi(argv[2]);
  int block_size = std::atoi(argv[4]);
  bool verify = std::atoi(argv[5]);

  char chars[] = {'A', 'C', 'G', 'T'};
  std::string a, b;
  for (int i = 0; i < problem_size; ++i) {
     a += chars[std::rand() % 4];
     b += chars[std::rand() % 4];
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  std::cout << "Computing using serial version....\n";
  beg = std::chrono::high_resolution_clock::now();
  int val1 = SW_serial(a, b);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "....done!" << std::endl;
  std::cout << "Serial Execution Time (milliseconds) : "
            << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1e3 << std::endl;

  //int val2 = SW_OpenMP(a, b, r, base_size);
  ttg_initialize(argc, argv, -1);

  Edge<Key, BlockMatrix<int>> leftedge, topedge, diagedge;
  Edge<Key, int> resultedge;
  auto s = make_sw1(sw_iterative<int>, block_size, a, b, problem_size, leftedge, topedge,
                  diagedge, resultedge);
  auto s1 = make_sw2(sw_iterative<int>, block_size, a, b, problem_size, leftedge, topedge,
                  diagedge, resultedge);
  auto r = make_result(verify, val1, resultedge);

  auto connected = make_graph_executable(s.get());
  assert(connected);
  TTGUNUSED(connected);
  std::cout << "Graph is connected.\n";

  if (ttg_default_execution_context().rank() == 0) {
    //std::cout << "==== begin dot ====\n";
    //std::cout << Dot()(s.get()) << std::endl;
    //std::cout << "==== end dot ====\n";

    beg = std::chrono::high_resolution_clock::now();
    s->in<0>()->send(Key(0,0), BlockMatrix<int>());
  }

  ttg_execute(ttg_default_execution_context());
  ttg_fence(ttg_default_execution_context());
  end = std::chrono::high_resolution_clock::now();
  std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;
  ttg_finalize();
}

int SW_serial(const std::string &a, const std::string &b) {
   size_t n = a.length();
   int *X = new int[n*n]; // (int*) malloc(n*n*sizeof(int));
   for(size_t i = 0; i < n; ++i) { // updating the row X[i][...]
      for(size_t j = 0; j < n; ++j) { // updating the cell X[i][j]
         int left_value = (j > 0 ? X[i*n + j-1] : (i+1)*get_score(a[i],'_')) + 
                            get_score('_', b[j]);

         int top_value = (i > 0 ? X[(i-1)*n + j] : (j+1)*get_score('_', b[j])) + 
                           get_score(a[i], '_');

         int diag_value = get_score(a[i], b[j]);
         if (i > 0 && j > 0) { diag_value += X[(i-1)*n + j-1]; }
         else if (i > 0) { diag_value += (i)*get_score(a[i],'_');  }
         else if (j > 0) { diag_value += (j)*get_score('_', b[j]); }

         X[i*n+j] = std::max({left_value, top_value, diag_value});
        //std::cout << left_value << " " << top_value << " " << diag_value << "-->" << X[i*n+j] << " ";
      }
      //std::cout << std::endl;
   }
   
   // returning the data at the bottom-right as the final value
   int final_value = X[(n-1)*n+(n-1)];
   delete[] X;
   return final_value;
}

/*int SW_OpenMP(const std::string &a, const std::string &b,
              int r, int base_size) {
   size_t n = a.length();
   int *X = new int[n*n]; // (int*) malloc(n*n*sizeof(int));
   #pragma omp parallel
   {
      #pragma omp single
      {
         #pragma omp task
         SW_OpenMP(X, n, 0, 0, r, base_size, a, b, n);
      }
   }
   // returning the data at the bottom-right as the final value
   int final_value = X[(n-1)*n+(n-1)];
   delete[] X;
   return final_value;
}*/

/*void SW_OpenMP(int *X, int block_size, int i_lb, int j_lb, int  r,
              int base_size, const std::string &a, const std::string &b,
              int problem_size) {
   if (block_size <= base_size || block_size <= r) {
      for(int i = i_lb; i < i_lb + block_size; ++i) {
         for(int j = j_lb; j < j_lb + block_size; ++j) {
            int left_value =
               (j > 0 ? X[i*problem_size + j-1] : (i+1)*get_score(a[i],'_')) +
                  get_score('_', b[j]);
            int top_value =
               (i > 0 ? X[(i-1)*problem_size + j] : (j+1)*get_score('_', b[j])) +
                  get_score(a[i], '_');

            int diag_value = get_score(a[i], b[j]);
            if (i > 0 && j > 0) { diag_value += X[(i-1)*problem_size + j-1]; }
            else if (i > 0) { diag_value += (i)*get_score(a[i],'_');  }
            else if (j > 0) { diag_value += (j)*get_score('_', b[j]); }

            X[i*problem_size+j] = std::max({left_value, top_value, diag_value});
         }
      }
      return;
   }
   else {
      int tile_size = block_size/r;
      // source-code for diagonal iteration of the matrix:
      // https://www.geeksforgeeks.org/zigzag-or-diagonal-traversal-of-matrix/
      for(int kk = 1; kk <= 2*r-1; ++kk) {
         int start_column = std::max(0, kk - r);
         int count = std::min({kk, r-start_column, r});
         for(int ii = 0; ii < count; ++ii) {
            #pragma omp task
            SW_OpenMP(X, tile_size, i_lb+(std::min(r, kk)-ii-1)*tile_size,
                      j_lb+(start_column+ii)*tile_size, r, base_size, a, b, problem_size);
         }
         #pragma omp taskwait
      }
      return;
   }
}*/
