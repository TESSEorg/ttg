#include <algorithm>  // for std::max
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include "blockmatrix.h"

/*!
        \file wavefront_df.impl.h
        \brief Wavefront computation on distributed memory
        \defgroup
        ingroup examples

        \par Points of interest
        - dynamic recursive DAG.
*/

using Key = std::pair<int, int>;

// An empty class used for pure control flows
struct Control {};

template <typename T>
inline BlockMatrix<T> stencil_computation(int i, int j, int M, int N, BlockMatrix<T> bm, BlockMatrix<T> left, 
                          BlockMatrix<T> top, BlockMatrix<T> right, BlockMatrix<T> bottom) {
  // i==0 -> no top block
  // j==0 -> no left block
  // i==M-1 -> no bottom block
  // j==N-1 -> no right block
  int MB = bm.rows();
  int NB = bm.cols();
  BlockMatrix<T> current = bm;
  for (int ii = 0; ii < MB; ++ii) {
    for (int jj = 0; jj < NB; ++jj) {
      current(ii,jj) = (current(ii,jj) + ((ii == 0) ? (i > 0 ? top(MB - 1, jj) : 0.0) : current(ii - 1, jj)));
      current(ii,jj) = (current(ii,jj) + ((ii == MB - 1) ? (i < M - 1 ? bottom(0, jj) : 0.0) : current(ii + 1, jj)));
      current(ii,jj) = (current(ii,jj) + ((jj == 0) ? (j > 0 ? left(ii, NB - 1) : 0.0) : current(ii, jj - 1)));
      current(ii,jj) = (current(ii,jj) + ((jj == NB - 1) ? (j < N - 1 ? right(ii, 0) : 0.0) : current(ii, jj + 1)));
      current(ii,jj) = current(ii,jj) * 0.25;
    }
  }
  return current;
}

// serial implementation of wavefront computation.
template <typename T>
void wavefront_serial(Matrix<T>* m, Matrix<T>* result, int n_brows, int n_bcols) {
  for (int i = 0; i < n_brows; i++) {
    for (int j = 0; j < n_bcols; j++) {
      BlockMatrix<T> left, top, right, bottom;
      if (i < n_brows - 1) bottom = ((*m)(i + 1, j));
      if (j < n_bcols - 1) right = ((*m)(i, j + 1));
      if (j > 0) left = ((*m)(i, j - 1));
      if (i > 0) top = ((*m)(i - 1, j));
      
      (*result)(i,j) = stencil_computation(i, j, n_brows, n_bcols, ((*m)(i,j)), (left), (top), (right), (bottom));
    }
  }
}

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

// Method to generate  wavefront tasks with two inputs.
template <typename funcT, typename T>
auto make_wavefront2(const funcT& func, int MB, int NB, Edge<Key, BlockMatrix<T>>& input, 
                      Edge<Key, BlockMatrix<T>>& left, Edge<Key, BlockMatrix<T>>& top, 
                      Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right,
                      Edge<Key, BlockMatrix<T>>& result) {
  
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input, BlockMatrix<T>&& left, BlockMatrix<T>&& top,
                     std::vector<BlockMatrix<T>>&& bottom_right,
                     std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, 
                     Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    int size = bottom_right.size(); 
    //std::cout << "wf2 " << i << " " << j << " " << "size: " << bottom_right.size() <<  std::endl;
    BlockMatrix<T> res;
    if (i == MB - 1 && j == NB - 1)
      res = func(i, j, MB, NB, input, left, top, input, input);
    else {
      if (size == 1)
        res = func(i, j, MB, NB, input, left, top, bottom_right[0], bottom_right[0]);
      else
        res = func(i, j, MB, NB, input, left, top, bottom_right[0], bottom_right[1]);
    }

    //Processing finished for this block, so send it to output Op
    send<2>(Key(i,j), res, out);

    if (next_i < MB) {
      send<1>(Key(next_i, j), res, out);
    }

    if (next_j < NB) {
      send<0>(Key(i, next_j), res, out);
    }
  };

  return wrap(f, edges(input, left, top, bottom_right), edges(left, top, result), "wavefront2", {"block_input", "left", "top", "bottom-right"}, {"left", "top", "result"});
}

template <typename T>
auto initiator(Matrix<T>* m, Edge<Key, BlockMatrix<T>>& out0, Edge<Key, BlockMatrix<T>>& out1, 
              Edge<Key, BlockMatrix<T>>& out2) { //Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right0,
              //Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right1,
              //Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right2) {
  auto f = [m](const Key& key, std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>,
               Out<Key, BlockMatrix<T>>>& out) 
              //Out<Key, std::vector<BlockMatrix<T>>>, 
              //Out<Key, std::vector<BlockMatrix<T>>>,
              //Out<Key, std::vector<BlockMatrix<T>>>>& out)
  {
    for (int i = 0; i < m->rows(); i++) {
      for (int j = 0; j < m->cols(); j++) {
        if (i == 0 && j == 0) {
          //std::cout << "send 0 : " << i << " " << j << std::endl; 
          send<0>(Key(i,j), (*m)(i,j), out);
        }
        else if ((i == 0 && j > 0) || (i > 0 && j == 0)) {
          //std::cout << "send 1 : " << i << " " << j << std::endl;
          send<1>(Key(i,j), (*m)(i,j), out);
        }
        else {
          //std::cout << "send 2 : " << i << " " << j << std::endl;
          send<2>(Key(i,j), (*m)(i,j), out);
        }
      }
    }
  };

  return wrap<Key>(f, edges(), edges(out0, out1, out2), //bottom_right0, bottom_right1, bottom_right2),
                  "initiator", {}, {"out0", "out1", "out2"}); 
                  //"bottom_right0", "bottom-right1", "bottom-right2"});
}

template <typename T>
auto make_getdata(Matrix<T>* m, Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right) {
  auto f = [m](const Key& key, std::tuple<Out<Key, std::vector<BlockMatrix<T>>>>& out) {
    //std::cout << "getdata called...\n";
    std::vector<BlockMatrix<T>> v;
    auto [i,j] = key;

    if (i == 0 && j == 0) {
      //std::cout << "send for 0 : " << i << " " << j << std::endl;
      //send<0>(Key(i,j), (*m)(i,j), out);
      v.push_back((*m)(i,j+1));
      v.push_back((*m)(i+1,j));
      send<0>(Key(i,j), v, out);
    }
  };

  return wrap<Key>(f, edges(), edges(bottom_right), "get_data", {},
                  {"get_bottom_right"});
}

template <typename T>
auto make_getdata1(Matrix<T>* m, Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right) {
  auto f = [m](const Key& key, std::tuple<Out<Key, std::vector<BlockMatrix<T>>>>& out) {
    //std::cout << "getdata1 called...\n";
    std::vector<BlockMatrix<T>> v;
    auto [i,j] = key;

    if ((i == 0 && j > 0) || (i > 0 && j == 0)) {
      //std::cout << "send for : " << i << " " << j << std::endl;
      if (j < m->cols() - 1) {
        //std::cout << "send 3 : " << i << " " << j << std::endl;
        v.push_back((*m)(i,j+1));
        //send<3>(Key(i,j), (*m)(i,j+1), out);
      }
      if (i < m->rows() - 1) {
        //std::cout << "send 4 : " << i << " " << j << std::endl;
        v.push_back((*m)(i+1,j));
        //send<4>(Key(i,j), (*m)(i+1,j), out);
      }
      send<0>(Key(i,j), v, out);
    }
  };

  return wrap<Key>(f, edges(), edges(bottom_right), "get_data1", {},
                  {"get_bottom_right1"});
}

template <typename T>
auto make_getdata2(Matrix<T>* m, Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right) {
  auto f = [m](const Key& key, std::tuple<Out<Key, std::vector<BlockMatrix<T>>>>& out) {
    //std::cout << "getdata2 called...\n";
    std::vector<BlockMatrix<T>> v;
    auto [i,j] = key;

    if (i > 0 && j > 0) {
      if (j < m->cols() - 1) {
        //std::cout << "send 3 : " << i << " " << j << std::endl;
        v.push_back((*m)(i,j+1));
      }
      if (i < m->rows() - 1) {
        //std::cout << "send 4 : " << i << " " << j << std::endl;
        v.push_back((*m)(i+1,j));
      } 
      send<0>(Key(i,j), v, out);
    }
  };

  return wrap<Key>(f, edges(), edges(bottom_right), "get_data2", {},
                  {"get_bottom_right2"});
}

template <typename funcT, typename T>
auto make_wavefront0(const funcT& func, int MB, int NB, Edge<Key, BlockMatrix<T>>& input, 
                    Edge<Key, BlockMatrix<T>>& toporleft, Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right, 
                    Edge<Key, BlockMatrix<T>>& result) {
  auto f = [func, MB, NB](const Key& key, BlockMatrix<T>&& input, std::vector<BlockMatrix<T>>&& bottom_right, std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>& out) {
    auto [i,j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    //std::cout << "wf0 " << i << " " << j << " " << "size: " << bottom_right.size() <<  std::endl;
    BlockMatrix<T> res = func(i, j, MB, NB, input, input, input, bottom_right[0], bottom_right[1]);  
    
    send<0>(Key(i,next_j), res, out);
    send<0>(Key(next_i,j), res, out);
    
    send<1>(Key(i,j), res, out);
  };

  return wrap(f, edges(input, bottom_right), edges(toporleft, result), "wavefront0", 
            {"block_input", "bottom_right"},
            {"toporleft", "result"});

}

// Method to generate wavefront task with single input.
template <typename funcT, typename T>
auto make_wavefront1(const funcT& func, int MB, int NB, Edge<Key, BlockMatrix<T>>& input, 
                    Edge<Key, BlockMatrix<T>>& toporleft, Edge<Key, std::vector<BlockMatrix<T>>>& bottom_right, 
                    Edge<Key, BlockMatrix<T>>& output1,
                    Edge<Key, BlockMatrix<T>>& output2, Edge<Key, BlockMatrix<T>>& result) {
  auto f = [MB, NB, func](const Key& key, BlockMatrix<T>&& input, BlockMatrix<T>&& previous, std::vector<BlockMatrix<T>>&& bottom_right, 
                          std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, 
                          Out<Key, BlockMatrix<T>>>& out) {
    auto [i, j] = key;
    int next_i = i + 1;
    int next_j = j + 1;

    //std::cout << "wf1 " << i << " " << j << "size: " << bottom_right.size() <<  std::endl;
    BlockMatrix<T> res;
    int size = bottom_right.size();
      if (size == 1)
        res = func(i, j, MB, NB, input, previous, previous, bottom_right[0], bottom_right[0]);
      else
        res = func(i, j, MB, NB, input, previous, previous, bottom_right[0], bottom_right[1]);
   
    //func(i, j, MB, NB, input, previous, previous, bottom_right[0], bottom_right[1]);
    //Processing finished for this block, so send it to output
    send<3>(Key(i,j), res, out);

    if (next_i < MB) {
      if (j == 0)  {
        // Single predecessor, no left block
        send<0>(Key(next_i, j), res, out); //send top block
      }
      else { 
        // Two predecessors
        send<2>(Key(next_i, j), res, out); //send top block
        
      }
    }
    if (next_j < NB) {
      if (i == 0) { 
        // Single predecessor, no top block
        send<0>(Key(i, next_j), res, out); //send left block
      }
      else { 
        // Two predecessors
        send<1>(Key(i, next_j), res, out); //send left block
      }
    }
  };

  return wrap(f, edges(input, toporleft, bottom_right), edges(toporleft, output1, output2, result), "wavefront1", 
              {"block_input", "toporleft", "bottom_right"}, 
              {"recur", "output1", "output2", "result"});
}

template <typename T>
auto make_result(Matrix<T> *r, const Edge<Key, BlockMatrix<T>>& result) {
  auto f = [r](const Key& key, BlockMatrix<T>&& bm, std::tuple<>& out) {
    auto [i,j] = key;
    if (bm(i, j) != (*r)(i, j)) {
      std::cout << "ERROR in block (" << i << "," << j << ")\n";
    }
  };

  return wrap(f, edges(result), edges(), "Final Output", {"result"}, {});
}

int main(int argc, char** argv) {
  int n_rows, n_cols, B;
  int n_brows, n_bcols;

  n_rows = n_cols = 16;//8192;
  B = 4;//128;
  bool verify = true;

  n_brows = (n_rows / B) + (n_rows % B > 0);
  n_bcols = (n_cols / B) + (n_cols % B > 0);

  Matrix<double>* m = new Matrix<double>(n_brows, n_bcols, B, B);
  Matrix<double>* m2 = new Matrix<double>(n_brows, n_bcols, B, B);

  Matrix<double>* r2 = new Matrix<double>(n_brows, n_bcols, B, B);

  m->fill();
  m2->fill();

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (verify) {
    std::cout << "Computing using serial version....";
    beg = std::chrono::high_resolution_clock::now();
    wavefront_serial(m2, r2, n_brows, n_bcols);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "....done!" << std::endl;
    std::cout << "Serial Execution Time (milliseconds) : "
            << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1e3 << std::endl;
  }

  Edge<Key, BlockMatrix<double>> input0("input0"), input1("input1"), input2("input2"), 
        toporleft("toporleft"), output1("output1"), output2("output2"), result("result");
  Edge<Key, std::vector<BlockMatrix<double>>> bottom_right("bottom_right", true), 
                    bottom_right1("bottom_right1", true),
                    bottom_right2("bottom_right2", true);

  ttg_initialize(argc, argv, -1);
  //OpBase::set_trace_all(true);
  //OpBase::set_lazy_pull(true);
  {
    auto i = initiator(m, input0, input1, input2);//, //bottom_right0, bottom_right1, bottom_right2);
    auto d = make_getdata(m, bottom_right);
    auto d2 = make_getdata1(m, bottom_right1);
    auto d3 = make_getdata2(m, bottom_right2);
    auto s0 = make_wavefront0(stencil_computation<double>, n_brows, n_bcols, input0, toporleft, bottom_right, result);
    auto s1 = make_wavefront1(stencil_computation<double>, n_brows, n_bcols, input1, toporleft, bottom_right1, output1, output2, result);
    auto s2 = make_wavefront2(stencil_computation<double>, n_brows, n_bcols, input2, output1, output2, bottom_right2, result);
    auto res = make_result(r2, result);

    auto connected = make_graph_executable(i.get());
    assert(connected);
    TTGUNUSED(connected);
    std::cout << "Graph is connected.\n";

    if (ttg_default_execution_context().rank() == 0) {
      //std::cout << "==== begin dot ====\n";
      std::cout << Dot()(i.get()) << std::endl;
      //std::cout << "==== end dot ====\n";
      
      beg = std::chrono::high_resolution_clock::now();
      i->invoke(Key(0,0));
      //i->in<0>()->send(Key(0, 0), Control());
      // This doesn't work!
      // s->send<0>(Key(0,0), Control());
    }

    ttg_execute(ttg_default_execution_context());
    ttg_fence(ttg_default_execution_context());
    
    if (ttg_default_execution_context().rank() == 0) {
      end = std::chrono::high_resolution_clock::now();
      std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000 << std::endl;
    }
  }

  ttg_finalize();

  /*m->print();
  std::cout << std::endl << std::endl;
  r->print();
  std::cout << std::endl << std::endl;
  m2->print();
  std::cout << std::endl << std::endl;
  r2->print();*/
  /*r->print();
  std::cout << std::endl << std::endl;
  m2->print();*/

  delete m;
  delete m2;
  delete r2;
  return 0;
}
