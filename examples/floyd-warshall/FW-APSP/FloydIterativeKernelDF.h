#ifndef FLOYD_ITERATIVE_KERNEL
#define FLOYD_ITERATIVE_KERNEL

#include <algorithm>    /* std::min */
#include <omp.h>
using namespace std;

template <typename T>
BlockMatrix<T> floyd_iterative_kernel(int block_size, BlockMatrix<T> m_ij,
                                      const BlockMatrix<T>& m_ik,
                                      const BlockMatrix<T>& m_kj) {
  //BlockMatrix<T> bm(block_size, block_size);
  for(int k = 0; k < block_size; ++k) {
		//#pragma ivdep
    for(int i = 0; i < block_size; ++i) {
			for(int j = 0; j < block_size; ++j) {
				m_ij(i,j) = min(m_ij(i,j), m_ik(i,k) + m_kj(k,j));
      }
		}
	}
  return (m_ij);
}


#endif
