#ifndef GE_ITERATIVE_KERNEL
#define GE_ITERATIVE_KERNEL

template <typename T>
BlockMatrix<T> ge_iterative_kernelA(int block_size, int I, int J, int K, BlockMatrix<T> m_ij) {
  for(int k = 0; k < block_size; ++k) {
		for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        for(int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij(i,j) -= ((m_ij(i,k) * m_ij(k,j)) / m_ij(k,k));
          }
        }	
      }		
    }	
	}
  return m_ij;
}

template <typename T>
BlockMatrix<T> ge_iterative_kernelB(int block_size, int I, int J, int K, BlockMatrix<T> m_ij,
              BlockMatrix<T> m_ik) {
  for(int k = 0; k < block_size; ++k) {
    for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        for(int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij(i,j) -= ((m_ik(i,k) * m_ij(k,j)) / m_ik(k,k));
          }
        }
      }
    }
  }
  return m_ij;
}

template <typename T>
BlockMatrix<T> ge_iterative_kernelC(int block_size, int I, int J, int K, BlockMatrix<T> m_ij,
              BlockMatrix<T> m_kj) {
  for(int k = 0; k < block_size; ++k) {
    for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        for(int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij(i,j) -= ((m_ij(i,k) * m_kj(k,j)) / m_kj(k,k));
          }
        }
      }
    }
  }
  return m_ij;
}

template <typename T>
BlockMatrix<T> ge_iterative_kernelD(int block_size, int I, int J, int K, BlockMatrix<T> m_ij,
              BlockMatrix<T> m_ik, BlockMatrix<T> m_kj, BlockMatrix<T> m_kk) {
  for(int k = 0; k < block_size; ++k) {
    for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        for(int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij(i,j) -= ((m_ik(i,k) * m_kj(k,j)) / m_kk(k,k));
          }
        }
      }
    }
  }
  return m_ij;
}


#endif
