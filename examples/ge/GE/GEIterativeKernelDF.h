#ifndef GE_ITERATIVE_KERNEL_DF
#define GE_ITERATIVE_KERNEL_DF

/*template <typename T>
void ge_iterative_kernelA(int block_size, int I, int J, int K, T*  m_ij) {
  for(int k = 0; k < block_size; ++k) {
		int k_row = k * block_size;
    for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        int i_row = i * block_size;
        T* temp = (T*)aligned_alloc(64, sizeof(T) * block_size);//new T[block_size];
        for(int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            temp[j] = ((m_ij[i_row + k] * m_ij[k_row + j]) 
                                      / m_ij[k_row + k]);
          }
        }
        for (int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij[i_row + j] -= temp[j];
          }
        }
        free(temp);	
      }		
    }	
	}
}

template <typename T>
void ge_iterative_kernelB(int block_size, int I, int J, int K, T* m_ij,
              const T* m_ik) {
  for(int k = 0; k < block_size; ++k) {
    int k_row = k * block_size;
    for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        int i_row = i * block_size;
        T* temp = (T*)aligned_alloc(64, sizeof(T) * block_size);//new T[block_size];
        for(int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            temp[j] = ((m_ik[i_row + k] * m_ij[k_row + j]) 
                                      / m_ik[k_row + k]);
          }
        }
        for (int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij[i_row + j] -= temp[j];
          }
        }
        free(temp);
      }
    }
  }
}

template <typename T>
void ge_iterative_kernelC(int block_size, int I, int J, int K, T* m_ij,
              const T* m_kj) {
  for(int k = 0; k < block_size; ++k) {
    int k_row = k * block_size;
    for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        int i_row = i * block_size;
        T* temp = (T*)aligned_alloc(64, sizeof(T) * block_size); //new T[block_size];
        for(int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            temp[j] = ((m_ij[i_row + k] * m_kj[k_row + j]) 
                                      / m_kj[k_row + k]);
          }
        }
        for (int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij[i_row + j] -= temp[j];
          }
        }
        free(temp);
      }
    }
  }
}
*/

template <typename T>
void ge_iterative_kernel(int block_size, int I, int J, int K, T* m_ij,
              const T* m_ik, const T* m_kj, const T* m_kk) {
  //T* temp = (T*)aligned_alloc(64, sizeof(T) * block_size); //new T[block_size];
  for(int k = 0; k < block_size; ++k) {
    int k_row = k * block_size;
    T reciprocal = 1.0 / m_kk[k_row + k];
    for(int i = 0; i < block_size; ++i) {
      if (i > k || I > K) {
        int i_row = i * block_size;
        int j_lb = J * block_size;
        #pragma omp simd
        for(int j = (j_lb - k) >= 0 ? 0 : k; j < block_size; ++j) {
        //for(int j = 0; j < block_size; ++j) {
          //if (J * block_size + j >= k) {
            m_ij[i_row + j] -= (m_ik[i_row + k] * m_kj[k_row + j] * reciprocal); 
                                      /// m_kk[k_row + k]);
          //}
        }
        /*for (int j = 0; j < block_size; ++j) {
          if (J * block_size + j >= k) {
            m_ij[i_row + j] -= temp[j];
          }
        }*/
      }
    }
  }
  //free(temp);
}


#endif
