#ifndef GE_ITERATIVE_KERNEL
#define GE_ITERATIVE_KERNEL

void ge_iterative_kernelA(int problem_size, int blocking_factor, 
							int I, int J, int K, double* X) {
  int block_size = problem_size/blocking_factor;
  int k_lb = K * block_size;
  int i_lb = I * block_size;
  int j_lb = J * block_size;

  for(int k = k_lb; k < k_lb + block_size; ++k) {
      if(k < problem_size - 1) {
        int k_row = k * problem_size;
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            for(int j = j_lb; j < j_lb + block_size; ++j) {
              if(j >= k) {
                X[i_row + j] -= (X[i_row + k] * X[k_row + j]) / X[k_row + k];
              }
            }
          }
        }
      }
    }
}

void ge_iterative_kernelB(int problem_size, int blocking_factor,
              int I, int J, int K, double* X, double *U) {
  int block_size = problem_size/blocking_factor;
  int k_lb = K * block_size;
  int i_lb = I * block_size;
  int j_lb = J * block_size;

  for(int k = k_lb; k < k_lb + block_size; ++k) {
      if(k < problem_size - 1) {
        int k_row = k * problem_size;
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            for(int j = j_lb; j < j_lb + block_size; ++j) {
              if(j >= k) {
                X[i_row + j] -= (U[i_row + k] * X[k_row + j]) / U[k_row + k];
              }
            }
          }
        }
      }
    }
}

void ge_iterative_kernelC(int problem_size, int blocking_factor,
              int I, int J, int K, double* X, double* V) {
  int block_size = problem_size/blocking_factor;
  int k_lb = K * block_size;
  int i_lb = I * block_size;
  int j_lb = J * block_size;

  for(int k = k_lb; k < k_lb + block_size; ++k) {
      if(k < problem_size - 1) {
        int k_row = k * problem_size;
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            for(int j = j_lb; j < j_lb + block_size; ++j) {
              if(j >= k) {
                X[i_row + j] -= (X[i_row + k] * V[k_row + j]) / V[k_row + k];
              }
            }
          }
        }
      }
    }
}

void ge_iterative_kernelD(int problem_size, int blocking_factor,
              int I, int J, int K, double* X, double* U, double* V, double* W) {
  int block_size = problem_size/blocking_factor;
  int k_lb = K * block_size;
  int i_lb = I * block_size;
  int j_lb = J * block_size;

  for(int k = k_lb; k < k_lb + block_size; ++k) {
      if(k < problem_size - 1) {
        int k_row = k * problem_size;
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            for(int j = j_lb; j < j_lb + block_size; ++j) {
              if(j >= k) {
                X[i_row + j] -= (U[i_row + k] * V[k_row + j]) / W[k_row + k];
              }
            }
          }
        }
      }
    }
}
#endif
