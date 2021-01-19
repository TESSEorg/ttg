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
        double reciprocal = 1.0 / X[k_row + k];
        //#pragma omp simd
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            //#pragma vector always
            //#pragma ivdep
            //#pragma omp simd
            for(int j = (j_lb - k) >= 0 ? j_lb : k; j < j_lb + block_size; ++j) {  
              //if(j >= k) {
                X[i_row + j] -= (X[i_row + k] * X[k_row + j] * reciprocal); //) / X[k_row + k];
              //}
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
        double reciprocal = 1.0 / U[k_row + k];
        //#pragma omp simd
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            //#pragma vector always
            //#pragma ivdep
            //#pragma omp simd           
            for(int j = (j_lb - k) >= 0 ? j_lb : k; j < j_lb + block_size; ++j) {
              //if(j >= k) {
                X[i_row + j] -= (U[i_row + k] * X[k_row + j] * reciprocal);//) / U[k_row + k];
              //}
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
        double reciprocal = 1.0 / V[k_row + k];
        //#pragma omp simd
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            //#pragma vector always
            //#pragma ivdep
            //#pragma omp simd
            for(int j = (j_lb - k) >= 0 ? j_lb : k; j < j_lb + block_size; ++j) {
              //if(j >= k) {
                X[i_row + j] -= (X[i_row + k] * V[k_row + j] * reciprocal);//) / V[k_row + k];
              //}
            }
          }
        }
      }
    }
}

void ge_iterative_kernelD(int problem_size, int blocking_factor,
              int I, int J, int K, double* __restrict__ X, double* __restrict__ U, double* __restrict__ V, double* __restrict__ W) 
{
  int block_size = problem_size/blocking_factor;
  int k_lb = K * block_size;
  int i_lb = I * block_size;
  int j_lb = J * block_size;
 
  for(int k = k_lb; k < k_lb + block_size; ++k) {
      if(k < problem_size - 1) {
        int k_row = k * problem_size;
        double reciprocal = 1.0 / X[k_row + k];
        //#pragma omp simd
        for(int i = i_lb; i < i_lb + block_size; ++i) {
          if (i > k) {
            int i_row = i * problem_size;
            //#pragma vector always
            //#pragma ivdep
            //#pragma omp simd
            for(int j = (j_lb - k) >= 0 ? j_lb : k; j < j_lb + block_size; ++j) {
              //if(j >= k) {
                //std::cout << j << " " << k << " " << j_lb + block_size << std::endl;
                X[i_row + j] -= (X[i_row + k] * X[k_row + j] * reciprocal); //) / X[k_row + k];
              //}
            }
          }
        }
      }
    }
}
#endif
