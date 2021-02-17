#ifndef GE_RECURSIVE_PARALLEL_KERNEL
#define GE_RECURSIVE_PARALLEL_KERNEL

//#include <omp.h>
using namespace std;

void ge_recursive_parallel_kernelA(double* X, int problem_size,
								   int block_size, int i_lb, int j_lb, int k_lb,
								   int recurisve_fan_out, int base_size);

// X = V, U = W
void ge_recursive_parallel_kernelB(double* X, double* U, int problem_size,
								   int block_size, int i_lb, int j_lb, int k_lb,
								   int recurisve_fan_out, int base_size);

// X = U, V = W
void ge_recursive_parallel_kernelC(double* X, double* V, int problem_size,
								   int block_size, int i_lb, int j_lb, int k_lb,
								   int recurisve_fan_out, int base_size);

// Everything is disjoint
void ge_recursive_parallel_kernelD(double* X, double* U, double* V, double* W,
								   int problem_size, int block_size, int i_lb,
								   int j_lb, int k_lb, int recurisve_fan_out,
								   int base_size);


void ge_recursive_parallel_kernelA(double* X, int problem_size,
								   int block_size, int i_lb, int j_lb, int k_lb,
								   int recurisve_fan_out, int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
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
	else {
		int tile_size = block_size / recurisve_fan_out;
		for(int kk = 0; kk < recurisve_fan_out; ++kk) {
			ge_recursive_parallel_kernelA(X, problem_size, tile_size,
										  i_lb + kk * tile_size,
										  j_lb + kk * tile_size,
										  k_lb + kk * tile_size,
										  recurisve_fan_out, base_size);
			// Calling functions B and C
			for(int ii = kk+1; ii < recurisve_fan_out; ++ii) {
				#pragma omp task
				ge_recursive_parallel_kernelB(X, X, problem_size, tile_size,
											  i_lb + kk * tile_size,
											  j_lb + ii * tile_size,
											  k_lb + kk * tile_size,
											  recurisve_fan_out, base_size);
				#pragma omp task
				ge_recursive_parallel_kernelC(X, X, problem_size, tile_size,
											  i_lb + ii * tile_size,
											  j_lb + kk * tile_size,
											  k_lb + kk * tile_size,
											  recurisve_fan_out, base_size);
			}
			#pragma omp taskwait
			// Calling functions D
			for(int ii = kk+1; ii < recurisve_fan_out; ++ii) {
				for(int jj = kk+1; jj < recurisve_fan_out; ++jj) {
					#pragma omp task
					ge_recursive_parallel_kernelD(X, X, X, X, problem_size,
												  tile_size, i_lb + ii * tile_size,
											   	  j_lb + jj * tile_size,
											      k_lb + kk * tile_size,
											      recurisve_fan_out, base_size);
				}
			}
			#pragma omp taskwait
		}
	}
}

// X = V, U = W
void ge_recursive_parallel_kernelB(double* X, double* U, int problem_size,
								   int block_size, int i_lb, int j_lb, int k_lb,
								   int recurisve_fan_out, int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
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
	else {
		int tile_size = block_size / recurisve_fan_out;
		for(int kk = 0; kk < recurisve_fan_out; ++kk) {
			for(int jj = 0; jj < recurisve_fan_out; ++jj) {
				#pragma omp task
				ge_recursive_parallel_kernelB(X, U, problem_size,
											  tile_size,
											  i_lb + kk * tile_size,
											  j_lb + jj * tile_size,
											  k_lb + kk * tile_size,
											  recurisve_fan_out, base_size);
			}
			#pragma omp taskwait
			for(int ii = kk+1; ii < recurisve_fan_out; ++ii) {
				for(int jj = 0; jj < recurisve_fan_out; ++jj) {
					#pragma omp task
					ge_recursive_parallel_kernelD(X, U, X, U, problem_size, tile_size,
											   	  i_lb + ii * tile_size,
											   	  j_lb + jj * tile_size,
											      k_lb + kk * tile_size,
											      recurisve_fan_out, base_size);
				}
			}
			#pragma omp taskwait
		}
	}

}

// X = U, V = W
void ge_recursive_parallel_kernelC(double* X, double* V, int problem_size,
								   int block_size, int i_lb, int j_lb, int k_lb,
								   int recurisve_fan_out, int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
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
	else {
		int tile_size = block_size / recurisve_fan_out;
		for(int kk = 0; kk < recurisve_fan_out; ++kk) {
			for(int ii = 0; ii < recurisve_fan_out; ++ii) {
				#pragma omp task
				ge_recursive_parallel_kernelC(X, V, problem_size,
											  tile_size,
											  i_lb + ii * tile_size,
											  j_lb + kk * tile_size,
											  k_lb + kk * tile_size,
											  recurisve_fan_out, base_size);
			}
			#pragma omp taskwait
			for(int jj = kk+1; jj < recurisve_fan_out; ++jj) {
				for(int ii = 0; ii < recurisve_fan_out; ++ii) {
					// #pragma omp task
					ge_recursive_parallel_kernelD(X, X, V, V, problem_size, tile_size,
											   	  i_lb + ii * tile_size,
											   	  j_lb + jj * tile_size,
											      k_lb + kk * tile_size,
											      recurisve_fan_out, base_size);
				}
			}
			#pragma omp taskwait
		}
	}
}

// Everything is disjoint
void ge_recursive_parallel_kernelD(double* X, double* U, double* V, double* W,
								   int problem_size, int block_size, int i_lb,
								   int j_lb, int k_lb, int recurisve_fan_out,
								   int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
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
	else {
		int tile_size = block_size / recurisve_fan_out;
		for(int kk = 0; kk < recurisve_fan_out; ++kk) {
			for(int ii = 0; ii < recurisve_fan_out; ++ii) {
				for(int jj = 0; jj < recurisve_fan_out; ++jj) {
					#pragma omp task
					ge_recursive_parallel_kernelD(X, U, V, W, problem_size, tile_size,
												   i_lb + ii * tile_size,
												   j_lb + jj * tile_size,
												   k_lb + kk * tile_size,
												   recurisve_fan_out, base_size);
				}
			}
			#pragma omp taskwait
		}
	}
}
#endif
