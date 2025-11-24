// SPDX-License-Identifier: BSD-3-Clause
#ifndef FLOYD_RECURSIVE_PARALLEL_KERNEL
#define FLOYD_RECURSIVE_PARALLEL_KERNEL

#include <omp.h>
#include <algorithm>    /* std::min */
using namespace std;

void floyd_recursive_parallel_kernelA(double* X, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size);

void floyd_recursive_parallel_kernelB(double* X, double* U, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size);

void floyd_recursive_parallel_kernelC(double* X, double* V, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size);

void floyd_recursive_parallel_kernelD(double* X, double* U, double* V, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size);


void floyd_recursive_parallel_kernelA(double* X, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
		for(int k = k_lb; k < k_lb + block_size; ++k) {
			int k_row = k * problem_size;
			for(int i = i_lb; i < i_lb + block_size; ++i) {
				int i_row = i * problem_size;
				for(int j = j_lb; j < j_lb + block_size; ++j) {
					X[i_row + j] = min(X[i_row + j], X[i_row + k] + X[k_row + j]);
				}
			}
		}
	}
	else {
		int tile_size = block_size / recurisve_fan_out;
		for(int kk = 0; kk < recurisve_fan_out; ++kk) {
			floyd_recursive_parallel_kernelA(X, problem_size, tile_size,
										  i_lb + kk * tile_size,
										  j_lb + kk * tile_size,
										  k_lb + kk * tile_size,
										  recurisve_fan_out, base_size);
			// Calling functions B and C
			for(int ii = 0; ii < recurisve_fan_out; ++ii) {
				if(ii != kk) {
					#pragma omp task
					floyd_recursive_parallel_kernelB(X, X, problem_size, tile_size,
												   i_lb + kk * tile_size,
												   j_lb + ii * tile_size,
												   k_lb + kk * tile_size,
												   recurisve_fan_out, base_size);
					#pragma omp task
					floyd_recursive_parallel_kernelC(X, X, problem_size, tile_size,
												   i_lb + ii * tile_size,
												   j_lb + kk * tile_size,
												   k_lb + kk * tile_size,
												   recurisve_fan_out, base_size);
				}
			}
			#pragma omp taskwait
			// Calling functions D
			for(int ii = 0; ii < recurisve_fan_out; ++ii) {
				for(int jj = 0; jj < recurisve_fan_out; ++jj) {
					if(ii != kk && jj != kk) {
						#pragma omp task
						floyd_recursive_parallel_kernelD(X, X, X, problem_size, tile_size,
												   	   i_lb + ii * tile_size,
												   	   j_lb + jj * tile_size,
												       k_lb + kk * tile_size,
												       recurisve_fan_out, base_size);
					}
				}
			}
			#pragma omp taskwait
		}
	}
}

void floyd_recursive_parallel_kernelB(double* X, double* U, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
		for(int k = k_lb; k < k_lb + block_size; ++k) {
			int k_row = k * problem_size;
			for(int i = i_lb; i < i_lb + block_size; ++i) {
				int i_row = i * problem_size;
				for(int j = j_lb; j < j_lb + block_size; ++j) {
					X[i_row + j] = min(X[i_row + j], U[i_row + k] + X[k_row + j]);
				}
			}
		}
	}
	else {
		int tile_size = block_size / recurisve_fan_out;
		for(int kk = 0; kk < recurisve_fan_out; ++kk) {
			for(int jj = 0; jj < recurisve_fan_out; ++jj) {
				#pragma omp task
				floyd_recursive_parallel_kernelB(X, U, problem_size,
											   tile_size,
											   i_lb + kk * tile_size,
											   j_lb + jj * tile_size,
											   k_lb + kk * tile_size,
											   recurisve_fan_out, base_size);
			}
			#pragma omp taskwait
			for(int ii = 0; ii < recurisve_fan_out; ++ii) {
				if (ii != kk) {
					for(int jj = 0; jj < recurisve_fan_out; ++jj) {
						#pragma omp task
						floyd_recursive_parallel_kernelD(X, U, X, problem_size, tile_size,
												   	   i_lb + ii * tile_size,
												   	   j_lb + jj * tile_size,
												       k_lb + kk * tile_size,
												       recurisve_fan_out, base_size);
					}
				}
			}
			#pragma omp taskwait
		}
	}

}


void floyd_recursive_parallel_kernelC(double* X, double* V, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
		for(int k = k_lb; k < k_lb + block_size; ++k) {
			int k_row = k * problem_size;
			for(int i = i_lb; i < i_lb + block_size; ++i) {
				int i_row = i * problem_size;
				for(int j = j_lb; j < j_lb + block_size; ++j) {
					X[i_row + j] = min(X[i_row + j], X[i_row + k] + V[k_row + j]);
				}
			}
		}
	}
	else {
		int tile_size = block_size / recurisve_fan_out;
		for(int kk = 0; kk < recurisve_fan_out; ++kk) {
			for(int ii = 0; ii < recurisve_fan_out; ++ii) {
				#pragma omp task
				floyd_recursive_parallel_kernelC(X, V, problem_size,
											   tile_size,
											   i_lb + ii * tile_size,
											   j_lb + kk * tile_size,
											   k_lb + kk * tile_size,
											   recurisve_fan_out, base_size);
			}
			#pragma omp taskwait
			for(int ii = 0; ii < recurisve_fan_out; ++ii) {
				for(int jj = 0; jj < recurisve_fan_out; ++jj) {
					if (jj != kk) {
						#pragma omp task
						floyd_recursive_parallel_kernelD(X, X, V, problem_size, tile_size,
												   	   i_lb + ii * tile_size,
												   	   j_lb + jj * tile_size,
												       k_lb + kk * tile_size,
												       recurisve_fan_out, base_size);
					}
				}
			}
			#pragma omp taskwait
		}
	}
}

void floyd_recursive_parallel_kernelD(double* X, double* U, double* V, int problem_size,
									int block_size, int i_lb, int j_lb, int k_lb,
									int recurisve_fan_out, int base_size) {
	if(block_size <= base_size || block_size <= recurisve_fan_out) {
		for(int k = k_lb; k < k_lb + block_size; ++k) {
			int k_row = k * problem_size;
			for(int i = i_lb; i < i_lb + block_size; ++i) {
				int i_row = i * problem_size;
				for(int j = j_lb; j < j_lb + block_size; ++j) {
					X[i_row + j] = min(X[i_row + j], U[i_row + k] + V[k_row + j]);
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
					floyd_recursive_parallel_kernelD(X, U, V, problem_size, tile_size,
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