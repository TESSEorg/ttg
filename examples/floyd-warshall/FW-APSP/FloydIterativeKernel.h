#ifndef FLOYD_ITERATIVE_KERNEL
#define FLOYD_ITERATIVE_KERNEL

#include <algorithm>    /* std::min */
using namespace std;

void floyd_iterative_kernel(int problem_size, int blocking_factor, int I, int J, int K, double* adjacency_matrix) {
	int block_size = problem_size/blocking_factor;
  for(int k = 0; k < block_size; ++k) {
		int absolute_k = K * block_size + k;
		int k_row = problem_size * absolute_k;
		for(int i = 0; i < block_size; ++i) {
			int absolute_i = I * block_size + i;
			int i_row = problem_size * absolute_i;
			for(int j = 0; j < block_size; ++j) {
				int absolute_j = J * block_size + j;
				//std::cout << i_row + absolute_j << " " << i_row + absolute_j << " " << i_row + absolute_k << " " << k_row + absolute_j <<  std::endl;
        adjacency_matrix[i_row + absolute_j] = 
					min(adjacency_matrix[i_row + absolute_j],
						adjacency_matrix[i_row + absolute_k] + 
							adjacency_matrix[k_row + absolute_j]);
			}
		}
	}
}


#endif
