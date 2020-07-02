#ifndef GE_ITERATIVE_KERNEL
#define GE_ITERATIVE_KERNEL

void ge_iterative_kernel(int problem_size, int blocking_factor, 
							int I, int J, int K, double* adjacency_matrix) {
  int block_size = problem_size/blocking_factor;
	for(int k = 0; k < block_size; ++k) {
		int absolute_k = K * block_size + k;
		if(absolute_k < problem_size - 1) {
			int k_row = problem_size * absolute_k;
			for(int i = 0; i < block_size; ++i) {
				int absolute_i = I * block_size + i;
				if (absolute_i > absolute_k) {
					int i_row = problem_size * absolute_i;
					for(int j = 0; j < block_size; ++j) {
						int absolute_j = J * block_size + j;
						if(absolute_j >= absolute_k) {
							adjacency_matrix[i_row + absolute_j] -= 
								(adjacency_matrix[i_row + absolute_k] * 
									adjacency_matrix[k_row + absolute_j])
									/ adjacency_matrix[k_row + absolute_k];
						}				
					}
				}
			}
		}
	}
}


#endif
