#include <algorithm> // for std::max
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <random>
#include <cmath>

/*!
	\file wavefront_df.impl.h
	\brief Wavefront computation on distributed memory
	\defgroup
	ingroup examples

	\par Points of interest
	- dynamic recursive DAG.
*/

using Key = std::pair<int, int>;

template <typename T>
class BlockMatrix {
	private:
		int _rows;
		int _cols;
		T** m_block;

	public: 
		BlockMatrix() = default;
		
		BlockMatrix(int rows, int cols): _rows(rows), _cols(cols)
		{
			m_block = new T *[_rows];
  		for ( int i = 0; i < _rows; ++i ) m_block[i] = new T [_cols];
  		for(int i=0; i<_rows; ++i){
    		for(int j=0; j<_cols ; ++j){
      		m_block[i][j] = 1; //i*_cols + j;
    		}  
  		}
		}
		
		BlockMatrix(const BlockMatrix<T> &b) 
		{
			_rows = b._rows;
			_cols = b._cols;
			m_block = b.m_block;
		} 

		int size() const { return _rows * _cols; }
    int rows() const { return _rows; }
    int cols() const { return _cols; }	
		
		bool operator ==(const BlockMatrix& m) const {
			return (m.m_block == m_block);
		}

		bool operator !=(const BlockMatrix& m) const {
			bool notequal = false;
			for(int i=0; i<_rows; i++){
        for(int j=0; j<_cols ; j++){
          if (m_block[i][j] != m.m_block[i][j])
					{
						notequal = true;
						break;
					}
        }
      }

			return notequal;
		}

		T& operator()(int row, int col) {
    	return m_block[row][col];
		}

		template <typename Archive>
    void serialize(Archive& ar) {
    }

};

template <typename T>
std::ostream& operator<<(std::ostream& s, BlockMatrix<T>& m) {
	for (int i = 0; i<m.rows(); i++) {
    for (int j = 0; j<m.cols(); j++)
      s << m(i,j) << " ";
    s << std::endl;
  }
	return s;

}

template <typename T>
class Matrix {
	private: 
		int nb_row;//# of blocks in a row
		int nb_col;//# of blocks in a col
		int b_rows;//# of rows in a block
		int b_cols;//# of cols in a block
		
		std::vector<std::vector<BlockMatrix<T>>> m;//Matrix as a collection of blocks

	public:		
		Matrix() = default;
		Matrix(int nb_row, int nb_col, int b_rows, int b_cols) 
					: nb_row(nb_row), nb_col(nb_col), b_rows(b_rows), b_cols(b_cols) 
		{
			
			for(int i=0; i<nb_row; i++) {
				std::vector<BlockMatrix<T>> v;
				for(int j=0; j<nb_col; j++) {
					v.push_back(BlockMatrix<T> (b_rows, b_cols));
				}
				m.push_back(v);
			}
		}
	
		//copy constructor	
		Matrix(const Matrix<T> &m)
		{
			nb_row = m.nb_row;
			nb_col = m.nb_col;
			b_rows = m.b_rows;
			b_cols = m.b_cols;
			m = m.m;
		}	

		//Return total # of elements in the matrix
		int size() const { return (nb_row * b_rows) * (nb_col * b_cols); }
    //Return # of block rows
		int rows() const { return nb_row; }
    //Return # of block cols
		int cols() const { return nb_col; }

    bool operator ==(const Matrix& matrix) const {
      return (matrix.m == m);
    }

    bool operator !=(const Matrix& matrix) const {
      return (matrix.m != m);
    }

    BlockMatrix<T> operator() (int block_row, int block_col) {
      return m[block_row][block_col];
    }

		void print() {
			for(int i=0; i<nb_row; i++) {
				for(int j=0; j<nb_col; j++) {
					std::cout << m[i][j];//print();
				}
			}
		}
		
		template <typename Archive>
    void serialize(Archive& ar) {
    }

};

// An empty class used for pure control flows
struct Control {
};

template <typename T>
void stencil_computation(int i, int j, int M, int N, BlockMatrix<T> &&bm, BlockMatrix<T> &left, BlockMatrix<T> &top, BlockMatrix<T> &right, BlockMatrix<T> &bottom)
{
	//i==0 -> no top block
	//j==0 -> no left block
	//i==M-1 -> no bottom block
	//j==N-1 -> no right block
	int MB = bm.rows();
	int NB = bm.cols();
		
	for ( int ii = 0; ii < bm.rows(); ++ii ) { 
    for ( int jj = 0; jj < bm.cols(); ++jj ) {
			bm(ii,jj) += (ii==0) ? (i>0 ? top(top.rows()-1,jj) : 0.0) : bm(ii-1,jj);
			bm(ii,jj) += (ii==MB-1) ? (i<M-1 ? bottom(0,jj) : 0.0) : bm(ii+1,jj);
			bm(ii,jj) += (jj==0) ? (j>0 ? left(ii,left.cols()-1) : 0.0) : bm(ii,jj-1);
			bm(ii,jj) += (jj==NB-1) ? (j<N-1 ? right(0,0) : 0.0) : bm(ii,jj+1);
			bm(ii,jj) *= 0.25;
		}
	}
}

//serial implementation of wavefront computation.
template <typename T>
void wavefront_serial(Matrix<T> *m, int n_brows, int n_bcols) {
	for (int i=0; i<n_brows; i++) {
		for (int j=0; j<n_bcols;j++) {
			BlockMatrix<T> left,top,right,bottom;
			if (i < n_brows-1)
				bottom = (*m)(i+1,j);
			if (j < n_bcols-1)
				right = (*m)(i,j+1);
			if (j > 0)
				left = (*m)(i,j-1);
			if (i > 0)
				top = (*m)(i-1,j);

			stencil_computation(i,j,n_brows,n_bcols,(*m)(i,j),left,top,right,bottom);
		}
	}
}

#include TTG_RUNTIME_H
IMPORT_TTG_RUNTIME_NS

//Method to generate  wavefront tasks with two inputs.
template <typename funcT, typename T>
auto make_wavefront2(const funcT& func, Matrix<T> *m, Edge<Key, BlockMatrix<T>>& input1, Edge<Key, BlockMatrix<T>>& input2)
{
	auto f = [m,func](const Key& key, BlockMatrix<T>&& bm1, BlockMatrix<T>&& bm2, std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>>& out) 
	{
    auto [i,j] = key;
		int next_i = i+1;
    int next_j = j+1;
		
		BlockMatrix<T> left,top,right,bottom;
    left = bm2;
		top = bm1;
		
		if (j < m->cols()-1) {
      right = (*m)(i,j+1);
    }
    if (i < m->rows()-1) {
      bottom = (*m)(i+1,j);
    }
		
		func(i,j,m->rows(),m->cols(),(*m)(i,j),left,top,right,bottom);

    if (next_i < m->rows())
    {
			if (j == 0)
				send<0>(Key(next_i,j), (*m)(i,j), out);
			else
				send<0>(Key(next_i,j), (*m)(i,j), out);
    }

    if (next_j < m->cols()) {
      if (i == 0) 
				send<0>(Key(i,next_j), (*m)(i,j), out);
			else
				send<1>(Key(i,next_j), (*m)(i,j), out);
    }
		
	};
	
	return wrap(f, edges(input1, input2), edges(input1, input2), "wavefront2", {"first","second"}, 
							{"output1", "output2"});
}

//Method to generate wavefront task with single input.
template <typename funcT, typename T>
auto make_wavefront(const funcT& func, Matrix<T> *m, Edge<Key, BlockMatrix<T>>& input1, Edge<Key, BlockMatrix<T>>& input2) 
{
	auto f = [m,func](const Key& key, BlockMatrix<T>&& bm, std::tuple<Out<Key, BlockMatrix<T>>, Out<Key, BlockMatrix<T>>, 
										Out<Key, BlockMatrix<T>>>& out) 
	{		
		auto [i,j] = key;
		int next_i = i+1;
    int next_j = j+1;
	
		//This handles the cases where i=0 or j=0, 
		//since these are the only tasks that have single input dependency.	
		BlockMatrix<T> left,top,right,bottom;
    if (i == 0) 
		{
			left = bm;
			bottom = (*m)(i+1,j);
		}
		if (j == 0) 
		{
			top = bm;
			right = (*m)(i,j+1);
		}

    if(j < m->cols()-1)
      right = (*m)(i,j+1);
    if (i < m->rows()-1)
      bottom = (*m)(i+1,j);
		
		func(i,j,m->rows(),m->cols(),(*m)(i,j),left,top,right,bottom);
		
    if (next_i < m->rows())
    {
			if (j == 0) //Single predecessor
				send<0>(Key(next_i,j), (*m)(i,j), out);
			else //Two predecessors
				send<1>(Key(next_i,j), (*m)(i,j), out);
    }
    if (next_j < m->cols()) {
			if (i == 0) //Single predecessor
				send<0>(Key(i,next_j), (*m)(i,j), out);
			else //Two predecessors
				send<2>(Key(i,next_j), (*m)(i,j), out);
    }
	
	};

	Edge<Key, BlockMatrix<T>> recur("recur");
	return wrap(f, edges(recur), edges(recur,input1,input2), "wavefront", {"control"}, {"recur","output1", "output2"});
}

int main(int argc, char** argv) 
{
	int n_rows, n_cols, B;
	int n_brows, n_bcols;

	n_rows = n_cols = 8192;
	B = 128;
	
	n_brows = (n_rows/B) + (n_rows % B > 0);
	n_bcols = (n_cols/B) + (n_cols % B > 0);

	Matrix<double> *m = new Matrix<double>(n_brows, n_bcols, B, B);
	Matrix<double> *m2 = new Matrix<double>(n_brows, n_bcols, B, B);

	Edge<Key, BlockMatrix<double>> parent1("parent1"), parent2("parent2");
	std::chrono::time_point<std::chrono::high_resolution_clock> beg,end;

	ttg_initialize(argc, argv, -1);
	{
		auto s = make_wavefront(stencil_computation<double>, m, parent1, parent2);
		auto s2 = make_wavefront2(stencil_computation<double>, m, parent1, parent2);
	
		auto connected = make_graph_executable(s.get());
		assert(connected);
		TTGUNUSED(connected);
		std::cout << "Graph is connected.\n";
	
		if (ttg_default_execution_context().rank() == 0) {
			//std::cout << "==== begin dot ====\n";
			//std::cout << Dot()(s.get()) << std::endl;
			//std::cout << "==== end dot ====\n";
			
			beg = std::chrono::high_resolution_clock::now();
			s->in<0>()->send(Key(0,0), (*m)(0,0));	
			//This doesn't work!
			//s->send<0>(Key(0,0), Control());
		}
	
		ttg_execute(ttg_default_execution_context());
		ttg_fence(ttg_default_execution_context());
		end = std::chrono::high_resolution_clock::now();
		std::cout << "TTG Execution Time (milliseconds) : " << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count())/1000 << std::endl;
	}

	ttg_finalize();

	std::cout << "Computing using serial version....";
	beg = std::chrono::high_resolution_clock::now();
	wavefront_serial(m2, n_brows, n_bcols);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "....done!" << std::endl;
	std::cout << "Serial Execution Time (milliseconds) : " << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count())/1e3 << std::endl;

	//m->print();
	//std::cout << std::endl << std::endl;
	//m2->print();
	
	std::cout << "Verifying the result....";
	bool success = true;
	for (int i=0; i<n_brows; i++) {
		for (int j=0; j<n_bcols; j++)
		{
			if ((*m)(i,j) != (*m2)(i,j)) {
				success = false;
				std::cout << "ERROR in block (" << i << "," << j  << ")\n";
			}
		}
	}
		
	std::cout << "....done!" << std::endl;
	std::cout << (success ? "SUCCESS!!!" : "FAILED!!!") << std::endl;
	
	return 0;
}
