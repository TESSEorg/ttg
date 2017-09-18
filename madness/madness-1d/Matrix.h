#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <stdexcept>

using namespace std;


class Matrix {
public:
	/* data members */
	unsigned int row_size;
	unsigned int col_size;
	/*Vector** data;*/

	vector<double> data;

	/* member functions */

	/* Constructor */
	Matrix(int n, int m): row_size(n), col_size(m) {
		data.resize(m * n);
	}

	/* Another constructor */
	Matrix(int n, int m, double **inputData) {
		row_size = n;
		col_size = m;
		data.resize(row_size * col_size);

		for (unsigned int i = 0; i < row_size; ++i) {
			for (unsigned int j = 0; j < col_size; ++j) {
				data[i * col_size + j] = inputData[i][j];
			}
		}
	}

	/* dims */
	const unsigned int get_row_size() const{
	   return row_size;
	}

	const unsigned int get_col_size() const{
	   return col_size;
	}

	const unsigned int get_size() const {
		return row_size * col_size;
	}

	/* non-modifiable __getitem__ */
	double get_item(unsigned int index_row, unsigned int index_column) const {

		return data.at(index_row * col_size + index_column);
	}
 
	double & get_item(unsigned int index_row, unsigned int index_column) {

		return data.at(index_row * col_size + index_column);
	}


        void set_item(const unsigned int index_row, const unsigned int index_col, const double value) {
	
		data.at(index_row * col_size + index_col) = value;
	}

	/* __str__ */
	void print_matrix() {
//		 cout << setw(12) << setprecision(6) << fixed; 
		for (unsigned int i = 0; i < row_size; ++i) {
			for (unsigned int j = 0; j < col_size; ++j) {
				 //cout << setw(12) << setprecision(6) << fixed << 
				 cout << j << " = " << data[i * col_size +  j] << " ";
			}
			cout << endl;
		}

	}

	Matrix &operator=(const Matrix &m2) {

		if (this == &m2) return *this;

		data.clear();

		row_size = m2.row_size;
		col_size = m2.col_size;

		data.reserve(row_size * col_size);

		for (unsigned int i = 0; i < row_size; ++i) {
			for (unsigned int j = 0; j < col_size; ++j) {
				data[i * col_size + j] = m2.data[i * col_size + j];
			}
		}

		return *this;

	}


	/* __mul__ */
	friend Vector operator*(const Matrix &left, const Vector &right);

	/* __rmul__ */
	friend Vector operator*(const Vector &left, const Matrix &right);

};

Vector operator*(const Matrix &left, const Vector &right) {
	if (left.get_col_size() != right.length()) {
		string message = "matrix and vector do not conform. The vector's size is ";
		message += right.length();
		message += "!!";
		//throw std::invalid_argument("matrix and vector do not conform");
		throw std::invalid_argument(message);
	}

	Vector result(left.get_row_size());
	for (unsigned int i = 0; i < left.get_row_size(); ++i) {
		for (unsigned int j = 0; j < left.get_col_size(); ++j) {
			result[i] = result[i] + (left.get_item(i, j) * right.get_item(j));
		}
	}

	return result;
}

Vector operator*(const Vector &left, const Matrix &right) {

	if (right.get_row_size() != left.length()) {
		throw std::invalid_argument("Matrix::operator*::matrix and vector do not conform");
	}

        Vector result(right.get_col_size());
	for (unsigned int i = 0; i < right.get_col_size(); ++i) {
		for (unsigned int j = 0; j < right.get_row_size(); ++j) {
			result[i] = result[i] + (left.get_item(j) * right.get_item(j, i));
		}
	}

	return result;
}


#endif /* MATRIX_H_ */

