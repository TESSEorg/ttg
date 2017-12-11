
#ifndef VECTOR_H_
#define VECTOR_H_

#include <iostream>
#include <string>
#include <math.h>
#include <cstddef>
#include <sstream>
#include <vector>
#include <algorithm>

#include <madness/world/worldhash.h>

using namespace std;


class Vector {
public:
   vector<double> data;

   /* member functions */

   /* constructor */
   Vector(int inputSize = 0) {
      if (inputSize > 0)
         data.resize(inputSize, 0);
   }

   void reserve(int newSize) {
      data.resize(newSize);
   }

   void resize(int newSize) {
      data.resize(newSize);
   }

   /* another constructor */
   Vector(const double *array, int low, int high) : data (array + low, array + high) {}

   /* another constructor */
   Vector(const vector<double> &array, int low, int high) : data(array.begin() + low, array.begin() + high) {}

   /* other functions */
   double normf(unsigned int low, unsigned int high) {
     double sum = 0.0;
     for (unsigned int i = low; i < high; ++i) {
         sum += (data.at(i) * data.at(i));
      }
      return sqrt(sum);
   }

   double normf() {
      double sum = 0.0;
      for (unsigned int i = 0; i < data.size(); ++i) {
         sum += (data.at(i) * data.at(i));
      }
      return sqrt(sum);
   }


   double inner(const Vector &v2) {
      double sum = 0.0;
      for (unsigned int i = 0; i < data.size(); ++i) {
         sum += (data.at(i) * v2.data.at(i));
      }
      return sum;
   }

   Vector &gaxpy(double alpha, const Vector &v2, double beta){
      for (unsigned int i = 0; i < data.size(); ++i){
         data.at(i) = (alpha * data.at(i)) + (beta * v2.data.at(i));
      }
      return *this;
   }
   
   Vector gaxpy_byValue(double alpha, const Vector &v2, double beta) {
     Vector result(data.size());
     for (unsigned int i = 0; i < data.size(); ++i) {
       result.data.at(i) = (alpha * data.at(i)) + (beta * v2.data.at(i));
       //result.data.push_back(alpha * data[i] + beta * v2.data[i]);
     }
     return result;
   }


   void gaxpy_inplace(const Vector &v2) {
     for (unsigned int i = 0; i < data.size(); ++i) {
         data[i] += v2.data[i];
      }
   }

   Vector &scale(double s){
      for (unsigned int i = 0; i < data.size(); ++i) {
         data[i] *= s;
      }
      return *this;
   }

   Vector &emule(const Vector &v2) {
      for (unsigned int i = 0; i < data.size(); ++i) {
         data.at(i) *= v2.data.at(i);
      }
      return *this;
   }

   double get_item(unsigned int ind) const {
      return data.at(ind);
   }

   void set_item(unsigned int ind, double value) {
      data.at(ind) = value;
   }

   Vector get_slice(unsigned int low, unsigned int high) {
      if (low > high) {
         cout << "Vector::get_slice::The value of low is greater than the value of high" << endl;
         return 0;
      }
      else if ((low < 0) || (low >= data.size())) {
         cout << "Vector::get_slice:: the value of the parameter low is either less than 0 or greater/equal to size" << endl;
         return 0;
      }
      else if ((high < 0) || (high > data.size())) {
         cout << "Vector::get_slice:: the value of the parameter high is either less than 0 or greater/equal to size" << endl;
         return 0;
      }
   
     return Vector(data, low, high);

   }

   void set_slice_from_another_vector(unsigned int low, unsigned int high, const Vector &v) {
     for (unsigned int i = low; i < high; ++i) {
       data.at(i) = v.data.at(i);
     }
   }

   void set_slice(unsigned int low, unsigned int high, double value) {
      for (unsigned int i = low; i < high; ++i) {
          data.at(i) = value;
      }
      return;
   }

   unsigned int length() const {
      return data.size();
   }

   void print_vector() const {
      cout << "[";
      if (data.size() >= 1) {
          for (unsigned int i = 0; i < (data.size() - 1); ++i) {
              ostringstream strs;
              strs << data.at(i);
              cout << strs.str();
              cout << ", ";
          }

          ostringstream strs;
          strs << data.at(data.size() - 1);
          cout << strs.str();
      }
      cout << "]" << endl;
      return ;

   }

   double &operator[](unsigned int ind) {
      return data.at(ind);
   }

   template<typename Archive> void serialize(Archive &ar) {ar & data;}

   friend const Vector operator|(const Vector &left, const Vector &right);
   friend const Vector operator+(const Vector &left, const Vector &right);

   friend std::ostream & operator<<(std::ostream &out, const Vector &vector);

};

const Vector operator|(const Vector &left, const Vector &right) {
  Vector result(left.data.size() + right.data.size());
  for (unsigned int i = 0; i < left.data.size(); ++i) {
    result.data.at(i) = left.data.at(i);
  }

  for (unsigned int j = 0; j < right.data.size(); ++j) {
    result.data.at(left.data.size() + j) = right.data.at(j);
    //result.data.push_back(right.data[j]);
   }

  return result;
}

const Vector operator+(const Vector &left, const Vector &right) {

    Vector result(left.data.size());
    for (unsigned int i = 0; i < left.data.size(); ++i) {
        result.data.at(i) = left.data.at(i) + right.data.at(i);
    }

    return result;
}

std::ostream & operator<<(std::ostream &out, const Vector &vector) {
  out << "[";
  if (vector.data.size() >= 1) {
    for (unsigned int i = 0; i < (vector.data.size() - 1); ++i) {
      out << vector.data.at(i);
      out << ", ";
    }

    out << vector.data[vector.data.size() - 1];
  }

  out << "]";
  return out;
}


/*
namespace madness {
      namespace archive {
                template <class Archive>
                struct ArchiveLoadImpl<Archive, Vector> {
                   static inline void load(const Archive& ar, Vector& obj) {
                       int size;
			ar & size;
			obj.resize(size);
			ar & wrap(obj.data, size);
                   }
                };

                template <class Archive>
                struct ArchiveStoreImpl<Archive, Vector> {
                   static inline void store(const Archive& ar, const Vector& obj) {
                         ar & obj.data.size() & wrap(obj.data, obj.data.size()); 	
                   }
                };
      }
  }
*/


#endif /* VECTOR_H_ */

