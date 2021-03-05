#ifndef MADMISC_H_INCL
#define MADMISC_H_INCL

#include <iostream>
#include <utility>
#include <vector>
#include <array>

#include "mrarange.h"

namespace mra {

    /// Implements simple Kahan or compensated summation
    template <typename T>
    class KahanAccumulator {
        T sum;
        T c;
    public:
        KahanAccumulator() {} // Must be empty for use in shared memory
        
        KahanAccumulator(T s) : sum(s), c(0) {}
        
        KahanAccumulator& operator=(const T s) {
            sum = s;
            c = 0;
            return *this;
        }
        
        KahanAccumulator& operator+=(const T input) {
            T y = input - c;
            T t = sum + y;
            c = (t - sum) - y;
            sum = t;
            return *this;
        }
        
        KahanAccumulator& operator+=(const KahanAccumulator& input) {
            (*this) += input.sum;
            (*this) += -input.c;
            return *this;
        }
        
        operator T() const {
            return sum;
        }
    };

    /// Easy printing of pairs
    template <typename T, typename R>
    std::ostream& operator<<(std::ostream& s, const std::pair<T,R>& a) {
        s << "(" << a.first << "," << a.second << ")";
        return s;
    }
    
    /// Easy printing of arrays
    template <typename T, size_t N>
    std::ostream& operator<<(std::ostream& s, const std::array<T,N>& a) {
        s << "[";
        for (auto i : range(a.size())) {
            s << a[i];
            if (i != a.size()-1) s << ", ";
        }
        s << "]";
        return s;
    }
    
    /// Easy printing of vectors
    template <typename T>
    std::ostream& operator<<(std::ostream& s, const std::vector<T>& a) {
        s << "[";
        for (auto i : range(a.size())) {
            s << a[i];
            if (i != a.size()-1) s << ", ";
        }
        s << "]";
        return s;
    }
    
}

#endif
