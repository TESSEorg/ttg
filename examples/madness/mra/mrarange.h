#ifndef MADRANGE_H_INCL
#define MADRANGE_H_INCL

namespace mra {

    /// A simple unsigned integer range (0,...,n-1) to support iteration a la Python range(n)
    class RangeSimple {
        const size_t finish;
        struct iterator {
            size_t value;
            iterator (size_t value) : value(value) {}
            operator size_t() const {return value;}
            size_t operator*() const {return value;}
            iterator& operator++ () {value++; return *this;}
            bool operator!=(const iterator&other) {return value != other.value;}
        };
    public:
        RangeSimple(size_t finish) : finish(finish) {}
        iterator begin() const { return iterator(0); }
        iterator end() const { return iterator(finish); }
        size_t initial_size() const {return finish;}
    };
    
    /// A simple signed integer range to support iteration a la Python range(start,finish,step)
    class Range {
        long start, count, finish, step;
        struct iterator {
            long value;
            const long step;
            iterator (long value, long step) : value(value), step(step) {}
            operator long() const {return value;}
            long operator*() const {return value;}
            iterator& operator++ () {value+=step; return *this;}
            bool operator!=(const iterator&other) {return value != other.value;}
        };
    public:
        Range(long start, long finish, int step)
            : start(start)
            , count(std::max(long(0),((finish-start-step/std::abs(step))/step+1)))
            , finish(start + count*step)
            , step(step)
        {}
        iterator begin() const { return iterator(start,step); }
        iterator end() const { return iterator(finish,step); }
        long initial_size() const {return count;}
    };
    
    /// range(n) will iterate over 0,...,n-1 (i.e., excludes n) a la Python
    inline static RangeSimple range(size_t finish) {return RangeSimple(finish);}
    
    /// range(start,finish,step) will iterate over start, start+step, ... excluding finish a la Python
    
    /// supports negative step and if step!=1 also supports finish not being part of the sequence.
    inline static Range range(long start, long finish, long step=1) {return Range(start,finish,step);}
}

#endif
