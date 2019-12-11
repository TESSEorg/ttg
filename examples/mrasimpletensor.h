#ifndef MAD_SIMPLETENSOR_H_INCL
#define MAD_SIMPLETENSOR_H_INCL

#include <array>
#include <cmath>
#include <tuple>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <type_traits>

#include "mramisc.h"
#include "mratypes.h"
#include "mrarange.h"

namespace mra {
    /// Slice now acts like Python slice or Elemental Range (so end point is NOT inclusive, and must say END to indicate END)
    class Slice {
    public:
        static constexpr long END = std::numeric_limits<long>::max();
        long start;  //< Start of slice (must be signed type)
        long finish; //< Inclusive end of slice (must be signed type)
        long step;   //< Stride for slice (must be signed type)
        long count;  //< Number of elements in slice (not known until dimension is applied; negative indicates not computed)
        Slice() : start(0), finish(END), step(1), count(-1) {}; // indicates entire range
        Slice(long start) : start(start), finish(start+1), step(1) {} // a single element
        Slice(long start, long end, long step=1) : start(start), finish(end), step(step) {};

        /// Once we know the dimension we adjust the start/end/count/finish to match, and do sanity checks
        void apply_dim(long dim) {
            if (start == END) {start = dim-1;}
            else if (start < 0) {start += dim;}
            
            if (finish == END && step > 0) {finish = dim;}
            else if (finish == END && step < 0) {finish = -1;}
            else if (finish < 0) {finish += dim;}

            count = std::max(0l,((finish-start-step/std::abs(step))/step+1));
            assert((count==0) || ((count<=dim) && (start>=0 && start<=dim)));
            finish = start + count*step; // finish is one past the last element
        }

        struct iterator {
            long value;
            const long step;
            iterator (long value, long step) : value(value), step(step) {}
            operator long() const {return value;}
            long operator*() const {return value;}
            iterator& operator++ () {value+=step; return *this;}
            bool operator!=(const iterator&other) {return value != other.value;}
        };

        iterator begin() const {assert(count>=0); return iterator(start,step); }
        iterator end() const {assert(count>=0); return iterator(finish,step); }

        Slice& operator=(const Slice& other) {
            if (this != &other) {
                start = other.start;
                finish = other.finish;
                step = other.step;
                count = other.count;
            }
            return *this;
        }
    };

    static constexpr long END = Slice::END;
    static const Slice _(0,END,1);	/// Entire dimension
    class Everything{};
    static constexpr Everything ___; /// Entire tensor
    static const Slice reverse(-1,END,-1); /// Reversed dimension

    inline static std::ostream& operator<<(std::ostream& stream, const Slice& s) {
        stream << "Slice(" << s.start << ",";
        if (s.finish == Slice::END) stream << ":,";
        else
            stream << s.finish << ",";
        stream << s.step << ")";
        return stream;
    }

    namespace detail {
        // If using c++-17 can use fold expression instead of some of these
        
        // Compute the product of the template parameters
        template <size_t v0, size_t ... Values> struct Product {static const size_t value = v0*Product<Values...>::value;};
        template <size_t v0> struct Product<v0> {static constexpr size_t value = v0;};

        // Raise the first argument to the power of the second
        template <size_t x, size_t N> struct Power {static const size_t value = x * Power<x,N-1>::value;};
        template <size_t x> struct Power<x,0> {static const size_t value = 1;};

        // Compute the maximum value of the template parameters
        template <size_t v0, size_t ... Values> struct Max {static const size_t value = std::max(v0,Product<Values...>::value);};
        template <size_t v0> struct Max<v0> {static constexpr size_t value = v0;};
        
        // Access size of d'th dimension
        template <size_t d, size_t v0, size_t ... Values> struct GetDim {static constexpr size_t value = (d==0) ? v0 :  GetDim<d-1, Values...>::value;};
        template <size_t d, size_t v0> struct GetDim<d,v0> {static constexpr size_t value = v0;};

        // Access stride of d'th dimension with GetStride<d,NDIM-1,Dims...>::value
        template <size_t d, size_t D, size_t v0, size_t ... Values> struct GetStride {
            static constexpr size_t value = ((d>=D) ? 1 : v0) * GetStride<d,D+1,Values...>::value;
        };
        template <size_t d, size_t D, size_t v0> struct GetStride<d,D,v0> {static constexpr size_t value = (d>=D) ? 1 : v0;};
        
        // Check that all types are integral (>=1)
        template <typename T0, typename ... Ts> struct IsIntegral {static constexpr bool value = std::is_integral<T0>::value && IsIntegral<Ts...>::value;};
        template <typename T> struct IsIntegral<T> {static constexpr bool value = std::is_integral<T>::value;};

        // Check that all types are slices or convertible to slices (>=1)
        template <typename T0, typename ... Ts> struct IsSlice {static constexpr bool value = std::is_convertible<T0,Slice>::value && IsSlice<Ts...>::value;};
        template <typename T> struct IsSlice<T> {static constexpr bool value = std::is_convertible<T,Slice>::value;};
        
        template <typename tensorT, size_t num_dimensions>
        struct base_tensor_iterator {
            size_t count;
            const tensorT* t;
            std::array<size_t,std::max(size_t(1),num_dimensions)> indx = {};
            
            base_tensor_iterator (size_t count, const tensorT* t)
                : count(count)
                , t(t)
            {}
            
            void inc() {
                assert(count < t->size());
                count++;
                for (int d=int(num_dimensions)-1; d>=0; --d) { // must be signed loop variable!
                    indx[d]++;
                    if (indx[d]<t->dim(d)) {
                        break;
                    } else {
                        indx[d] = 0;
                    }
                }
            }

            const auto& index() const {return indx;}
        };
    }

    template <typename tensorT, typename opT>
    void apply_unaryop(tensorT& t, const opT& op) {
        if constexpr (t.is_contiguous()) { // by definition not a slice tensor
            std::for_each(t.data().begin(), t.data().end(), op);
        }
        else if (t.stride(t.ndim()-1) == 1) { // inner loop is contiguous
            if constexpr (t.ndim() == 1) {
                const size_t n = t.dim(t.ndim()-1);
                auto p = t.ptr();
                for (size_t i=0; i<n; ++i) op(p[i]);                
            }
            else if constexpr (t.ndim() >= 2) {
                const size_t dimi = t.dim(t.ndim()-1);
                const size_t dimj = t.dim(t.ndim()-2);
                const size_t stridei = t.stride(t.ndim()-1);
                const size_t stridej = t.stride(t.ndim()-2);
                if (stridei == 1) {
                    for (auto it=t.xbegin(); it!=t.xend(); ++it) {
                        auto p = t.ptr(it);
                        for (size_t j=0; j<dimj; ++j, p+=stridej) {
                            for (size_t i=0; i<dimi; ++i) {
                                op(p[i]);
                            }
                        }
                    }
                }
                else {
                    for (auto it=t.xbegin(); it!=t.xend(); ++it) {
                        auto p = t.ptr(it);
                        for (size_t j=0; j<dimj; ++j, p+=stridej) {
                            auto q = p;
                            for (size_t i=0; i<dimi; ++i, q+=stridei) {
                                op(*q);
                            }
                        }
                    }
                }
            }
        }
        else { // general but slowest iteration
            std::cout << "in slow apply_unaryop ... what are you doing?\n";
            std::for_each(t.begin(), t.end(), op);
        }        
    }

    
    template <typename tensorA, typename tensorB, typename opT>
    void apply_binaryop(tensorA& a, tensorB& b, const opT& op) {
        assert(a.conforms(b));
        if (a.is_contiguous() && b.is_contiguous()) { // by definition not a slice tensor
                std::cout << "bina ... not been here yet?\n";
            auto pa = a.ptr();
            auto pb = b.ptr();
            for (size_t i=0; i<a.size(); i++) op(pa[i],pb[i]);
        }
        else if (a.stride(a.ndim()-1) == 1 && b.stride(b.ndim()-1) == 1) { // inner loop is contiguous
            if (a.ndim() == 1) {
                std::cout << "binb ... not been here yet?\n";
                const size_t n = a.dim(a.ndim()-1);
                auto pa = a.ptr();
                auto pb = b.ptr();
                for (size_t i=0; i<n; ++i) op(pa[i],pb[i]);                
            }
            else if (a.ndim() >= 2) {
                const size_t dimi = a.dim(a.ndim()-1);
                const size_t dimj = a.dim(a.ndim()-2);
                const size_t strideia = a.stride(a.ndim()-1);
                const size_t strideja = a.stride(a.ndim()-2);
                const size_t strideib = b.stride(b.ndim()-1);
                const size_t stridejb = b.stride(b.ndim()-2);
                auto ita = a.xbegin();
                auto itb = b.xbegin();
                const auto enda = a.xend();
                const auto endb = b.xend();
                if (strideia == 1 && strideib == 1) {
                    //std::cout << "binc\n";
                    while (ita != enda) {
                        auto pa = a.ptr(ita);
                        auto pb = b.ptr(itb);
                        for (size_t j=0; j<dimj; ++j, pa+=strideja, pb+=stridejb) {
                            for (size_t i=0; i<dimi; ++i) {
                                op(pa[i],pb[i]);
                            }
                        }
                        ++ita;
                        ++itb;
                    }
                }
                else {
                    std::cout << "bind ... not been here yet?\n";
                    while (ita != enda) {
                        auto pa = a.ptr(ita);
                        auto pb = b.ptr(itb);
                        for (size_t j=0; j<dimj; ++j, pa+=strideja, pb+=stridejb) {
                            auto qa = pa;
                            auto qb = pb;
                            for (size_t i=0; i<dimi; ++i, qa+=strideia, qb+=strideib) {
                                op(qa[i],qb[i]);
                            }
                        }
                        ++ita;
                        ++itb;
                    }
                }
                assert(itb == endb); // should be true if they really do conform
                TTGUNUSED(endb);
            }
        }
        else { // general but slowest iteration
            std::cout << "in slow apply_binaryop ... what are you doing?\n";
            auto ita = a.begin();
            auto itb = b.begin();
            const auto enda = a.end();
            const auto endb = b.end();
            while (ita != enda) {
                op(*ita,*itb);
                ++ita;
                ++itb;
            }
            assert(itb == endb); // should be true if they really do conform
            TTGUNUSED(endb);
        }        
    }

    template <typename tensorT>
    class SliceTensor {

    public:
        using data_type = typename tensorT::data_type;
        static constexpr size_t num_dimensions = tensorT::ndim(); //< Number of dimensions in the tensor
        static constexpr bool is_tensor = true;

    private:
        tensorT& t; // The tensor being viewed
        std::array<Slice,num_dimensions> slices;
        std::array<size_t,num_dimensions> dimensions; // dimensions of the sliced tensor
        size_t num_elements; // number of elements in the sliced tensor
        
        SliceTensor() = delete; // no default constuctor

        // Computes index in dimension d for underlying tensor using slice info
        inline size_t index(size_t d, size_t i) const {return slices[d].start+i*slices[d].step;}
        
        // Given indices in slice as arguments looks up element in underlying tensor
        template <typename returnT, size_t...D, typename...Args>
        returnT
        access(std::index_sequence<D...>, Args...args) const {return t(index(D,args)...);}

        // Given indices in slice in an array looks up element in underlying tensor
        template <typename returnT, size_t...D>
        returnT
        access(std::index_sequence<D...>, const std::array<size_t,num_dimensions>& indices) const {return t(index(D,indices[D])...);}
        
        using ST = SliceTensor<tensorT>;
        template<size_t ndimactive>
        struct iterator : public detail::base_tensor_iterator<ST,ndimactive> {
            iterator (size_t count, ST* t) : detail::base_tensor_iterator<ST,ndimactive>(count, t) {}
            data_type& operator*() {return this->t-> template access<data_type&>(std::make_index_sequence<ndimactive>{},this->indx);}
            iterator& operator++() {this->inc(); return *this;}
            bool operator!=(const iterator& other) {return this->count != other.count;}
            bool operator==(const iterator& other) {return this->count == other.count;}
        };
        
        template<size_t ndimactive>
        struct const_iterator : public detail::base_tensor_iterator<ST,ndimactive> {
            const_iterator (size_t count, const ST* t) : detail::base_tensor_iterator<ST,ndimactive>(count, t) {}
            data_type operator*() const {return this->t-> template access<data_type>(std::make_index_sequence<ndimactive>{},this->indx);}
            const_iterator& operator++() {this->inc(); return *this;}
            bool operator!=(const const_iterator& other) {return this->count != other.count;}
            bool operator==(const const_iterator& other) {return this->count == other.count;}
        };
        
        iterator<num_dimensions> finish{0,0};
        const_iterator<num_dimensions> cfinish{0,0};

    public:
        SliceTensor(tensorT& t, const std::array<Slice,num_dimensions>& slices)
            : t(t)
            , slices(slices)
        {
            num_elements = 1;
            for (size_t d : range(num_dimensions)) {
                this->slices[d].apply_dim(t.dim(d));
                num_elements *= this->slices[d].count;
                dimensions[d] = this->slices[d].count;
            }
            finish = iterator<num_dimensions>{num_elements,0};
            cfinish = const_iterator<num_dimensions>{num_elements,0};
        }
        
        /// Returns number of dimensions at compile time
        static constexpr size_t ndim() {return num_dimensions;}

        /// Returns number of elements in the tensor at runtime
        size_t size() const {return num_elements;}

        /// Returns size of dimension d at runtime
        size_t dim(size_t d) const {return slices[d].count;}        

        /// Returns array containing size of each dimension at runtime
        const std::array<size_t, num_dimensions>& dims() const {return dimensions;}

        /// Returns stride of dimension d in *underlying* data at runtime
        size_t stride(size_t d) const {
            return t.stride(d)*slices[d].step;
        }

        /// Returns true if data is contiguous (presently always false)
        static constexpr bool is_contiguous() {return false;}

        template <typename...Args, typename X=data_type>
        typename std::enable_if<std::is_const<tensorT>::value,X>::type
        operator()(Args...args) const {
            static_assert(num_dimensions == sizeof...(Args), "SliceTensor number of indices must match dimension");
            return access<X>(std::index_sequence_for<Args...>{},args...);
        }

        template <typename...Args, typename X=data_type>
        typename std::enable_if<!std::is_const<tensorT>::value,X&>::type
        operator()(Args...args) {
            static_assert(num_dimensions == sizeof...(Args), "SliceTensor number of indices must match dimension");
            return access<X&>(std::index_sequence_for<Args...>{},args...);
        }

        /// Fill with scalar ... needs optimized iterator
        template <typename X=SliceTensor<tensorT>>
        typename std::enable_if<!std::is_const<tensorT>::value,X&>::type
        operator=(data_type t) {for (data_type& x : *this) x= t; return *this;}

        /// Returns true if this and other conform (dimensions all same size)
        template <typename otherT> bool conforms(const otherT& other) const {return this->dims() == other.dims();}

        /// Copy into patch ... desperately needs optimized iterator
        template <typename otherT, typename X=SliceTensor<tensorT>>
        typename std::enable_if<otherT::is_tensor && !std::is_const<tensorT>::value,X&>::type
        operator=(const otherT& other) {
            // assert(conforms(other));
            // auto lit=this->begin();
            // auto rit=other.begin();
            // while (lit != this->end() || rit != other.end()) {
            //     *lit = *rit;
            //     ++lit;
            //     ++rit;
            // }
            // assert(lit==this->end() && rit==other.end());
            apply_binaryop(*this,other,[](data_type& a, const typename SliceTensor<otherT>::data_type& b) {a=b;});
            return *this;
        }        

        /// Start for forward iteration through elements in row-major order --- this is convenient but not efficient
        iterator<num_dimensions> begin() {return iterator<num_dimensions>(0,this);}

        /// End for forward iteration through elements in row-major order --- this is convenient but not efficient
        const iterator<num_dimensions>& end() {return finish;}

        /// Start for forward iteration through elements in row-major order --- this is convenient but not efficient
        const_iterator<num_dimensions> begin() const {return const_iterator<num_dimensions>(0,this);}

        /// End for forward iteration through elements in row-major order --- this is convenient but not efficient
        const const_iterator<num_dimensions>& end() const {return cfinish;}

        /// Optimized iteration --- start for forward iteration through all but inner two dimensions
        iterator<num_dimensions-2> xbegin() {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            return iterator<num_dimensions-2>(0,this);
        }

        /// Optimized iteration --- end for forward iteration  through all but inner two dimensions
        iterator<num_dimensions-2> xend() {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            size_t count = num_elements/dim(num_dimensions-1)/dim(num_dimensions-2);
            return {count,0};
        }

        /// Optimized iteration --- start for forward iteration through all but inner two dimensions
        const_iterator<num_dimensions-2> xbegin() const {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            return const_iterator<num_dimensions-2>(0,this);
        }

        /// Optimized iteration --- end for forward iteration  through all but inner two dimensions
        const const_iterator<num_dimensions-2> xend() const {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            size_t count = num_elements/dim(num_dimensions-1)/dim(num_dimensions-2);
            return {count,0};
        }

        /// Optimized iteration ... returns address of first element in underlying data which will likely not be contiguous
        data_type* ptr() {
            auto p = t.ptr();
            for (auto d : range(num_dimensions)) p += slices[d].start*stride(d);
            return p;
        }

        /// Optimized iteration ... returns address of first element in underlying data which will likely not be contiguous
        const data_type* ptr() const {
            auto p = t.ptr();
            for (auto d : range(num_dimensions)) p += slices[d].start*stride(d);
            return p;
        }

        /// Optimized iteration ... given optimized (ndim-2) iterator returns address of first element in underlying data which will likely not be contiguous
        template <typename iteratorT>
        data_type* ptr(iteratorT& it) {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            auto p = t.ptr() + slices[num_dimensions-2].start*stride(num_dimensions-2) + slices[num_dimensions-1].start*stride(num_dimensions-1);
            auto& indx = it.index();
            for (auto d : range(num_dimensions-2)) p += (slices[d].start + indx[d]) * t.stride(d);
            return p;
        }

        /// Optimized iteration ... given optimized (ndim-2) iterator returns address of first element in underlying data which will likely not be contiguous
        template <typename iteratorT>
        const data_type* ptr(iteratorT& it) const {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            auto p = t.ptr() + slices[num_dimensions-2].start*stride(num_dimensions-2) + slices[num_dimensions-1].start*stride(num_dimensions-1);
            auto& indx = it.index();
            for (auto d : range(num_dimensions-2)) p += (slices[d].start + indx[d]) * t.stride(d);
            return p;
        }
    };

    template <typename T, size_t ... Dims>
    class SimpleTensor {

    public:
        using data_type = T; //< Type of each element in the tensor
        static constexpr size_t num_dimensions = sizeof...(Dims); //< Number of dimensions in the tensor
        static constexpr size_t num_elements = detail::Product<Dims...>::value; //< Number of elements in the tensor
        static constexpr bool is_tensor = true;
        static constexpr bool is_simple_tensor = true;
        
    private:
        template <size_t d> struct Dim {static constexpr size_t value = detail::GetDim<d,Dims...>::value;};
        template <size_t d> struct Stride {static constexpr size_t value = detail::GetStride<d,0,Dims...>::value;};
        template <size_t...S> static constexpr std::array<size_t,num_dimensions> make_strides(std::index_sequence<S...>) {return {Stride<S>::value...};}
        template <size_t d> static void check(size_t i) {assert(i < Dim<d>::value);}

        static constexpr bool bounds_check = true; //< If true bounds are checked on access
        inline static constexpr std::array<size_t, num_dimensions> dimensions = {Dims...}; //< Array containing size of each dimension
        inline static constexpr std::array<size_t, num_dimensions> stride_array = make_strides(std::make_index_sequence<num_dimensions>{});
        
        std::array<T, num_elements> a; //< Holds the data

        template <size_t D> static size_t offset(size_t i) {if (bounds_check) {check<D>(i);} return i*Stride<D>::value;}
        
        template <size_t...D,typename...Args> static size_t sum_offset(std::index_sequence<D...>, Args...args) {return (offset<D>(args)+...);}

        template <size_t ndimactive>
        struct iterator : public detail::base_tensor_iterator<SimpleTensor<T,Dims...>,ndimactive> {
            iterator (size_t count, SimpleTensor<T,Dims...>* t) : detail::base_tensor_iterator<SimpleTensor<T,Dims...>,ndimactive>(count, t) {}
            T& operator*() {return const_cast<SimpleTensor<T,Dims...>*>(this->t)->a[this->count];}
            iterator& operator++() {this->inc(); return *this;}
            bool operator!=(const iterator& other) {return this->count != other.count;}
            bool operator==(const iterator& other) {return this->count == other.count;}
        };

        template <size_t ndimactive>
        struct const_iterator : public detail::base_tensor_iterator<SimpleTensor<T,Dims...>,ndimactive> {
            const_iterator (size_t count, const SimpleTensor<T,Dims...>* t) : detail::base_tensor_iterator<SimpleTensor<T,Dims...>,ndimactive>(count, t) {}
            T operator*() const {return this->t->a[this->count];}
            const_iterator& operator++() {this->inc(); return *this;}
            bool operator!=(const const_iterator& other) {return this->count != other.count;}
            bool operator==(const const_iterator& other) {return this->count == other.count;}
        };
        
        inline static iterator<num_dimensions> finish = {num_elements,0};
        inline static const_iterator<num_dimensions> cfinish = {num_elements,0};

        inline static iterator<num_dimensions-2> finish2 = {num_elements/Dim<num_dimensions-1>::value/Dim<num_dimensions-2>::value,0};
        inline static const_iterator<num_dimensions-2> cfinish2 = {num_elements/Dim<num_dimensions-1>::value/Dim<num_dimensions-2>::value,0};

    public:
        /// Default constructor does not initialize data --- need this to be POD
        SimpleTensor() = default;

        /// Constructor initializing all elements to a constant
        SimpleTensor(data_type t) {*this = t;}

        /// Copy constructor is deep (with possible type conversion) from identically shaped tensor
        template <typename R>
        SimpleTensor(const SimpleTensor<R,Dims...>& other) {
            for (size_t i=0; i<num_elements; ++i) a[i] = T(other.a[i]);
        }
        
        /// Returns number of elements in the tensor
        static constexpr size_t size() {return num_elements;}
        
        /// Returns number of dimensions in the tensor
        static constexpr size_t ndim() {return num_dimensions;}

        /// Returns true if data is contiguous (presently always true)
        static constexpr bool is_contiguous() {return true;}
        
        /// Returns array with size of each dimension
        static constexpr const std::array<size_t, num_dimensions>& dims() {return dimensions;}

        /// Returns array with stride of each dimension
        static constexpr const std::array<size_t, num_dimensions>& strides() {return stride_array;}

        /// Returns size of dimension d at compile time
        template <size_t d> static constexpr size_t dim() {
            static_assert(d < num_dimensions);
            return Dim<d>::value;
        }
        
        /// Returns stride of dimension d at compile time
        template <size_t d> static constexpr size_t stride() {
            static_assert(d < num_dimensions);
            return Stride<d>::value;
        }

        /// Returns size of dimension d at runtime
        static size_t dim(size_t d) {
            if (bounds_check) assert(d>=0 && d<num_dimensions);
            return dimensions[d];
        }

        /// Returns stride of dimension d at runtime
        static size_t stride(size_t d) {
            if (bounds_check) assert(d>=0 && d<num_dimensions);
            return stride_array[d];
        }
        
        /// Returns value of element (const element access)
        template <typename...Args, typename X=T>
        typename std::enable_if<detail::IsIntegral<Args...>::value,X>::type
        operator()(Args...args) const {
            static_assert(num_dimensions == sizeof...(Args), "SimpleTensor number of indices must match dimension");
            size_t offset = sum_offset(std::index_sequence_for<Args...>{},args...);
            if (bounds_check) assert(offset>=0 && offset<num_elements);
            return a[offset];
        }
        
        /// Returns reference to value of element (non-const element access)
        template <typename...Args,typename X=T>
        typename std::enable_if<detail::IsIntegral<Args...>::value, X&>::type
        operator()(Args...args) {
            static_assert(num_dimensions ==  sizeof...(Args), "SimpleTensor number of indices must match dimension");
            size_t offset = sum_offset(std::index_sequence_for<Args...>{},args...);
            if (bounds_check) assert(offset>=0 && offset<num_elements);
            return a[offset];
        }

        /// Returns SliceTensor (const patch access)
        template <typename...Args, typename X=T>
        typename std::enable_if<detail::IsSlice<Args...>::value && !detail::IsIntegral<Args...>::value, SliceTensor<const SimpleTensor<X,Dims...>>>::type
        operator()(Args...args) const {
            static_assert(num_dimensions == sizeof...(Args), "SimpleTensor number of indices must match dimension");
            return SliceTensor<const SimpleTensor<T,Dims...>>(*this, std::array<Slice,num_dimensions>{args...});
        }

        /// Returns SliceTensor (non-const patch access)
        template <typename...Args, typename X=T>
        typename std::enable_if<detail::IsSlice<Args...>::value && !detail::IsIntegral<Args...>::value, SliceTensor<SimpleTensor<X,Dims...>>>::type
        operator()(Args...args) {
            static_assert(num_dimensions == sizeof...(Args), "SimpleTensor number of indices must match dimension");
            return SliceTensor<SimpleTensor<T,Dims...>>(*this, std::array<Slice,num_dimensions>{args...});
        }

        /// Returns SliceTensor (const patch access)
        SliceTensor<const SimpleTensor<T,Dims...>>  operator()(const std::array<Slice,num_dimensions>& slices) const {
            return SliceTensor<const SimpleTensor<T,Dims...>>(*this, slices);
        }

        /// Returns SliceTensor (non-const patch access)
        SliceTensor<SimpleTensor<T,Dims...>>  operator()(const std::array<Slice,num_dimensions>& slices) {
            return SliceTensor<SimpleTensor<T,Dims...>>(*this, slices);
        }

        /// Access data directly (const accessor)
        const std::array<T,num_elements>& data() const {return a;}

        /// Access data directly (non-const accessor)
        std::array<T,num_elements>& data() {return a;}

        /// Access data directly via pointer (const accessor)
        const T* ptr() const {return &a[0];}

        /// Access data directly via pointer (non-const accessor)
        T* ptr() {return &a[0];}

        /// Computes sum of square of absolute values ... still needs specializing for complex and should also implement pairwise summation for increased accuracy
        template <typename accumulatorT = T>
        T sumabssq() const {
            accumulatorT sum = 0;
            for (size_t i=0; i<num_elements; i++) sum += a[i]*a[i];
            return sum;
        }
        
        /// Compute Frobenius norm ... still needs specializing for complex
        template <typename accumulatorT = T>
        T normf() const {
            return std::sqrt(sumabssq<accumulatorT>());
        }

        /// Fill with value
        SimpleTensor<T,Dims...>& operator=(T value) {a.fill(value); return *this;} //for (size_t i=0; i<num_elements; ++i) a[i] = value; 

        /// Deep copy (with possible type conversion) from identically shaped SimpleTensor
        template <typename R>
        SimpleTensor<T,Dims...>& operator=(const SimpleTensor<R,Dims...>& other) {
            if (this != &other) for (size_t i=0; i<num_elements; ++i) a[i] = T(other.a[i]);
            return *this;
        }

        /// Inplace scaling by a constant
        SimpleTensor<T,Dims...>& operator*=(T value) {for (size_t i=0; i<num_elements; ++i) a[i] *= value; return *this;}

        /// Returns true if this and other conform (dimensions all same size)
        template <typename otherT> bool conforms(const otherT& other) const {return dims() == other.dims();}

        /// Deep copy (with possible type conversion) from identically shaped SliceTensor ... desperately needs optimized iterator
        template <typename otherT>
        SimpleTensor<T,Dims...>& operator=(const SliceTensor<otherT>& other) {
            // assert(this->conforms(other));
            // auto lit=this->begin();
            // auto rit=other.begin();
            // while (lit != this->end() || rit != other.end()) {
            //     *lit = *rit;
            //     ++lit;
            //     ++rit;
            // }
            // assert(lit==this->end() && rit==other.end());
            apply_binaryop(*this,other,[](data_type& a, const typename SliceTensor<otherT>::data_type& b) {a=b;});
            return *this;
        }

        /// Start for forward iteration through elements in row-major order --- this is convenient but not efficient
        iterator<num_dimensions> begin() {return iterator<num_dimensions>(0,this);}

        /// End for forward iteration through elements in row-major order --- this is convenient but not efficient
        const iterator<num_dimensions>& end() {return finish;}

        /// Start for forward iteration through elements in row-major order --- this is convenient but not efficient
        const_iterator<num_dimensions> begin() const {return const_iterator<num_dimensions>(0,this);}

        /// End for forward iteration through elements in row-major order --- this is convenient but not efficient
        const const_iterator<num_dimensions>& end() const {return cfinish;}

        /// Optimized iteration --- start for forward iteration through all but inner dimension
        iterator<num_dimensions-2> xbegin() {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            return iterator<num_dimensions-2>(0,this);
        }

        /// Optimized iteration --- end for forward iteration  through all but inner dimension
        const iterator<num_dimensions-2> xend() {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            return finish2;
        }

        /// Optimized iteration --- start for forward iteration through all but inner dimension
        const_iterator<num_dimensions-2> xbegin() const {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            return const_iterator<num_dimensions-2>(0,this);
        }

        /// Optimized iteration --- end for forward iteration  through all but inner dimension
        const const_iterator<num_dimensions-2>& xend() const {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            return cfinish2;
        }

        /// Optimized iteration ... given optimized (ndim-2) iterator returns address of first element in underlying data which will likely not be contiguous
        template <typename iteratorT>
        data_type* ptr(iteratorT& it) {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            auto p = ptr();
            auto& indx = it.index();
            for (auto d : range(num_dimensions-2)) p += indx[d] * stride(d);
            return p;
        }

        /// Optimized iteration ... given optimized (ndim-2) iterator returns address of first element in underlying data which will likely not be contiguous
        template <typename iteratorT>
        const data_type* ptr(iteratorT& it) const {
            static_assert(num_dimensions>=2, "trying to use optimized iteration with too few dimensions");
            auto p = ptr();
            auto& indx = it.index();
            for (auto d : range(num_dimensions-2)) p += indx[d] * stride(d);
            return p;
        }
        
    };

    namespace detail {
        template <typename T, size_t K, Dimension NDIM> struct FixedTensor{};
        template <typename T, size_t K> struct FixedTensor<T,K,Dimension(1)>{using type = SimpleTensor<T,K>;};
        template <typename T, size_t K> struct FixedTensor<T,K,Dimension(2)>{using type = SimpleTensor<T,K,K>;};
        template <typename T, size_t K> struct FixedTensor<T,K,Dimension(3)>{using type = SimpleTensor<T,K,K,K>;};
    }

    /// FixedTensor is a SimpleTensor that has the same fixed size for each dimension

    /// Cannot specialize type aliases so instead use one level of redirection
    template <typename T, size_t K, Dimension NDIM>
    using FixedTensor = typename detail::FixedTensor<T,K,NDIM>::type;

    /// Transform all dimensions of the tensor t by the matrix c

    /// \code
    /// result(i,j,k...) <-- sum(i',j', k',...) t(i',j',k',...) c(i',i) c(j',j) c(k',k) ...
    /// \endcode
    ///
    /// In this variant it is enforced that the input and output dimensions are all the same
    /// (i.e., that \c c is a square matrix).
    template <typename tT, typename cT, typename resultT, size_t K, Dimension NDIM>
    void transform(const FixedTensor<tT,K,NDIM>& t, const FixedTensor<cT,K,2>& c, FixedTensor<resultT,K,NDIM>& result) {
        FixedTensor<resultT,K,NDIM> workspace;
        const cT* pc = c.ptr();
        resultT *t0=workspace.ptr(), *t1=result.ptr();
        if (t.ndim() & 0x1) std::swap(t0,t1);
        const size_t dimj = c.dim(1);
        size_t dimi = 1;
        for (size_t n=1; n<t.ndim(); ++n) dimi *= dimj;
        mTxmq(dimi, dimj, dimj, t0, t.ptr(), pc);
        for (size_t n=1; n<t.ndim(); ++n) {
            mTxmq(dimi, dimj, dimj, t1, t0, pc);
            std::swap(t0,t1);
        }
    }

    template <typename tensorT>
    typename std::enable_if<tensorT::is_tensor,std::ostream>::type&
    operator<<(std::ostream& s, const tensorT& t) {
        if (t.size() == 0) {
            s << "[empty tensor]\n";
            return s;
        }
        
        const Dimension ndim = t.ndim();
        size_t maxdim = 0;
        for (auto d : range(ndim)) maxdim = std::max(maxdim,t.dim(d));
        size_t index_width;
        if (maxdim < 10)
            index_width = 1;
        else if (maxdim < 100)
            index_width = 2;
        else if (maxdim < 1000)
            index_width = 3;
        else if (maxdim < 10000)
            index_width = 4;
        else
            index_width = 6;
        
        std::ios::fmtflags oldflags = s.setf(std::ios::scientific);
        long oldprec = s.precision();
        long oldwidth = s.width();
        
        const Dimension lastdim = ndim-1;
        const size_t lastdimsize = t.dim(lastdim);

        for (auto it=t.begin(); it!=t.end(); ) {
            const auto& index = it.index();
            s.unsetf(std::ios::scientific);
            s << '[';
            for (Dimension d=0; d<(ndim-1); d++) {
                s.width(index_width);
                s << index[d];
                s << ",";
            }
            s << " *]";
            // s.setf(std::ios::scientific);
            s.setf(std::ios::fixed);
            for (size_t i=0; i<lastdimsize; ++i,++it) { //<<< it incremented here!
                // s.precision(4);
                s << " ";
                //s.precision(8);
                //s.width(12);
                s.precision(6);
                s.width(10);
                s << *it;
            }
            s.unsetf(std::ios::scientific);
            if (it != t.end()) s << std::endl;
        }
        
        s.setf(oldflags,std::ios::floatfield);
        s.precision(oldprec);
        s.width(oldwidth);
        
        return s;
    }
}
                         
#endif
