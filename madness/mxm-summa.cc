
#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <tuple>
#include <cmath>
#include <cassert>
#include <iostream>
#include <random>
#include <cstdio>

#include "madness/ttg.h"

#include <madness/world/worldmutex.h>
#include <madness/world/worldhash.h>
#include <madness/world/archive.h>

using pairT = std::pair<std::size_t, std::size_t>;

using namespace madness;
using namespace madness::ttg;
using namespace ::ttg;

namespace madness {
    template <>
    madness::hashT hash_value(const pairT& t) {
        return (t.first<<32) + t.second;
    }
}

struct tripleT {
    std::size_t first, second, third;
    tripleT() {}
    tripleT(size_t i,std::size_t j,std::size_t k) : first(i), second(j), third(k) {}

    bool operator<(const tripleT& a) const { // for storing in map
        if (first < a.first) return true;
        else if (first > a.first) return false;
        else if (second < a.second) return true;
        else if (second > a.second) return false;
        else if (third < a.third) return true;
        else return false;
    }

    bool operator==(const tripleT& a) const { // for storing in hash
        return (first==a.first && second==a.second && third==a.third);
    }

    madness::hashT hash() const {return (first<<40) + (second<<20) + first;}

    template <typename Archive> void serialize(Archive& ar) {madness::archive::wrap((unsigned char*) this, sizeof(*this));}
};

std::ostream& operator<<(std::ostream&s, const pairT& a) {
    return s << "(" << a.first << "," << a.second << ")";
}

std::ostream& operator<<(std::ostream&s, const tripleT& a) {
    return s << "(" << a.first << "," << a.second << "," << a.third << ")";
}

class Matrix {
    std::size_t n, m;
    std::vector<double> t;

public:

    Matrix() : n(0), m(0) {}

    Matrix(std::size_t n, std::size_t m) : n(n), m(m), t(n*m,0.0) {}

    Matrix(const Matrix& a) : n(a.n), m(a.m), t(a.t) {}

    Matrix& operator=(const Matrix& a) {
        if (this != & a) {
            n = a.n;
            m = a.m;
            t = a.t;
        }
        return *this;
    }

    double& operator()(std::size_t i, std::size_t j) {
        std::size_t index = i*m+j;
        assert(index < t.size());
        return t[index];
    }

    double operator()(std::size_t i, std::size_t j) const {
        std::size_t index = i*m+j;
        assert(index < t.size());
        return t[index];
    }

    Matrix operator+(const Matrix& a) const {
        assert(a.rowdim()==rowdim() && a.coldim()==coldim());
        Matrix r(coldim(),rowdim());
        for (std::size_t i=0; i<coldim(); i++)
            for (std::size_t j=0; j<rowdim(); j++)
                r(i,j) = (*this)(i,j) + a(i,j);
        return r;
    }

    Matrix operator-(const Matrix& a) const {
        assert(a.rowdim()==rowdim() && a.coldim()==coldim());
        Matrix r(coldim(),rowdim());
        for (std::size_t i=0; i<coldim(); i++)
            for (std::size_t j=0; j<rowdim(); j++)
                r(i,j) = (*this)(i,j) - a(i,j);
        return r;
    }

    std::size_t coldim() const {return n;}

    std::size_t rowdim() const {return m;}

    // Return deep copy of patch with ilo <= i < ihi and jlo <= j < jhi (note upper bound is NOT inclusive)
    Matrix get_patch(std::size_t ilo, std::size_t ihi, std::size_t jlo, std::size_t jhi) const {
        Matrix p(ihi-ilo,jhi-jlo);
        for (std::size_t i=ilo; i<ihi; i++)
            for (std::size_t j=jlo; j<jhi; j++)
                p(i-ilo,j-jlo) = (*this)(i,j);
        return p;
    }

    void set_patch(std::size_t ilo, std::size_t ihi, std::size_t jlo, std::size_t jhi, const Matrix& p) {
        //std::cout << "set_patch: " << ilo << " " << ihi << " " << jlo << " " << jhi << " " << (ihi-ilo) << " " << (jhi-jlo) << " " << p.coldim() << " " << p.rowdim() << std::endl;
        assert((ihi-ilo) == p.coldim() && (jhi-jlo) == p.rowdim());
        for (std::size_t i=ilo; i<ihi; i++)
            for (std::size_t j=jlo; j<jhi; j++)
                (*this)(i,j) = p(i-ilo,j-jlo);
    }

    double normf() const {
        double sumsq = 0.0;
        for (std::size_t i=0; i<n; i++) {
            for (std::size_t j=0; j<m; j++) {
                double aij = (*this)(i,j);
                sumsq += aij*aij;
            }
        }
        return std::sqrt(sumsq);
    }

    template <typename Archive>
    void serialize(Archive& ar) {
        ar & n & m & t;
    }
    
};

std::ostream& operator<<(std::ostream&s, const Matrix& a) {
    s << "Matrix(" << a.coldim() << "," << a.rowdim() << ")" << std::endl;
    for (std::size_t i=0; i<a.coldim(); i++) {
        for (std::size_t j=0; j<a.rowdim(); j++) {
            s << a(i,j) << " ";
        }
        s << std::endl;
    }
    return s;
}


Matrix fill_matrix(std::size_t n, std::size_t m) {
    Matrix a(n,m);
    for (std::size_t i=0; i<n; i++)
        for (std::size_t j=0; j<m; j++)
            a(i,j) = 1.0/(i+j+1);
    return a;
}

Matrix mxm_ref(const Matrix& a, const Matrix& b) {
    assert(a.rowdim() == b.coldim());
    const std::size_t n = a.coldim(), k=a.rowdim(), m=b.rowdim();

    Matrix c(n,m);
    for (std::size_t p=0; p<n; p++)
        for (std::size_t q=0; q<k; q++)
            for (std::size_t r=0; r<m; r++)
                c(p,r) += a(p,q)*b(q,r);
    return c;
}

class MxM {
    const std::size_t N; // Matrix dimensions in DGEMM sense
    const std::size_t M;
    const std::size_t K;
    const std::size_t tilesize;

    // My very simple understanding of SUMMA is
    // for tiles of k
    //    owner of tile aik broadcasts it to tasks (i,all j,k)
    //    owner of tile bkj broadcasts it to tasks (all i,j,k)
    //    task(i,j,k) computes and adds into result cij
    // where tasks (triplets ijk) and data (pairs ij, ik, kj) are mapped across processors

    // Decompose this into these operations
    // input flow produces tiles of A on processor (i,k) -> broadcast to mxm(i,*,k) as Aik
    // input flow produces tiles of B on processor (k,j) -> broadcast to mxm(*,j,k) as Bkj
    // Aik, Bkj -> mxm(i,j,k) -> first argument of reduce operation for Cij 
    // second argument of reduce operation carries the sum
    // when the sum is done it sends to the overall output

    // The following members must be declared in the correct order so
    // initialized correctly
    
    Edge<tripleT,Matrix> Aik_bcast, Bkj_bcast, Cij_reduce;

    class BroadcastA : public Op<pairT, std::tuple<Out<tripleT,Matrix>>, BroadcastA, Matrix> {
        using baseT =         Op<pairT, std::tuple<Out<tripleT,Matrix>>, BroadcastA, Matrix>;
        const std::size_t N; // Matrix dimensions in DGEMM sense
        const std::size_t M;
        const std::size_t K;
        const std::size_t tilesize;
    public:
        BroadcastA(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize,
                   Edge<pairT,Matrix>& in, Edge<tripleT,Matrix>& out)
            : baseT(edges(in), edges(out), "broadcastA", {"Aik"}, {"broadcast Aik"})
            , N(N)
            , M(M)
            , K(K)
            , tilesize(tilesize)
        {}

        void op(const pairT& ik, const std::tuple<Matrix>& t, std::tuple<Out<tripleT,Matrix>>& out) {
            const Matrix& Aik = std::get<0>(t);
            std::size_t i=ik.first, k=ik.second;
            std::vector<tripleT> ijk;
            for (std::size_t j=0; j<M; j+=tilesize) ijk.push_back({i,j,k});
            ::broadcast<0>(ijk, Aik,out);
        }
    } broadcastA;
        
    class BroadcastB : public Op<pairT, std::tuple<Out<tripleT,Matrix>>, BroadcastB, Matrix> {
        using baseT =         Op<pairT, std::tuple<Out<tripleT,Matrix>>, BroadcastB, Matrix>;
        const std::size_t N; // Matrix dimensions in DGEMM sense
        const std::size_t M;
        const std::size_t K;
        const std::size_t tilesize;
    public:
        BroadcastB(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize,
                   Edge<pairT,Matrix>& in, Edge<tripleT,Matrix>& out)
            : baseT(edges(in), edges(out), "broadcastB", {"Bkj"}, {"broadcast Bkj"})
            , N(N)
            , M(M)
            , K(K)
            , tilesize(tilesize)
        {}

        void op(const pairT& kj, const std::tuple<Matrix>& t, std::tuple<Out<tripleT,Matrix>>& out) {
            const Matrix& Bkj = std::get<0>(t);
            std::size_t k=kj.first, j=kj.second;
            std::vector<tripleT> ijk;
            for (std::size_t i=0; i<N; i+=tilesize) ijk.push_back({i,j,k});
            ::broadcast<0>(ijk, Bkj, out);
        }
    } broadcastB;

    class MxMTask : public Op<tripleT, std::tuple<Out<tripleT,Matrix>>, MxMTask, Matrix, Matrix> {
        using baseT =      Op<tripleT, std::tuple<Out<tripleT,Matrix>>, MxMTask, Matrix, Matrix>;
        const std::size_t N; // Matrix dimensions in DGEMM sense
        const std::size_t M;
        const std::size_t K;
        const std::size_t tilesize;
    public:
        MxMTask(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize,
                Edge<tripleT,Matrix>& Aik, Edge<tripleT,Matrix>& Bkj, Edge<tripleT,Matrix>& Cijreduce)
            : baseT(edges(Aik,Bkj), edges(Cijreduce), "mxmtask", {"Aik","Bkj"}, {"Cij"})
            , N(N)
            , M(M)
            , K(K)
            , tilesize(tilesize)
        {}
            
        void op(const tripleT& ijk, const std::tuple<Matrix, Matrix>& t, std::tuple<Out<tripleT,Matrix>>& out) {
            const Matrix &Aik=std::get<0>(t), &Bkj=std::get<1>(t);
            ::send<0>(ijk, mxm_ref(Aik, Bkj), out);
        }
    } mxmtask;

    // Simple reduction for testing
    class Reduce : public Op<tripleT, std::tuple<Out<pairT,Matrix>,Out<tripleT,Matrix>>, Reduce, Matrix, Matrix> {
        using baseT =     Op<tripleT, std::tuple<Out<pairT,Matrix>,Out<tripleT,Matrix>>, Reduce, Matrix, Matrix>;
        const std::size_t N; // Matrix dimensions in DGEMM sense
        const std::size_t M;
        const std::size_t K;
        const std::size_t tilesize;
    public:
        Reduce(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize,
               Edge<tripleT,Matrix>& Cijpartial, Edge<pairT,Matrix>& result)
            : baseT("reduce",{"Cij partial","sum"},{"Cij total","sum"})
            , N(N)
            , M(M)
            , K(K)
            , tilesize(tilesize)
        {
            // Must connect manually since any locally defined edge
            // won't be constructed before base class
            Cijpartial.set_out(&(this->in<0>()));
            result.set_in(&(this->out<0>()));
            this->out<1>().connect(this->in<1>());
            
            // prime the accumulator
            std::vector<tripleT> ijk;
            for (std::size_t i=0; i<N; i+=tilesize) {
                for (std::size_t j=0; j<M; j+=tilesize) {
                    ijk.push_back({i,j,0});
                }
            }
            Matrix zero(tilesize,tilesize);
            this->out<1>().broadcast(ijk,zero);
        }

        void op(tripleT ijk, const std::tuple<Matrix,Matrix>& t, std::tuple<Out<pairT,Matrix>,Out<tripleT,Matrix>>& out) {
            const Matrix &Cij=std::get<0>(t), &sum=std::get<1>(t);
            std::size_t i=ijk.first, j=ijk.second, k=ijk.third;
            Matrix total = sum + Cij;
            std::size_t nextk = k+tilesize;
            if (nextk < K) {
                ::send<1>(tripleT{i,j,nextk}, total, out);
            }
            else {
                ::send<0>(pairT{i,j}, total, out);
            }
        }
    } reduce;

public:

    MxM(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize,
        Edge<pairT,Matrix>& Aik, Edge<pairT,Matrix>& Bkj, Edge<pairT,Matrix>& Cij)
        : N(N)
        , M(M)
        , K(K)
        , tilesize(tilesize)
        , broadcastA(N, M, K, tilesize, Aik, Aik_bcast)
        , broadcastB(N, M, K, tilesize, Bkj, Bkj_bcast)
        , mxmtask(N, M, K, tilesize, Aik_bcast, Bkj_bcast, Cij_reduce)
        , reduce(N, M, K, tilesize, Cij_reduce, Cij)
    {
        assert((N%tilesize)==0 && (K%tilesize)==0 && (M%tilesize)==0); // for simplicity
    }
};

// Incoming tiles are written to memory
class Writer : public Op<pairT, std::tuple<>, Writer, Matrix> {
    using baseT =     Op<pairT, std::tuple<>, Writer, Matrix>;
    Matrix& A; // Ugh, but will do for testing
    std::size_t tilesize;
public:
    Writer(Edge<pairT,Matrix>& in, Matrix& A, std::size_t tilesize)
        : baseT(edges(in), edges(), "writer", {"Matrix tile"}, {})
        , A(A)
        , tilesize(tilesize)
    {}

    void op(const pairT ij, const std::tuple<Matrix>& t, std::tuple<>& out) {
        std::size_t ilo=ij.first, ihi=ilo+tilesize, jlo=ij.second, jhi=jlo+tilesize;
        A.set_patch(ilo,ihi,jlo,jhi,std::get<0>(t));
    }
};


// Read Aik sending tiles to output with k in outer loop ... eventually can use control to throttle
class ReadA : public Op<int, std::tuple<Out<pairT,Matrix>>, ReadA, int> {
    using baseT =    Op<int, std::tuple<Out<pairT,Matrix>>, ReadA, int>;

    const Matrix& A;
    const size_t tilesize;
 public:
    ReadA(const Matrix& A, std::size_t tilesize, Edge<int,int>& ctl, Edge<pairT,Matrix>& out) :
        baseT(edges(ctl),edges(out),"read A", {"ctl"}, {"Aik"})
        , A(A)
        , tilesize(tilesize)
    {}
    
    void op(const int& key, const std::tuple<int>& junk, std::tuple<Out<pairT,Matrix>>& out) {
        for (std::size_t klo=0; klo<A.rowdim(); klo+=tilesize) {
            std::size_t khi = klo+tilesize;
            for (std::size_t ilo=0; ilo<A.coldim(); ilo+=tilesize) {
                std::size_t ihi = ilo+tilesize;
                ::send<0>(pairT(ilo,klo), A.get_patch(ilo,ihi,klo,khi), out);
            }
        }
    }
};

// Read Bkj sending tiles to output with k in outer loop ... eventually can use control to throttle
class ReadB : public Op<int, std::tuple<Out<pairT,Matrix>>, ReadB, int> {
    using baseT =    Op<int, std::tuple<Out<pairT,Matrix>>, ReadB, int>;

    const Matrix& B;
    const size_t tilesize;
 public:
    ReadB(const Matrix& B, std::size_t tilesize, Edge<int,int>& ctl, Edge<pairT,Matrix>& out) 
        : baseT(edges(ctl),edges(out),"read B", {"ctl"}, {"Bkj"})
        , B(B)
        , tilesize(tilesize)
    {}
    
    void op(const int& key, const std::tuple<int>& junk, std::tuple<Out<pairT,Matrix>>& out) {
        for (std::size_t klo=0; klo<B.coldim(); klo+=tilesize) {
            std::size_t khi = klo+tilesize;
            for (std::size_t jlo=0; jlo<B.rowdim(); jlo+=tilesize) {
                std::size_t jhi = jlo+tilesize;
                ::send<0>(pairT(klo,jlo), B.get_patch(klo,khi,jlo,jhi), out);
            }
        }
    }
};

class Control : public Op<int, std::tuple<Out<int,int>>, Control> {
    using baseT =      Op<int, std::tuple<Out<int,int>>, Control>;

 public:
    Control(Edge<int,int>& ctl)
        : baseT(edges(),edges(ctl),"Control",{},{"ctl"})
        {}

    void op(const int& key, const std::tuple<>& junk, std::tuple<Out<int,int>>& out) {
        ::send<0>(0,0,out);
    }

    void start() {
        invoke(0);
    }
};
    
    
void pretty_print(const Matrix& a) {
    printf("Matrix(%lu,%lu)\n",a.coldim(),a.rowdim());
    for (std::size_t i=0; i<a.coldim(); i++) {
        for (std::size_t j=0; j<a.rowdim(); j++) {
            printf("%10.6f ",a(i,j));
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    madness::initialize(argc, argv);
    madness::World world(SafeMPI::COMM_WORLD);
    set_default_world(world);

    if (world.size() > 1) {
        madness::error("Currently only works with one MPI process (multiple threads OK)");
    }

    //OpBase::set_trace_all(true);
    
    const std::size_t tilesize = 7;
    const std::size_t N=13*tilesize, M=17*tilesize, K=19*tilesize;    
    
    Matrix a = fill_matrix(N,K);
    Matrix b = fill_matrix(K,M);
    Matrix c = mxm_ref(a,b);
    std::cout << " reference result " << std::endl;
    //pretty_print(c);    

    Matrix c_summa(N,M); // SUMMA result will go here

    Edge<pairT,Matrix> A, B, C; // input (A,B) and output (C) flows
    Edge<int,int> ctl("control");

    Control control(ctl);
    ReadA readA(a, tilesize, ctl, A);
    ReadB readB(b, tilesize, ctl, B);
    MxM mxmop(N, M, K, tilesize, A, B, C);
    Writer writerop(C, c_summa, tilesize); // write result to memory

    std::cout << "==== begin dot ====\n";
    std::cout << Dot()(&control) << std::endl;
    std::cout << "====  end dot  ====\n";

    control.start();
    
    world.gop.fence();

    // Cross fingers
    //std::cout << " SUMMA result " << std::endl;
    //pretty_print(c_summa);

    std::cout << "\nnorm of difference is " << (c - c_summa).normf() << std::endl;

    madness::finalize();
    
    return 0;
}
