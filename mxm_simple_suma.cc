#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <tuple>
#include "flow.h"

class Matrix {
  std::size_t n, m;
  std::vector<double> t;

 public:
  Matrix() : n(0), m(0) {}

  Matrix(std::size_t n, std::size_t m) : n(n), m(m), t(n * m, 0.0) {}

  Matrix(const Matrix& a) : n(a.n), m(a.m), t(a.t) {}

  Matrix& operator=(const Matrix& a) {
    if (this != &a) {
      n = a.n;
      m = a.m;
      t = a.t;
    }
    return *this;
  }

  double& operator()(std::size_t i, std::size_t j) {
    std::size_t index = i * m + j;
    assert(index < t.size());
    return t[index];
  }

  double operator()(std::size_t i, std::size_t j) const {
    std::size_t index = i * m + j;
    assert(index < t.size());
    return t[index];
  }

  Matrix operator+(const Matrix& a) const {
    assert(a.rowdim() == rowdim() && a.coldim() == coldim());
    Matrix r(coldim(), rowdim());
    for (std::size_t i = 0; i < coldim(); i++)
      for (std::size_t j = 0; j < rowdim(); j++) r(i, j) = (*this)(i, j) + a(i, j);
    return r;
  }

  Matrix operator-(const Matrix& a) const {
    assert(a.rowdim() == rowdim() && a.coldim() == coldim());
    Matrix r(coldim(), rowdim());
    for (std::size_t i = 0; i < coldim(); i++)
      for (std::size_t j = 0; j < rowdim(); j++) r(i, j) = (*this)(i, j) - a(i, j);
    return r;
  }

  std::size_t coldim() const { return n; }

  std::size_t rowdim() const { return m; }

  // Return deep copy of patch with ilo <= i < ihi and jlo <= j < jhi (note upper bound is NOT inclusive)
  Matrix get_patch(std::size_t ilo, std::size_t ihi, std::size_t jlo, std::size_t jhi) const {
    Matrix p(ihi - ilo, jhi - jlo);
    for (std::size_t i = ilo; i < ihi; i++)
      for (std::size_t j = jlo; j < jhi; j++) p(i - ilo, j - jlo) = (*this)(i, j);
    return p;
  }

  void set_patch(std::size_t ilo, std::size_t ihi, std::size_t jlo, std::size_t jhi, const Matrix& p) {
    // std::cout << "set_patch: " << ilo << " " << ihi << " " << jlo << " " << jhi << " " << (ihi-ilo) << " " <<
    // (jhi-jlo) << " " << p.coldim() << " " << p.rowdim() << std::endl;
    assert((ihi - ilo) == p.coldim() && (jhi - jlo) == p.rowdim());
    for (std::size_t i = ilo; i < ihi; i++)
      for (std::size_t j = jlo; j < jhi; j++) (*this)(i, j) = p(i - ilo, j - jlo);
  }

  double normf() const {
    double sumsq = 0.0;
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < m; j++) {
        double aij = (*this)(i, j);
        sumsq += aij * aij;
      }
    }
    return std::sqrt(sumsq);
  }
};

std::ostream& operator<<(std::ostream& s, const Matrix& a) {
  s << "Matrix(" << a.coldim() << "," << a.rowdim() << ")" << std::endl;
  for (std::size_t i = 0; i < a.coldim(); i++) {
    for (std::size_t j = 0; j < a.rowdim(); j++) {
      s << a(i, j) << " ";
    }
    s << std::endl;
  }
  return s;
}

Matrix fill_matrix(std::size_t n, std::size_t m) {
  Matrix a(n, m);
  for (std::size_t i = 0; i < n; i++)
    for (std::size_t j = 0; j < m; j++) a(i, j) = 1.0 / (i + j + 1);
  return a;
}

Matrix mxm_ref(const Matrix& a, const Matrix& b) {
  assert(a.rowdim() == b.coldim());
  const std::size_t n = a.coldim(), k = a.rowdim(), m = b.rowdim();

  Matrix c(n, m);
  for (std::size_t p = 0; p < n; p++)
    for (std::size_t q = 0; q < k; q++)
      for (std::size_t r = 0; r < m; r++) c(p, r) += a(p, q) * b(q, r);
  return c;
}

using pairT = std::pair<std::size_t, std::size_t>;

struct tripleT {
  std::size_t first, second, third;
  tripleT() {}
  tripleT(size_t i, std::size_t j, std::size_t k) : first(i), second(j), third(k) {}

  bool operator<(const tripleT& a) const {
    if (first < a.first)
      return true;
    else if (first > a.first)
      return false;
    else if (second < a.second)
      return true;
    else if (second > a.second)
      return false;
    else if (third < a.third)
      return true;
    else
      return false;
  }
};

std::ostream& operator<<(std::ostream& s, const pairT& a) { return s << "(" << a.first << "," << a.second << ")"; }

std::ostream& operator<<(std::ostream& s, const tripleT& a) {
  return s << "(" << a.first << "," << a.second << "," << a.third << ")";
}

class MxM {
  const std::size_t N;  // Matrix dimensions in DGEMM sense
  const std::size_t M;
  const std::size_t K;
  const std::size_t tilesize;

  // My very simple understanding of SUMA is
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

  Flow<tripleT, Matrix> Aik_bcast, Bkj_bcast, Cij_reduce;

  class BroadcastA : public Op<InFlows<pairT, Matrix>, Flows<Flow<tripleT, Matrix>>, BroadcastA> {
    using baseT = Op<InFlows<pairT, Matrix>, Flows<Flow<tripleT, Matrix>>, BroadcastA>;
    const std::size_t N;  // Matrix dimensions in DGEMM sense
    const std::size_t M;
    const std::size_t K;
    const std::size_t tilesize;

   public:
    BroadcastA(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize, Flow<pairT, Matrix> in,
               Flow<tripleT, Matrix> out)
        : baseT(make_flows(in), make_flows(out), "broadcastA"), N(N), M(M), K(K), tilesize(tilesize) {}

    void op(const pairT& ik, const Matrix& Aik, Flows<Flow<tripleT, Matrix>>& out) {
      std::size_t i = ik.first, k = ik.second;
      std::vector<tripleT> ijk;
      for (std::size_t j = 0; j < M; j += tilesize) ijk.push_back({i, j, k});
      out.broadcast<0>(ijk, Aik);
    }
  } broadcastA;

  class BroadcastB : public Op<InFlows<pairT, Matrix>, Flows<Flow<tripleT, Matrix>>, BroadcastB> {
    using baseT = Op<InFlows<pairT, Matrix>, Flows<Flow<tripleT, Matrix>>, BroadcastB>;
    const std::size_t N;  // Matrix dimensions in DGEMM sense
    const std::size_t M;
    const std::size_t K;
    const std::size_t tilesize;

   public:
    BroadcastB(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize, Flow<pairT, Matrix> in,
               Flow<tripleT, Matrix> out)
        : baseT(make_flows(in), make_flows(out), "broadcastB"), N(N), M(M), K(K), tilesize(tilesize) {}

    void op(const pairT& kj, const Matrix& Bkj, Flows<Flow<tripleT, Matrix>>& out) {
      std::size_t k = kj.first, j = kj.second;
      std::vector<tripleT> ijk;
      for (std::size_t i = 0; i < N; i += tilesize) ijk.push_back({i, j, k});
      out.broadcast<0>(ijk, Bkj);
    }
  } broadcastB;

  class MxMTask : public Op<InFlows<tripleT, Matrix, Matrix>, Flows<Flow<tripleT, Matrix>>, MxMTask> {
    using baseT = Op<InFlows<tripleT, Matrix, Matrix>, Flows<Flow<tripleT, Matrix>>, MxMTask>;
    const std::size_t N;  // Matrix dimensions in DGEMM sense
    const std::size_t M;
    const std::size_t K;
    const std::size_t tilesize;

   public:
    MxMTask(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize, Flow<tripleT, Matrix> Aik,
            Flow<tripleT, Matrix> Bkj, Flow<tripleT, Matrix> Cijreduce)
        : baseT(make_flows(Aik, Bkj), make_flows(Cijreduce), "mxmtask"), N(N), M(M), K(K), tilesize(tilesize) {}

    void op(const tripleT& ijk, const Matrix& Aik, const Matrix& Bkj, Flows<Flow<tripleT, Matrix>>& Cijreduce) {
      Cijreduce.send<0>(ijk, mxm_ref(Aik, Bkj));
    }
  } mxmtask;

  // Simple reduction for testing
  class Reduce
      : public Op<InFlows<tripleT, Matrix, Matrix>, Flows<Flow<tripleT, Matrix>, Flow<pairT, Matrix>>, Reduce> {
    using baseT = Op<InFlows<tripleT, Matrix, Matrix>, Flows<Flow<tripleT, Matrix>, Flow<pairT, Matrix>>, Reduce>;
    const std::size_t N;  // Matrix dimensions in DGEMM sense
    const std::size_t M;
    const std::size_t K;
    const std::size_t tilesize;

   public:
    Reduce(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize, Flow<tripleT, Matrix> Cijreduce,
           Flow<pairT, Matrix> result)
        : baseT("reduce"), N(N), M(M), K(K), tilesize(tilesize) {
      Flow<tripleT, Matrix> sumflow;
      this->connect({Cijreduce, sumflow}, {sumflow, result});

      // prime the accumulator
      std::vector<tripleT> ijk;
      for (std::size_t i = 0; i < N; i += tilesize) {
        for (std::size_t j = 0; j < M; j += tilesize) {
          ijk.push_back({i, j, 0});
        }
      }
      Matrix zero(tilesize, tilesize);
      sumflow.broadcast(ijk, zero);
    }

    void op(tripleT ijk, const Matrix& Cij, const Matrix& sum, Flows<Flow<tripleT, Matrix>, Flow<pairT, Matrix>>& out) {
      std::size_t i = ijk.first, j = ijk.second, k = ijk.third;
      Matrix total = sum + Cij;
      std::size_t nextk = k + tilesize;
      if (nextk < K) {
        out.send<0>(tripleT{i, j, nextk}, total);
      } else {
        out.send<1>(pairT{i, j}, total);
      }
    }
  } reduce;

 public:
  MxM(std::size_t N, std::size_t M, std::size_t K, std::size_t tilesize, Flow<pairT, Matrix> Aik,
      Flow<pairT, Matrix> Bkj, Flow<pairT, Matrix> Cij)
      : N(N)
      , M(M)
      , K(K)
      , tilesize(tilesize)
      , broadcastA(N, M, K, tilesize, Aik, Aik_bcast)
      , broadcastB(N, M, K, tilesize, Bkj, Bkj_bcast)
      , mxmtask(N, M, K, tilesize, Aik_bcast, Bkj_bcast, Cij_reduce)
      , reduce(N, M, K, tilesize, Cij_reduce, Cij) {
    assert((N % tilesize) == 0 && (K % tilesize) == 0 && (M % tilesize) == 0);  // for simplicity
  }
};

// Incoming tiles are written to memory
class Writer : public Op<InFlows<pairT, Matrix>, Flows<>, Writer> {
  using baseT = Op<InFlows<pairT, Matrix>, Flows<>, Writer>;
  Matrix& A;  // Ugh, but will do for testing
  std::size_t tilesize;

 public:
  Writer(Flow<pairT, Matrix> in, Matrix& A, std::size_t tilesize)
      : baseT(make_flows(in), Flows<>(), "writer"), A(A), tilesize(tilesize) {}

  void op(const pairT ij, const Matrix& patch, Flows<> junk) {
    std::size_t ilo = ij.first, ihi = ilo + tilesize, jlo = ij.second, jhi = jlo + tilesize;
    A.set_patch(ilo, ihi, jlo, jhi, patch);
  }
};

// Read Aik sending tiles to output with k in outer loop
void readA(const Matrix& A, Flow<pairT, Matrix>& out, std::size_t tilesize) {
  for (std::size_t klo = 0; klo < A.rowdim(); klo += tilesize) {
    std::size_t khi = klo + tilesize;
    for (std::size_t ilo = 0; ilo < A.coldim(); ilo += tilesize) {
      std::size_t ihi = ilo + tilesize;
      out.send({ilo, klo}, A.get_patch(ilo, ihi, klo, khi));
    }
  }
}

// Read Bkj sending tiles to output with k in outer loop
void readB(const Matrix& B, Flow<pairT, Matrix>& out, std::size_t tilesize) {
  for (std::size_t klo = 0; klo < B.coldim(); klo += tilesize) {
    std::size_t khi = klo + tilesize;
    for (std::size_t jlo = 0; jlo < B.rowdim(); jlo += tilesize) {
      std::size_t jhi = jlo + tilesize;
      out.send({klo, jlo}, B.get_patch(klo, khi, jlo, jhi));
    }
  }
}

void pretty_print(const Matrix& a) {
  printf("Matrix(%lu,%lu)\n", a.coldim(), a.rowdim());
  for (std::size_t i = 0; i < a.coldim(); i++) {
    for (std::size_t j = 0; j < a.rowdim(); j++) {
      printf("%10.6f ", a(i, j));
    }
    printf("\n");
  }
}

int main() {
  // BaseOp::set_trace(true);

  const std::size_t tilesize = 2;
  const std::size_t N = 3 * tilesize, M = 5 * tilesize, K = 7 * tilesize;

  Matrix a = fill_matrix(N, K);
  Matrix b = fill_matrix(K, M);
  Matrix c = mxm_ref(a, b);
  std::cout << " reference result " << std::endl;
  pretty_print(c);

  Matrix csuma(N, M);  // suma result will go here

  Flow<pairT, Matrix> A, B, C;  // input (A,B) and output (C) flows

  MxM mxmop(N, M, K, tilesize, A, B, C);

  Writer writerop(C, csuma, tilesize);  // write result to memory

  // Push a and b into the input flows
  readA(a, A, tilesize);
  readB(b, B, tilesize);

  // Cross fingers
  std::cout << " suma result " << std::endl;
  pretty_print(csuma);
  std::cout << "norm of difference is " << (c - csuma).normf() << std::endl;

  return 0;
}
