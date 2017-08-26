#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/MADworld.h>
#include <madness/world/worlddc.h>
#include <madness/world/worldhash.h>
#include <madness/world/worldtypes.h>
#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace madness;

using std::size_t;

// Index for matrix block
struct ijT {
  size_t i, j;
  ijT() {}
  ijT(size_t i, size_t j) : i(i), j(j) {}
  bool operator==(const ijT& a) const { return i == a.i && j == a.j; }
  template <typename Archive>
  void serialize(Archive& ar) {
    ar& i& j;
  }
  hashT hash() const { return i * j; }
};

std::ostream& operator<<(std::ostream& s, const ijT& ij) {
  s << "(" << ij.i << "," << ij.j << ")";
  return s;
}

std::shared_ptr<WorldDCPmapInterface<ijT>> matrix_default_pmap;  // set at top of main

/// A process map for block cycle distributed matrix
class BlockCyclicPmap : public WorldDCPmapInterface<ijT> {
 private:
  const size_t iprocdim, jprocdim, nproc;

 public:
  BlockCyclicPmap(World& world, size_t iprocdim, size_t jprocdim)
      : iprocdim(iprocdim)  // i dimension of processor grid
      , jprocdim(jprocdim)  // j dimension of processor grid
      , nproc(world.size()) {}

  ProcessID owner(const ijT& ij) const {
    if (nproc == 1) return 0;
    size_t i = ij.i % iprocdim;
    size_t j = ij.j % jprocdim;
    ProcessID p = (i * jprocdim + j) % nproc;
    return p;
  }
};

/// More for curiosity to experiment with range based for loop
class rangeT {
  size_t lo, hi;

 public:
  rangeT() {}
  rangeT(size_t start, size_t stop) : lo(start), hi(stop) {}  // lo <= i < hi;
  const rangeT& begin() const { return *this; }
  rangeT end() const { return {hi, hi}; }
  rangeT& operator++() {
    lo += 1;
    return *this;
  }
  size_t operator*() const { return lo; }
  bool operator!=(const rangeT& a) { return lo != a.lo; }
  size_t size() const { return hi - lo; }
  size_t start() const { return lo; }
  size_t stop() const { return hi; }
};

std::ostream& operator<<(std::ostream& s, const rangeT& r) {
  s << "(" << r.start() << "," << r.stop() << ")";
  return s;
}

struct MatrixBlockMetaData {
  ijT ijblock;  // The block indices for i and j and the key of this block in the container
  rangeT irange, jrange;
  size_t iblockdim, jblockdim;  // Number of elements in each dimension of this block
  size_t nelem;

  MatrixBlockMetaData() {}

  MatrixBlockMetaData(ijT ijblock, const rangeT& irange, const rangeT& jrange)
      : ijblock(ijblock)
      , irange(irange)
      , jrange(jrange)
      , iblockdim(irange.size())
      , jblockdim(jrange.size())
      , nelem(iblockdim * jblockdim) {}

  template <typename Archive>
  void serialize(Archive& ar) {
    ar& archive::wrap((unsigned char*)this, sizeof(*this));
  }
};

std::ostream& operator<<(std::ostream& s, const MatrixBlockMetaData& m) {
  s << "(" << m.ijblock << "," << m.irange << "," << m.jrange << ")";
  return s;
}

class MatrixBlock {
  MatrixBlockMetaData meta;
  std::vector<double> d;

 public:
  MatrixBlock() {}

  // Constructor initializes data to zero
  MatrixBlock(ijT ijblock, const rangeT& irange, const rangeT& jrange)
      : meta(ijblock, irange, jrange), d(meta.nelem, 0.0) {}

  // Constructor initializes data to zero
  MatrixBlock(const MatrixBlockMetaData& meta) : meta(meta), d(meta.nelem, 0.0) {}

  double& operator()(size_t i, size_t j) {
    return d[(i - meta.irange.start()) * meta.jblockdim + (j - meta.jrange.start())];
  }

  double operator()(size_t i, size_t j) const {
    return d[(i - meta.irange.start()) * meta.jblockdim + (j - meta.jrange.start())];
  }

  const MatrixBlockMetaData& get_meta() const { return meta; }

  const ijT ijblock() const { return meta.ijblock; }

  const rangeT& irange() const { return meta.irange; }

  const rangeT& jrange() const { return meta.jrange; }

  void scale(double s) {
    for (auto& x : d) x *= s;
  }

  void add(const MatrixBlock& b) {
    for (auto i : rangeT(0, meta.nelem)) d[i] += b.d[i];
  }

  void add_transpose(const MatrixBlock& b) {
    if (&b == this) throw "think!";
    for (auto i : meta.irange) {
      for (auto j : meta.jrange) {
        (*this)(i, j) += b(j, i);
      }
    }
  }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar& meta& d;
  }
};

std::ostream& operator<<(std::ostream& s, const MatrixBlock& d) {
  s << "MatrixBlock(" << d.get_meta() << ")\n";
  s << "   ";
  for (auto j : d.jrange()) s << std::setw(10) << j;
  s << "\n";

  for (auto i : d.irange()) {
    s << std::setw(4) << i << ": ";
    for (auto j : d.jrange()) {
      s << std::fixed << std::setw(10) << std::setprecision(4) << d(i, j);
    }
    s << "\n";
  }
  return s;
}

struct MatrixMetaData {
  size_t idim, jdim;            // Matrix dimensions
  size_t iblockdim, jblockdim;  // Blocked-matrix dimensions
  std::vector<size_t> ilo;      // Start index of i blocks with value idim appended
  std::vector<size_t> jlo;      // Start index of j blocks with value jdim appended
  std::shared_ptr<WorldDCPmapInterface<ijT>> pmap;

  MatrixMetaData() {}

  MatrixMetaData(size_t idim, size_t jdim, const std::vector<size_t>& ilo, const std::vector<size_t>& jlo,
                 const std::shared_ptr<WorldDCPmapInterface<ijT>>& pmap = matrix_default_pmap)
      : idim(idim), jdim(jdim), iblockdim(ilo.size() - 1), jblockdim(jlo.size() - 1), ilo(ilo), jlo(jlo), pmap(pmap) {}
};

/// A block distributed matrix addressable by blocks - only barely functional!
class Matrix {
  MatrixMetaData meta;
  using dcT = WorldContainer<ijT, MatrixBlock>;
  using accessorT = typename dcT::accessor;
  dcT a;

  Matrix(Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

 public:
  Matrix(World& world, const MatrixMetaData& meta) : meta(meta), a(world, meta.pmap) { zero(); }

  Matrix(Matrix&& x) : meta(x.meta), a(x.a) {
    x.meta = MatrixMetaData();
    x.a = dcT();
  }

  Matrix& operator=(Matrix&& x) {
    if (this != &x) {
      meta = MatrixMetaData();
      a = dcT();
      std::swap(a, x.a);
      std::swap(meta, x.meta);
    }
    return *this;
  }

  void zero() {
    a.clear();
    for (auto iblock : rangeT(0, meta.iblockdim)) {
      for (auto jblock : rangeT(0, meta.jblockdim)) {
        ijT ijblock(iblock, jblock);
        if (a.is_local(ijblock)) {
          accessorT acc;
          a.insert(acc, ijblock);
          acc->second = MatrixBlock(ijblock, rangeT(meta.ilo[iblock], meta.ilo[iblock + 1]),
                                    rangeT(meta.jlo[jblock], meta.jlo[jblock + 1]));
        }
      }
    }
  }

  World& get_world() const { return a.get_world(); }

  const MatrixMetaData& get_meta() const { return meta; }

  MatrixBlock get(size_t iblock, size_t jblock) const {  // for easy debug/print access
    return a.find(ijT(iblock, jblock)).get()->second;
  }

  void add(const MatrixBlock& b) { a.send(b.ijblock(), &MatrixBlock::add, b); }

  void add_transpose(const MatrixBlock& b) {
    a.send(ijT(b.ijblock().j, b.ijblock().i), &MatrixBlock::add_transpose, b);
  }

  void fence() const { a.get_world().gop.fence(); }

  void scale(double s) {
    for (auto& p : *this) {
      p.second.scale(s);
    }
    fence();
  }

  using iterator = dcT::iterator;
  using const_iterator = dcT::const_iterator;
  iterator begin() { return a.begin(); }
  iterator end() { return a.end(); }
  const_iterator begin() const { return a.begin(); }
  const_iterator end() const { return a.end(); }
};

void make_matrix_default_pmap(World& world) {
  size_t n = std::sqrt(world.size());
  size_t m = world.size() / n;
  print("bcgrid:", n, m);
  matrix_default_pmap = std::shared_ptr<WorldDCPmapInterface<ijT>>(new BlockCyclicPmap(world, n, m));
}

double g(size_t i, size_t j, size_t k, size_t l) {  // Compute 2-electron integral
  return 1.0 / ((i + 1) * (j + 1) + (k + 1) * (l + 1));
}

void BuildFockTask(const MatrixBlock& Dij, const MatrixBlock& Dik, const MatrixBlock& Dil, const MatrixBlock& Djk,
                   const MatrixBlock& Djl, const MatrixBlock& Dkl, MatrixBlock& Jij, MatrixBlock& Jkl, MatrixBlock& Kik,
                   MatrixBlock& Kil, MatrixBlock& Kjk, MatrixBlock& Kjl) {
  const size_t ilo = Dij.irange().start(), ihi = Dij.irange().stop();
  const size_t jlo = Dij.jrange().start(), jhi = Dij.jrange().stop();
  const size_t klo = Dkl.irange().start(), khi = Dkl.irange().stop();
  const size_t llo = Dkl.jrange().start(), lhi = Dkl.jrange().stop();

  const bool oij = (ilo == jlo);
  const bool okl = (klo == llo);
  const bool oikjl = (ilo == klo) && (jlo == llo);

  for (auto i : rangeT(ilo, ihi)) {
    size_t jtop = jhi;
    if (oij) jtop = i + 1;
    for (auto j : rangeT(jlo, jtop)) {
      double facij = 1.0;
      if (i == j) facij = 0.5;
      size_t ktop = khi;
      if (oikjl) ktop = i + 1;
      for (auto k : rangeT(klo, ktop)) {
        size_t ltop = lhi;
        if (okl) ltop = k + 1;
        if (oikjl && k == i) ltop = j + 1;
        for (auto l : rangeT(llo, ltop)) {
          double facijkl = facij;
          if (k == l) facijkl *= 0.5;
          if (i == k && j == l) facijkl *= 0.5;
          double gijkl = g(i, j, k, l) * facijkl;
          Jij(i, j) += Dkl(k, l) * gijkl;
          Jkl(k, l) += Dij(i, j) * gijkl;
          Kik(i, k) += Djl(j, l) * gijkl;
          Kil(i, l) += Djk(j, k) * gijkl;
          Kjk(j, k) += Dil(i, l) * gijkl;
          Kjl(j, l) += Dik(i, k) * gijkl;
        }
      }
    }
  }
}

Matrix copy(const Matrix& D) {
  Matrix C(D.get_world(), D.get_meta());
  for (auto& p : D) {
    C.add(p.second);
  }
  D.fence();
  return C;
}

Matrix symmetrize(const Matrix& D) {
  Matrix Dcopy = copy(D);
  for (auto& p : D) {
    Dcopy.add_transpose(p.second);
  }
  Dcopy.fence();
  Dcopy.scale(0.5);
  return Dcopy;
}

void BuildFockBrutal(const size_t natom, const Matrix& D, Matrix& J, Matrix& K) {
  for (auto iat : rangeT(0, natom)) {
    for (auto jat : rangeT(0, iat + 1)) {
      const MatrixBlock Dij = D.get(iat, jat);
      MatrixBlock Jij(Dij.get_meta());
      for (auto kat : rangeT(0, iat + 1)) {
        const MatrixBlock Dik = D.get(iat, kat);
        const MatrixBlock Djk = D.get(jat, kat);
        MatrixBlock Kik(Dik.get_meta());
        MatrixBlock Kjk(Djk.get_meta());
        size_t lattop = kat;
        if (kat == iat) lattop = jat;
        for (auto lat : rangeT(0, lattop + 1)) {
          const MatrixBlock Dil = D.get(iat, lat);
          const MatrixBlock Djl = D.get(jat, lat);
          const MatrixBlock Dkl = D.get(kat, lat);
          MatrixBlock Jkl(Dkl.get_meta());
          MatrixBlock Kil(Dil.get_meta());
          MatrixBlock Kjl(Djl.get_meta());

          BuildFockTask(Dij, Dik, Dil, Djk, Djl, Dkl, Jij, Jkl, Kik, Kil, Kjk, Kjl);

          J.add(Jkl);
          K.add(Kil);
          K.add(Kjl);
        }
        K.add(Kik);
        K.add(Kjk);
      }
      J.add(Jij);
    }
  }

  J = symmetrize(J);
  K = symmetrize(K);
  J.scale(4.0);
  K.scale(2.0);
}

void initializeD(size_t natom, Matrix& D) {
  for (auto& p : D) {  // Loops over local data
    MatrixBlock& Dij = p.second;
    for (auto i : Dij.irange()) {
      for (auto j : Dij.jrange()) {
        Dij(i, j) = 1.0 / (i + j + 2.0);
      }
    }
  }
}

void printmat(const Matrix& D) {
  for (auto& p : D) {
    const MatrixBlock& Dij = p.second;
    std::cout << Dij;
  }
}

int main(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);

  make_matrix_default_pmap(world);

  for (int arg = 1; arg < argc; ++arg) {
    if (strcmp(argv[arg], "-dx") == 0) xterm_debug(argv[0], 0);
  }

  const int natom = 3;      // Number of atoms
  std::vector<size_t> ilo;  // Start index of each atom's bf
  int nbf = 0;              // Total number of bf
  for (int i = 0; i < natom; i++) {
    int nbfa = 10 / ((i + 1) % 2 + 1);  // #basis per atom consistent with fortran example
    ilo.push_back(nbf);
    nbf += nbfa;
  }
  ilo.push_back(nbf);

  MatrixMetaData meta(nbf, nbf, ilo, ilo);
  Matrix D(world, meta);
  Matrix J(world, meta);
  Matrix K(world, meta);

  initializeD(natom, D);

  BuildFockBrutal(natom, D, J, K);

  print("D");
  printmat(D);
  print("J");
  printmat(J);
  print("K");
  printmat(K);

  // for (auto i : rangeT(0,nbf)) {
  //     for (auto j : rangeT(0,nbf)) {
  //         std::cout << J(i,j) << " ";
  //     }
  //     std::cout << std::endl;
  // }

  finalize();
  return 0;
}
