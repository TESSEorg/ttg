
#ifndef WORLD_INSTANTIATE_STATIC_TEMPLATES
#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#endif

#include <iostream>
#include <tuple>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <utility>

#include "ttg/madness/ttg.h"

#include "Vector.h"
#include "Matrix.h"
#include "twoscalecoeffs.h"
#include "quadrature.h"

using namespace ttg;

// GLOBAL VARIABLES & FUNCTIONS
const int k = 5;
const double thresh = 1e-10; // The threshold for small difference coefficients

const int max_level = 30;

Matrix * hg;
Matrix * hg0;
Matrix * hg1;
Matrix * hgT;

Matrix * rm;
Matrix * r0;
Matrix * rp;
Vector * quad_w;
Vector * quad_x;
int quad_npt;
Matrix * quad_phi;
Matrix * quad_phiT;
Matrix * quad_phiw;



double gaussian(double x, double a, double coeff) {
    return coeff*exp(-a*x*x);
}

double funcD(double x) {
   return gaussian((x-0.5), 100.0, 1.0);
}

double funcDD(double x) {
   return -200 * (x-0.5) * gaussian((x-0.5), 100.0, 1.0);
}

double funcA(double x) {
        static const int N = 100;
        static double a[N], X[N], c[N];
        static bool initialized = false;

        if (!initialized) {
                for (int i=0; i<N; i++) {
                        a[i] = 1000*drand48();
                        X[i] = drand48();
                        c[i] = pow(2*a[i]/M_PI,0.25);
                }
                initialized = true;
        }

        double sum = 0.0;
        for (int i=0; i<N; i++) sum += gaussian(x-X[i], a[i], c[i]);
        return sum;
}

double funcB(double x) {
	static const int N = 100;
	static double a[N], X[N], c[N];
	static bool initialized = false;

	if (!initialized) {
		for (int i=0; i<N; i++) {
			a[i] = 1000*drand48();
			X[i] = drand48();
			c[i] = pow(2*a[i]/M_PI,0.25);
		}
		initialized = true;
	}

	double sum = 0.0;
	for (int i=0; i<N; i++) sum += gaussian(x-X[i], a[i], c[i]);
	return sum;
}

double funcC(double x) {
	static const int N = 100;
	static double a[N], X[N], c[N];
	static bool initialized = false;

	if (!initialized) {
		for (int i=0; i<N; i++) {
			a[i] = 1000*drand48();
			X[i] = drand48();
			c[i] = pow(2*a[i]/M_PI,0.25);
		}
		initialized = true;
	}

	double sum = 0.0;
	for (int i=0; i<N; i++) sum += gaussian(x-X[i], a[i], c[i]);
	return sum;
}

// emule
Vector mult(const Vector &v1, const Vector &v2) {
	Vector result(v1);
	for (unsigned int i = 0; i < v1.length(); ++i) {
		result.data[i] *= v2.data[i];
	}
	return result;
}

// gaxpy_inplace
Vector add(const Vector &v1, const Vector &v2) {
	Vector result(v1);
	for (unsigned int i = 0; i < v2.length(); ++i) {
	  result.data[i] += v2.data[i];
	}
	return result;
}


Vector sub(const Vector &v1, const Vector &v2) {
        Vector result(v1);
        for (unsigned int i = 0; i < v2.length(); ++i) {
          result.data[i] -= v2.data[i];
        }
        return result;
}


void init_twoscale(int k) {
	double  (*hgInput)[22] = twoscalecoeffs(k);

	hg = new Matrix(2*k, 2*k);
	hg0 = new Matrix(2*k, k);
	hg1 = new Matrix(2*k, k);
	hgT = new Matrix(2*k, 2*k);

	for (int i = 0; i < 2 * k; ++i) {
		for (int j = 0; j < 2 * k; ++j) {
			hg->set_item(i, j, hgInput[i][j]);
			hgT->set_item(i, j, hgInput[j][i]);

		}
	}

	for (int i = 0; i < 2 * k; ++i) {
		for (int j = 0; j < k; ++j) {
			hg0->set_item(i, j, hgInput[i][j]);
			hg1->set_item(i, j, hgInput[i][j+k]);
		}
	}
}

void init_quadrature(int order) {
	double *x = gauss_legendre_point(order);
	double *w = gauss_legendre_weight(order);

	quad_w = new Vector(w, 0, order);
	quad_x = new Vector(x, 0, order);

	int npt = order;
	quad_npt = npt;

	quad_phi = new Matrix(npt, k);
	quad_phiT = new Matrix(k, npt);
	quad_phiw = new Matrix(npt, k);

	for (int i = 0; i < npt; ++i) {
		double * p = phi((*quad_x)[i], k);
		for (int m = 0; m < k; ++m) {
			quad_phi->set_item(i, m, p[m]);
			quad_phiT->set_item(m, i, p[m]);
			quad_phiw->set_item(i, m, w[i] * p[m]);
		}
	}
}

void make_dc_periodic() {
	rm = new Matrix(k, k);
	r0 = new Matrix(k, k);
	rp = new Matrix(k, k);

	double iphase = 1.0;
	for (int i = 0; i < k; ++i) {
		double jphase = 1.0;

		for (int j = 0; j < k; ++j) {
			double gammaij = sqrt(( 2 * i + 1) * ( 2 * j + 1));
			double Kij;
			if ((( i -  j ) > 0) && (((i - j ) % 2) == 1 )) {
				Kij = 2.0;
			} else {
				Kij = 0.0;
			}

	                r0->set_item(i, j, 0.5 * (1.0 - iphase * jphase - 2.0 * Kij) * gammaij);
			rm->set_item(i, j, 0.5 * jphase * gammaij);
			rp->set_item(i, j, -0.5 * iphase * gammaij);

	               jphase = -1 * jphase;
		}

		iphase = -1 * iphase;
	}
}

void error(const char* s) {
    std::cerr << s << std::endl;
    throw s;
}

// Computes powers of 2
double pow2(double n) {
    return std::pow(2.0,n);
}

// 1 dimensional index into the tree (n=level,l=translation)
struct Key {
    int n; // leave this as signed otherwise -n does unexpected things
    unsigned long l;
    madness::hashT hashval;

    Key() : n(0), l(0) { rehash(); }

    Key(unsigned long n, unsigned long l) : n(n), l(l) { rehash(); }

    bool operator==(const Key& b) const {return n==b.n && l==b.l;}

    bool operator!=(const Key& b) const {return !((*this)==b);}

    bool operator<(const Key& b) const {return (n<b.n) || (n==b.n && l<b.l);}

    Key parent() const {return Key(n-1,l>>1);}

    Key left_child() const {return Key(n+1,2*l);}

    Key right_child() const {return Key(n+1,2*l+1);}

    Key left() const  {return Key(n,l==0ul ? (1ul<<n)-1 : l-1);} // periodic b.c. ==> (1ul<<n) is indeed 2^n
    Key right() const {return Key(n,l==((1ul<<n)-1) ? 0 : l+1);} // periodic b.c.

    void rehash() {hashval = (size_t(n)<<48)+l;}

    madness::hashT hash() const {return hashval;}

    template <typename Archive>
    void serialize(Archive& ar) {ar & madness::archive::wrap((unsigned char*) this, sizeof(*this));}

};

namespace std {
// specialize std::hash for Key
template <>
struct hash<Key> {
  std::size_t operator()(const Key& s) const noexcept { return s.hash(); }
};
}  // namespace std

std::ostream& operator<<(std::ostream&s, const Key& key) {
    s << "Key(" << key.n << "," << key.l << ")";
    return s;
}


// A node in the tree
class Node {
public:

	Key key;
	Vector s;
	Vector d;
	bool has_children;

	Node() : key(), s(k), d(k), has_children(false) {}

	Node(const Key &input_key, const Vector &s_input, const Vector &d_input, bool has_children_input): key(input_key), s(s_input), d(d_input), has_children(has_children_input) {}

        template <typename Archive>
             void serialize(Archive& ar) {ar & key & s & d & has_children;}
};

std::ostream& operator<<(std::ostream&s, const Node& node) {
    s << "Node(" << node.key << "," << node.s << "," << node.d << "," << node.has_children << ")";
    return s;
}

class Printer : public TT<Key, std::tuple<>, Printer, ttg::typelist<Node>> {
    using baseT = typename Printer::ttT;
public:
    Printer(const std::string& name) : baseT(name, {"input"}, {}) {}

    void op(const Key& key, const std::tuple<Node>& t, baseT::output_terminals_type& out) {
        std::cout << get_name() << ": Node with info: (" << key << "," << std::get<0>(t) << ")" << std::endl;
    }

    Printer(const typename baseT::input_edges_type& inedges, const std::string& name)
        : baseT(inedges, edges(), name, {"input"}, {}) {}

    ~Printer() {std::cout << "Printer destructor\n";}
};


class GaxpyOp : public TT<Key, std::tuple<Out<Key, Node>, Out<Key, Node>, Out<Key, Node>>, GaxpyOp, ttg::typelist<Node, Node>> {
  using baseT =  typename GaxpyOp::ttT;

  double alpha;
  double beta;

public:
  GaxpyOp(const double &alpha, const double &beta, const std::string &name)
  : baseT(name, {"input_op1", "input_op2"}, {"iterator_op1", "result", "iterator_op2"}), alpha(alpha), beta(beta) {}

  GaxpyOp(const double &alpha, const double &beta, const typename baseT::input_edges_type& inedges, const typename baseT::output_edges_type& outedges, const std::string& name)
  : baseT(inedges, outedges, name, {"input_op1", "input_op2"}, {"iterator_op1", "result", "iterator_op2"}), alpha(alpha), beta(beta) {}

  ~GaxpyOp() {std::cout << "GaxpyOp destructor\n";}

  void op(const Key &key, const std::tuple<Node, Node> &t, baseT::output_terminals_type &out) {
    const Node &left = std::get<0>(t);
    const Node &right = std::get<1>(t);

    Vector tempD(left.d);
    tempD.gaxpy(alpha, right.d, beta);

    Vector tempS;
    if (key.n == 0 && key.l == 0) {
      tempS = left.s;
      tempS.gaxpy(alpha, right.s, beta);
    }

    ::send<1>(key, Node(key, tempS, tempD, left.has_children || right.has_children), out);

    if (left.has_children && !right.has_children) {
      ::send<2>(key.left_child(), Node(key, Vector(), Vector(k), false), out);
      ::send<2>(key.right_child(), Node(key, Vector(), Vector(k), false), out);
    }

    if (right.has_children && !left.has_children) {
      ::send<0>(key.left_child(), Node(key, Vector(), Vector(k), false), out);
      ::send<0>(key.right_child(), Node(key, Vector(), Vector(k), false), out);
    }
  }

};


class BinaryOp : public TT<Key, std::tuple<Out<Key, Node>, Out<Key, Node>, Out<Key, Node>>,
                           BinaryOp, ttg::typelist<Node, Node>> {
  using baseT = typename BinaryOp::ttT;

  using funcT = Vector (*)(const Vector &, const Vector&);
  funcT func;

  Vector unfilter(const Vector &inputVector, int k, const Matrix * hg) const {
    Vector inputVector_copy(inputVector);
    Vector vector_d(2 * k);
    vector_d.set_slice_from_another_vector(0, k, inputVector);
    Vector vector_s2 = (vector_d * (*hg));
    return vector_s2;
  }


public:
  BinaryOp(const funcT &func, const std::string &name)
  : baseT(name, {"input_a", "input_b"}, {"iterator_a", "result", "iterator_b"})
  , func(func)
  {}

  BinaryOp(const funcT &func, const typename baseT::input_edges_type& inedges, const typename baseT::output_edges_type& outedges, const std::string& name)
  : baseT(inedges, outedges, name, {"input_a", "input_b"}, {"iterator_a", "result", "iterator_b"})
  , func(func) {}

  ~BinaryOp() {std::cout << "Binary Op destructor\n";}

   void op(const Key &key, const std::tuple<Node, Node> &t, baseT::output_terminals_type &out) {
      Node left = std::get<0>(t);
      Node right = std::get<1>(t);

      if (left.s.length() && right.s.length()) {

         double scale_factor = 1.0; //sqrt(pow(2.0, (key.n)));
         Vector f_vector(left.s * (*quad_phiT));
         Vector g_vector(right.s * (*quad_phiT));

         Vector temp = func(f_vector, g_vector);
         Vector resultVector((temp * (*quad_phiw)).scale(scale_factor));
         ::send<1>(key, Node(key, resultVector, Vector(), false), out);
      }
      else {
         if (left.s.length()) {

            Vector left_unfiltered = unfilter(left.s, k, hg);
            ::send<0>(key.left_child(), Node(key.left_child(), left_unfiltered.get_slice(0, k), Vector(), false), out);
            ::send<0>(key.right_child(), Node(key.right_child(), left_unfiltered.get_slice(k, 2 * k), Vector(), false), out);
         }
         if (right.s.length()) {
            Vector right_unfiltered = unfilter(right.s, k, hg);
            ::send<2>(key.left_child(), Node(key.left_child(), right_unfiltered.get_slice(0, k), Vector(), false), out);
            ::send<2>(key.right_child(), Node(key.right_child(), right_unfiltered.get_slice(k, 2 * k), Vector(), false), out);
         }

         ::send<1>(key, Node(key, Vector(), Vector(), true), out);
      }
   }

};


class Diff_prologue : public TT<Key, std::tuple<Out<Key, Node>, Out<Key, Node>, Out<Key, Node>>,
                                Diff_prologue, ttg::typelist<Node>> {
   using baseT = typename Diff_prologue::ttT;

public:

   Diff_prologue(const std::string &name)
   : baseT(name, {"input"}, {"L", "C", "R"}) {}

   Diff_prologue(const typename baseT::input_edges_type& inedges,
                const typename baseT::output_edges_type& outedges, const std::string& name)
   : baseT(inedges, outedges, name, {"input"}, {"L", "C", "R"}) {}

   ~Diff_prologue() {std::cout << "Diff_prologue destructor\n";}

   void op(const Key &key, const std::tuple<Node> &t, baseT::output_terminals_type &out) {
      Node node = std::get<0>(t);
      ::send<0>(key.right(), node, out);
      ::send<1>(key, node, out);
      ::send<2>(key.left(), node, out);
   }
};

class Diff_doIt : public TT<Key, std::tuple<Out<Key, Node>, Out<Key, Node>, Out<Key, Node>, Out<Key, Node>>,
                            Diff_doIt, ttg::typelist<Node, Node, Node>> {
   using baseT = typename Diff_doIt::ttT;

   Vector unfilter(const Vector &inputVector, int k, const Matrix * hg) const {
      Vector inputVector_copy(inputVector);
      Vector vector_d(2 * k);
      vector_d.set_slice_from_another_vector(0, k, inputVector);
      Vector vector_s2 = (vector_d * (*hg));
      return vector_s2;
   }


public:
   Diff_doIt(const std::string &name)
   : baseT(name, {"L_input", "C_input", "R_input"}, {"L_output", "C_output", "R_output", "output"}) {}

   Diff_doIt(const typename baseT::input_edges_type& inedges,
                const typename baseT::output_edges_type& outedges, const std::string& name)
   : baseT(inedges, outedges, name, {"L_input", "C_input", "R_input"}, {"L_output", "C_output", "R_output", "output"}) {}

   ~Diff_doIt() {std::cout << "Diff_doIt destructor\n";}

   void op(const Key &key, const std::tuple<Node, Node, Node> &t, baseT::output_terminals_type &out) {
      Node left = std::get<0>(t);
      Node center = std::get<1>(t);
      Node right = std::get<2>(t);

      if ((left.s.length() != 0 && left.has_children) ||(left.s.length() == 0 && !left.has_children)) {
          std::cout << "ERROR at left" << std::endl;
      }

      if ((center.s.length() != 0 && center.has_children)|| (center.s.length() == 0 && !center.has_children)) {
          std::cout << "ERROR at center" << std::endl;
      }

      if ((right.s.length() != 0 && right.has_children) ||(right.s.length() == 0 && !right.has_children)) {
          std::cout << "ERROR at right" << std::endl;
      }


      //if (!(left.has_children || center.has_children || right.has_children)) { /* if all of them are leaves */
      if (left.s.length() != 0 && center.s.length() != 0 && right.s.length() != 0) {
         //std::cout << "left.s.length() is " << left.s.length() << ", center.s.length() is " << center.s.length() << " and right.s.length() is " << right.s.length() << std::endl;
         Vector r = ((*rp) * left.s) + ((*r0) * center.s) + ((*rm) * right.s);
         ::send<3>(key, Node(key, r.scale(pow(2.0, key.n)), Vector(), false), out);
      }
      else {
         ::send<3>(key, Node(key, Vector(), Vector(), true), out);

         //if (!left.has_children) { /* if left is a leaf */
         if (left.s.length() != 0) {
            Vector unfiltered = unfilter(left.s, k, hg);
            ::send<0>(key.left_child(), Node(left.key.right_child(), unfiltered.get_slice(k, 2 * k), Vector(), false), out);
         }

         //if (!center.has_children) { /* if center is a leaf */
         if (center.s.length() != 0) {

           Vector unfiltered = unfilter(center.s, k, hg);

           ::send<2>(key.left_child(), Node(key.right_child(), unfiltered.get_slice(k, 2 * k), Vector(), false), out);
           ::send<0>(key.right_child(), Node(key.left_child(), unfiltered.get_slice(0, k), Vector(), false), out);

           ::send<1>(key.left_child(), Node(key.left_child(), unfiltered.get_slice(0, k), Vector(), false), out);
           ::send<1>(key.right_child(), Node(key.right_child(), unfiltered.get_slice(k, 2 * k), Vector(), false), out);
         }

         //if (!right.has_children) { /* if right is a leaf */
         if (right.s.length() != 0) {
            Vector unfiltered = unfilter(right.s, k, hg);
            ::send<2>(key.right_child(),Node(right.key.left_child(), unfiltered.get_slice(0, k), Vector(), false), out);
         }
      }
   }

};


class Compress_prologue : public TT<Key, std::tuple<Out<Key, Node>, Out<Key, Node>, Out<Key, Node>>,
                                    Compress_prologue, ttg::typelist<Node>> {
   using baseT = typename Compress_prologue::ttT;

public:
   Compress_prologue(const std::string &name)
   : baseT(name, {"input"}, {"left_intermediate_output", "output", "right_intermediate_output"}) {}

   Compress_prologue(const typename baseT::input_edges_type& inedges,
    		const typename baseT::output_edges_type& outedges, const std::string& name)
   : baseT(inedges, outedges, name, {"input"}, {"left_intermediate_output", "output", "right_intermediate_output"}) {}

   ~Compress_prologue() {std::cout << "Compress_prologue destructor\n";}

   void op(const Key &key, const std::tuple<Node> &t, baseT::output_terminals_type &out) {
      Node node = std::get<0>(t);
      if (!node.has_children) { // if the node is a leaf
         if (key.n == 0) {
            ::send<1>(key, node, out);
         }
         else {
            ::send<1>(key, Node(key, Vector(), Vector(k), false), out);

            if (key.l & 0x1uL) {
               ::send<2>(key.parent(), node, out);
            }
            else {
               ::send<0>(key.parent(), node, out);
            }
         }
      }
   }

};

class Compress_doIt : public TT<Key, std::tuple<Out<Key, Node>, Out<Key, Node>, Out<Key, Node>>,
                                Compress_doIt, ttg::typelist<Node, Node>> {
   using baseT = typename Compress_doIt::ttT;

public:
   Compress_doIt(const std::string &name)
   : baseT(name, {"input_left", "input_right"}, {"iterate_left", "result", "iterate_right"}) {}

   Compress_doIt(const typename baseT::input_edges_type& inedges,
                const typename baseT::output_edges_type& outedges, const std::string& name)
   : baseT(inedges, outedges, name, {"input_left", "input_right"}, {"iterate_left", "result", "iterate_right"}) {}

   ~Compress_doIt() {std::cout << "Compress_doIt destructor\n";}

   void op(const Key &key, const std::tuple<Node, Node> &t, baseT::output_terminals_type &out) {
      Node left = std::get<0>(t);
      Node right = std::get<1>(t);

      Vector s(left.s | right.s);
      Vector d = s * (*hgT);

      Vector sValue(d.data, 0, k);
      Vector dValue(d.data, k, 2 * k);


      if (key.n == 0) {
         //::send<1>(key, node, out);
         ::send<1>(key, Node(key, sValue, dValue, true), out);
      }
      else {
         ::send<1>(key, Node(key, Vector(), dValue, true), out);

         if (key.l & 0x1uL) {
            ::send<2>(key.parent(), Node(key, sValue, Vector(), false), out);
         }
         else {
            ::send<0>(key.parent(), Node(key, sValue, Vector(), false), out);
         }
      }
   }

};


class Reconstruct_prologue : public TT<Key, std::tuple<Out<Key, Vector>>,
                                       Reconstruct_prologue, ttg::typelist<Node>> {
   using baseT = typename Reconstruct_prologue::ttT;

public:
   Reconstruct_prologue(const std::string &name)
   : baseT(name, {"input_node"}, {"output_vector"}) {}


   Reconstruct_prologue(const typename baseT::input_edges_type& inedges,
                const typename baseT::output_edges_type& outedges, const std::string& name)
   : baseT(inedges, outedges, name, {"input_node"}, {"output_vector"}) {}


   ~Reconstruct_prologue() {std::cout << "Reconstruct_prologue destructor\n";}

   void op(const Key &key, const std::tuple<Node> &t, baseT::output_terminals_type &out) {
      Node node = std::get<0>(t);

      if (key.n == 0) {
         ::send<0>(key, node.s, out);
      }
   }

};


class Reconstruct_doIt : public TT<Key, std::tuple<Out<Key, Vector>, Out<Key, Node>>,
                                   Reconstruct_doIt, ttg::typelist<Vector, Node>> {
   using baseT = typename Reconstruct_doIt::ttT;

public:
   Reconstruct_doIt(const std::string &name)
   : baseT(name, {"input_vector", "input_node"}, {"output_vector", "output_node"}) {}


   Reconstruct_doIt(const typename baseT::input_edges_type& inedges,
                const typename baseT::output_edges_type& outedges, const std::string& name)
   : baseT(inedges, outedges, name, {"input_vector", "input_node"}, {"output_vector", "output_node"}) {}

   ~Reconstruct_doIt() {std::cout << "Reconstruct_doIt destructor\n"; }

   void op(const Key &key, const std::tuple<Vector, Node> &t, baseT::output_terminals_type &out) {
      Vector s = std::get<0>(t);
      Node node = std::get<1>(t);

      if (node.has_children) {
         Vector v1(s | node.d);
         Vector v2(v1 * (*hg));

         Vector leftChildS(v2.data, 0, k);
         Vector rightChildS(v2.data, k, 2 * k);


         ::send<0>(key.left_child(), leftChildS, out);
         ::send<0>(key.right_child(), rightChildS, out);

         ::send<1>(key, Node(key, Vector(), Vector(), true), out);
      }
      else {
         ::send<1>(key, Node(key, s, Vector(), false), out);
      }
   }

};


class Project : public  TT<Key, std::tuple<Out<Key,void>, Out<Key,Node>>,
                           Project, ttg::typelist<void>> {
    using baseT = typename Project::ttT;

 public:
    using funcT = double(*)(double);

    Project(const funcT& func, const std::string& name) : baseT(name, {"input"}, {"recurse","result"}), f(func) {}

    Project(const funcT& func, const typename baseT::input_edges_type& inedges,
            const typename baseT::output_edges_type& outedges, const std::string& name)
        : baseT(inedges, outedges, name, {"input"}, {"result", "recurse"}), f(func) {}

    ~Project() {std::cout << "Project destructor\n";}

    void op(const Key& key, baseT::output_terminals_type& out) {

      Vector s0 = sValues(key.n + 1, 2 * key.l);
      Vector s1 = sValues(key.n + 1, 2 * key.l + 1);

      Vector s(s0 | s1);
      Vector d(s * (*hgT));

      /* if the error is less than the threshhold or we have reached max_level */
      if (d.normf(k, 2*k) < thresh ||  key.n >= max_level - 1) {
        ::send<1>(key, Node(key, Vector(), Vector(), true), out);
        ::send<1>(Key(key.n+1, 2 * key.l), Node(key, s0, Vector(), false), out);
        ::send<1>(Key(key.n+1, 2 * key.l + 1), Node(key, s1, Vector(), false), out);
      }
      else {
        ::sendk<0>(key.left_child(), out);
        ::sendk<0>(key.right_child(), out);
        ::send<1>(key, Node(key, Vector(), Vector(), true), out);
      }
    }

private:
  funcT f;
  Vector sValues(int nInput, unsigned long lInput) const {
    Vector s(k);
    Vector & quad_x_ref = *quad_x;
    Matrix & quad_phiw_ref = *quad_phiw;

    double h = pow(0.5, nInput);
    double scale = sqrt(h);
    for (int mu = 0; mu < quad_npt; ++mu) {
      double x = (lInput + quad_x_ref[mu]) * h;
      double fValue = f(x); // = (func.f)(x);

      for (int i = 0; i < (k); ++i) {
        s[i] = s[i] + (scale * fValue * (quad_phiw_ref.get_item(mu, i))); // (quad_phiw_ref[mu])[i]);
      }
    }

    return s;
  }

};

class Producer : public TT<Key, std::tuple<Out<Key, void>>, Producer> {
  using baseT = typename Producer::ttT;

public:
  Producer(const std::string &name) : baseT(name, {}, {"output"}) {}
  Producer(const typename baseT::output_edges_type &outedges, const std::string &name)
    : baseT(edges(), outedges, name, {}, {"output"}) {}

  ~Producer() {std::cout << "Producer destructor\n";}

  void op(const Key &key, baseT::output_terminals_type &out) {
    Key root(0, 0);
    //std::cout << "Produced the root node whose key is " << root << std::endl;
    ::sendk<0>(root, out);
  }
};


// EXAMPLE 1
class Everything : public TT<Key, std::tuple<>, Everything> {
  using baseT = typename Everything::ttT;

  Producer producer;
  Project project;
  Printer printer;

  ttg::World world;

 public:
  Everything()
      : baseT("everything", {}, {})
      , producer("producer")
      , project(&funcA, "Project")
      , printer("Printer")
      , world(ttg::default_execution_context()) {
    producer.out<0>()->connect(project.in<0>());
    project.out<1>()->connect(printer.in<0>());
    project.out<0>()->connect(project.in<0>());

    if (!make_graph_executable(&producer)) throw "should be connected";
    fence();
  }

  void print() {}//{Print()(&producer);}
  std::string dot() {return Dot()(&producer);}
  void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
  void fence() { ttg_fence(world); }
};


// EXAMPLE 6
class Everything_comp_rec_test {
   Producer producer;
   Project project;
   Compress_prologue compress_prologue;
   Compress_doIt compress_doIt;
   Reconstruct_prologue reconstruct_prologue;
   Reconstruct_doIt reconstruct_doIt;
   BinaryOp minusOp;
   Printer printer;

   ttg::World world;

  public:
   Everything_comp_rec_test()
       : producer("producer")
       , project(&funcA, "Project_funcA")
       , compress_prologue("Compress_prologue")
       , compress_doIt("Compress_doIt")
       , reconstruct_prologue("Reconstruct_prologue")
       , reconstruct_doIt("Reconstruct_doIt")
       , minusOp(&sub, "minusOp")
       , printer("Printer")
       , world(ttg::default_execution_context()) {
    producer.out<0>()->connect(project.in<0>());
    project.out<0>()->connect(project.in<0>());
    project.out<1>()->connect(compress_prologue.in<0>());

    project.out<1>()->connect(minusOp.in<1>());

    compress_prologue.out<0>()->connect(compress_doIt.in<0>());
    compress_prologue.out<2>()->connect(compress_doIt.in<1>());

    compress_doIt.out<0>()->connect(compress_doIt.in<0>());
    compress_doIt.out<2>()->connect(compress_doIt.in<1>());

    compress_prologue.out<1>()->connect(reconstruct_prologue.in<0>());
    compress_doIt.out<1>()->connect(reconstruct_prologue.in<0>());

    compress_prologue.out<1>()->connect(reconstruct_doIt.in<1>());

    reconstruct_prologue.out<0>()->connect(reconstruct_doIt.in<0>());
    compress_doIt.out<1>()->connect(reconstruct_doIt.in<1>());

    reconstruct_doIt.out<0>()->connect(reconstruct_doIt.in<0>());
    reconstruct_doIt.out<1>()->connect(minusOp.in<0>());

    minusOp.out<0>()->connect(minusOp.in<0>());
    minusOp.out<2>()->connect(minusOp.in<1>());

    minusOp.out<1>()->connect(printer.in<0>());

    if (!make_graph_executable(&producer)) throw "should be connected";
    fence();

   }

   void print() {}//{Print()(&producer);}
   std::string dot() {return Dot()(&producer);}
   void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
   void fence() { ttg_fence(world); }
};

// EXAMPLE 2
class Everything_cnc {

   Producer producer;
   Project project_funcA;
   Project project_funcB;
   Project project_funcC;

   BinaryOp multOp;
   BinaryOp addOp;

   Printer printer;

   World world;


public:
 Everything_cnc()
     : producer("producer")
     , project_funcA(&funcA, "Project_funcA")
     , project_funcB(&funcB, "Project_funcB")
     , project_funcC(&funcC, "Project_funcC")
     , multOp(&mult, "A multiply B")
     , addOp(&add, "(A multiply B) add C")
     , printer("Printer")
     , world(ttg::default_execution_context()) {
    producer.out<0>()->connect(project_funcA.in<0>());
    producer.out<0>()->connect(project_funcB.in<0>());
    producer.out<0>()->connect(project_funcC.in<0>());

    project_funcA.out<0>()->connect(project_funcA.in<0>());
    project_funcA.out<1>()->connect(multOp.in<0>());

    project_funcB.out<0>()->connect(project_funcB.in<0>());
    project_funcB.out<1>()->connect(multOp.in<1>());

    multOp.out<0>()->connect(multOp.in<0>());
    multOp.out<2>()->connect(multOp.in<1>());
    multOp.out<1>()->connect(addOp.in<0>());

    project_funcC.out<0>()->connect(project_funcC.in<0>());
    project_funcC.out<1>()->connect(addOp.in<1>());

    addOp.out<0>()->connect(addOp.in<0>());
    addOp.out<2>()->connect(addOp.in<1>());
    addOp.out<1>()->connect(printer.in<0>());

    if (!make_graph_executable(&producer)) throw "should be connected";
    fence();
  }


  void print() {}//{Print()(&producer);}
  std::string dot() {return Dot()(&producer);}
  void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
  void fence() { ttg_fence(world); }
};


// EXAMPLE 3
class Everything_compress {
   Producer producer;
   Project project;
   Compress_prologue compress_prologue;
   Compress_doIt compress_doIt;
   Reconstruct_prologue reconstruct_prologue;
   Reconstruct_doIt reconstruct_doIt;
   Printer printer;

   ttg::World world;

  public:
   Everything_compress()
       : producer("producer")
       , project(&funcA, "Project_funcA")
       , compress_prologue("Compress_prologue")
       , compress_doIt("Compress_doIt")
       , reconstruct_prologue("Reconstruct_prologue")
       , reconstruct_doIt("Reconstruct_doIt")
       , printer("Printer")
       , world(ttg::default_execution_context()) {
    producer.out<0>()->connect(project.in<0>());

    project.out<0>()->connect(project.in<0>());
    project.out<1>()->connect(compress_prologue.in<0>());

    compress_prologue.out<0>()->connect(compress_doIt.in<0>());
    compress_prologue.out<2>()->connect(compress_doIt.in<1>());

    compress_prologue.out<1>()->connect(reconstruct_prologue.in<0>());
    compress_prologue.out<1>()->connect(reconstruct_doIt.in<1>());

    compress_doIt.out<0>()->connect(compress_doIt.in<0>());
    compress_doIt.out<2>()->connect(compress_doIt.in<1>());
    compress_doIt.out<1>()->connect(reconstruct_prologue.in<0>());
    compress_doIt.out<1>()->connect(reconstruct_doIt.in<1>());

    reconstruct_prologue.out<0>()->connect(reconstruct_doIt.in<0>());
    reconstruct_doIt.out<0>()->connect(reconstruct_doIt.in<0>());

    reconstruct_doIt.out<1>()->connect(printer.in<0>());

    if (!make_graph_executable(&producer)) throw "should be connected";
    fence();

  }

  void print() {}//{Print()(&producer);}
  std::string dot() {return Dot()(&producer);}
  void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
  void fence() { ttg_fence(world); }
};

// EXAMPLE 7
class Everything_gaxpy_test2 {
   Producer producer;
   Project project_funcA1;
   Project project_funcA2;

   Compress_prologue compress_prologue_funcA1;
   Compress_doIt compress_doIt_funcA1;

   Compress_prologue compress_prologue_funcA2;
   Compress_doIt compress_doIt_funcA2;

   GaxpyOp gaxpyOp;

   Reconstruct_prologue reconstruct_prologue;
   Reconstruct_doIt reconstruct_doIt;

   Printer printer;

   ttg::World world;

  public:
   Everything_gaxpy_test2()
   : producer("producer")
       , project_funcA1(&funcA, "project_funcA1")
       , project_funcA2(&funcA, "project_funcA2")
       , compress_prologue_funcA1("compress_prologue_funcA1")
       , compress_doIt_funcA1("compress_doIt_funcA1")
       , compress_prologue_funcA2("compress_prologue_funcA2")
       , compress_doIt_funcA2("compress_doIt_funcA2")
       , gaxpyOp(1.0, -1.0, "gaxpyOp")
       , reconstruct_prologue("reconstruct_prologue")
       , reconstruct_doIt("reconstruct_doIt")
       , printer("printer")
       , world(ttg::default_execution_context()) {
    producer.out<0>()->connect(project_funcA1.in<0>());
    producer.out<0>()->connect(project_funcA2.in<0>());

    project_funcA1.out<0>()->connect(project_funcA1.in<0>());
    project_funcA2.out<0>()->connect(project_funcA2.in<0>());

    project_funcA1.out<1>()->connect(compress_prologue_funcA1.in<0>());
    project_funcA2.out<1>()->connect(compress_prologue_funcA2.in<0>());

    compress_prologue_funcA1.out<0>()->connect(compress_doIt_funcA1.in<0>());
    compress_prologue_funcA1.out<2>()->connect(compress_doIt_funcA1.in<1>());
    compress_doIt_funcA1.out<0>()->connect(compress_doIt_funcA1.in<0>());
    compress_doIt_funcA1.out<2>()->connect(compress_doIt_funcA1.in<1>());

    compress_prologue_funcA2.out<0>()->connect(compress_doIt_funcA2.in<0>());
    compress_prologue_funcA2.out<2>()->connect(compress_doIt_funcA2.in<1>());
    compress_doIt_funcA2.out<0>()->connect(compress_doIt_funcA2.in<0>());
    compress_doIt_funcA2.out<2>()->connect(compress_doIt_funcA2.in<1>());

    compress_prologue_funcA1.out<1>()->connect(gaxpyOp.in<0>());
    compress_prologue_funcA2.out<1>()->connect(gaxpyOp.in<1>());

    compress_doIt_funcA1.out<1>()->connect(gaxpyOp.in<0>());
    compress_doIt_funcA2.out<1>()->connect(gaxpyOp.in<1>());

    gaxpyOp.out<0>()->connect(gaxpyOp.in<0>());
    gaxpyOp.out<2>()->connect(gaxpyOp.in<1>());

    gaxpyOp.out<1>()->connect(reconstruct_prologue.in<0>());
    gaxpyOp.out<1>()->connect(reconstruct_doIt.in<1>());


    reconstruct_prologue.out<0>()->connect(reconstruct_doIt.in<0>());
    reconstruct_doIt.out<0>()->connect(reconstruct_doIt.in<0>());
    reconstruct_doIt.out<1>()->connect(printer.in<0>());


    // EXAMPLE 9  --> DIRECTLY CONNECTING THE RESULT OF GAXPY TO PRINTER
    //gaxpyOp.out<1>()->connect(printer.in<0>());

    if (!make_graph_executable(&producer)) throw "should be connected";
    fence();

  }

  void print() {}//{Print()(&producer);}
  std::string dot() {return Dot()(&producer);}
  void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
  void fence() { ttg_fence(world); }

};


class Everything_gaxpy_test3 {
   Producer producer;
   Project project_funcA;
   Project project_funcB;

   BinaryOp minusOp;

   Compress_prologue compress_prologue;
   Compress_doIt compress_doIt;

   Compress_prologue compress_prologueA;
   Compress_doIt compress_doItA;

   Compress_prologue compress_prologueB;
   Compress_doIt compress_doItB;

   GaxpyOp gaxpyOp_minus;
   GaxpyOp gaxpyOp_minus2;

   Reconstruct_prologue reconstruct_prologue;
   Reconstruct_doIt reconstruct_doIt;

   Printer printer;
   ttg::World world;

  public:
   Everything_gaxpy_test3()
   : producer("producer")
   , project_funcA(&funcA, "project_funcA")
   , project_funcB(&funcB, "project_funcB")
   , minusOp(&sub, "minusOp")
   , compress_prologue("compress_prologue")
       , compress_doIt("compress_doIt")
       , compress_prologueA("compress_prologueA")
       , compress_doItA("compress_doItA")
       , compress_prologueB("compress_prologueB")
       , compress_doItB("compress_doItB")
       , gaxpyOp_minus(1.0, -1.0, "gaxpyOp_minus")
       , gaxpyOp_minus2(1.0, -1.0, "gaxpyOp_minus2")
       , reconstruct_prologue("reconstruct_prologue")
       , reconstruct_doIt("reconstruct_doIt")
       , printer("printer")
       , world(ttg::default_execution_context()) {
    producer.out<0>()->connect(project_funcA.in<0>());
    producer.out<0>()->connect(project_funcB.in<0>());

    project_funcA.out<0>()->connect(project_funcA.in<0>());
    project_funcB.out<0>()->connect(project_funcB.in<0>());

    project_funcA.out<1>()->connect(minusOp.in<0>());
    minusOp.out<0>()->connect(minusOp.in<0>());
    project_funcB.out<1>()->connect(minusOp.in<1>());
    minusOp.out<2>()->connect(minusOp.in<1>());

    project_funcA.out<1>()->connect(compress_prologueA.in<0>());
    project_funcB.out<1>()->connect(compress_prologueB.in<0>());

    minusOp.out<1>()->connect(compress_prologue.in<0>());

    compress_prologueA.out<0>()->connect(compress_doItA.in<0>());
    compress_prologueA.out<2>()->connect(compress_doItA.in<1>());

    compress_prologueB.out<0>()->connect(compress_doItB.in<0>());
          compress_prologueB.out<2>()->connect(compress_doItB.in<1>());

    compress_doItA.out<0>()->connect(compress_doItA.in<0>());
    compress_doItA.out<2>()->connect(compress_doItA.in<1>());

    compress_doItB.out<0>()->connect(compress_doItB.in<0>());
          compress_doItB.out<2>()->connect(compress_doItB.in<1>());

    compress_prologueA.out<1>()->connect(gaxpyOp_minus2.in<0>());
    compress_prologueB.out<1>()->connect(gaxpyOp_minus2.in<1>());

    compress_doItA.out<1>()->connect(gaxpyOp_minus2.in<0>());
    compress_doItB.out<1>()->connect(gaxpyOp_minus2.in<1>());

    gaxpyOp_minus2.out<0>()->connect(gaxpyOp_minus2.in<0>());
    gaxpyOp_minus2.out<2>()->connect(gaxpyOp_minus2.in<1>());

    gaxpyOp_minus2.out<1>()->connect(gaxpyOp_minus.in<1>());

    compress_prologue.out<0>()->connect(compress_doIt.in<0>());
    compress_prologue.out<2>()->connect(compress_doIt.in<1>());

    compress_prologue.out<1>()->connect(gaxpyOp_minus.in<0>());
    compress_doIt.out<1>()->connect(gaxpyOp_minus.in<0>());

    compress_doIt.out<0>()->connect(compress_doIt.in<0>());
    compress_doIt.out<2>()->connect(compress_doIt.in<1>());

    gaxpyOp_minus.out<0>()->connect(gaxpyOp_minus.in<0>());
    gaxpyOp_minus.out<2>()->connect(gaxpyOp_minus.in<1>());

    gaxpyOp_minus.out<1>()->connect(reconstruct_prologue.in<0>());
    gaxpyOp_minus.out<1>()->connect(reconstruct_doIt.in<1>());

    reconstruct_prologue.out<0>()->connect(reconstruct_doIt.in<0>());
    reconstruct_doIt.out<0>()->connect(reconstruct_doIt.in<0>());

    reconstruct_doIt.out<1>()->connect(printer.in<0>());

    if (!make_graph_executable(&producer)) throw "should be connected";
    fence();
  }


   void print() {}//{Print()(&producer);}
   std::string dot() {return Dot()(&producer);}
   void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
   void fence() { ttg_fence(world); }

};


// EXAMPLE 5
class Everything_gaxpy_test {
    Producer producer;
    Project project_funcA;
    Project project_funcB;

    Compress_prologue compress_prologue_funcA;
    Compress_doIt compress_doIt_funcA;

    Compress_prologue compress_prologue_funcB;
    Compress_doIt compress_doIt_funcB;

    GaxpyOp gaxpyOp;

    Reconstruct_prologue reconstruct_prologue;
    Reconstruct_doIt reconstruct_doIt;

    BinaryOp minusOp;
    BinaryOp subOp;

    Printer printer;

    ttg::World world;

    public:
    Everything_gaxpy_test()
        : producer("producer")
        , project_funcA(&funcA, "project_funcA")
        , project_funcB(&funcB, "project_funcB")
        , compress_prologue_funcA("compress_prologue_funcA")
        , compress_doIt_funcA("compress_doIt_funcA")
        , compress_prologue_funcB("compress_prologue_funcB")
        , compress_doIt_funcB("compress_doIt_funcB")
        , gaxpyOp(1.0, -1.0, "gaxpyOp")
        , reconstruct_prologue("reconstruct_prologue")
        , reconstruct_doIt("reconstruct_doIt")
        , minusOp(&sub, "minusOp")
        , subOp(&sub, "subOp")
        , printer("printer")
        , world(ttg::default_execution_context()) {
      producer.out<0>()->connect(project_funcA.in<0>());
      producer.out<0>()->connect(project_funcB.in<0>());

      project_funcA.out<0>()->connect(project_funcA.in<0>());
      project_funcB.out<0>()->connect(project_funcB.in<0>());

      project_funcA.out<1>()->connect(compress_prologue_funcA.in<0>());
      project_funcB.out<1>()->connect(compress_prologue_funcB.in<0>());

      project_funcA.out<1>()->connect(minusOp.in<0>());
      project_funcB.out<1>()->connect(minusOp.in<1>());

      minusOp.out<0>()->connect(minusOp.in<0>());
      minusOp.out<2>()->connect(minusOp.in<1>());

      compress_prologue_funcA.out<0>()->connect(compress_doIt_funcA.in<0>());
      compress_prologue_funcB.out<0>()->connect(compress_doIt_funcB.in<0>());

      compress_prologue_funcA.out<2>()->connect(compress_doIt_funcA.in<1>());
      compress_prologue_funcB.out<2>()->connect(compress_doIt_funcB.in<1>());

      compress_doIt_funcA.out<0>()->connect(compress_doIt_funcA.in<0>());
      compress_doIt_funcA.out<2>()->connect(compress_doIt_funcA.in<1>());

      compress_doIt_funcB.out<0>()->connect(compress_doIt_funcB.in<0>());
      compress_doIt_funcB.out<2>()->connect(compress_doIt_funcB.in<1>());

      compress_prologue_funcA.out<1>()->connect(gaxpyOp.in<0>());
      compress_prologue_funcB.out<1>()->connect(gaxpyOp.in<1>());

      compress_doIt_funcA.out<1>()->connect(gaxpyOp.in<0>());
      compress_doIt_funcB.out<1>()->connect(gaxpyOp.in<1>());

      gaxpyOp.out<0>()->connect(gaxpyOp.in<0>());
      gaxpyOp.out<2>()->connect(gaxpyOp.in<1>());

      gaxpyOp.out<1>()->connect(reconstruct_prologue.in<0>());
      gaxpyOp.out<1>()->connect(reconstruct_doIt.in<1>());


      reconstruct_prologue.out<0>()->connect(reconstruct_doIt.in<0>());
      reconstruct_doIt.out<0>()->connect(reconstruct_doIt.in<0>());

      reconstruct_doIt.out<1>()->connect(subOp.in<0>());
      //reconstruct_doIt.out<1>()->connect(subOp.in<1>());

      subOp.out<0>()->connect(subOp.in<0>());

      subOp.out<2>()->connect(subOp.in<1>());

      minusOp.out<1>()->connect(subOp.in<1>());
      //minusOp.out<1>()->connect(subOp.in<0>());

      subOp.out<1>()->connect(printer.in<0>());

      if (!make_graph_executable(&producer)) throw "should be connected";
      fence();
    }

    void print() {}//{Print()(&producer);}
    std::string dot() {return Dot()(&producer);}
    void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
    void fence() { ttg_fence(world); }


};

// EXAMPLE 4
class Everything_diff {
  Producer producer;
  Project project;
  Diff_prologue diff_prologue;
  Diff_doIt diff_doIt;
  Printer printer;

  ttg::World world;

public:
  Everything_diff()
       : producer("producer")
       , project(&funcD, "Project_funcD")
       , diff_prologue("Diff_prologue")
       , diff_doIt("Diff_doIt")
       , printer("Printer")
       , world(ttg::default_execution_context()) {
    producer.out<0>()->connect(project.in<0>());

    project.out<0>()->connect(project.in<0>());
    project.out<1>()->connect(diff_prologue.in<0>());

    diff_prologue.out<0>()->connect(diff_doIt.in<0>());
    diff_prologue.out<1>()->connect(diff_doIt.in<1>());
    diff_prologue.out<2>()->connect(diff_doIt.in<2>());

    diff_doIt.out<0>()->connect(diff_doIt.in<0>());
    diff_doIt.out<1>()->connect(diff_doIt.in<1>());
    diff_doIt.out<2>()->connect(diff_doIt.in<2>());

    diff_doIt.out<3>()->connect(printer.in<0>());

    if (!make_graph_executable(&producer)) throw "should be connected";
    fence();
  }


  void print() {}//{Print()(&producer);}
  std::string dot() {return Dot()(&producer);}
  void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
  void fence() { ttg_fence(world); }
};


// EXAMPLE 5
class Everything_diff_test {

  Edge<Key, Node> inter, l, c, r, op1, op2, out, r3, r4;
  Edge<Key, void> p1, p2, r1, r2;


  Producer producer;
  Project project_funcD;
  Project project_funcDD;
  Diff_prologue diff_prologue;
  Diff_doIt diff_doIt;
  BinaryOp minusOp;
  Printer printer;
  ttg::World world;

   //Edge<Key, Node> p1, p2, inter, l, c, r, op1, op2, out;

public:
  Everything_diff_test()
     : producer(edges(fuse(p1, p2)), "producer")
     , project_funcD(&funcD, edges(fuse(r1, p1)), edges(r1, inter), "Project_funcD")
     , project_funcDD(&funcDD, edges(fuse(r2, p2)), edges(r2, op2), "Project_funcDD")
     , diff_prologue(edges(inter), edges(l, c, r), "Diff_prologue")
     , diff_doIt(edges(l, c, r), edges(l, c, r, op1), "Diff_doIt")
     , minusOp(&sub, edges(fuse(op1, r3), fuse(op2, r4)), edges(r3, out, r4), "diff_funcD sub funcDD")
     , printer(edges(out), "Printer")
     , world(ttg::default_execution_context()) {
  fence();
 }

 void print() {}//{Print()(&producer);}
   std::string dot() {return Dot()(&producer);}
   void start() {if (world.rank() == 0) producer.invoke(Key(0, 0));}
   void fence() { ttg_fence(world); }
};


//void hi() { std::cout << "hi\n"; }

int main(int argc, char** argv) {
  /*
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);

  for (int arg=1; arg<argc; ++arg) {
      if (strcmp(argv[arg],"-dx")==0)
          xterm_debug(argv[0], 0);
  }

  TTBase::set_trace_all(false); */

  initialize(argc, argv);
  auto world = ttg::default_execution_context();
  // world.taskq.add(world.rank(), hi);
  ttg_fence(world);

  for (int arg = 1; arg < argc; ++arg) {
    if (strcmp(argv[arg], "-dx") == 0) madness::xterm_debug(argv[0], 0);
  }

  ttg::TTBase::set_trace_all(false);

  init_twoscale(k);
  init_quadrature(k);
  make_dc_periodic();

  try {
    /* doing all the initializtions */

    // FIRST EXAMPLE
    /* {
        Everything x;
        x.print();
        std::cout << x.dot() << std::endl;
        x.start();
        x.wait();
    } */

    // SECOND EXAMPLE
    {
      Everything_cnc x;
      // x.print();
      // std::cout << x.dot() << std::endl;
      x.start();
      x.fence();
    }

    // THIRD EXAMPLE
    /* {
       Everything_compress x;
       x.print();
       std::cout << x.dot() << std::endl;
       x.start();
       x.wait();
    } */

    // FORTH EXAMPLE
    /*{
       Everything_diff_test x; // previously it was Everything_diff x;
       x.print();
       std::cout << x.dot() << std::endl;
       x.start();
       x.fence();
    }*/

    // FIFTH EXAMPLE
    /* {
      Everything_gaxpy_test x; // previously it was Everything_diff x;
      //x.print();
      //std::cout << x.dot() << std::endl;
      x.start();
      //x.wait();
      x.fence();
   } */

    // SIXTH EXAMPLE
    /* {
       Everything_comp_rec_test x;
       x.print();
       std::cout << x.dot() << std::endl;
       x.start();
       x.wait();
    } */

    // SEVENTH EXAMPLE
    /* {
       Everything_gaxpy_test2 x;
       x.print();
       std::cout << x.dot() << std::endl;
       x.start();
       x.wait();
    } */
  } catch (std::exception &x) {
    std::cerr << "Caught a std::exception: " << x.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Caught an unknown exception: " << std::endl;
    return 1;
  }

  fence();
  ttg_finalize();
  return 0;
}
