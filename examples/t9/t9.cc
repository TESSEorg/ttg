#include <cmath>
#include <iostream>

#include "ttg.h"

/* TODO: get rid of the using statement! */
using namespace ttg;

/*!
  \file t9.impl.h
  \brief Multiresolution computation on a 1-D spatial tree
  \defgroup example1Dtree Multiresolution computation on a 1-D spatial tree
  \ingroup examples

  \par Points of interest
  - recursive tree-traversal using TTG
  - dynamic sparsity --- no global knowledge of what data is being computed on
  - abstract data type as the key (index) mapping computation
*/

const double L = 10.0;       //< The computational domain is [-L,L]
const double thresh = 1e-6;  //< The threshold for small difference coefficients

// Simplistic error handler useful only for debugging --- production code should call `ttg_error()`
void error(const char* s) {
  std::cerr << s << std::endl;
  throw s;
}

// Computes powers of 2
double pow2(double n) { return std::pow(2.0, n); }

// 1 dimensional index into the tree (`n`=level,`l`=translation)
struct Key {
  int n;  // Level in the tree --- leave this as signed otherwise `-n` will do unexpected things
  unsigned long l; // Translation in at level `n` in range [0,2**n).
  std::size_t hashvalue; // Hash of n and l for storage for rapid comparison and hashing

  // Default constructor makes uninitialized key

  // Not overriding the default constructor is essential to keep the class POD
  // in which case we do not need to provide a serialization method.
  Key() = default;

  // Constructor given `n` and `l`
  Key(unsigned long n, unsigned long l) : n(n), l(l) { rehash(); }

  // Equality test
  bool operator==(const Key& b) const { return n == b.n && l == b.l; }

  // Inequality test
  bool operator!=(const Key& b) const { return !((*this) == b); }

  // Ordering
  bool operator<(const Key& b) const { return (n < b.n) || (n == b.n && l < b.l); }

  // Returns Key of the parent
  Key parent() const { return Key(n - 1, l >> 1); }

  // Returns Key of the left child
  Key left_child() const { return Key(n + 1, 2 * l); }

  // Returns Key of the right child
  Key right_child() const { return Key(n + 1, 2 * l + 1); }

  // Returns Key of the left child with periodic boundary conditions (used in derivative)
  Key left_child_periodic() const { return Key(n, l == 0ul ? (1ul << n) - 1 : l - 1); }  // periodic b.c.

  // Returns Key of the left child with periodic boundary conditions (used in derivative)
  Key right_child_periodic() const { return Key(n, l == ((1ul << n) - 1) ? 0 : l + 1); }  // periodic b.c.

  // Recomputes the hash --- internal use only
  void rehash() { hashvalue = (size_t(n) << 48) + l; }

  // Returns the hash
  std::size_t hash() const { return hashvalue; }
};

std::ostream& operator<<(std::ostream& s, const Key& key) {
  s << "Key(" << key.n << "," << key.l << ")";
  return s;
}

// Maps middle of the box labeled by key in [0,1] to real value in [-L,L]
double key_to_x(const Key& key) {
  const double scale = (2.0 * L) * pow2(-key.n);
  return -L + scale * (0.5 + key.l);
}

// A node in the tree
struct Node {
  Key key;  // As a convenience node stores its own key (in multidim code only tiny storage overhead)
  double s; // sum or scaling function coefficient
  double d; // difference or wavelet coefficient
  bool has_children; // if true this node has children

  // Use of generated default constructor required to make class POD
  Node() = default;

  // Construct node from data
  Node(const Key& key, double s, double d, bool has_children) : key(key), s(s), d(d), has_children(has_children) {}

  // Equality test checks key, s, d (but not has_children)
  bool operator==(const Node& a) const {
    return (key == a.key) && (std::abs(s - a.s) < 1e-12) && (std::abs(d - a.d) < 1e-12);
  }

  bool operator!=(const Node& a) const { return !(*this == a); }

  // Evaluate the scaling coefficients for a child of this node
  double child_value(const Key& key) const {
    if (this->key.n > key.n) error("not a child of this node");
    return s;  // With Haar and current normalization convention nothing needed
  }
};

std::ostream& operator<<(std::ostream& s, const Node& node) {
  s << "Node(" << node.key << "," << node.s << "," << node.d << "," << node.has_children << ")";
  return s;
}

using nodeEdge = Edge<Key, Node>;
using doubleEdge = Edge<Key, double>;
using ctlEdge = Edge<Key, void>;

using nodeOut = Out<Key, Node>;
using doubleOut = Out<Key, double>;
using ctlOut = Out<Key, void>;

template <typename keyT, typename valueT>
auto make_printer(const Edge<keyT, valueT>& in, const char* str = "") {
  auto func = [str](const keyT& key, const valueT& value, std::tuple<>& out) {
    std::cout << str << " (" << key << "," << value << ")" << std::endl;
  };
  return make_tt(func, edges(in), edges(), "printer", {"input"});
}

template <typename funcT>
auto make_project(const funcT& func, ctlEdge& ctl, nodeEdge& result, const std::string& name = "project") {
  auto f = [func](const Key& key, std::tuple<ctlOut, nodeOut>& out) {
    const double sl = func(key_to_x(key.left_child()));
    const double sr = func(key_to_x(key.right_child()));
    const double s = 0.5 * (sl + sr), d = 0.5 * (sl - sr);
    const double boxsize = 2.0 * L * pow2(-key.n);
    const double err = std::sqrt(d * d * boxsize);  // Estimate the norm2 error

    if ((key.n >= 5) && (err <= thresh)) {
      send<1>(key, Node(key, s, 0.0, false), out);
    } else {
      sendk<0>(key.left_child(), out);
      sendk<0>(key.right_child(), out);
      send<1>(key, Node(key, 0.0, 0.0, true), out);
    }
  };
  ctlEdge refine("refine");
  return make_tt(f, edges(fuse(refine, ctl)), edges(refine, result), name, {"control"}, {"refine", "result"});
}

template <typename funcT>
auto make_binary_op(const funcT& func, nodeEdge left, nodeEdge right, nodeEdge Result,
                    const std::string& name = "binaryop") {
  auto f = [&func](const Key& key, const Node& left, const Node& right, std::tuple<nodeOut, nodeOut, nodeOut>& out) {
    if (!(left.has_children || right.has_children)) {
      send<2>(key, Node(key, func(left.s, right.s), 0.0, false), out);
    } else {
      auto children = {key.left_child(), key.right_child()};
      if (!left.has_children) broadcast<0>(children, left, out);
      if (!right.has_children) broadcast<1>(children, right, out);
      send<2>(key, Node(key, 0.0, 0.0, true), out);
    }
  };
  nodeEdge L("L"), R("R");
  return make_tt(f, edges(fuse(left, L), fuse(right, R)), edges(L, R, Result), name, {"left", "right"},
              {"refineL", "refineR", "result"});
}

void send_to_output_tree(const Key& key, const Node& node, std::tuple<nodeOut, nodeOut, nodeOut>& out) {
  send<0>(key.right_child_periodic(), node, out);  // CANNOT MOVE NODE HERE SINCE USED BELOW!!!!
  send<1>(key, node, out);
  send<2>(key.left_child_periodic(), node, out);
}

void diff(const Key& key, const Node& left, const Node& center, const Node& right,
          std::tuple<nodeOut, nodeOut, nodeOut, nodeOut>& out) {
  nodeOut &L = std::get<0>(out), &C = std::get<1>(out), &R = std::get<2>(out), &result = std::get<3>(out);
  if (!(left.has_children || center.has_children || right.has_children)) {
    double derivative = (right.s - left.s) / (4.0 * ::L * pow2(-key.n));
    result.send(key, Node(key, derivative, 0.0, false));
  } else {
    result.send(key, Node(key, 0.0, 0.0, true));
    if (!left.has_children) L.send(key.left_child(), left);
    if (!center.has_children) {
      auto children = {key.left_child(), key.right_child()};
      L.send(key.right_child(), center);
      C.broadcast(children, center);
      R.send(key.left_child(), center);
    }
    if (!right.has_children) R.send(key.right_child(), right);
  }
}

auto make_diff(nodeEdge in, nodeEdge out, const std::string& name = "diff") {
  nodeEdge L("L"), C("C"), R("R");
  return std::make_tuple(
      make_tt(send_to_output_tree, edges(in), edges(L, C, R), "send_to_output_tree", {"input"}, {"L", "C", "R"}),
      make_tt(diff, edges(L, C, R), edges(L, C, R, out), name, {"L", "C", "R"}, {"L", "C", "R", "result"}));
}

void do_compress(const Key& key, double left, double right, std::tuple<doubleOut, doubleOut, nodeOut>& out) {
  doubleOut &L = std::get<0>(out), &R = std::get<1>(out);
  nodeOut& result = std::get<2>(out);

  double s = (left + right) * 0.5;
  double d = (left - right) * 0.5;

  if (key.n == 0) {
    result.send(key, Node(key, s, d, true));
  } else {
    result.send(key, Node(key, 0.0, d, true));
    if (key.l & 0x1uL)
      R.send(key.parent(), s);
    else
      L.send(key.parent(), s);
  }
}

void send_leaves_up(const Key& key, const Node& node, std::tuple<doubleOut, doubleOut, nodeOut>& out) {
  doubleOut &L = std::get<0>(out), &R = std::get<1>(out);
  nodeOut& result = std::get<2>(out);

  if (!node.has_children) {
    if (key.n == 0) {  // Tree is just one node
      result.send(key, node);
    } else {
      result.send(key, Node(key, 0.0, 0.0, false));
      if (key.l & 0x1uL)
        R.send(key.parent(), node.s);
      else
        L.send(key.parent(), node.s);
    }
  }
}

auto make_compress(const nodeEdge& in, nodeEdge& out, const std::string& name = "compress") {
  doubleEdge L("L"), R("R");
  return std::make_tuple(
      make_tt(send_leaves_up, edges(in), edges(L, R, out), "send_leaves_up", {"input"}, {"L", "R", "result"}),
      make_tt(do_compress, edges(L, R), edges(L, R, out), name, {"leftchild", "rightchild"}, {"L", "R", "result"}));
}

void start_reconstruct(const Key& key, const Node& node, std::tuple<doubleOut>& out) {
  if (key.n == 0) send<0>(key, node.s, out);
}

void do_reconstruct(const Key& key, double s, const Node& node, std::tuple<doubleOut, nodeOut>& out) {
  if (node.has_children) {
    send<0>(key.left_child(), s + node.d, out);
    send<0>(key.right_child(), s - node.d, out);
    send<1>(key, Node(key, 0.0, 0.0, true), out);
  } else {
    send<1>(key, Node(key, s, 0.0, false), out);
  }
}

auto make_reconstruct(const nodeEdge& in, nodeEdge& out, const std::string& name = "reconstruct") {
  doubleEdge S("S");  // passes scaling functions down
  return std::make_tuple(make_tt(start_reconstruct, edges(in), edges(S), "start reconstruct", {"nodes"}, {"node0"}),
                         make_tt(do_reconstruct, edges(S, in), edges(S, out), name, {"s", "nodes"}, {"s", "result"}));
}

// cannot easily replace this with make_tt due to persistent state
class Norm2 : public TT<Key, std::tuple<>, Norm2, ttg::typelist<Node>> {
  using baseT = typename Norm2::ttT;
  double sumsq;
  std::mutex charon;

 public:
  Norm2(const nodeEdge& in, const std::string& name = "norm2")
      : baseT(edges(in), edges(), name, {"nodes"}, {}), sumsq(0.0) {}

  // Lazy implementation of reduce operation ... just accumulates to local variable instead of summing up tree
  template <typename InputTuple>
  void op(const Key& key, InputTuple&& t, std::tuple<>& output) {
    std::lock_guard<std::mutex> obolus(charon);  // <<<<<<<<<< mutex
    const Node& node = baseT::get<0>(t);
    const double boxsize = 2.0 * L * pow2(-key.n);
    sumsq += (node.s * node.s + node.d * node.d) * boxsize;
  }

  double get() const {
    double value = sumsq;
    ttg_sum(ttg::default_execution_context(), value);
    return std::sqrt(value);
  }
};

auto make_norm2(const nodeEdge& in) { return std::make_unique<Norm2>(in); }  // for dull uniformity

auto make_start(const ctlEdge& ctl) {
  auto func = [](const Key& key, std::tuple<ctlOut>& out) { sendk<0>(key, out); };
  return make_tt<Key>(func, edges(), edges(ctl), "start", {}, {"control"});
}

// Operations used with BinaryOp
double add(double a, double b) { return a + b; }

double sub(double a, double b) { return a - b; }

double mul(double a, double b) { return a * b; }

// Functions we are testing with
double A(const double x) { return std::exp(-x * x); }

double diffA(const double x) { return -2.0 * x * exp(-x * x); }

double B(const double x) { return std::exp(-x * x) * std::cos(x); }

double C(const double x) { return std::exp(-x * x) * std::sin(x); }

double R(const double x) { return (A(x) + B(x)) * C(x); }

int main(int argc, char** argv) {
  initialize(argc, argv, -1);
  {
    ctlEdge ctl("start ctl");
    nodeEdge a("a"), b("b"), c("c"), abc("abc"), diffa("diffa"), errdiff("errdiff"), errabc("errabc"), a_plus_b("a+b"),
        a_plus_b_times_c("(a+b)*c"), deriva("deriva"), compa("compa"), recona("recona");

    // The following can indeed be specified in any order!
    auto p1 = make_project(&A, ctl, a, "project A");
    auto p2 = make_project(&B, ctl, b, "project B");
    auto p3 = make_project(&C, ctl, c, "project C");
    auto p4 = make_project(&R, ctl, abc, "project ABC");
    auto p5 = make_project(&diffA, ctl, diffa, "project dA/dx");

    auto b1 = make_binary_op(add, a, b, a_plus_b, "a+b");
    auto b2 = make_binary_op(mul, a_plus_b, c, a_plus_b_times_c, "(a+b)*c");
    auto b3 = make_binary_op(sub, a_plus_b_times_c, abc, errabc, "(a+b)*c - abc");
    auto b4 = make_binary_op(sub, diffa, deriva, errdiff, "dA/dx analytic - numeric");

    auto d = make_diff(a, deriva, "dA/dx numeric");

    auto norma = make_norm2(a);
    auto normabcerr = make_norm2(errabc);
    auto normdifferr = make_norm2(errdiff);

    auto comp1 = make_compress(a, compa, "compress(A)");
    auto norma2 = make_norm2(compa);

    auto recon1 = make_reconstruct(compa, recona, "reconstruct(A)");
    auto norma3 = make_norm2(recona);

    auto start = make_start(ctl);

    // auto printer = make_printer(a);
    // auto printer2 = make_printer(b);
    // auto printer = make_printer(a_plus_b);
    // auto printer2 = make_printer(err);
    // auto printer4 = make_printer(deriva,"numerical deriv");
    // auto printer5 = make_printer(diffa, "    exact deriv");
    // auto printer6 = make_printer(err,"differr");
    // auto pp = make_printer(compa,"compa");
    // auto pp = make_printer(compa,"compa");

    // ready to run!
    auto connected = make_graph_executable(start.get());
    assert(connected);
    TTGUNUSED(connected);

    if (ttg::default_execution_context().rank() == 0) {
#if 0
      std::cout << "Is everything connected? " << verify()(start.get()) << std::endl;
      std::cout << "==== begin dot ====\n";
      std::cout << Dot()(start.get()) << std::endl;
      std::cout << "====  end dot  ====\n";
#endif

      // This kicks off the entire computation
      start->invoke(Key(0, 0));
    }
    execute();
    fence();

    double nap = norma->get(), nac = norma2->get(), nar = norma3->get(), nabcerr = normabcerr->get(),
           ndifferr = normdifferr->get();

    if (ttg::default_execution_context().rank() == 0) {
      std::cout << "Norm2 of a projected     " << nap << std::endl;
      std::cout << "Norm2 of a compressed    " << nac << std::endl;
      std::cout << "Norm2 of a reconstructed " << nar << std::endl;
      std::cout << "Norm2 of error in abc    " << nabcerr << std::endl;
      std::cout << "Norm2 of error in diff   " << ndifferr << std::endl;
    }
  }
  ttg_finalize();

  return 0;
}
