#include "flow.h"
#include <cmath>
#include <iostream>

const double L = 10.0; // The computational domain is [-L,L]
const double thresh = 1e-6; // The threshold for small difference coefficients

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

    Key() : n(0), l(0) {}

    Key(unsigned long n, unsigned long l) : n(n), l(l) {}

    bool operator==(const Key& b) const {return n==b.n && l==b.l;}

    bool operator!=(const Key& b) const {return !((*this)==b);}

    bool operator<(const Key& b) const {return (n<b.n) || (n==b.n && l<b.l);}

    Key parent() const {return Key(n-1,l>>1);}

    Key left_child() const {return Key(n+1,2*l);}
    
    Key right_child() const {return Key(n+1,2*l+1);}

    Key left() const  {return Key(n,l==0ul ? (1ul<<n)-1 : l-1);} // periodic b.c.

    Key right() const {return Key(n,l==((1ul<<n)-1) ? 0 : l+1);} // periodic b.c.
};

std::ostream& operator<<(std::ostream&s, const Key& key) {
    s << "Key(" << key.n << "," << key.l << ")";
    return s;
}

// Maps middle of the box labeled by key in [0,1] to real value in [-L,L]
double key_to_x(const Key& key) {
    const double scale = (2.0*L)*pow2(-key.n);
    return -L + scale*(0.5+key.l);
}

// A node in the tree
struct Node {
    Key key; // A convenience and in multidim code is only a tiny storage overhead
    double s;
    double d;
    bool has_children;

    Node() : key(), s(0.0), d(0.0), has_children(false) {}

    Node(const Key& key, double s, double d, bool has_children) : key(key), s(s), d(d), has_children(has_children) {}

    bool operator==(const Node& a) const {
        return (key==a.key) && (std::abs(s-a.s)<1e-12) && (std::abs(d-a.d)<1e-12);

    }

    bool operator!=(const Node& a) const {
        return !(*this == a);
    }

    // Evaluate the scaling coefficients for a child of this node
    double child_value(const Key& key) const {
        if (this->key.n > key.n) error("not a child of this node");
        return s; // With Haar and current normalization convention nothing needed
    }
};

std::ostream& operator<<(std::ostream&s, const Node& node) {
    s << "Node(" << node.key << "," << node.s << "," << node.d << "," << node.has_children << ")";
    return s;
}


// An empty class used for pure control flows
class Control {};

std::ostream& operator<<(std::ostream&s, const Control& ctl) {
    s << "Ctl";
    return s;
}

// Prints input to cout, produces nothing
template <typename keyT, typename valueT>
class Printer : public Op<InFlows<keyT,valueT>, Flows<>, Printer<keyT,valueT>> {
    using baseT = Op<InFlows<keyT,valueT>, Flows<>, Printer<keyT,valueT>>;
    const std::string str;
public:
    explicit Printer(Flow<keyT,valueT> in, const char* str="") : baseT(make_flows(in), Flows<>(), "printer"), str(str) {}
    
    explicit Printer(InFlows<keyT,valueT> in) : baseT(in,Flows<>()) {}

    void op(const keyT& key, const std::tuple<valueT>& t, Flows<>& out) const {
        std::cout << str << " (" << key << "," << std::get<0>(t) << ")" << std::endl;
    }
};

// Project function into the scaling function basis to given threshold
// --- input is the key at which to start projecting (usually the
// root) and produces output in a downward traversal of the tree.
template <typename funcT>
class Project : public Op<InFlows<Key,Control>, Flows<Flow<Key,Control>, Flow<Key,Node>>, Project<funcT>> {
    using baseT = Op<InFlows<Key,Control>, Flows<Flow<Key,Control>, Flow<Key,Node>>, Project<funcT>>;
    funcT f;
 public:
    Project(const funcT& func, Flow<Key,Control> in, Flow<Key,Node> out) : baseT("project"), f(func) {
        Flow<Key,Control> ctl = clone(in);
        this->connect({ctl}, {ctl,out});
    }

    //void op(const Key& key, const Control& junk, baseT::output_type& out) {
    void op(const Key& key, const Control& junk, Flows<Flow<Key,Control>,Flow<Key,Node>>& out) {
        const double sl = f(key_to_x(key.left_child()));
        const double sr = f(key_to_x(key.right_child()));
        const double s = 0.5*(sl+sr), d = 0.5*(sl-sr);
        const double boxsize = 2.0*L*pow2(-key.n);
        const double err = std::sqrt(d*d*boxsize); // Estimate the norm2 error

        if ( (key.n >= 5) && (err <= thresh) ) {
            out.send<1>(key,Node(key,s,0.0,false));
        }
        else {
            out.send<0>(key.left_child(),Control());
            out.send<0>(key.right_child(),Control());
            out.send<1>(key,Node(key,0.0,0.0,true));
        }
    }
};

// Binary operation in scaling function basis without automatic refinement.  Assumes inputs
// are provided using a downward traversal of the tree in the scaling function basis.
class BinaryOp : public Op<InFlows<Key,Node,Node>, Flows<Flow<Key,Node>, Flow<Key,Node>, Flow<Key,Node>>, BinaryOp> {
    using baseT = Op<InFlows<Key,Node,Node>, Flows<Flow<Key,Node>, Flow<Key,Node>, Flow<Key,Node>>, BinaryOp>;
    double (*func)(double,double);
 public:    
    BinaryOp(double (*func)(double,double), Flow<Key,Node> left, Flow<Key,Node> right, Flow<Key,Node> Result) : baseT("binaryop"), func(func) {
        Flow<Key,Node> L=clone(left), R=clone(right);
        this->connect({L,R},{L,R,Result});
    }

    void op(const Key& key, const Node& left, const Node& right, typename baseT::output_type& outputs) {
        Flow<Key,Node> L, R, Result;
        std::tie(L,R,Result) = outputs.all();

        if (!(left.has_children || right.has_children)) {
            Result.send(key,Node(key, func(left.s,right.s),0.0,false));
        }
        else {
            auto children = {key.left_child(), key.right_child()};
            if (!left.has_children) L.broadcast(children,left);
            if (!right.has_children)R.broadcast(children,right);
            Result.send(key,Node(key,0.0,0.0,true));
        }
    }
};

// Differentiates the input function assuming input is in the scaling function basis and
// provided in a downward traversal of the tree. 
class Diff : public Op<InFlows<Key,Node,Node,Node>, Flows<Flow<Key,Node>,Flow<Key,Node>,Flow<Key,Node>,Flow<Key,Node>>, Diff> {
    using baseT = Op<InFlows<Key,Node,Node,Node>, Flows<Flow<Key,Node>,Flow<Key,Node>,Flow<Key,Node>,Flow<Key,Node>>, Diff>;

    struct SendToOutputTree : public Op<InFlows<Key,Node>, Flows<Flow<Key,Node>,Flow<Key,Node>,Flow<Key,Node>>, SendToOutputTree> {
        void op(const Key& key, const Node& node, Flows<Flow<Key,Node>,Flow<Key,Node>,Flow<Key,Node>>& out) {
            Flow<Key,Node> L, C, R;  std::tie(L, C, R) = out.all();
            L.send(key.right(), node);
            C.send(key, node);
            R.send(key.left(), node);
        }
    } sendtooutputtree;
        
public:
    Diff(Flow<Key,Node> in, Flow<Key,Node> out) : baseT("diff") {
        Flow<Key,Node> L, C, R;
        sendtooutputtree.connect({in},{L,C,R});
        this->connect({L,C,R},{L,C,R,out});
    }

    void op(const Key& key, const Node& left, const Node& center, const Node& right, baseT::output_type& outputs) {
        Flow<Key,Node> L, C, R, out;
        std::tie(L,C,R,out) = outputs.all();
        if (!(left.has_children || center.has_children || right.has_children)) {
            double derivative = (right.s - left.s)/(4.0*::L*pow2(-key.n));
            out.send(key,Node(key,derivative,0.0,false));
        }
        else {
            out.send(key,Node(key,0.0,0.0,true));
            if (!left.has_children) L.send(key.left_child(), left);
            if (!center.has_children) {
                auto children = {key.left_child(),key.right_child()};
                L.send(key.right_child(),center);
                C.broadcast(children,center);
                R.send(key.left_child(), center);
            }
            if (!right.has_children) R.send(key.right_child(),right);
        }
    }
};

// Input can be provided in any order. Output appears in order of an upward
// traversal of the tree.
class Compress : public Op<InFlows<Key,Node>,Flows<Flow<Key,double>,Flow<Key,double>,Flow<Key,Node>>,Compress> {
    using baseT = Op<InFlows<Key,Node>,Flows<Flow<Key,double>,Flow<Key,double>,Flow<Key,Node>>,Compress>;

    class DoIt : public Op<InFlows<Key,double,double>,Flows<Flow<Key,double>,Flow<Key,double>,Flow<Key,Node>>,DoIt> {
        using baseT = Op<InFlows<Key,double,double>,Flows<Flow<Key,double>,Flow<Key,double>,Flow<Key,Node>>,DoIt>;
    public:
        DoIt() : baseT("compress:doit") {}
        
        // Given left and right child values, compute s & d for this node, store d and pass s up.
        void op(const Key& key, double left, double right, baseT::output_type& outputs) {
            Flow<Key,double> L, R;
            Flow<Key,Node> out;
            std::tie(L,R,out) = outputs.all();

            double s = (left+right)*0.5;
            double d = (left-right)*0.5;

            if (key.n == 0) {
                out.send(key, Node(key, s, d, true));
            }
            else {
                out.send(key, Node(key, 0.0, d, true));
                if (key.l & 0x1uL)
                    R.send(key.parent(),s);
                else
                    L.send(key.parent(),s);
            }
        }
    } doit;


public:
    Compress(Flow<Key,Node> in, Flow<Key,Node> out)
        : baseT("compress"), doit()
    {
        Flow<Key,double> L, R;
        this->connect({in},{L,R,out});
        doit.connect({L,R},{L,R,out});
    }

    // Discard interior nodes, send leaf values to parent and fill in output leaves
    void op(const Key& key, const Node& node, baseT::output_type& outputs) {
        Flow<Key,double> L, R;
        Flow<Key,Node> out;
        std::tie(L,R,out) = outputs.all();
        
        if (!node.has_children) {
            if (key.n == 0) {   // Tree is just one node
                out.send(key,node);
            }
            else {
                out.send(key,Node(key,0.0,0.0,false));
                if (key.l & 0x1uL)
                    R.send(key.parent(),node.s);
                else
                    L.send(key.parent(),node.s);
            }
        }
    }
};

class Reconstruct : public Op<InFlows<Key,double,Node>, Flows<Flow<Key,double>,Flow<Key,Node>>, Reconstruct> {
    using baseT = Op<InFlows<Key,double,Node>, Flows<Flow<Key,double>,Flow<Key,Node>>, Reconstruct>;

    struct Start : public Op<InFlows<Key,Node>, Flows<Flow<Key,double>>, Start> {
        void op(const Key& key, const Node& node, Flows<Flow<Key,double>>& outputs) {
            if (key.n==0) outputs.send<0>(key,node.s);
        }
    } start;

public:
    Reconstruct(Flow<Key,Node> in, Flow<Key,Node> out) : baseT("reconstruct") {
        Flow<Key,double> S;     // passes scaling functions down
        start.connect(in,S);    // Clunky but ok in test code
        this->connect({S,in},{S,out});
    }

    void op(const Key& key, double s, const Node& node, baseT::output_type& outputs) {
        if (node.has_children) {
            outputs.send<0>(key.left_child(), s+node.d);
            outputs.send<0>(key.right_child(),s-node.d);
            outputs.send<1>(key, Node(key, 0.0, 0.0, true));
        }
        else {
            outputs.send<1>(key,Node(key, s, 0.0, false));
        }
    }
};

// Compute the 2-norm of the function in either basis
class Norm2 : public Op<InFlows<Key,Node>,Flows<>, Norm2> {
    using baseT = Op<InFlows<Key,Node>,Flows<>, Norm2>;
    double sumsq;
public:
    Norm2(Flow<Key,Node> in) : baseT(make_flows(in),Flows<>(), "norm2"), sumsq(0.0) {}

    // Lazy implementation of reduce operation ... just accumulates to local variable instead of summing up tree
    void op(const Key& key, const Node& node, baseT::output_type& output) {
        const double boxsize = 2.0*L*pow2(-key.n);
        sumsq += (node.s*node.s + node.d*node.d)*boxsize;
    }

    double get() const {return std::sqrt(sumsq);}
};

// Operations used with BinaryOp
double add(double a, double b) {return a+b;}

double sub(double a, double b) {return a-b;}

double mul(double a, double b) {return a*b;}

// Functions we are testing with
double A(const double x) {return std::exp(-x*x);}

double diffA(const double x) {return -2.0*x*exp(-x*x);}

double B(const double x) {return std::exp(-x*x)*std::cos(x);}

double C(const double x) {return std::exp(-x*x)*std::sin(x);}

double R(const double x) {return (A(x) + B(x))*C(x);}

int main () {
    using projectT = Project<double(*)(double)>;
    
    Flow<Key,Control> ctl;
    Flow<Key,Node> a, b, c, abc, diffa, errdiff, errabc, a_plus_b, a_plus_b_times_c, deriva, compa, recona;

    // The following can indeed be specified in any order!
    projectT p1(&A, ctl, a);
    projectT p2(&B, ctl, b);
    projectT p3(&C, ctl, c);
    projectT p4(&R, ctl, abc);
    projectT p5(&diffA, ctl, diffa);
    BinaryOp b1(add, a, b, a_plus_b);
    BinaryOp b2(mul, a_plus_b, c, a_plus_b_times_c);
    BinaryOp b3(sub, a_plus_b_times_c, abc, errabc);
    BinaryOp b4(sub, diffa, deriva, errdiff);
    Diff d(a,deriva);
    Norm2 norma(a);
    Norm2 normabcerr(errabc);
    Norm2 normdifferr(errdiff);

    Compress comp1(a,compa);
    Norm2 norma2(compa);

    Reconstruct recon1(compa,recona);
    Norm2 norma3(recona);
    
    // Printer<Key,Node> printer(a);
    // Printer<Key,Node> printer2(b);
    // Printer<Key,Node> printer(a_plus_b);
    // Printer<Key,Node> printer2(err);
    // Printer<Key,Node> printer4(deriva,"numerical deriv");
    // Printer<Key,Node> printer5(diffa, "    exact deriv");
    // Printer<Key,Node> printer6(err,"differr");
    // Printer<Key,Node> pp(compa,"compa");

    // This kicks off the entire computation
    ctl.send(Key(0,0),Control());

    std::cout << "Norm2 of a projected     " << norma.get() << std::endl;
    std::cout << "Norm2 of a compressed    " << norma2.get() << std::endl;
    std::cout << "Norm2 of a reconstructed " << norma3.get() << std::endl;
    std::cout << "Norm2 of error in abc    " << normabcerr.get() << std::endl;
    std::cout << "Norm2 of error in diff   " << normdifferr.get() << std::endl;

    return 0;
}
