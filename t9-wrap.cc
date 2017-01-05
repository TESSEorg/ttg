#include "flow.h"
#include <cmath>
#include <iostream>

// Same as t9.cc but using wrapper templates to get rid of most boilerplate template stuff

const double L = 10.0; // The computational domain is [-L,L]
const double thresh = 1e-2; // The threshold for small difference coefficients

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

using nodeFlow = Flow<Key,Node>;
using doubleFlow = Flow<Key,double>;
using ctlFlow = Flow<Key,Control>;

template <typename keyT, typename valueT>
auto make_printer(const Flow<keyT,valueT>& in, const char* str="") {
    auto func = [str](const keyT& key, const valueT& value, Flows<>& out) 
        {std::cout << str << " (" << key << "," << value << ")" << std::endl;};
    return make_op_wrapper(func, make_flows(in), Flows<>(), "printer");
}

template <typename funcT>
auto make_project(const funcT& func, const ctlFlow& ctl, nodeFlow& result) {
    auto f = [func](const Key& key, const Control& junk, Flows<ctlFlow,nodeFlow>& out) {
        const double sl = func(key_to_x(key.left_child()));
        const double sr = func(key_to_x(key.right_child()));
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
    };
    auto c = clone(ctl);
    return make_op_wrapper(f, make_flows(c), make_flows(c,result), "project");
}

template <typename funcT>
auto make_binary_op(const funcT& func, nodeFlow left, nodeFlow right, nodeFlow Result) {
    auto f = [&func](const Key& key, const Node& left, const Node& right, Flows<nodeFlow,nodeFlow,nodeFlow>& outputs) {
        nodeFlow L, R, Result; std::tie(L,R,Result) = outputs.all();

        if (!(left.has_children || right.has_children)) {
            Result.send(key,Node(key, func(left.s,right.s),0.0,false));
        }
        else {
            auto children = {key.left_child(), key.right_child()};
            if (!left.has_children) L.broadcast(children,left);
            if (!right.has_children)R.broadcast(children,right);
            Result.send(key,Node(key,0.0,0.0,true));
        }
    };
    nodeFlow L=clone(left), R=clone(right);
    return make_op_wrapper(f, make_flows(L,R), make_flows(L,R,Result), "binaryop");
}

void send_to_output_tree(const Key& key, const Node& node, Flows<nodeFlow,nodeFlow,nodeFlow>& out) {
    nodeFlow L, C, R;  std::tie(L, C, R) = out.all();
    L.send(key.right(), node);
    C.send(key, node);
    R.send(key.left(), node);
}

void diff(const Key& key, const Node& left, const Node& center, const Node& right, Flows<nodeFlow,nodeFlow,nodeFlow,nodeFlow>& outputs) {
    nodeFlow L, C, R, out;
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

auto make_diff(nodeFlow in, nodeFlow out) {
    nodeFlow L, C, R;
    return std::make_tuple(make_op_wrapper(&send_to_output_tree, make_flows(in), make_flows(L,C,R), "send_to_output_tree"),
                           make_op_wrapper(&diff, make_flows(L,C,R), make_flows(L,C,R,out),"diff"));
}

void do_compress(const Key& key, double left, double right, Flows<doubleFlow,doubleFlow,nodeFlow>& outputs) {
    doubleFlow L, R;
    nodeFlow out;
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

void send_leaves_up(const Key& key, const Node& node, Flows<doubleFlow,doubleFlow,nodeFlow>& outputs) {
    doubleFlow L, R;
    nodeFlow out;
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

auto make_compress(const nodeFlow& in, nodeFlow& out) {
    doubleFlow L, R;
    return std::make_tuple(make_op_wrapper(&send_leaves_up, make_flows(in), make_flows(L,R,out), "send_leaves_up"),
                           make_op_wrapper(&do_compress,make_flows(L,R), make_flows(L,R,out), "do_compress"));
}

void start_reconstruct(const Key& key, const Node& node, Flows<doubleFlow>& outputs) {
    if (key.n==0) outputs.send<0>(key,node.s);
}

void do_reconstruct(const Key& key, double s, const Node& node, Flows<doubleFlow,nodeFlow>& outputs) {
    if (node.has_children) {
        outputs.send<0>(key.left_child(), s+node.d);
        outputs.send<0>(key.right_child(),s-node.d);
        outputs.send<1>(key, Node(key, 0.0, 0.0, true));
    }
    else {
        outputs.send<1>(key,Node(key, s, 0.0, false));
    }
}

auto make_reconstruct(const nodeFlow& in, nodeFlow& out) {
    doubleFlow S;     // passes scaling functions down
    return std::make_tuple(make_op_wrapper(&start_reconstruct, make_flows(in), make_flows(S), "start_recon"),
                           make_op_wrapper(&do_reconstruct, make_flows(S,in), make_flows(S,out), "do_recon"));
}

// cannot easily replace this with wrapper due to persistent state
class Norm2 : public Op<InFlows<Key,Node>,Flows<>, Norm2> {
    using baseT = Op<InFlows<Key,Node>,Flows<>, Norm2>;
    double sumsq;
public:
    Norm2(const nodeFlow& in) : baseT(make_flows(in),Flows<>(), "norm2"), sumsq(0.0) {}

    // Lazy implementation of reduce operation ... just accumulates to local variable instead of summing up tree
    void op(const Key& key, const Node& node, baseT::output_type& output) {
        const double boxsize = 2.0*L*pow2(-key.n);
        sumsq += (node.s*node.s + node.d*node.d)*boxsize;
    }

    double get() const {return std::sqrt(sumsq);}
};

auto make_norm2(const nodeFlow& in) {return Norm2(in);} // for dull uniformity

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
    ctlFlow ctl;
    nodeFlow a, b, c, abc, diffa, errdiff, errabc, a_plus_b, a_plus_b_times_c, deriva, compa, recona;

    // The following can indeed be specified in any order!
    auto p1 = make_project(&A, ctl, a);
    auto p2 = make_project(&B, ctl, b);
    auto p3 = make_project(&C, ctl, c);
    auto p4 = make_project(&R, ctl, abc);
    auto p5 = make_project(&diffA, ctl, diffa);
    
    auto b1 = make_binary_op(add, a, b, a_plus_b);
    auto b2 = make_binary_op(mul, a_plus_b, c, a_plus_b_times_c);
    auto b3 = make_binary_op(sub, a_plus_b_times_c, abc, errabc);
    auto b4 = make_binary_op(sub, diffa, deriva, errdiff);

    auto d = make_diff(a, deriva);

    auto norma = make_norm2(a);
    auto normabcerr = make_norm2(errabc);
    auto normdifferr = make_norm2(errdiff);

    auto comp1 = make_compress(a,compa);
    auto norma2 = make_norm2(compa);

    auto recon1 = make_reconstruct(compa,recona);
    auto norma3 = make_norm2(recona);
    
    // auto printer = make_printer(a);
    // auto printer2 = make_printer(b);
    // auto printer = make_printer(a_plus_b);
    // auto printer2 = make_printer(err);
    // auto printer4 = make_printer(deriva,"numerical deriv");
    // auto printer5 = make_printer(diffa, "    exact deriv");
    // auto printer6 = make_printer(err,"differr");
    // auto pp = make_printer(compa,"compa");
    // auto pp = make_printer(compa,"compa");

    // This kicks off the entire computation
    ctl.send(Key(0,0),Control());

    std::cout << "Norm2 of a projected     " << norma.get() << std::endl;
    std::cout << "Norm2 of a compressed    " << norma2.get() << std::endl;
    std::cout << "Norm2 of a reconstructed " << norma3.get() << std::endl;
    std::cout << "Norm2 of error in abc    " << normabcerr.get() << std::endl;
    std::cout << "Norm2 of error in diff   " << normdifferr.get() << std::endl;

    return 0;
}
