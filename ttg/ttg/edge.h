#ifndef TTG_EDGE_H
#define TTG_EDGE_H

#include <iostream>
#include <vector>
#include <memory>

#include "ttg/base/terminal.h"
#include "ttg/util/print.h"
#include "ttg/util/trace.h"
#include "ttg/terminal.h"

namespace ttg {

  template <typename keyT, typename valueT>
  class Edge {
   private:
    // An EdgeImpl represents a single edge that most usually will
    // connect a single output terminal with a single
    // input terminal.  However, we had to relax this constraint in
    // order to easily accommodate connecting an input/output edge to
    // an operation that to the outside looked like a single op but
    // internally was implemented as multiple operations.  Thus, the
    // input/output edge has to connect to multiple terminals.
    // Permitting multiple end points makes this much easier to
    // compose, easier to implement, and likely more efficient at
    // runtime.  This is why outs/ins are vectors rather than pointers
    // to a single terminal.
    struct EdgeImpl {
      std::string name;
      std::vector<TerminalBase *> outs;  // In<keyT, valueT> or In<keyT, const valueT>
      std::vector<Out<keyT, valueT> *> ins;

      EdgeImpl() : name(""), outs(), ins() {}

      EdgeImpl(const std::string &name) : name(name), outs(), ins() {}

      void set_in(Out<keyT, valueT> *in) {
        if (ins.size() && tracing()) {
          print("Edge: ", name, " : has multiple inputs");
        }
        ins.push_back(in);
        try_to_connect_new_in(in);
      }

      void set_out(TerminalBase *out) {
        if (outs.size() && tracing()) {
          print("Edge: ", name, " : has multiple outputs");
        }
        outs.push_back(out);
        try_to_connect_new_out(out);
      }

      void try_to_connect_new_in(Out<keyT, valueT> *in) const {
        for (auto out : outs)
          if (in && out) in->connect(out);
      }

      void try_to_connect_new_out(TerminalBase *out) const {
        assert(out->get_type() != TerminalBase::Type::Write);  // out must be an In<>
        for (auto in : ins)
          if (in && out) in->connect(out);
      }

      ~EdgeImpl() {
        if (ins.size() == 0 || outs.size() == 0) {
            std::cerr << "Edge: destroying edge pimpl ('" << name << "') with either in or out not "
                       "assigned --- graph may be incomplete"
                    << std::endl;
        }
      }
    };

    // We have a vector here to accomodate fusing multiple edges together
    // when connecting them all to a single terminal.
    mutable std::vector<std::shared_ptr<EdgeImpl>> p;  // Need shallow copy semantics

   public:
    typedef Out<keyT, valueT> output_terminal_type;
    typedef keyT key_type;
    typedef valueT value_type;
    static_assert(std::is_same<keyT, std::decay_t<keyT>>::value,
                  "Edge<keyT,valueT> assumes keyT is a non-decayable type");
    static_assert(std::is_same<valueT, std::decay_t<valueT>>::value,
                  "Edge<keyT,valueT> assumes valueT is a non-decayable type");
    static constexpr bool is_an_edge = true;

    Edge(const std::string name = "anonymous edge") : p(1) { p[0] = std::make_shared<EdgeImpl>(name); }

    template <typename... valuesT>
    Edge(const Edge<keyT, valuesT> &... edges) : p(0) {
      std::vector<Edge<keyT, valueT>> v = {edges...};
      for (auto &edge : v) {
        p.insert(p.end(), edge.p.begin(), edge.p.end());
      }
    }

    /// probes if this is already has at least one input
    bool live() const {
      bool result = false;
      for(const auto& edge: p) {
        if (!edge->ins.empty())
          return true;
      }
      return result;
    }

    void set_in(Out<keyT, valueT> *in) const {
      for (auto &edge : p) edge->set_in(in);
    }

    void set_out(TerminalBase *out) const {
      for (auto &edge : p) edge->set_out(out);
    }

    // this is currently just a hack, need to understand better whether this is a good idea
    Out<keyT, valueT> *in(size_t edge_index = 0, size_t terminal_index = 0) {
      return p.at(edge_index)->ins.at(terminal_index);
    }
  };


  // Make type of tuple of edges from type of tuple of terminals
  template <typename termsT>
  struct terminals_to_edges;
  template <typename... termsT>
  struct terminals_to_edges<std::tuple<termsT...>> {
    typedef std::tuple<typename termsT::edge_type...> type;
  };

  // Make type of tuple of output terminals from type of tuple of edges
  template <typename edgesT>
  struct edges_to_output_terminals;
  template <typename... edgesT>
  struct edges_to_output_terminals<std::tuple<edgesT...>> {
    typedef std::tuple<typename edgesT::output_terminal_type...> type;
  };


} // namespace ttg

#endif // TTG_EDGE_H
