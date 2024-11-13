#ifndef TTG_UTIL_DOT_H
#define TTG_UTIL_DOT_H

#include <sstream>
#include <map>
#include <string>

#include "ttg/base/terminal.h"
#include "ttg/traverse.h"

namespace ttg {
  /// Prints the graph to a std::string in the format understood by GraphViz's dot program
  class Dot : private detail::Traverse {
    std::stringstream edges;
    std::map<const TTBase*, std::stringstream> tt_nodes;
    std::multimap<const TTBase *, const TTBase *> ttg_hierarchy;
    int cluster_cnt;
    bool disable_type;

   public:
    /// \param[in] disable_type disable_type controls whether to embed types into the DOT output;
    ///            set to `true` to reduce the amount of the output
    Dot(bool disable_type = false) : disable_type(disable_type){};

    // Insert backslash before characters that dot is interpreting
    std::string escape(const std::string &in) {
      std::stringstream s;
      for (char c : in) {
        if (c == '<' || c == '>' || c == '"' || c == '|')
          s << "\\" << c;
        else
          s << c;
      }
      return s.str();
    }

    // A unique name for the node derived from the pointer
    std::string nodename(const TTBase *op) {
      std::stringstream s;
      s << "n" << (void *)op;
      return s.str();
    }

    void build_ttg_hierarchy(const TTBase *tt) {
      if(nullptr == tt) {
        return;
      }
      auto search = ttg_hierarchy.find(tt->ttg_ptr());
      if(search == ttg_hierarchy.end()) {
        build_ttg_hierarchy(tt->ttg_ptr()); // make sure the parent is in the hierarchy
      }
      search = ttg_hierarchy.find(tt);
      if(search == ttg_hierarchy.end()) {
        ttg_hierarchy.insert( decltype(ttg_hierarchy)::value_type(tt->ttg_ptr(), tt) );
      }
    }

    void ttfunc(TTBase *tt) {
      std::string ttnm = nodename(tt);

      const TTBase *ttc = reinterpret_cast<const TTBase*>(tt);
      build_ttg_hierarchy(ttc);
      if(!tt->is_ttg()) {
        std::stringstream ttss;

        ttss << "        " << ttnm << " [shape=record,style=filled,fillcolor=gray90,label=\"{";

        size_t count = 0;
        if (tt->get_inputs().size() > 0) ttss << "{";
        for (auto in : tt->get_inputs()) {
          if (in) {
            if (count != in->get_index()) throw "ttg::Dot: lost count of ins";
            if (disable_type) {
              ttss << " <in" << count << ">"
                   << " " << escape(in->get_key_type_str()) << " " << escape(in->get_name());
            } else {
              ttss << " <in" << count << ">"
                   << " " << escape("<" + in->get_key_type_str() + "," + in->get_value_type_str() + ">") << " "
                   << escape(in->get_name());
           }
          } else {
            ttss << " <in" << count << ">"
                 << " unknown ";
          }
          count++;
          if (count < tt->get_inputs().size()) ttss << " |";
        }
        if (tt->get_inputs().size() > 0) ttss << "} |";

        ttss << tt->get_name() << " ";

        if (tt->get_outputs().size() > 0) ttss << " | {";

        count = 0;
        for (auto out : tt->get_outputs()) {
          if (out) {
            if (count != out->get_index()) throw "ttg::Dot: lost count of outs";
            if (disable_type) {
              ttss << " <out" << count << ">"
                   << " " << escape(out->get_key_type_str()) << " " << out->get_name();
            } else {
              ttss << " <out" << count << ">"
                   << " " << escape("<" + out->get_key_type_str() + "," + out->get_value_type_str() + ">") << " "
                   << out->get_name();
            }
          } else {
            ttss << " <out" << count << ">"
                 << " unknown ";
          }
          count++;
          if (count < tt->get_outputs().size()) ttss << " |";
        }

        if (tt->get_outputs().size() > 0) ttss << "}";

        ttss << " } \"];\n";

        auto search = tt_nodes.find(ttc);
        if( tt_nodes.end() == search ) {
          tt_nodes.insert( {ttc, std::move(ttss)} );
        } else {
          search->second << ttss.str();
        }
      } else {
        std::cout << ttnm << " is a TTG" << std::endl;
      }

      for (auto out : tt->get_outputs()) {
        if (out) {
          for (auto successor : out->get_connections()) {
            if (successor) {
              edges << ttnm << ":out" << out->get_index() << ":s -> " << nodename(successor->get_tt()) << ":in"
                    << successor->get_index() << ":n;\n";
            }
          }
        }
      }
    }

    void infunc(TerminalBase *in) {}

    void outfunc(TerminalBase *out) {}

    void tree_down(int level, const TTBase *node, std::stringstream &buf) {
      if(node == nullptr || node->is_ttg()) {
        if(nullptr != node) {
          buf << "subgraph cluster_" << cluster_cnt++ << " {\n";
        }
        auto children = ttg_hierarchy.equal_range(node);
        for(auto child = children.first; child != children.second; child++) {
          assert(child->first == node);
          tree_down(level+1, child->second, buf);
        }
        if(nullptr != node) {
          buf << "        label = \"" << node->get_name() << "\";\n";
          buf << "}\n";
        }
      } else {
        auto child = tt_nodes.find(node);
        if( child != tt_nodes.end()) {
          assert(child->first == node);
          buf << child->second.str();
        }
      }
    }

   public:
    /// @return string containing the graph specification in the format understood by GraphViz's dot program
    template <typename... TTBasePtrs>
    std::enable_if_t<(std::is_convertible_v<std::remove_const_t<std::remove_reference_t<TTBasePtrs>>, TTBase *> && ...),
                     std::string>
    operator()(TTBasePtrs &&... ops) {
      reset();
      std::stringstream buf;
      buf.str(std::string());
      buf.clear();

      edges.str(std::string());
      edges.clear();

      tt_nodes.clear();
      ttg_hierarchy.clear();

      buf << "digraph G {\n";
      buf << "        ranksep=1.5;\n";
      bool t = true;
      t &= (traverse(std::forward<TTBasePtrs>(ops)) && ... );

      cluster_cnt = 0;
      tree_down(0, nullptr, buf);

      buf << edges.str();
      buf << "}\n";

      reset();
      std::string result = buf.str();
      buf.str(std::string());
      buf.clear();

      return result;
    }
  };
}  // namespace ttg
#endif  // TTG_UTIL_DOT_H
