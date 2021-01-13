#ifndef TTG_UTIL_DOT_H
#define TTG_UTIL_DOT_H

#include <string>
#include <sstream>

#include "base/terminal.h"
#include "util/traverse.h"

namespace ttg {
  /// Prints the graph to a std::string in the format understood by GraphViz's dot program
  class Dot : private detail::Traverse {
    std::stringstream buf;

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
    std::string nodename(const base::OpBase *op) {
      std::stringstream s;
      s << "n" << (void *)op;
      return s.str();
    }

    void opfunc(base::OpBase *op) {
      std::string opnm = nodename(op);

      buf << "        " << opnm << " [shape=record,style=filled,fillcolor=gray90,label=\"{";

      size_t count = 0;
      if (op->get_inputs().size() > 0) buf << "{";
      for (auto in : op->get_inputs()) {
        if (in) {
          if (count != in->get_index()) throw "ttg::Dot: lost count of ins";
          buf << " <in" << count << ">"
              << " " << escape("<" + in->get_key_type_str() + "," + in->get_value_type_str() + ">") << " "
              << escape(in->get_name());
        } else {
          buf << " <in" << count << ">"
              << " unknown ";
        }
        count++;
        if (count < op->get_inputs().size()) buf << " |";
      }
      if (op->get_inputs().size() > 0) buf << "} |";

      buf << op->get_name() << " ";

      if (op->get_outputs().size() > 0) buf << " | {";

      count = 0;
      for (auto out : op->get_outputs()) {
        if (out) {
          if (count != out->get_index()) throw "ttg::Dot: lost count of outs";
          buf << " <out" << count << ">"
              << " " << escape("<" + out->get_key_type_str() + "," + out->get_value_type_str() + ">") << " "
              << out->get_name();
        } else {
          buf << " <out" << count << ">"
              << " unknown ";
        }
        count++;
        if (count < op->get_outputs().size()) buf << " |";
      }

      if (op->get_outputs().size() > 0) buf << "}";

      buf << " } \"];\n";

      for (auto out : op->get_outputs()) {
        if (out) {
          for (auto successor : out->get_connections()) {
            if (successor) {
              buf << opnm << ":out" << out->get_index() << ":s -> " << nodename(successor->get_op()) << ":in"
                  << successor->get_index() << ":n;\n";
            }
          }
        }
      }
    }

    void infunc(base::TerminalBase *in) {}

    void outfunc(base::TerminalBase *out) {}

   public:
    /// @return string containing the graph specification in the format understood by GraphViz's dot program
    template <typename... OpBasePtrs>
    std::enable_if_t<(std::is_convertible_v<std::remove_const_t<std::remove_reference_t<OpBasePtrs>>,base::OpBase *> && ...),
                     std::string>
    operator()(OpBasePtrs &&... ops) {
      reset();
      buf.str(std::string());
      buf.clear();

      buf << "digraph G {\n";
      buf << "        ranksep=1.5;\n";
      bool t = true;
      t &= (traverse(std::forward<OpBasePtrs>(ops)) && ... );
      buf << "}\n";

      reset();
      std::string result = buf.str();
      buf.str(std::string());
      buf.clear();

      return result;
    }
  };
} // namespace ttg
#endif // TTG_UTIL_DOT_H
