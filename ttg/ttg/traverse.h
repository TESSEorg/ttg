#ifndef TTG_TRAVERSE_H
#define TTG_TRAVERSE_H

#include <iostream>
#include <set>

#include "ttg/tt.h"
#include "ttg/util/meta.h"

namespace ttg {

  namespace detail {
    /// Traverses a graph of TTs in depth-first manner following out edges
    class Traverse {
      std::set<TTBase *> seen;

      bool visited(TTBase *p) { return !seen.insert(p).second; }

     public:
      virtual void ttfunc(TTBase *tt) = 0;

      virtual void infunc(TerminalBase *in) = 0;

      virtual void outfunc(TerminalBase *out) = 0;

      void reset() { seen.clear(); }

      // Returns true if no null pointers encountered (i.e., if all
      // encountered terminals/operations are connected)
      bool traverse(TTBase *tt) {
        if (!tt) {
          std::cout << "ttg::Traverse: got a null op!\n";
          return false;
        }

        if (visited(tt)) return true;

        bool status = true;

        ttfunc(tt);

        int count = 0;
        for (auto in : tt->get_inputs()) {
          if (!in) {
            std::cout << "ttg::Traverse: got a null in!\n";
            status = false;
          } else {
            infunc(in);
            if (!in->is_connected()) {
              std::cout << "ttg::Traverse: " << tt->get_name() << " input terminal #" << count << " " << in->get_name()
                        << " is not connected\n";
              status = false;
            }
          }
          count++;
        }

      for (auto in : tt->get_inputs()) {
        if (in) {
          for (auto predecessor : in->get_predecessors()) {
            if (!predecessor) {
              std::cout << "ttg::Traverse: got a null predecessor!\n";
              status = false;
            } else {
              status = status && traverse(predecessor->get_tt());
            }
          }
        }
      }

      count = 0;
      for (auto out : tt->get_outputs()) {
          if (!out) {
            std::cout << "ttg::Traverse: got a null out!\n";
            status = false;
          } else {
            outfunc(out);
            if (!out->is_connected()) {
              std::cout << "ttg::Traverse: " << tt->get_name() << " output terminal #" << count << " "
                        << out->get_name() << " is not connected\n";
              status = false;
            }
          }
          count++;
        }

        for (auto out : tt->get_outputs()) {
          if (out) {
            for (auto successor : out->get_connections()) {
              if (!successor) {
                std::cout << "ttg::Traverse: got a null successor!\n";
                status = false;
              } else {
                status = status && traverse(successor->get_tt());
              }
            }
          }
        }

        return status;
      }

      template <typename TT>
      std::enable_if_t<std::is_base_of_v<TTBase, TT> && !std::is_same_v<TT, TTBase>,
                       bool>
      traverse(TT* tt) {
        return traverse(static_cast<TTBase*>(tt));
      }

      template <typename TT>
      std::enable_if_t<std::is_base_of_v<TTBase, TT>,
                       bool>
      traverse(const std::shared_ptr<TTBase>& tt) {
        return traverse(tt.get());
      }

      template <typename TT, typename Deleter>
      std::enable_if_t<std::is_base_of_v<TTBase, TT>,
                       bool>
      traverse(const std::unique_ptr<TT, Deleter>& tt) {
        return traverse(tt.get());
      }

      /// visitor that does nothing
      /// @tparam Visitable any type
      template <typename Visitable>
      struct null_visitor {
        /// visits a non-const Visitable object
        void operator()(Visitable*) {};
        /// visits a const Visitable object
        void operator()(const Visitable*) {};
      };

    };
  }  // namespace detail

  /// @brief Traverses a graph of ops in depth-first manner following out edges
  /// @tparam TTVisitor A Callable type that visits each TT
  /// @tparam InVisitor A Callable type that visits each In terminal
  /// @tparam OutVisitor A Callable type that visits each Out terminal
  template <typename TTVisitor = detail::Traverse::null_visitor<TTBase>,
      typename InVisitor = detail::Traverse::null_visitor<TerminalBase>,
      typename OutVisitor = detail::Traverse::null_visitor<TerminalBase>>
  class Traverse : private detail::Traverse {
   public:
    static_assert(
        std::is_void_v<meta::void_t<decltype(std::declval<TTVisitor>()(std::declval<TTBase *>()))>>,
        "Traverse<TTVisitor,...>: TTVisitor(TTBase *op) must be a valid expression");
    static_assert(
        std::is_void_v<meta::void_t<decltype(std::declval<InVisitor>()(std::declval<TerminalBase *>()))>>,
        "Traverse<,InVisitor,>: InVisitor(TerminalBase *op) must be a valid expression");
    static_assert(
        std::is_void_v<meta::void_t<decltype(std::declval<OutVisitor>()(std::declval<TerminalBase *>()))>>,
        "Traverse<...,OutVisitor>: OutVisitor(TerminalBase *op) must be a valid expression");

    template <typename TTVisitor_ = detail::Traverse::null_visitor<TTBase>,
              typename InVisitor_ = detail::Traverse::null_visitor<TerminalBase>,
              typename OutVisitor_ = detail::Traverse::null_visitor<TerminalBase>>
    Traverse(TTVisitor_ &&tt_v = TTVisitor_{}, InVisitor_ &&in_v = InVisitor_{}, OutVisitor_ &&out_v = OutVisitor_{})
        : tt_visitor_(std::forward<TTVisitor_>(tt_v))
        , in_visitor_(std::forward<InVisitor_>(in_v))
        , out_visitor_(std::forward<OutVisitor_>(out_v)){};

    const TTVisitor &tt_visitor() const { return tt_visitor_; }
    const InVisitor &in_visitor() const { return in_visitor_; }
    const OutVisitor &out_visitor() const { return out_visitor_; }

    /// Traverses graph starting at one or more TTs
    template <typename TTBasePtr, typename ... TTBasePtrs>
    std::enable_if_t<std::is_base_of_v<TTBase, std::decay_t<decltype(*(std::declval<TTBasePtr>()))>> && (std::is_base_of_v<TTBase, std::decay_t<decltype(*(std::declval<TTBasePtrs>()))>> && ...),
                     bool>
        operator()(
        TTBasePtr&& op, TTBasePtrs && ... ops) {
      reset();
      bool result = traverse(op);
      result &= (traverse(std::forward<TTBasePtrs>(ops)) && ... );
      reset();
      return result;
    }

   private:
    TTVisitor tt_visitor_;
    InVisitor in_visitor_;
    OutVisitor out_visitor_;

    void ttfunc(TTBase *tt) { tt_visitor_(tt); }

    void infunc(TerminalBase *in) { in_visitor_(in); }

    void outfunc(TerminalBase *out) { out_visitor_(out); }
  };

  namespace {
    auto trivial_1param_lambda = [](auto &&op) {};
  }
  template <typename TTVisitor = decltype(trivial_1param_lambda)&, typename InVisitor = decltype(trivial_1param_lambda)&, typename OutVisitor = decltype(trivial_1param_lambda)&>
  auto make_traverse(TTVisitor &&tt_v = trivial_1param_lambda, InVisitor &&in_v = trivial_1param_lambda, OutVisitor &&out_v = trivial_1param_lambda) {
    return Traverse<std::remove_reference_t<TTVisitor>, std::remove_reference_t<InVisitor>,
                    std::remove_reference_t<OutVisitor>>{std::forward<TTVisitor>(tt_v), std::forward<InVisitor>(in_v),
                                                         std::forward<OutVisitor>(out_v)};
  };

  /// verifies connectivity of the Graph
  static Traverse<> verify{};

  /// Prints the graph to std::cout in an ad hoc format
  static auto print_ttg = make_traverse(
      [](auto *tt) {
        std::cout << "tt: " << (void *)tt << " " << tt->get_name() << " numin " << tt->get_inputs().size() << " numout "
                  << tt->get_outputs().size() << std::endl;
      },
      [](auto *in) {
        std::cout << "  in: " << in->get_index() << " " << in->get_name() << " " << in->get_key_type_str() << " "
                  << in->get_value_type_str() << std::endl;
      },
      [](auto *out) {
        std::cout << " out: " << out->get_index() << " " << out->get_name() << " " << out->get_key_type_str() << " "
                  << out->get_value_type_str() << std::endl;
      });


} // namespace ttg

#endif // TTG_TRAVERSE_H
