#ifndef TTG_UTIL_TRAVERSE_H
#define TTG_UTIL_TRAVERSE_H

#include <iostream>
#include <set>

#include "ttg/tt.h"
#include "ttg/util/meta.h"

namespace ttg {

  namespace detail {
    /// Traverses a graph of ops in depth-first manner following out edges
    class Traverse {
      std::set<OpBase *> seen;

      bool visited(OpBase *p) { return !seen.insert(p).second; }

     public:
      virtual void opfunc(OpBase *op) = 0;

      virtual void infunc(TerminalBase *in) = 0;

      virtual void outfunc(TerminalBase *out) = 0;

      void reset() { seen.clear(); }

      // Returns true if no null pointers encountered (i.e., if all
      // encountered terminals/operations are connected)
      bool traverse(OpBase *op) {
        if (!op) {
          std::cout << "ttg::Traverse: got a null op!\n";
          return false;
        }

        if (visited(op)) return true;

        bool status = true;

        opfunc(op);

      int count = 0;
      for (auto in : op->get_inputs()) {
          if (!in) {
              std::cout << "ttg::Traverse: got a null in!\n";
              status = false;
          } else {
              infunc(in);
              if (!in->is_connected()) {
                  std::cout << "ttg::Traverse: " << op->get_name() << " input terminal #" << count << " " << in->get_name() << " is not connected\n";
                  status = false;
              }
          }
          count++;
      }

      count = 0;
      for (auto out : op->get_outputs()) {
          if (!out) {
              std::cout << "ttg::Traverse: got a null out!\n";
              status = false;
          } else {
              outfunc(out);
              if (!out->is_connected()) {
                  std::cout << "ttg::Traverse: " << op->get_name() << " output terminal #" << count << " " << out->get_name() << " is not connected\n";
                  status = false;
              }
          }
          count++;
      }

        for (auto out : op->get_outputs()) {
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

      // converters to OpBase*
      static OpBase* to_OpBase_ptr(OpBase* op) { return op; }
      static OpBase* to_OpBase_ptr(const OpBase* op) {
        return const_cast<OpBase*>(op);
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
  /// @tparam OpVisitor A Callable type that visits each Op
  /// @tparam InVisitor A Callable type that visits each In terminal
  /// @tparam OutVisitor A Callable type that visits each Out terminal
  template <typename OpVisitor = detail::Traverse::null_visitor<OpBase>,
      typename InVisitor = detail::Traverse::null_visitor<TerminalBase>,
      typename OutVisitor = detail::Traverse::null_visitor<TerminalBase>>
  class Traverse : private detail::Traverse {
   public:
    static_assert(
        std::is_void<meta::void_t<decltype(std::declval<OpVisitor>()(std::declval<OpBase *>()))>>::value,
        "Traverse<OpVisitor,...>: OpVisitor(OpBase *op) must be a valid expression");
    static_assert(
        std::is_void<meta::void_t<decltype(std::declval<InVisitor>()(std::declval<TerminalBase *>()))>>::value,
        "Traverse<,InVisitor,>: InVisitor(TerminalBase *op) must be a valid expression");
    static_assert(
        std::is_void<meta::void_t<decltype(std::declval<OutVisitor>()(std::declval<TerminalBase *>()))>>::value,
        "Traverse<...,OutVisitor>: OutVisitor(TerminalBase *op) must be a valid expression");

    template <typename OpVisitor_ = detail::Traverse::null_visitor<OpBase>,
              typename InVisitor_ = detail::Traverse::null_visitor<TerminalBase>,
              typename OutVisitor_ = detail::Traverse::null_visitor<TerminalBase>>
    Traverse(OpVisitor_ &&op_v = OpVisitor_{}, InVisitor_ &&in_v = InVisitor_{}, OutVisitor_ &&out_v = OutVisitor_{})
        : op_visitor_(std::forward<OpVisitor_>(op_v))
        , in_visitor_(std::forward<InVisitor_>(in_v))
        , out_visitor_(std::forward<OutVisitor_>(out_v)){};

    const OpVisitor &op_visitor() const { return op_visitor_; }
    const InVisitor &in_visitor() const { return in_visitor_; }
    const OutVisitor &out_visitor() const { return out_visitor_; }

    /// Traverses graph starting at one or more Ops
    template <typename ... OpBasePtrs>
    std::enable_if_t<(std::is_convertible_v<std::remove_reference_t<OpBasePtrs>,OpBase*> && ...),bool>
        operator()(OpBase* op, OpBasePtrs && ... ops) {
      reset();
      bool result = traverse(op);
      result &= (traverse(std::forward<OpBasePtrs>(ops)) && ... );
      reset();
      return result;
    }

   private:
    OpVisitor op_visitor_;
    InVisitor in_visitor_;
    OutVisitor out_visitor_;

    void opfunc(OpBase *op) { op_visitor_(op); }

    void infunc(TerminalBase *in) { in_visitor_(in); }

    void outfunc(TerminalBase *out) { out_visitor_(out); }
  };

  namespace {
    auto trivial_1param_lambda = [](auto &&op) {};
  }
  template <typename OpVisitor = decltype(trivial_1param_lambda)&, typename InVisitor = decltype(trivial_1param_lambda)&, typename OutVisitor = decltype(trivial_1param_lambda)&>
  auto make_traverse(OpVisitor &&op_v = trivial_1param_lambda, InVisitor &&in_v = trivial_1param_lambda, OutVisitor &&out_v = trivial_1param_lambda) {
    return Traverse<std::remove_reference_t<OpVisitor>, std::remove_reference_t<InVisitor>,
                    std::remove_reference_t<OutVisitor>>{std::forward<OpVisitor>(op_v), std::forward<InVisitor>(in_v),
                                                         std::forward<OutVisitor>(out_v)};
  };

  /// verifies connectivity of the Graph
  static Traverse<> verify{};

  /// Prints the graph to std::cout in an ad hoc format
  static auto print_ttg = make_traverse(
      [](auto *op) {
        std::cout << "op: " << (void *)op << " " << op->get_name() << " numin " << op->get_inputs().size() << " numout "
                  << op->get_outputs().size() << std::endl;
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

#endif // TTG_UTIL_TRAVERSE_H
