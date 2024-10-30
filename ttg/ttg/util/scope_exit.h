#ifndef TTG_UTIL_SCOPE_EXIT_H
#define TTG_UTIL_SCOPE_EXIT_H

//
// N4189: Scoped Resource - Generic RAII Wrapper for the Standard Library
// Peter Sommerlad and Andrew L. Sandoval
// Adopted from https://github.com/tandasat/ScopedResource/tree/master
//

#include <type_traits>

namespace ttg::detail {
  template <typename EF>
  struct scope_exit
  {
    // construction
    explicit
    scope_exit(EF &&f)
    : exit_function(std::move(f))
    , execute_on_destruction{ true }
    { }

    // move
    scope_exit(scope_exit &&rhs)
    : exit_function(std::move(rhs.exit_function))
    , execute_on_destruction{ rhs.execute_on_destruction }
    {
      rhs.release();
    }

    // release
    ~scope_exit()
    {
      if (execute_on_destruction) this->exit_function();
    }

    void release()
    {
      this->execute_on_destruction = false;
    }

  private:
    scope_exit(scope_exit const &) = delete;
    void operator=(scope_exit const &) = delete;
    scope_exit& operator=(scope_exit &&) = delete;
    EF exit_function;
    bool execute_on_destruction; // exposition only
  };

  template <typename EF>
  auto make_scope_exit(EF &&exit_function)
  {
    return scope_exit<std::remove_reference_t<EF>>(std::forward<EF>(exit_function));
  }

} // namespace ttg::detail

#endif // TTG_UTIL_SCOPE_EXIT_H