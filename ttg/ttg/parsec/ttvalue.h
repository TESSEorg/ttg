#ifndef TTG_PARSEC_TTVALUE_H
#define TTG_PARSEC_TTVALUE_H

#include <type_traits>

#include "ttg/parsec/ttg_data_copy.h"

namespace ttg_parsec {

  /**
   * Base class for data to moved into, through, and out of
   * a task graph. By inheriting from this base class,
   * TTG is able to easily track the data and avoid some
   * of the copies otherwise necessary.
   */
  template<typename DerivedT>
  struct TTValue : private ttg_parsec::detail::ttg_data_copy_container_setter<ttg_parsec::detail::ttg_data_copy_t>
                 , public ttg_parsec::detail::ttg_data_copy_t {

    using derived_type = std::decay_t<DerivedT>;

    /* Constructor called with a pointer to the derived class object */
    TTValue()
    : ttg_data_copy_container_setter(this)
    , ttg_data_copy_t()
    { }

    /* default copy ctor */
    TTValue(const TTValue& v)
    : ttg_data_copy_container_setter(this)
    , ttg_data_copy_t(v)
    { }

    /* default move ctor */
    TTValue(TTValue&& v)
    : ttg_data_copy_container_setter(this)
    , ttg_data_copy_t(std::move(v))
    { }

    /* virtual mark destructor */
    virtual ~TTValue() = default;

    /* default copy operator */
    TTValue& operator=(const TTValue& v) {
      ttg_parsec::detail::ttg_data_copy_container() = this;
      ttg_data_copy_t::operator=(v);
      return *this;
    }

    /* default move operator */
    TTValue& operator=(TTValue&& v) {
      ttg_parsec::detail::ttg_data_copy_container() = this;
      ttg_data_copy_t::operator=(std::move(v));
      return *this;
    }

    virtual void* get_ptr() override final {
      return static_cast<DerivedT*>(this);
    }

    derived_type& get_derived() {
        return *static_cast<DerivedT*>(this);
    }

    const derived_type& get_derived() const {
        return *static_cast<DerivedT*>(this);
    }
  };

  namespace detail {

    template<typename T, typename Enabler = void>
    struct is_ttvalue_base : std::false_type {};

    template<typename T>
    struct is_ttvalue_base<T, std::is_base_of<TTValue<std::decay_t<T>>, std::decay_t<T>>>
    : std::true_type
    { };

    template<typename T>
    static constexpr const bool is_ttvalue_base_v = is_ttvalue_base<T>::value;

    template<typename ValueT>
    struct persistent_value_ref {
      using reference_type = ValueT;
      using value_type = std::decay_t<ValueT>;
      using lvalue_reference_type = std::add_lvalue_reference_t<std::remove_reference_t<ValueT>>;
      lvalue_reference_type value_ref;
    };
  } // namespace detail

  template<typename ValueT>
  inline auto persistent(ValueT&& value) {
    static_assert(std::is_base_of_v<TTValue<std::decay_t<ValueT>>, std::decay_t<ValueT>>,
                  "ttg::persistent can only be used on types derived from ttg::TTValue");
    return detail::persistent_value_ref<ValueT>{value};
  }

} // namespace ttg_parsec

#endif // TTG_PARSEC_TTVALUE_H