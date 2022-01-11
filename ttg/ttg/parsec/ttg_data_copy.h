#ifndef TTG_DATA_COPY_H
#define TTG_DATA_COPY_H

#include <utility>

#include <parsec.h>

/* Extension of PaRSEC's data copy. Note that we use the readers field
 * to facilitate the ref-counting of the data copy.*/
struct ttg_data_copy_t : public parsec_data_copy_t {

  ttg_data_copy_t()
  {
    /* TODO: do we need this construction? */
    //PARSEC_OBJ_CONSTRUCT(this, parsec_data_copy_t);
    this->readers = 1;
    this->push_task = nullptr;
  }

  /* mark destructor as virtual */
  virtual ~ttg_data_copy_t() = default;
};


/**
 * Extension of ttg_data_copy_t holding the actual value.
 * The virtual destructor will take care of destructing the value if
 * the destructor of ttg_data_copy_t base class is called.
 */
template<typename ValueT>
struct ttg_data_value_copy_t final : public ttg_data_copy_t {
  using value_type = std::decay_t<ValueT>;
  value_type m_value;

  template<typename T>
  ttg_data_value_copy_t(T&& value)
  : ttg_data_copy_t(), m_value(std::forward<T>(value))
  {
    this->device_private = const_cast<value_type*>(&m_value);
  }

  /* will destruct the value */
  virtual ~ttg_data_value_copy_t() = default;
};

#endif // TTG_DATA_COPY_H
