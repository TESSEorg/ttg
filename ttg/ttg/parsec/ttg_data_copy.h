#ifndef TTG_DATA_COPY_H
#define TTG_DATA_COPY_H

#include <utility>
#include <limits>
#include <vector>
#include <iterator>

#include <parsec.h>


namespace ttg_parsec {

  namespace detail {

    /* Extension of PaRSEC's data copy. Note that we use the readers field
    * to facilitate the ref-counting of the data copy.
    * TODO: create abstractions for all fields in parsec_data_copy_t that we access.
    */
    struct ttg_data_copy_t : public parsec_data_copy_t {
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      int64_t size;
      int64_t uid;
#endif

      /* special value assigned to parsec_data_copy_t::readers to mark the copy as
      * mutable, i.e., a task will modify it */
      static constexpr int mutable_tag = std::numeric_limits<int>::min();

      /* Returns true if the copy is mutable */
      bool is_mutable() const {
        return this->readers == mutable_tag;
      }

      /* Mark the copy as mutable */
      void mark_mutable() {
        this->readers = mutable_tag;
      }

      /* Increment the reader counter and return previous value
      * \tparam Atomic Whether to decrement atomically. Default: true
      */
      template<bool Atomic = true>
      int increment_readers() {
        if constexpr(Atomic) {
          return parsec_atomic_fetch_inc_int32(&this->readers);
        } else {
          return this->readers++;
        }
      }

      /**
      * Reset the number of readers to read-only with a single reader.
      */
      void reset_readers() {
        this->readers = 1;
      }

      /* Decrement the reader counter and return previous value.
      * \tparam Atomic Whether to decrement atomically. Default: true
      */
      template<bool Atomic = true>
      int decrement_readers() {
        if constexpr(Atomic) {
          return parsec_atomic_fetch_dec_int32(&this->readers);
        } else {
          return this->readers--;
        }
      }

      /* Returns the number of readers if the copy is immutable, or \c mutable_tag
      * if the copy is mutable */
      int num_readers() const {
        return this->readers;
      }

      ttg_data_copy_t()
      {
        /* TODO: do we need this construction? */
        PARSEC_OBJ_CONSTRUCT(this, parsec_data_copy_t);
        this->readers = 1;
        this->push_task = nullptr;
      }

      /* mark destructor as virtual */
      virtual ~ttg_data_copy_t() = default;



      /* manage device copies owned by this object
       * we only touch the vector if we have more than one copies to track
       * and otherwise use the single-element member.
       */
      using iterator = parsec_data_copy_t**;

      void add_device_copy(parsec_data_copy_t* copy) {
        // TODO: properly release again!
        PARSEC_OBJ_RETAIN(copy);
        switch (m_num_dev_copies) {
          case 0:
            m_single_dev_copy = copy;
            break;
          case 1:
            /* move single copy into vector and add new copy below */
            m_dev_copies.push_back(m_single_dev_copy);
            /* fall-through */
          default:
            /* store in multi-copy vector */
            m_dev_copies.push_back(copy);
            break;
        }
        m_num_dev_copies++;
      }

      int num_dev_copies() const {
        return m_num_dev_copies;
      }

      iterator begin() {
        switch(m_num_dev_copies) {
          // no device copies
          case 0: return end();
          case 1: return &m_single_dev_copy;
          default: return m_dev_copies.data();
        }
      }

      iterator end() {
        switch(m_num_dev_copies) {
          case 0:
          case 1:
            return &(m_single_dev_copy) + 1;
          default:
            return m_dev_copies.data() + m_dev_copies.size();
        }
      }

    private:
      std::vector<parsec_data_copy_t*> m_dev_copies;   //< used if there are multiple device copies
                                                       //  that belong to this object
      parsec_data_copy_t* m_single_dev_copy;           //< used if there is a single device copy
      int m_num_dev_copies = 0;                        //< number of device copies
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

  } // namespace detail

} // namespace ttg_parsec

#endif // TTG_DATA_COPY_H
