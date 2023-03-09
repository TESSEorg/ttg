#ifndef TTG_DATA_COPY_H
#define TTG_DATA_COPY_H

#include <utility>
#include <limits>
#include <vector>
#include <iterator>
#include <atomic>
#include <type_traits>

#include <parsec.h>

#include "ttg/parsec/thread_local.h"
#include "ttg/util/span.h"


namespace ttg_parsec {

  namespace detail {

    template<typename T>
    struct ttg_data_copy_container_setter {
      ttg_data_copy_container_setter(T* ptr) {
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = ptr;
      }
    };

    /* Non-owning copy-tracking wrapper, accounting for N readers or 1 writer.
     * Also counts external references, which are not treated as
     * readers or writers but merely prevent the object from being
     * destroyed once no readers/writers exist.
     */
    struct ttg_data_copy_t : private ttg_data_copy_container_setter<ttg_data_copy_t> {

      /* special value assigned to parsec_data_copy_t::readers to mark the copy as
      * mutable, i.e., a task will modify it */
      static constexpr int mutable_tag = std::numeric_limits<int>::min();

      ttg_data_copy_t()
      : ttg_data_copy_container_setter(this)
      { }

      ttg_data_copy_t(const ttg_data_copy_t& c)
      : ttg_data_copy_container_setter(this)
      {
        /* we allow copying but do not copy any data over from the original
         * device copies will have to be allocated again
         * and it's a new object to reference */
      }

      ttg_data_copy_t(ttg_data_copy_t&& c)
      : ttg_data_copy_container_setter(this)
      , m_ptr(c.m_ptr)
      , m_next_task(c.m_next_task)
      , m_readers(c.m_readers)
      , m_refs(c.m_refs.load(std::memory_order_relaxed))
      , m_dev_data(std::move(c.m_dev_data))
      , m_single_dev_data(c.m_single_dev_data)
      , m_num_dev_data(c.m_num_dev_data)
      {
        c.m_num_dev_data = 0;
        c.m_readers = 0;
        c.m_single_dev_data = nullptr;
      }

      ttg_data_copy_t& operator=(ttg_data_copy_t&& c)
      {
        m_ptr = c.m_ptr;
        c.m_ptr = nullptr;
        m_next_task = c.m_next_task;
        c.m_next_task = nullptr;
        m_readers = c.m_readers;
        c.m_readers = 0;
        m_refs.store(c.m_refs.load(std::memory_order_relaxed), std::memory_order_relaxed);
        c.m_refs.store(0, std::memory_order_relaxed);
        m_dev_data = std::move(c.m_dev_data);
        m_single_dev_data = c.m_single_dev_data;
        c.m_single_dev_data = nullptr;
        m_num_dev_data = c.m_num_dev_data;
        c.m_num_dev_data = 0;
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = this;
        return *this;
      }

      ttg_data_copy_t& operator=(const ttg_data_copy_t& c) {
        /* we allow copying but do not copy any data over from the original
         * device copies will have to be allocated again
         * and it's a new object to reference */

        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = this;
        return *this;
      }

      /* mark destructor as virtual */
      virtual ~ttg_data_copy_t() = default;

      /* Returns true if the copy is mutable */
      bool is_mutable() const {
        return m_readers == mutable_tag;
      }

      /* Mark the copy as mutable */
      void mark_mutable() {
        m_readers = mutable_tag;
      }

      /* Increment the reader counter and return previous value
      * \tparam Atomic Whether to decrement atomically. Default: true
      */
      template<bool Atomic = true>
      int increment_readers() {
        if constexpr(Atomic) {
          //return parsec_atomic_fetch_inc_int32(&m_readers);
          std::atomic_ref<int32_t> a{m_readers};
          return a.fetch_add(1, std::memory_order_relaxed);
        } else {
          return m_readers++;
        }
      }

      /**
      * Reset the number of readers to read-only with a single reader.
      */
      void reset_readers() {
        m_readers = 1;
      }

      /* Decrement the reader counter and return previous value.
      * \tparam Atomic Whether to decrement atomically. Default: true
      */
      template<bool Atomic = true>
      int decrement_readers() {
        if constexpr(Atomic) {
          //return parsec_atomic_fetch_dec_int32(&m_readers);
          std::atomic_ref<int32_t> a{m_readers};
          return a.fetch_sub(1, std::memory_order_relaxed);
        } else {
          return m_readers--;
        }
      }

      /* Returns the number of readers if the copy is immutable, or \c mutable_tag
      * if the copy is mutable */
      int num_readers() const {
        return m_readers;
      }

      void *get_ptr() const {
        return m_ptr;
      }

      parsec_task_t* get_next_task() const {
        return m_next_task;
      }

      void set_next_task(parsec_task_t* task) {
        m_next_task = task;
      }

      int32_t add_ref() {
        return m_refs.fetch_add(1, std::memory_order_relaxed);
      }

      int32_t drop_ref() {
        return m_refs.fetch_sub(1, std::memory_order_relaxed);
      }

      bool has_ref() {
        return (m_refs.load(std::memory_order_relaxed) != 0);
      }

      int32_t num_ref() const {
        return m_refs.load(std::memory_order_relaxed);
      }

      /* manage device copies owned by this object
       * we only touch the vector if we have more than one copies to track
       * and otherwise use the single-element member.
       */
      using iterator = parsec_data_t**;

      void add_device_data(parsec_data_t* data) {
        // TODO: properly release again!
        PARSEC_OBJ_RETAIN(data);
        switch (m_num_dev_data) {
          case 0:
            m_single_dev_data = data;
            break;
          case 1:
            /* move single copy into vector and add new copy below */
            m_dev_data.push_back(m_single_dev_data);
            /* fall-through */
          default:
            /* store in multi-copy vector */
            m_dev_data.push_back(data);
            break;
        }
        m_num_dev_data++;
      }

      void remove_device_data(parsec_data_t* data) {
        if (m_num_dev_data == 1) {
          m_single_dev_data = nullptr;
        } else if (m_num_dev_data > 1) {
          auto it = std::find(m_dev_data.begin(), m_dev_data.end(), data);
          if (it != m_dev_data.end()) {
            m_dev_data.erase(it);
          }
        }
        --m_num_dev_data;
      }

      int num_dev_data() const {
        return m_num_dev_data;
      }

      iterator begin() {
        switch(m_num_dev_data) {
          // no device copies
          case 0: return end();
          case 1: return &m_single_dev_data;
          default: return m_dev_data.data();
        }
      }

      iterator end() {
        switch(m_num_dev_data) {
          case 0:
          case 1:
            return &(m_single_dev_data) + 1;
          default:
            return m_dev_data.data() + m_dev_data.size();
        }
      }

      using iovec_iterator = typename std::vector<ttg::iovec>::iterator;

      iovec_iterator iovec_begin() {
        return m_iovecs.begin();
      }

      iovec_iterator iovec_end() {
        return m_iovecs.end();
      }

      void iovec_reset() {
        m_iovecs.clear();
      }

      void iovec_add(const ttg::iovec& iov) {
        m_iovecs.push_back(iov);
      }

      ttg::span<ttg::iovec> iovec_span() {
        return ttg::span<ttg::iovec>(m_iovecs.data(), m_iovecs.size());
      }

      std::size_t iovec_count() const {
        return m_iovecs.size();
      }

#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_TTG_PROFILE_BACKEND)
      int64_t size;
      int64_t uid;
#endif
    protected:
      void          *m_ptr;
      parsec_task_t *m_next_task = nullptr;
      int32_t        m_readers  = 1;
      std::atomic<int32_t>  m_refs = 1; // number of entities referencing this copy (TTGs, external)

      std::vector<ttg::iovec> m_iovecs;

      std::vector<parsec_data_t*> m_dev_data;   //< used if there are multiple device copies
                                                  //  that belong to this object
      parsec_data_t *m_single_dev_data;           //< used if there is a single device copy
      int m_num_dev_data = 0;                   //< number of device copies
    };


    /**
    * Extension of ttg_data_copy_t holding the actual value.
    * The virtual destructor will take care of destructing the value if
    * the destructor of ttg_data_copy_t base class is called.
    */
    template<typename ValueT>
    struct ttg_data_value_copy_t final : public ttg_data_copy_t {
      using value_type = ValueT;
      value_type m_value;

      template<typename T>
      ttg_data_value_copy_t(T&& value)
      : ttg_data_copy_t()
      , m_value(std::forward<T>(value))
      {
        this->m_ptr = const_cast<value_type*>(&m_value);
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t(ttg_data_value_copy_t&& c)
        noexcept(std::is_nothrow_move_constructible_v<value_type>)
      : ttg_data_copy_t(std::move(c))
      , m_value(std::move(c.m_value))
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t(const ttg_data_value_copy_t& c)
        noexcept(std::is_nothrow_copy_constructible_v<value_type>)
      : ttg_data_copy_t(c)
      , m_value(c.m_value)
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t& operator=(ttg_data_value_copy_t&& c)
        noexcept(std::is_nothrow_move_assignable_v<value_type>)
      {
        ttg_data_copy_t::operator=(std::move(c));
        m_value = std::move(c.m_value);
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t& operator=(const ttg_data_value_copy_t& c)
        noexcept(std::is_nothrow_copy_assignable_v<value_type>)
      {
        ttg_data_copy_t::operator=(c);
        m_value = c.m_value;
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      value_type& operator*() {
        return m_value;
      }

      /* will destruct the value */
      virtual ~ttg_data_value_copy_t() = default;
    };

  } // namespace detail

} // namespace ttg_parsec

#endif // TTG_DATA_COPY_H