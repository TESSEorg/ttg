#ifndef TTG_DATA_COPY_H
#define TTG_DATA_COPY_H

#include <utility>
#include <limits>
#include <vector>
#include <iterator>
#include <atomic>
#include <type_traits>

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include <cuda_runtime.h>
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT

#include <parsec.h>

#include "ttg/parsec/thread_local.h"
#include "ttg/parsec/parsec-ext.h"
#include "ttg/util/span.h"


namespace ttg_parsec {

  namespace detail {

    // fwd-decl
    struct ttg_data_copy_t;

    /* Wrapper managing the relationship between a ttg data copy and the parsec_data_t object */
    struct ttg_parsec_data_wrapper_t {

    protected:
      using parsec_data_ptr = std::unique_ptr<parsec_data_t, decltype(&parsec_data_destroy)>;

      ttg_data_copy_t *m_ttg_copy = nullptr;
      parsec_data_ptr m_data;

      friend ttg_data_copy_t;

      static parsec_data_t* create_parsec_data(void *ptr, size_t size) {
        parsec_data_t *data = parsec_data_create_with_type(nullptr, 0, ptr, size,
                                                          parsec_datatype_int8_t);
        data->device_copies[0]->flags |= PARSEC_DATA_FLAG_PARSEC_MANAGED;
        data->device_copies[0]->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
        data->device_copies[0]->version = 1;
        return data;
      }

      parsec_data_t* parsec_data() {
        return m_data.get();
      }

      const parsec_data_t* parsec_data() const {
        return m_data.get();
      }

      static void delete_parsec_data(parsec_data_t *data) {
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
        if (data->device_copies[0]->flags & TTG_PARSEC_DATA_FLAG_REGISTERED) {
          // register the memory for faster access
          cudaError_t status;
          status = cudaHostUnregister(data->device_copies[0]->device_private);
          assert(cudaSuccess == status);
          data->device_copies[0]->flags ^= TTG_PARSEC_DATA_FLAG_REGISTERED;
      }
#endif // PARSEC_HAVE_DEV_CUDA_SUPPORT
        parsec_data_destroy(data);
      }

      static void delete_null_parsec_data(parsec_data_t *) {
        // nothing to be done, only used for nullptr
      }

    protected:

      /* remove the the data from the owning data copy */
      void remove_from_owner();

      /* add the data to the owning data copy */
      void reset_parsec_data(void *ptr, size_t size);

      ttg_parsec_data_wrapper_t();

      ttg_parsec_data_wrapper_t(const ttg_parsec_data_wrapper_t& other) = delete;

      ttg_parsec_data_wrapper_t(ttg_parsec_data_wrapper_t&& other);

      ttg_parsec_data_wrapper_t& operator=(const ttg_parsec_data_wrapper_t& other) = delete;

      ttg_parsec_data_wrapper_t& operator=(ttg_parsec_data_wrapper_t&& other);

      virtual ~ttg_parsec_data_wrapper_t();

      /* set a new owning data copy object */
      void set_owner(ttg_data_copy_t& new_copy) {
        m_ttg_copy = &new_copy;
      }
    };


    /* templated to break cyclic dependency with ttg_data_copy_container */
    template<typename T = ttg_data_copy_t>
    struct ttg_data_copy_container_setter {
      ttg_data_copy_container_setter(T* ptr) {
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = ptr;
      }
    };

    /* special type: stores a pointer to the ttg_data_copy_t. This is necessary
     * because ttg_data_copy_t has virtual functions so we cannot cast from parsec_data_copy_t
     * to ttg_data_copy_t (offsetof is not supported for virtual classes).
     * The self pointer is a back-pointer to the ttg_data_copy_t. */
    struct ttg_data_copy_self_t {
      parsec_list_item_t super;
      ttg_data_copy_t *self;
      ttg_data_copy_self_t(ttg_data_copy_t* dc)
      : self(dc)
      {
        PARSEC_OBJ_CONSTRUCT(&super, parsec_list_item_t);
      }
    };

    /* Non-owning copy-tracking wrapper, accounting for N readers or 1 writer.
     * Also counts external references, which are not treated as
     * readers or writers but merely prevent the object from being
     * destroyed once no readers/writers exist.
     */
    struct ttg_data_copy_t : public ttg_data_copy_self_t {

      /* special value assigned to parsec_data_copy_t::readers to mark the copy as
      * mutable, i.e., a task will modify it */
      static constexpr int mutable_tag = std::numeric_limits<int>::min();

      ttg_data_copy_t()
      : ttg_data_copy_self_t(this)
      { }

      ttg_data_copy_t(const ttg_data_copy_t& c)
      : ttg_data_copy_self_t(this)
      {
        /* we allow copying but do not copy any data over from the original
         * device copies will have to be allocated again
         * and it's a new object to reference */
      }

      ttg_data_copy_t(ttg_data_copy_t&& c)
      : ttg_data_copy_self_t(this)
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

        foreach_wrapper([&](ttg_parsec_data_wrapper_t* data){
          data->set_owner(*this);
        });
      }

      ttg_data_copy_t& operator=(ttg_data_copy_t&& c)
      {
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

        /* move all data to the new owner */
        foreach_wrapper([&](ttg_parsec_data_wrapper_t* data){
          data->set_owner(*this);
        });
        return *this;
      }

      ttg_data_copy_t& operator=(const ttg_data_copy_t& c) {
        /* we allow copying but do not copy any data over from the original
         * device copies will have to be allocated again
         * and it's a new object to reference */

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
          return parsec_atomic_fetch_inc_int32(&m_readers);
//          std::atomic_ref<int32_t> a{m_readers};
//          return a.fetch_add(1, std::memory_order_relaxed);
        } else {
          return m_readers++;
        }
      }

      /**
      * Reset the number of readers to read-only with a single reader.
      */
      void reset_readers() {
        if (mutable_tag == m_readers) {
          m_readers = 1;
        }
      }

      /* Decrement the reader counter and return previous value.
      * \tparam Atomic Whether to decrement atomically. Default: true
      */
      template<bool Atomic = true>
      int decrement_readers() {
        if constexpr(Atomic) {
          return parsec_atomic_fetch_dec_int32(&m_readers);
//          std::atomic_ref<int32_t> a{m_readers};
//          return a.fetch_sub(1, std::memory_order_relaxed);
        } else {
          return m_readers--;
        }
      }

      /* Returns the number of readers if the copy is immutable, or \c mutable_tag
      * if the copy is mutable */
      int num_readers() const {
        return m_readers;
      }

      /* Returns the pointer to the user data wrapped by the the copy object */
      virtual void* get_ptr() = 0;

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

      /* increment the version of the current copy */
      void inc_current_version() {
        //std::cout << "data-copy " << this << " inc_current_version " << " count " << m_num_dev_data << std::endl;
        foreach_parsec_data([](parsec_data_t* data){
          assert(data->device_copies[0] != nullptr);
          data->device_copies[0]->version++;
        });
      }

      void transfer_ownership(int access, int device = 0) {
        foreach_parsec_data([&](parsec_data_t* data){
          parsec_data_transfer_ownership_to_copy(data, device, access);
        });
      }

      /* manage device copies owned by this object
       * we only touch the vector if we have more than one copies to track
       * and otherwise use the single-element member.
       */
      using iterator = ttg_parsec_data_wrapper_t**;

      void add_device_data(ttg_parsec_data_wrapper_t* data) {
        // TODO: REMOVE AFTER DEBUG
        //assert(m_num_dev_data == 0);
        switch (m_num_dev_data) {
          case 0:
            m_single_dev_data = data;
            break;
          case 1:
            /* move single copy into vector and add new copy below */
            m_dev_data.push_back(m_single_dev_data);
            m_single_dev_data = nullptr;
            /* fall-through */
          default:
            /* store in multi-copy vector */
            m_dev_data.push_back(data);
            break;
        }
        //std::cout << "data-copy " << this << " add data " << data << " count " << m_num_dev_data << std::endl;
        m_num_dev_data++;
      }

      void remove_device_data(ttg_parsec_data_wrapper_t* data) {
        //std::cout << "data-copy " << this << " remove data " << data << " count " << m_num_dev_data << std::endl;
        if (m_num_dev_data == 0) {
          /* this may happen if we're integrated into the object and have been moved */
          return;
        }
        if (m_num_dev_data == 1) {
          assert(m_single_dev_data == data);
          m_single_dev_data = nullptr;
        } else if (m_num_dev_data > 1) {
          auto it = std::find(m_dev_data.begin(), m_dev_data.end(), data);
          if (it != m_dev_data.end()) {
            m_dev_data.erase(it);
          }
        }
        --m_num_dev_data;
        /* make single-entry if needed */
        if (m_num_dev_data == 1) {
          m_single_dev_data = m_dev_data[0];
          m_dev_data.clear();
        }
        // TODO: REMOVE AFTER DEBUG
        //assert(m_num_dev_data == 0);
      }

      int num_dev_data() const {
        return m_num_dev_data;
      }

      template<typename Fn>
      void foreach_wrapper(Fn&& fn) {
        if (m_num_dev_data == 1) {
          fn(m_single_dev_data);
        } else if (m_num_dev_data > 1) {
          std::for_each(m_dev_data.begin(), m_dev_data.end(), fn);
        }
      }

      template<typename Fn>
      void foreach_parsec_data(Fn&& fn) {
        if (m_num_dev_data == 1) {
          if (m_single_dev_data->parsec_data()) {
            fn(m_single_dev_data->parsec_data());
          }
        } else if (m_num_dev_data > 1) {
          std::for_each(m_dev_data.begin(), m_dev_data.end(),
            [&](ttg_parsec_data_wrapper_t* data){
              if (data->parsec_data()) {
                fn(data->parsec_data());
              }
            }
          );
        }
      }


#if 0
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
#endif // 0

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
      parsec_task_t *m_next_task = nullptr;
      int32_t        m_readers  = 1;
      std::atomic<int32_t>  m_refs = 1;                     //< number of entities referencing this copy (TTGs, external)

      std::vector<ttg::iovec> m_iovecs;

      std::vector<ttg_parsec_data_wrapper_t*> m_dev_data;   //< used if there are multiple device copies
                                                            //  that belong to this object
      ttg_parsec_data_wrapper_t *m_single_dev_data;         //< used if there is a single device copy
      int m_num_dev_data = 0;                               //< number of device copies
    };


    /**
    * Extension of ttg_data_copy_t holding the actual value.
    * The virtual destructor will take care of destructing the value if
    * the destructor of ttg_data_copy_t base class is called.
    */
    template<typename ValueT>
    struct ttg_data_value_copy_t final : private ttg_data_copy_container_setter<ttg_data_copy_t>
                                       , public ttg_data_copy_t {
      using value_type = ValueT;
      value_type m_value;

      template<typename T>
      ttg_data_value_copy_t(T&& value)
      : ttg_data_copy_container_setter(this)
      , ttg_data_copy_t()
      , m_value(std::forward<T>(value))
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t(ttg_data_value_copy_t&& c)
        noexcept(std::is_nothrow_move_constructible_v<value_type>)
      : ttg_data_copy_container_setter(this)
      , ttg_data_copy_t(std::move(c))
      , m_value(std::move(c.m_value))
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t(const ttg_data_value_copy_t& c)
        noexcept(std::is_nothrow_copy_constructible_v<value_type>)
      : ttg_data_copy_container_setter(this)
      , ttg_data_copy_t(c)
      , m_value(c.m_value)
      {
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t& operator=(ttg_data_value_copy_t&& c)
        noexcept(std::is_nothrow_move_assignable_v<value_type>)
      {
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = this;
        ttg_data_copy_t::operator=(std::move(c));
        m_value = std::move(c.m_value);
        /* reset the container tracker */
        ttg_data_copy_container() = nullptr;
      }

      ttg_data_value_copy_t& operator=(const ttg_data_value_copy_t& c)
        noexcept(std::is_nothrow_copy_assignable_v<value_type>)
      {
        /* set the container ptr here, will be reset in the the ttg_data_value_copy_t ctor */
        ttg_data_copy_container() = this;
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

      virtual void* get_ptr() override final {
        return &m_value;
      }
    };

    /**
     * definition of ttg_parsec_data_wrapper_t members that depend on ttg_data_copy_t
     */

    void ttg_parsec_data_wrapper_t::remove_from_owner() {
      if (nullptr != m_ttg_copy) {
        m_ttg_copy->remove_device_data(this);
        m_ttg_copy = nullptr;
      }
    }

    void ttg_parsec_data_wrapper_t::reset_parsec_data(void *ptr, size_t size) {
      if (ptr == m_data.get()) return;

      if (nullptr == ptr) {
        m_data = parsec_data_ptr(nullptr, &delete_null_parsec_data);
      } else {
        m_data = parsec_data_ptr(create_parsec_data(ptr, size), &delete_parsec_data);
      }
    }

    ttg_parsec_data_wrapper_t::ttg_parsec_data_wrapper_t()
    : m_data(nullptr, delete_null_parsec_data)
    , m_ttg_copy(detail::ttg_data_copy_container())
    {
      if (m_ttg_copy) {
        m_ttg_copy->add_device_data(this);
      }
    }
    ttg_parsec_data_wrapper_t::ttg_parsec_data_wrapper_t(ttg_parsec_data_wrapper_t&& other)
    : m_data(std::move(other.m_data))
    , m_ttg_copy(detail::ttg_data_copy_container())
    {
      /* the ttg_data_copy may have moved us already */
      if (other.m_ttg_copy != m_ttg_copy) {
        // try to remove the old buffer from the *old* ttg_copy
        other.remove_from_owner();

        // register with the new ttg_copy
        if (nullptr != m_ttg_copy) {
          m_ttg_copy->add_device_data(this);
        }
      }
    }

    ttg_parsec_data_wrapper_t& ttg_parsec_data_wrapper_t::operator=(ttg_parsec_data_wrapper_t&& other) {
      m_data = std::move(other.m_data);
      /* check whether the owning ttg_data_copy has already moved us */
      if (other.m_ttg_copy != m_ttg_copy) {
        /* remove from old ttg copy */
        other.remove_from_owner();

        if (nullptr != m_ttg_copy) {
          /* register with the new ttg_copy */
          m_ttg_copy->add_device_data(this);
        }
      }
      return *this;
    }


    ttg_parsec_data_wrapper_t::~ttg_parsec_data_wrapper_t() {
      if (nullptr != m_ttg_copy) {
        m_ttg_copy->remove_device_data(this);
        m_ttg_copy = nullptr;
      }
    }


  } // namespace detail

} // namespace ttg_parsec

#endif // TTG_DATA_COPY_H
