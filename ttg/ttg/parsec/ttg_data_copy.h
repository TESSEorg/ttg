#ifndef TTG_DATA_COPY_H
#define TTG_DATA_COPY_H

#include <utility>
#include <limits>

#include "ttg/parsec/task.h"

namespace ttg_parsec {

  namespace detail {

    /* Extension of PaRSEC's data copy. Note that we use the readers field
    * to facilitate the ref-counting of the data copy.
    * TODO: create abstractions for all fields in parsec_data_copy_t that we access.
    */
    struct ttg_data_copy_t : public parsec_data_copy_t {

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


    inline ttg_data_copy_t *find_copy_in_task(parsec_ttg_task_base_t *task, const void *ptr) {
      ttg_data_copy_t *res = nullptr;
      if (task == nullptr || ptr == nullptr) {
        return res;
      }
      for (int i = 0; i < task->data_count; ++i) {
        auto copy = static_cast<ttg_data_copy_t *>(task->parsec_task.data[i].data_in);
        if (NULL != copy && copy->device_private == ptr) {
          res = copy;
          break;
        }
      }
      return res;
    }

    inline bool add_copy_to_task(ttg_data_copy_t *copy, parsec_ttg_task_base_t *task) {
      if (task == nullptr || copy == nullptr) {
        return false;
      }

      if (MAX_PARAM_COUNT < task->data_count) {
        throw std::logic_error("Too many data copies, check MAX_PARAM_COUNT!");
      }

      task->parsec_task.data[task->data_count].data_in = copy;
      task->data_count++;
      return true;
    }

    inline void remove_data_copy(ttg_data_copy_t *copy, parsec_ttg_task_base_t *task) {
      int i;
      /* find and remove entry; copies are usually appended and removed, so start from back */
      for (i = task->data_count; i >= 0; --i) {
        if (copy == task->parsec_task.data[i].data_in) {
          break;
        }
      }
      if (i < 0) return;
      /* move all following elements one up */
      for (; i < task->data_count - 1; ++i) {
        task->parsec_task.data[i].data_in = task->parsec_task.data[i + 1].data_in;
      }
      /* null last element */
      task->parsec_task.data[i].data_in = nullptr;
      task->data_count--;
    }

    template <typename Value>
    inline ttg_data_copy_t *create_new_datacopy(Value &&value) {
      using value_type = std::decay_t<Value>;
      ttg_data_copy_t *copy = new ttg_data_value_copy_t<value_type>(std::forward<Value>(value));
      return copy;
    }

    inline void release_data_copy(ttg_data_copy_t *copy) {
      if (nullptr != copy->push_task) {
        /* Release the deferred task.
         * The copy was mutable and will be mutated by the released task,
         * so simply transfer ownership.
         */
        parsec_task_t *push_task = copy->push_task;
        copy->push_task = nullptr;
        parsec_ttg_task_base_t *deferred_op = (parsec_ttg_task_base_t *)push_task;
        deferred_op->release_task();
      } else {
        if (copy->is_mutable()) {
          /* current task mutated the data but there are no consumers so prepare
          * the copy to be freed below */
          copy->reset_readers();
        }

        int32_t readers = copy->num_readers();
        if (readers > 1) {
          /* potentially more than one reader, decrement atomically */
          readers = copy->decrement_readers();
        }
        /* if there was only one reader (the current task) we release the copy */
        if (1 == readers) {
          delete copy;
        }
      }
    }

    template <typename Value>
    inline ttg_data_copy_t *register_data_copy(ttg_data_copy_t *copy_in, parsec_ttg_task_base_t *task, bool readonly) {
      ttg_data_copy_t *copy_res = copy_in;
      bool replace = false;
      int32_t readers = copy_in->num_readers();

      assert(readers != 0);

      if (readonly && !copy_in->is_mutable()) {
        /* simply increment the number of readers */
        readers = copy_in->increment_readers();
      }

      if (readers == copy_in->mutable_tag) {
        /* someone is going to write into this copy -> we need to make a copy */
        copy_res = NULL;
        if (readonly) {
          /* we replace the copy in a deferred task if the copy will be mutated by
           * the deferred task and we are readonly.
           * That way, we can share the copy with other readonly tasks and release
           * the deferred task. */
          replace = true;
        }
      } else if (!readonly) {
        /* this task will mutate the data
         * check whether there are other readers already and potentially
         * defer the release of this task to give following readers a
         * chance to make a copy of the data before this task mutates it
         *
         * Try to replace the readers with a negative value that indicates
         * the value is mutable. If that fails we know that there are other
         * readers or writers already.
         *
         * NOTE: this check is not atomic: either there is a single reader
         *       (current task) or there are others, in which we case won't
         *       touch it.
         */
        if (1 == copy_in->num_readers()) {
          /**
           * no other readers, mark copy as mutable and defer the release
           * of the task
           */
          copy_in->mark_mutable();
          assert(nullptr == copy_in->push_task);
          assert(nullptr != task);
          copy_in->push_task = &task->parsec_task;
        } else {
          /* there are readers of this copy already, make a copy that we can mutate */
          copy_res = NULL;
        }
      }

      if (NULL == copy_res) {
        ttg_data_copy_t *new_copy = detail::create_new_datacopy(*static_cast<Value *>(copy_in->device_private));
        if (replace && nullptr != copy_in->push_task) {
          /* replace the task that was deferred */
          parsec_ttg_task_base_t *deferred_op = (parsec_ttg_task_base_t *)copy_in->push_task;
          new_copy->mark_mutable();
          /* replace the copy in the deferred task */
          for (int i = 0; i < deferred_op->data_count; ++i) {
            if (deferred_op->parsec_task.data[i].data_in == copy_in) {
              deferred_op->parsec_task.data[i].data_in = new_copy;
              break;
            }
          }
          copy_in->push_task = nullptr;
          deferred_op->release_task();
          copy_in->reset_readers();            // set the copy back to being read-only
          copy_in->increment_readers<false>(); // register as reader
          copy_res = copy_in;                  // return the copy we were passed
        } else {
          if (!readonly) {
            new_copy->mark_mutable();
          }
          copy_res = new_copy;  // return the new copy
        }
      }
      return copy_res;
    }


  } // namespace detail

} // namespace ttg_parsec

#endif // TTG_DATA_COPY_H
