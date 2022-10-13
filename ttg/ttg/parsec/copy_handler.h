#ifndef TTG_PARSEC_COPY_HANDLER_H
#define TTG_PARSEC_COPY_HANDLER_H


#include "ttg/runtimes.h"
#include "ttg/func.h"

#include "ttg/parsec/ttg_data_copy.h"
#include "ttg/parsec/vars.h"


/**
 * The PaRSEC backend tracks data copies so we make a copy of the data
 * if the data is not being tracked yet or if the data is not const, i.e.,
 * the user may mutate the data after it was passed to send/broadcast.
 */
template <>
struct ttg::detail::value_copy_handler<ttg::Runtime::PaRSEC> {
 private:
  ttg_parsec::detail::ttg_data_copy_t *copy_to_remove = nullptr;

 public:
  ~value_copy_handler() {
    if (nullptr != copy_to_remove) {
      ttg_parsec::detail::remove_data_copy(copy_to_remove, ttg_parsec::detail::parsec_ttg_caller);
      ttg_parsec::detail::release_data_copy(copy_to_remove);
    }
  }

  template <typename Value>
  inline Value &&operator()(Value &&value) {
    if (nullptr == ttg_parsec::detail::parsec_ttg_caller) {
      ttg::print("ERROR: ttg_send or ttg_broadcast called outside of a task!\n");
    }
    ttg_parsec::detail::ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::find_copy_in_task(ttg_parsec::detail::parsec_ttg_caller, &value);
    Value *value_ptr = &value;
    if (nullptr == copy) {
      /**
       * the value is not known, create a copy that we can track
       * depending on Value, this uses either the copy or move constructor
       */
      copy = ttg_parsec::detail::create_new_datacopy(std::forward<Value>(value));
      bool inserted = ttg_parsec::detail::add_copy_to_task(copy, ttg_parsec::detail::parsec_ttg_caller);
      assert(inserted);
      value_ptr = reinterpret_cast<Value *>(copy->device_private);
      copy_to_remove = copy;
    } else {
      /* this copy won't be modified anymore so mark it as read-only */
      copy->reset_readers();
    }
    return std::move(*value_ptr);
  }

  template <typename Value>
  inline const Value &operator()(const Value &value) {
    if (nullptr == ttg_parsec::detail::parsec_ttg_caller) {
      ttg::print("ERROR: ttg_send or ttg_broadcast called outside of a task!\n");
    }
    ttg_parsec::detail::ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::find_copy_in_task(ttg_parsec::detail::parsec_ttg_caller, &value);
    const Value *value_ptr = &value;
    if (nullptr == copy) {
      /**
       * the value is not known, create a copy that we can track
       * depending on Value, this uses either the copy or move constructor
       */
      copy = ttg_parsec::detail::create_new_datacopy(value);
      bool inserted = ttg_parsec::detail::add_copy_to_task(copy, ttg_parsec::detail::parsec_ttg_caller);
      assert(inserted);
      value_ptr = reinterpret_cast<Value *>(copy->device_private);
      copy_to_remove = copy;
    }
    return *value_ptr;
  }

  /* we have to make a copy of non-const data as the user may modify it after
   * send/broadcast */
  template <typename Value, typename Enabler = std::enable_if_t<!std::is_const_v<Value>>>
  inline Value &operator()(Value &value) {
    if (nullptr == ttg_parsec::detail::parsec_ttg_caller) {
      ttg::print("ERROR: ttg_send or ttg_broadcast called outside of a task!\n");
    }
    /* the value is not known, create a copy that we can track */
    ttg_parsec::detail::ttg_data_copy_t *copy;
    copy = ttg_parsec::detail::create_new_datacopy(value);
    bool inserted = ttg_parsec::detail::add_copy_to_task(copy, ttg_parsec::detail::parsec_ttg_caller);
    assert(inserted);
    Value *value_ptr = reinterpret_cast<Value *>(copy->device_private);
    copy_to_remove = copy;
    return *value_ptr;
  }
};

#endif // TTG_PARSEC_COPY_HANDLER_H
