#ifndef TTG_PARSEC_THREAD_LOCAL_H
#define TTG_PARSEC_THREAD_LOCAL_H

namespace ttg_parsec {

namespace detail {

  // fwd decls
  struct parsec_ttg_task_base_t;
  struct ttg_data_copy_t;

  inline thread_local parsec_ttg_task_base_t *parsec_ttg_caller = nullptr;

  inline ttg_data_copy_t*& ttg_data_copy_container() {
    static thread_local ttg_data_copy_t *ptr = nullptr;
    return ptr;
  }

} // namespace detail
} // namespace ttg_parsec

#endif // TTG_PARSEC_THREAD_LOCAL_H