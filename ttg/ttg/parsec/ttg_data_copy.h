#ifndef TTG_DATA_COPY_H
#define TTG_DATA_COPY_H

#include <parsec.h>

extern "C" {

typedef void (data_copy_delete_fn)(void*);

} // extern C

/* Extension of PaRSEC's data copy. Note that we use the underlying
 * ref-counting here to avoid further levels of indirection
 * (e.g., std::shared_ptr). */
struct ttg_data_copy_t : public parsec_data_copy_t {
  data_copy_delete_fn* delete_fn;
};

extern "C" {

PARSEC_OBJ_CLASS_DECLARATION(ttg_data_copy_t);


} // extern C

#endif // TTG_DATA_COPY_H
