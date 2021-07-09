
#include "ttg/parsec/ttg_data_copy.h"

void ttg_data_copy_ctor(ttg_data_copy_t *copy)
{
  copy->delete_fn = nullptr;
}

extern "C" {
PARSEC_OBJ_CLASS_INSTANCE(ttg_data_copy_t, parsec_data_copy_t,
                          &ttg_data_copy_ctor, NULL);
}
