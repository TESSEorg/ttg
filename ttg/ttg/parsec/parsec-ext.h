#ifndef TTG_PARSEC_EXT_H
#define TTG_PARSEC_EXT_H

/* HACK: we need this flag on a data copy to indicate whether it has been registered */
#define TTG_PARSEC_DATA_FLAG_REGISTERED        ((parsec_data_flag_t)1<<2)

/* HACK: mark the flows of device scratch as temporary so that we can easily discard it */
#define TTG_PARSEC_FLOW_ACCESS_TMP (1<<7)

#endif // TTG_PARSEC_EXT_H