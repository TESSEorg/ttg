#include "ttg/config.h"
#include <cinttypes>

void increment_buffer_cuda(
    double* buffer, std::size_t buffer_size, double* scratch, std::size_t scratch_size);