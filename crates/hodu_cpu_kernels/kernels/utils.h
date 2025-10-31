#ifndef HODU_CPU_KERNELS_UTILS_H
#define HODU_CPU_KERNELS_UTILS_H

#include "constants.h"
#include "math_utils.h"
#include "types.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check if a tensor is contiguous in memory
static inline bool is_contiguous(size_t num_dims, const size_t *dims, const size_t *strides) {
    size_t acc = 1;
    for (size_t d = 0; d < num_dims; d++) {
        size_t dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

// Get strided index for non-contiguous tensors
static inline size_t get_strided_index(size_t idx, size_t num_dims, const size_t *dims,
                                       const size_t *strides) {
    size_t strided_i = 0;
    for (size_t d = 0; d < num_dims; d++) {
        size_t dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

// Generic helper functions
#define MAXIMUM(a, b) ((a) > (b) ? (a) : (b))
#define MINIMUM(a, b) ((a) < (b) ? (a) : (b))

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_UTILS_H
