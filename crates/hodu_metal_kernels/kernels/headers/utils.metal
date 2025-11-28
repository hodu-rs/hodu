#pragma once

#include <metal_stdlib>

using namespace metal;

// Common helper functions
template <typename T> T maximum(T x, T y) { return (x > y) ? x : y; }
template <typename T> T minimum(T x, T y) { return (x < y) ? x : y; }

inline bool is_contiguous(const size_t num_dims, constant size_t *dims, constant size_t *strides) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

inline unsigned int get_strided_index(unsigned int idx, const size_t num_dims,
                                      constant size_t *dims, constant size_t *strides) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;

        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}
