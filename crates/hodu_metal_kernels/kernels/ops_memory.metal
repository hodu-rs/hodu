#include "./utils.metal"
#include <metal_stdlib>

using namespace metal;

// Contiguous copy operation - converts strided layout to contiguous layout
// This macro creates kernels that copy data from a strided input layout to a contiguous output
#define CONTIGUOUS_OP(TYPENAME, FN_NAME)                                                           \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t &num_els [[buffer(2)]], constant size_t &num_dims [[buffer(3)]],           \
        constant size_t *metadata [[buffer(4)]], uint id [[thread_position_in_grid]]) {            \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const constant size_t *dims = metadata;                                                    \
        const constant size_t *strides = metadata + num_dims;                                      \
        const size_t offset = metadata ? *(metadata + 2 * num_dims) : 0;                           \
                                                                                                   \
        if (metadata == nullptr || is_contiguous(num_dims, dims, strides)) {                       \
            output[id] = input[offset + id];                                                       \
        } else {                                                                                   \
            unsigned strided_i = offset + get_strided_index(id, num_dims, dims, strides);          \
            output[id] = input[strided_i];                                                         \
        }                                                                                          \
    }

// Copy operation - simple element-wise copy (for already contiguous data)
#define COPY_OP(TYPENAME, FN_NAME)                                                                 \
    kernel void FN_NAME(                                                                           \
        const device TYPENAME *input [[buffer(0)]], device TYPENAME *output [[buffer(1)]],         \
        constant size_t &num_els [[buffer(2)]], uint id [[thread_position_in_grid]]) {             \
        if (id >= num_els)                                                                         \
            return;                                                                                \
        output[id] = input[id];                                                                    \
    }

// ============================================================================
// Contiguous operations for all data types
// ============================================================================

CONTIGUOUS_OP(bool, contiguous_bool);
CONTIGUOUS_OP(bfloat, contiguous_bf16);
CONTIGUOUS_OP(half, contiguous_f16);
CONTIGUOUS_OP(float, contiguous_f32);
CONTIGUOUS_OP(uint8_t, contiguous_u8);
CONTIGUOUS_OP(uint16_t, contiguous_u16);
CONTIGUOUS_OP(uint32_t, contiguous_u32);
CONTIGUOUS_OP(uint64_t, contiguous_u64);
CONTIGUOUS_OP(int8_t, contiguous_i8);
CONTIGUOUS_OP(int16_t, contiguous_i16);
CONTIGUOUS_OP(int32_t, contiguous_i32);
CONTIGUOUS_OP(int64_t, contiguous_i64);

// ============================================================================
// Simple copy operations for all data types
// ============================================================================

COPY_OP(bool, copy_bool);
COPY_OP(bfloat, copy_bf16);
COPY_OP(half, copy_f16);
COPY_OP(float, copy_f32);
COPY_OP(uint8_t, copy_u8);
COPY_OP(uint16_t, copy_u16);
COPY_OP(uint32_t, copy_u32);
COPY_OP(uint64_t, copy_u64);
COPY_OP(int8_t, copy_i8);
COPY_OP(int16_t, copy_i16);
COPY_OP(int32_t, copy_i32);
COPY_OP(int64_t, copy_i64);
