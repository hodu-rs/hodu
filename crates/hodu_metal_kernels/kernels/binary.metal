#include "./utils.metal"
#include <metal_stdlib>

using namespace metal;

template <typename T> T sub_with_clamp(T x, T y) { return (x > y) ? (x - y) : 0; }
template <typename T> T maximum(T x, T y) { return (x > y) ? x : y; }
template <typename T> T minimum(T x, T y) { return (x < y) ? x : y; }

#define BINARY_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)                                        \
    kernel void FN_NAME(                                                                           \
        const device IN_TYPENAME *lhs [[buffer(0)]], const device IN_TYPENAME *rhs [[buffer(1)]],  \
        device OUT_TYPENAME *out [[buffer(2)]], constant size_t &num_els [[buffer(3)]],            \
        constant size_t &num_dims [[buffer(4)]], constant size_t *metadata [[buffer(5)]],          \
        uint id [[thread_position_in_grid]]) {                                                     \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const constant size_t *dims = metadata;                                                    \
        const constant size_t *lhs_strides = metadata + 1 * num_dims;                              \
        const constant size_t *rhs_strides = metadata + 2 * num_dims;                              \
        const size_t lhs_offset = metadata ? metadata[3 * num_dims] : 0;                           \
        const size_t rhs_offset = metadata ? metadata[3 * num_dims + 1] : 0;                       \
        bool lhs_cont = metadata == nullptr || is_contiguous(num_dims, dims, lhs_strides);         \
        bool rhs_cont = metadata == nullptr || is_contiguous(num_dims, dims, rhs_strides);         \
        if (lhs_cont && rhs_cont) {                                                                \
            IN_TYPENAME x = lhs[lhs_offset + id];                                                  \
            IN_TYPENAME y = rhs[rhs_offset + id];                                                  \
            out[id] = FUNC;                                                                        \
        } else if (lhs_cont) {                                                                     \
            unsigned int tmp_i = id;                                                               \
            unsigned int rhs_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % dims[d];                                              \
                rhs_i += i_dim * rhs_strides[d];                                                   \
                tmp_i /= dims[d];                                                                  \
            }                                                                                      \
            IN_TYPENAME x = lhs[lhs_offset + id];                                                  \
            IN_TYPENAME y = rhs[rhs_offset + rhs_i];                                               \
            out[id] = FUNC;                                                                        \
        } else if (rhs_cont) {                                                                     \
            unsigned int tmp_i = id;                                                               \
            unsigned int lhs_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % dims[d];                                              \
                lhs_i += i_dim * lhs_strides[d];                                                   \
                tmp_i /= dims[d];                                                                  \
            }                                                                                      \
            IN_TYPENAME x = lhs[lhs_offset + lhs_i];                                               \
            IN_TYPENAME y = rhs[rhs_offset + id];                                                  \
            out[id] = FUNC;                                                                        \
        } else {                                                                                   \
            unsigned int tmp_i = id;                                                               \
            unsigned int lhs_i = 0;                                                                \
            unsigned int rhs_i = 0;                                                                \
            for (int d = num_dims - 1; d >= 0; d--) {                                              \
                unsigned int i_dim = tmp_i % dims[d];                                              \
                lhs_i += i_dim * lhs_strides[d];                                                   \
                rhs_i += i_dim * rhs_strides[d];                                                   \
                tmp_i /= dims[d];                                                                  \
            }                                                                                      \
            IN_TYPENAME x = lhs[lhs_offset + lhs_i];                                               \
            IN_TYPENAME y = rhs[rhs_offset + rhs_i];                                               \
            out[id] = FUNC;                                                                        \
        }                                                                                          \
    }

// bool operations
BINARY_OP(bool, bool, add_bool, x + y);
BINARY_OP(bool, bool, sub_bool, x - y);
BINARY_OP(bool, bool, mul_bool, x *y);
BINARY_OP(bool, bool, div_bool, x &&y);
BINARY_OP(bool, bool, maximum_bool, x || y);
BINARY_OP(bool, bool, minimum_bool, x &&y);

BINARY_OP(bool, bool, logical_and_bool, x &&y);
BINARY_OP(bool, bool, logical_or_bool, x || y);
BINARY_OP(bool, bool, logical_xor_bool, x != y);

BINARY_OP(bool, bool, eq_bool, x == y);
BINARY_OP(bool, bool, ne_bool, x != y);
BINARY_OP(bool, bool, lt_bool, x < y);
BINARY_OP(bool, bool, le_bool, x <= y);
BINARY_OP(bool, bool, gt_bool, x > y);
BINARY_OP(bool, bool, ge_bool, x >= y);

// Metal 16-bit braian floating point
BINARY_OP(bfloat, bfloat, add_bf16, x + y);
BINARY_OP(bfloat, bfloat, sub_bf16, x - y);
BINARY_OP(bfloat, bfloat, mul_bf16, x *y);
BINARY_OP(bfloat, bfloat, div_bf16, x / y);
BINARY_OP(bfloat, bfloat, maximum_bf16, maximum(x, y));
BINARY_OP(bfloat, bfloat, minimum_bf16, minimum(x, y));

BINARY_OP(bfloat, bool, logical_and_bf16, (x != 0.0bf && y != 0.0bf));
BINARY_OP(bfloat, bool, logical_or_bf16, (x != 0.0bf || y != 0.0bf));
BINARY_OP(bfloat, bool, logical_xor_bf16, (x != 0.0bf) != (y != 0.0bf));

BINARY_OP(bfloat, bool, eq_bf16, x == y);
BINARY_OP(bfloat, bool, ne_bf16, x != y);
BINARY_OP(bfloat, bool, lt_bf16, x < y);
BINARY_OP(bfloat, bool, le_bf16, x <= y);
BINARY_OP(bfloat, bool, gt_bf16, x > y);
BINARY_OP(bfloat, bool, ge_bf16, x >= y);

// Metal half-precision floating point
BINARY_OP(half, half, add_f16, x + y);
BINARY_OP(half, half, sub_f16, x - y);
BINARY_OP(half, half, mul_f16, x *y);
BINARY_OP(half, half, div_f16, x / y);
BINARY_OP(half, half, maximum_f16, maximum(x, y));
BINARY_OP(half, half, minimum_f16, minimum(x, y));

BINARY_OP(half, bool, logical_and_f16, (x != 0.0h && y != 0.0h));
BINARY_OP(half, bool, logical_or_f16, (x != 0.0h || y != 0.0h));
BINARY_OP(half, bool, logical_xor_f16, (x != 0.0h) != (y != 0.0h));

BINARY_OP(half, bool, eq_f16, x == y);
BINARY_OP(half, bool, ne_f16, x != y);
BINARY_OP(half, bool, lt_f16, x < y);
BINARY_OP(half, bool, le_f16, x <= y);
BINARY_OP(half, bool, gt_f16, x > y);
BINARY_OP(half, bool, ge_f16, x >= y);

// float32 operations
BINARY_OP(float, float, add_f32, x + y);
BINARY_OP(float, float, sub_f32, x - y);
BINARY_OP(float, float, mul_f32, x *y);
BINARY_OP(float, float, div_f32, x / y);
BINARY_OP(float, float, maximum_f32, maximum(x, y));
BINARY_OP(float, float, minimum_f32, minimum(x, y));

BINARY_OP(float, bool, logical_and_f32, (x != 0.0f && y != 0.0f));
BINARY_OP(float, bool, logical_or_f32, (x != 0.0f || y != 0.0f));
BINARY_OP(float, bool, logical_xor_f32, (x != 0.0f) != (y != 0.0f));

BINARY_OP(float, bool, eq_f32, x == y);
BINARY_OP(float, bool, ne_f32, x != y);
BINARY_OP(float, bool, lt_f32, x < y);
BINARY_OP(float, bool, le_f32, x <= y);
BINARY_OP(float, bool, gt_f32, x > y);
BINARY_OP(float, bool, ge_f32, x >= y);

// uint8 operations
BINARY_OP(uint8_t, uint8_t, add_u8, x + y);
BINARY_OP(uint8_t, uint8_t, sub_u8, sub_with_clamp(x, y));
BINARY_OP(uint8_t, uint8_t, mul_u8, x *y);
BINARY_OP(uint8_t, uint8_t, div_u8, x / y);
BINARY_OP(uint8_t, uint8_t, maximum_u8, maximum(x, y));
BINARY_OP(uint8_t, uint8_t, minimum_u8, minimum(x, y));

BINARY_OP(uint8_t, bool, logical_and_u8, (x != 0u && y != 0u));
BINARY_OP(uint8_t, bool, logical_or_u8, (x != 0u || y != 0u));
BINARY_OP(uint8_t, bool, logical_xor_u8, (x != 0u) != (y != 0u));

BINARY_OP(uint8_t, bool, eq_u8, x == y);
BINARY_OP(uint8_t, bool, ne_u8, x != y);
BINARY_OP(uint8_t, bool, lt_u8, x < y);
BINARY_OP(uint8_t, bool, le_u8, x <= y);
BINARY_OP(uint8_t, bool, gt_u8, x > y);
BINARY_OP(uint8_t, bool, ge_u8, x >= y);

// uint16 operations
BINARY_OP(uint16_t, uint16_t, add_u16, x + y);
BINARY_OP(uint16_t, uint16_t, sub_u16, sub_with_clamp(x, y));
BINARY_OP(uint16_t, uint16_t, mul_u16, x *y);
BINARY_OP(uint16_t, uint16_t, div_u16, x / y);
BINARY_OP(uint16_t, uint16_t, maximum_u16, maximum(x, y));
BINARY_OP(uint16_t, uint16_t, minimum_u16, minimum(x, y));

BINARY_OP(uint16_t, bool, logical_and_u16, (x != 0u && y != 0u));
BINARY_OP(uint16_t, bool, logical_or_u16, (x != 0u || y != 0u));
BINARY_OP(uint16_t, bool, logical_xor_u16, (x != 0u) != (y != 0u));

BINARY_OP(uint16_t, bool, eq_u16, x == y);
BINARY_OP(uint16_t, bool, ne_u16, x != y);
BINARY_OP(uint16_t, bool, lt_u16, x < y);
BINARY_OP(uint16_t, bool, le_u16, x <= y);
BINARY_OP(uint16_t, bool, gt_u16, x > y);
BINARY_OP(uint16_t, bool, ge_u16, x >= y);

// uint32 operations
BINARY_OP(uint32_t, uint32_t, add_u32, x + y);
BINARY_OP(uint32_t, uint32_t, sub_u32, sub_with_clamp(x, y));
BINARY_OP(uint32_t, uint32_t, mul_u32, x *y);
BINARY_OP(uint32_t, uint32_t, div_u32, x / y);
BINARY_OP(uint32_t, uint32_t, maximum_u32, maximum(x, y));
BINARY_OP(uint32_t, uint32_t, minimum_u32, minimum(x, y));

BINARY_OP(uint32_t, bool, logical_and_u32, (x != 0u && y != 0u));
BINARY_OP(uint32_t, bool, logical_or_u32, (x != 0u || y != 0u));
BINARY_OP(uint32_t, bool, logical_xor_u32, (x != 0u) != (y != 0u));

BINARY_OP(uint32_t, bool, eq_u32, x == y);
BINARY_OP(uint32_t, bool, ne_u32, x != y);
BINARY_OP(uint32_t, bool, lt_u32, x < y);
BINARY_OP(uint32_t, bool, le_u32, x <= y);
BINARY_OP(uint32_t, bool, gt_u32, x > y);
BINARY_OP(uint32_t, bool, ge_u32, x >= y);

// uint64 operations
BINARY_OP(uint64_t, uint64_t, add_u64, x + y);
BINARY_OP(uint64_t, uint64_t, sub_u64, sub_with_clamp(x, y));
BINARY_OP(uint64_t, uint64_t, mul_u64, x *y);
BINARY_OP(uint64_t, uint64_t, div_u64, x / y);
BINARY_OP(uint64_t, uint64_t, maximum_u64, maximum(x, y));
BINARY_OP(uint64_t, uint64_t, minimum_u64, minimum(x, y));

BINARY_OP(uint64_t, bool, logical_and_u64, (x != 0u && y != 0u));
BINARY_OP(uint64_t, bool, logical_or_u64, (x != 0u || y != 0u));
BINARY_OP(uint64_t, bool, logical_xor_u64, (x != 0u) != (y != 0u));

BINARY_OP(uint64_t, bool, eq_u64, x == y);
BINARY_OP(uint64_t, bool, ne_u64, x != y);
BINARY_OP(uint64_t, bool, lt_u64, x < y);
BINARY_OP(uint64_t, bool, le_u64, x <= y);
BINARY_OP(uint64_t, bool, gt_u64, x > y);
BINARY_OP(uint64_t, bool, ge_u64, x >= y);

// int8 operations
BINARY_OP(int8_t, int8_t, add_i8, x + y);
BINARY_OP(int8_t, int8_t, sub_i8, x - y);
BINARY_OP(int8_t, int8_t, mul_i8, x *y);
BINARY_OP(int8_t, int8_t, div_i8, x / y);
BINARY_OP(int8_t, int8_t, maximum_i8, maximum(x, y));
BINARY_OP(int8_t, int8_t, minimum_i8, minimum(x, y));

BINARY_OP(int8_t, bool, logical_and_i8, (x != 0 && y != 0));
BINARY_OP(int8_t, bool, logical_or_i8, (x != 0 || y != 0));
BINARY_OP(int8_t, bool, logical_xor_i8, (x != 0) != (y != 0));

BINARY_OP(int8_t, bool, eq_i8, x == y);
BINARY_OP(int8_t, bool, ne_i8, x != y);
BINARY_OP(int8_t, bool, lt_i8, x < y);
BINARY_OP(int8_t, bool, le_i8, x <= y);
BINARY_OP(int8_t, bool, gt_i8, x > y);
BINARY_OP(int8_t, bool, ge_i8, x >= y);

// int16 operations
BINARY_OP(int16_t, int16_t, add_i16, x + y);
BINARY_OP(int16_t, int16_t, sub_i16, x - y);
BINARY_OP(int16_t, int16_t, mul_i16, x *y);
BINARY_OP(int16_t, int16_t, div_i16, x / y);
BINARY_OP(int16_t, int16_t, maximum_i16, maximum(x, y));
BINARY_OP(int16_t, int16_t, minimum_i16, minimum(x, y));

BINARY_OP(int16_t, bool, logical_and_i16, (x != 0 && y != 0));
BINARY_OP(int16_t, bool, logical_or_i16, (x != 0 || y != 0));
BINARY_OP(int16_t, bool, logical_xor_i16, (x != 0) != (y != 0));

BINARY_OP(int16_t, bool, eq_i16, x == y);
BINARY_OP(int16_t, bool, ne_i16, x != y);
BINARY_OP(int16_t, bool, lt_i16, x < y);
BINARY_OP(int16_t, bool, le_i16, x <= y);
BINARY_OP(int16_t, bool, gt_i16, x > y);
BINARY_OP(int16_t, bool, ge_i16, x >= y);

// int32 operations
BINARY_OP(int32_t, int32_t, add_i32, x + y);
BINARY_OP(int32_t, int32_t, sub_i32, x - y);
BINARY_OP(int32_t, int32_t, mul_i32, x *y);
BINARY_OP(int32_t, int32_t, div_i32, x / y);
BINARY_OP(int32_t, int32_t, maximum_i32, maximum(x, y));
BINARY_OP(int32_t, int32_t, minimum_i32, minimum(x, y));

BINARY_OP(int32_t, bool, logical_and_i32, (x != 0 && y != 0));
BINARY_OP(int32_t, bool, logical_or_i32, (x != 0 || y != 0));
BINARY_OP(int32_t, bool, logical_xor_i32, (x != 0) != (y != 0));

BINARY_OP(int32_t, bool, eq_i32, x == y);
BINARY_OP(int32_t, bool, ne_i32, x != y);
BINARY_OP(int32_t, bool, lt_i32, x < y);
BINARY_OP(int32_t, bool, le_i32, x <= y);
BINARY_OP(int32_t, bool, gt_i32, x > y);
BINARY_OP(int32_t, bool, ge_i32, x >= y);

// int64 operations
BINARY_OP(int64_t, int64_t, add_i64, x + y);
BINARY_OP(int64_t, int64_t, sub_i64, x - y);
BINARY_OP(int64_t, int64_t, mul_i64, x *y);
BINARY_OP(int64_t, int64_t, div_i64, x / y);
BINARY_OP(int64_t, int64_t, maximum_i64, maximum(x, y));
BINARY_OP(int64_t, int64_t, minimum_i64, minimum(x, y));

BINARY_OP(int64_t, bool, logical_and_i64, (x != 0 && y != 0));
BINARY_OP(int64_t, bool, logical_or_i64, (x != 0 || y != 0));
BINARY_OP(int64_t, bool, logical_xor_i64, (x != 0) != (y != 0));

BINARY_OP(int64_t, bool, eq_i64, x == y);
BINARY_OP(int64_t, bool, ne_i64, x != y);
BINARY_OP(int64_t, bool, lt_i64, x < y);
BINARY_OP(int64_t, bool, le_i64, x <= y);
BINARY_OP(int64_t, bool, gt_i64, x > y);
BINARY_OP(int64_t, bool, ge_i64, x >= y);
