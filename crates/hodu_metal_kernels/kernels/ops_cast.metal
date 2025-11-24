#include "./headers/utils.metal"
#include <metal_stdlib>

using namespace metal;

// Cast Operations
// ===============
// Performs type conversion on tensor elements.
// Supports both contiguous and strided tensors with automatic layout detection.
//
// Metadata Layout (Total: 2 + num_dims * 2 + 1):
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape (dimensions of the tensor)
// - metadata[2+num_dims..2+2*num_dims]: strides (stride for each dimension)
// - metadata[2+2*num_dims]: offset (starting offset in input buffer)
//
// Buffer Layout:
// - buffer(0): input tensor (const device IN_TYPENAME*)
// - buffer(1): output tensor (device OUT_TYPENAME*)
// - buffer(2): metadata (constant size_t*)
//
// The kernel automatically detects contiguous layouts and uses optimized access patterns.

#define CAST_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, CAST_EXPR)                                     \
    kernel void hodu_metal_##FN_NAME(                                                              \
        const device IN_TYPENAME *input [[buffer(0)]], device OUT_TYPENAME *output [[buffer(1)]],  \
        constant size_t *metadata [[buffer(2)]], uint id [[thread_position_in_grid]]) {            \
        const size_t num_els = metadata[0];                                                        \
        if (id >= num_els)                                                                         \
            return;                                                                                \
                                                                                                   \
        const size_t num_dims = metadata[1];                                                       \
        const constant size_t *dims = metadata + 2;                                                \
        const constant size_t *strides = metadata + 2 + num_dims;                                  \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
                                                                                                   \
        if (is_contiguous(num_dims, dims, strides)) {                                              \
            IN_TYPENAME x = input[offset + id];                                                    \
            output[id] = CAST_EXPR;                                                                \
        } else {                                                                                   \
            unsigned strided_i = offset + get_strided_index(id, num_dims, dims, strides);          \
            IN_TYPENAME x = input[strided_i];                                                      \
            output[id] = CAST_EXPR;                                                                \
        }                                                                                          \
    }

// Helper template for clamping
template <typename T> T minimum(T x, T y) { return (x < y) ? x : y; }
template <typename T> T maximum(T x, T y) { return (x > y) ? x : y; }

// ============================================================================
// BOOL conversions
// ============================================================================

// BOOL -> other types
CAST_OP(bool, bfloat, cast_bool_to_bf16, x ? 1.0bf : 0.0bf);
CAST_OP(bool, half, cast_bool_to_f16, x ? 1.0h : 0.0h);
CAST_OP(bool, float, cast_bool_to_f32, x ? 1.0f : 0.0f);
CAST_OP(bool, uint8_t, cast_bool_to_u8, x ? uint8_t(1) : uint8_t(0));
CAST_OP(bool, uint16_t, cast_bool_to_u16, x ? uint16_t(1) : uint16_t(0));
CAST_OP(bool, uint32_t, cast_bool_to_u32, x ? uint32_t(1) : uint32_t(0));
CAST_OP(bool, uint64_t, cast_bool_to_u64, x ? uint64_t(1) : uint64_t(0));
CAST_OP(bool, int8_t, cast_bool_to_i8, x ? int8_t(1) : int8_t(0));
CAST_OP(bool, int16_t, cast_bool_to_i16, x ? int16_t(1) : int16_t(0));
CAST_OP(bool, int32_t, cast_bool_to_i32, x ? int32_t(1) : int32_t(0));
CAST_OP(bool, int64_t, cast_bool_to_i64, x ? int64_t(1) : int64_t(0));

// ============================================================================
// BFLOAT (bfloat16) conversions
// ============================================================================

// BFLOAT -> other types
CAST_OP(bfloat, bool, cast_bf16_to_bool, x != 0.0bf);
CAST_OP(bfloat, half, cast_bf16_to_f16, half(x));
CAST_OP(bfloat, float, cast_bf16_to_f32, float(x));
CAST_OP(bfloat, uint8_t, cast_bf16_to_u8, uint8_t(clamp(float(x), 0.0f, 255.0f)));
CAST_OP(bfloat, uint16_t, cast_bf16_to_u16, uint16_t(clamp(float(x), 0.0f, 65535.0f)));
CAST_OP(bfloat, uint32_t, cast_bf16_to_u32, uint32_t(maximum(0.0bf, x)));
CAST_OP(bfloat, uint64_t, cast_bf16_to_u64, uint64_t(maximum(0.0bf, x)));
CAST_OP(bfloat, int8_t, cast_bf16_to_i8, int8_t(clamp(float(x), -128.0f, 127.0f)));
CAST_OP(bfloat, int16_t, cast_bf16_to_i16, int16_t(clamp(float(x), -32768.0f, 32767.0f)));
CAST_OP(bfloat, int32_t, cast_bf16_to_i32, int32_t(x));
CAST_OP(bfloat, int64_t, cast_bf16_to_i64, int64_t(x));

// ============================================================================
// HALF (float16) conversions
// ============================================================================

// HALF -> other types
CAST_OP(half, bool, cast_f16_to_bool, x != 0.0h);
CAST_OP(half, bfloat, cast_f16_to_bf16, bfloat(x));
CAST_OP(half, float, cast_f16_to_f32, float(x));
CAST_OP(half, uint8_t, cast_f16_to_u8, uint8_t(clamp(float(x), 0.0f, 255.0f)));
CAST_OP(half, uint16_t, cast_f16_to_u16, uint16_t(clamp(float(x), 0.0f, 65504.0f)));
CAST_OP(half, uint32_t, cast_f16_to_u32, uint32_t(maximum(0.0h, x)));
CAST_OP(half, uint64_t, cast_f16_to_u64, uint64_t(maximum(0.0h, x)));
CAST_OP(half, int8_t, cast_f16_to_i8, int8_t(clamp(float(x), -128.0f, 127.0f)));
CAST_OP(half, int16_t, cast_f16_to_i16, int16_t(clamp(float(x), -32768.0f, 32767.0f)));
CAST_OP(half, int32_t, cast_f16_to_i32, int32_t(x));
CAST_OP(half, int64_t, cast_f16_to_i64, int64_t(x));

// ============================================================================
// FLOAT32 conversions
// ============================================================================

// FLOAT32 -> other types
CAST_OP(float, bool, cast_f32_to_bool, x != 0.0f);
CAST_OP(float, bfloat, cast_f32_to_bf16, bfloat(x));
CAST_OP(float, half, cast_f32_to_f16, half(x));
CAST_OP(float, uint8_t, cast_f32_to_u8, uint8_t(clamp(x, 0.0f, 255.0f)));
CAST_OP(float, uint16_t, cast_f32_to_u16, uint16_t(clamp(x, 0.0f, 65535.0f)));
CAST_OP(float, uint32_t, cast_f32_to_u32, uint32_t(maximum(0.0f, x)));
CAST_OP(float, uint64_t, cast_f32_to_u64, uint64_t(maximum(0.0f, x)));
CAST_OP(float, int8_t, cast_f32_to_i8, int8_t(clamp(x, -128.0f, 127.0f)));
CAST_OP(float, int16_t, cast_f32_to_i16, int16_t(clamp(x, -32768.0f, 32767.0f)));
CAST_OP(float, int32_t, cast_f32_to_i32, int32_t(x));
CAST_OP(float, int64_t, cast_f32_to_i64, int64_t(x));

// ============================================================================
// UINT8 conversions
// ============================================================================

// UINT8 -> other types
CAST_OP(uint8_t, bool, cast_u8_to_bool, x != 0);
CAST_OP(uint8_t, bfloat, cast_u8_to_bf16, bfloat(x));
CAST_OP(uint8_t, half, cast_u8_to_f16, half(x));
CAST_OP(uint8_t, float, cast_u8_to_f32, float(x));
CAST_OP(uint8_t, uint16_t, cast_u8_to_u16, uint16_t(x));
CAST_OP(uint8_t, uint32_t, cast_u8_to_u32, uint32_t(x));
CAST_OP(uint8_t, uint64_t, cast_u8_to_u64, uint64_t(x));
CAST_OP(uint8_t, int8_t, cast_u8_to_i8, int8_t(minimum(uint8_t(127), x)));
CAST_OP(uint8_t, int16_t, cast_u8_to_i16, int16_t(x));
CAST_OP(uint8_t, int32_t, cast_u8_to_i32, int32_t(x));
CAST_OP(uint8_t, int64_t, cast_u8_to_i64, int64_t(x));

// ============================================================================
// UINT16 conversions
// ============================================================================

// UINT16 -> other types
CAST_OP(uint16_t, bool, cast_u16_to_bool, x != 0);
CAST_OP(uint16_t, bfloat, cast_u16_to_bf16, bfloat(x));
CAST_OP(uint16_t, half, cast_u16_to_f16, half(x));
CAST_OP(uint16_t, float, cast_u16_to_f32, float(x));
CAST_OP(uint16_t, uint8_t, cast_u16_to_u8, uint8_t(minimum(uint16_t(255), x)));
CAST_OP(uint16_t, uint32_t, cast_u16_to_u32, uint32_t(x));
CAST_OP(uint16_t, uint64_t, cast_u16_to_u64, uint64_t(x));
CAST_OP(uint16_t, int8_t, cast_u16_to_i8, int8_t(minimum(uint16_t(127), x)));
CAST_OP(uint16_t, int16_t, cast_u16_to_i16, int16_t(minimum(uint16_t(32767), x)));
CAST_OP(uint16_t, int32_t, cast_u16_to_i32, int32_t(x));
CAST_OP(uint16_t, int64_t, cast_u16_to_i64, int64_t(x));

// ============================================================================
// UINT32 conversions
// ============================================================================

// UINT32 -> other types
CAST_OP(uint32_t, bool, cast_u32_to_bool, x != 0);
CAST_OP(uint32_t, bfloat, cast_u32_to_bf16, bfloat(x));
CAST_OP(uint32_t, half, cast_u32_to_f16, half(x));
CAST_OP(uint32_t, float, cast_u32_to_f32, float(x));
CAST_OP(uint32_t, uint8_t, cast_u32_to_u8, uint8_t(minimum(uint32_t(255), x)));
CAST_OP(uint32_t, uint16_t, cast_u32_to_u16, uint16_t(minimum(uint32_t(65535), x)));
CAST_OP(uint32_t, uint64_t, cast_u32_to_u64, uint64_t(x));
CAST_OP(uint32_t, int8_t, cast_u32_to_i8, int8_t(minimum(uint32_t(127), x)));
CAST_OP(uint32_t, int16_t, cast_u32_to_i16, int16_t(minimum(uint32_t(32767), x)));
CAST_OP(uint32_t, int32_t, cast_u32_to_i32, int32_t(minimum(uint32_t(2147483647), x)));
CAST_OP(uint32_t, int64_t, cast_u32_to_i64, int64_t(x));

// ============================================================================
// UINT64 conversions
// ============================================================================

// UINT64 -> other types
CAST_OP(uint64_t, bool, cast_u64_to_bool, x != 0);
CAST_OP(uint64_t, bfloat, cast_u64_to_bf16, bfloat(x));
CAST_OP(uint64_t, half, cast_u64_to_f16, half(x));
CAST_OP(uint64_t, float, cast_u64_to_f32, float(x));
CAST_OP(uint64_t, uint8_t, cast_u64_to_u8, uint8_t(minimum(uint64_t(255), x)));
CAST_OP(uint64_t, uint16_t, cast_u64_to_u16, uint16_t(minimum(uint64_t(65535), x)));
CAST_OP(uint64_t, uint32_t, cast_u64_to_u32, uint32_t(minimum(uint64_t(4294967295), x)));
CAST_OP(uint64_t, int8_t, cast_u64_to_i8, int8_t(minimum(uint64_t(127), x)));
CAST_OP(uint64_t, int16_t, cast_u64_to_i16, int16_t(minimum(uint64_t(32767), x)));
CAST_OP(uint64_t, int32_t, cast_u64_to_i32, int32_t(minimum(uint64_t(2147483647), x)));
CAST_OP(uint64_t, int64_t, cast_u64_to_i64, int64_t(minimum(uint64_t(9223372036854775807ULL), x)));

// ============================================================================
// INT8 conversions
// ============================================================================

// INT8 -> other types
CAST_OP(int8_t, bool, cast_i8_to_bool, x != 0);
CAST_OP(int8_t, bfloat, cast_i8_to_bf16, bfloat(x));
CAST_OP(int8_t, half, cast_i8_to_f16, half(x));
CAST_OP(int8_t, float, cast_i8_to_f32, float(x));
CAST_OP(int8_t, uint8_t, cast_i8_to_u8, uint8_t(maximum(int8_t(0), x)));
CAST_OP(int8_t, uint16_t, cast_i8_to_u16, uint16_t(maximum(int8_t(0), x)));
CAST_OP(int8_t, uint32_t, cast_i8_to_u32, uint32_t(maximum(int8_t(0), x)));
CAST_OP(int8_t, uint64_t, cast_i8_to_u64, uint64_t(maximum(int8_t(0), x)));
CAST_OP(int8_t, int16_t, cast_i8_to_i16, int16_t(x));
CAST_OP(int8_t, int32_t, cast_i8_to_i32, int32_t(x));
CAST_OP(int8_t, int64_t, cast_i8_to_i64, int64_t(x));

// ============================================================================
// INT16 conversions
// ============================================================================

// INT16 -> other types
CAST_OP(int16_t, bool, cast_i16_to_bool, x != 0);
CAST_OP(int16_t, bfloat, cast_i16_to_bf16, bfloat(x));
CAST_OP(int16_t, half, cast_i16_to_f16, half(x));
CAST_OP(int16_t, float, cast_i16_to_f32, float(x));
CAST_OP(int16_t, uint8_t, cast_i16_to_u8, uint8_t(clamp(int32_t(x), 0, 255)));
CAST_OP(int16_t, uint16_t, cast_i16_to_u16, uint16_t(maximum(int16_t(0), x)));
CAST_OP(int16_t, uint32_t, cast_i16_to_u32, uint32_t(maximum(int16_t(0), x)));
CAST_OP(int16_t, uint64_t, cast_i16_to_u64, uint64_t(maximum(int16_t(0), x)));
CAST_OP(int16_t, int8_t, cast_i16_to_i8, int8_t(clamp(int32_t(x), -128, 127)));
CAST_OP(int16_t, int32_t, cast_i16_to_i32, int32_t(x));
CAST_OP(int16_t, int64_t, cast_i16_to_i64, int64_t(x));

// ============================================================================
// INT32 conversions
// ============================================================================

// INT32 -> other types
CAST_OP(int32_t, bool, cast_i32_to_bool, x != 0);
CAST_OP(int32_t, bfloat, cast_i32_to_bf16, bfloat(x));
CAST_OP(int32_t, half, cast_i32_to_f16, half(x));
CAST_OP(int32_t, float, cast_i32_to_f32, float(x));
CAST_OP(int32_t, uint8_t, cast_i32_to_u8, uint8_t(clamp(x, 0, 255)));
CAST_OP(int32_t, uint16_t, cast_i32_to_u16, uint16_t(clamp(x, 0, 65535)));
CAST_OP(int32_t, uint32_t, cast_i32_to_u32, uint32_t(maximum(int32_t(0), x)));
CAST_OP(int32_t, uint64_t, cast_i32_to_u64, uint64_t(maximum(int32_t(0), x)));
CAST_OP(int32_t, int8_t, cast_i32_to_i8, int8_t(clamp(x, -128, 127)));
CAST_OP(int32_t, int16_t, cast_i32_to_i16, int16_t(clamp(x, -32768, 32767)));
CAST_OP(int32_t, int64_t, cast_i32_to_i64, int64_t(x));

// ============================================================================
// INT64 conversions
// ============================================================================

// INT64 -> other types
CAST_OP(int64_t, bool, cast_i64_to_bool, x != 0);
CAST_OP(int64_t, bfloat, cast_i64_to_bf16, bfloat(x));
CAST_OP(int64_t, half, cast_i64_to_f16, half(x));
CAST_OP(int64_t, float, cast_i64_to_f32, float(x));
CAST_OP(int64_t, uint8_t, cast_i64_to_u8, uint8_t(clamp(x, int64_t(0), int64_t(255))));
CAST_OP(int64_t, uint16_t, cast_i64_to_u16, uint16_t(clamp(x, int64_t(0), int64_t(65535))));
CAST_OP(int64_t, uint32_t, cast_i64_to_u32, uint32_t(clamp(x, int64_t(0), int64_t(4294967295))));
CAST_OP(int64_t, uint64_t, cast_i64_to_u64, uint64_t(maximum(int64_t(0), x)));
CAST_OP(int64_t, int8_t, cast_i64_to_i8, int8_t(clamp(x, int64_t(-128), int64_t(127))));
CAST_OP(int64_t, int16_t, cast_i64_to_i16, int16_t(clamp(x, int64_t(-32768), int64_t(32767))));
CAST_OP(int64_t, int32_t, cast_i64_to_i32,
        int32_t(clamp(x, int64_t(-2147483648), int64_t(2147483647))));
