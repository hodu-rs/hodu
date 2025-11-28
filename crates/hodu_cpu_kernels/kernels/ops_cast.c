#include "ops_cast.h"
#include "thread_utils.h"
#include "types.h"

// ============================================================================
// CAST OPERATION IMPLEMENTATION
// ============================================================================
//
// Type casting operations convert tensor elements from one type to another.
// All operations support both contiguous and strided tensor access patterns.

// Helper macros for type conversion
#define TO_FLOAT(TYPE, val)                                                                        \
    _Generic((TYPE){0},                                                                            \
        f8e4m3_t: f8e4m3_to_float(val),                                                            \
        f8e5m2_t: f8e5m2_to_float(val),                                                            \
        bf16_t: bf16_to_float(val),                                                                \
        f16_t: f16_to_float(val),                                                                  \
        f32_t: (float)(val),                                                                       \
        f64_t: (float)(val),                                                                       \
        u8_t: (float)(val),                                                                        \
        u16_t: (float)(val),                                                                       \
        u32_t: (float)(val),                                                                       \
        u64_t: (float)(val),                                                                       \
        i8_t: (float)(val),                                                                        \
        i16_t: (float)(val),                                                                       \
        i32_t: (float)(val),                                                                       \
        i64_t: (float)(val),                                                                       \
        default: (float)(val))

/**
 * @brief Macro to implement cast operation from one type to another
 */
#define IMPL_CAST_OP(FROM_TYPE, FROM_SUFFIX, TO_TYPE, TO_SUFFIX, CONVERT)                          \
    typedef struct {                                                                               \
        const FROM_TYPE *input;                                                                    \
        TO_TYPE *output;                                                                           \
        size_t start;                                                                              \
        size_t end;                                                                                \
        size_t offset;                                                                             \
    } cast_##FROM_SUFFIX##_to_##TO_SUFFIX##_args_t;                                                \
                                                                                                   \
    static void *cast_##FROM_SUFFIX##_to_##TO_SUFFIX##_worker(void *arg) {                         \
        cast_##FROM_SUFFIX##_to_##TO_SUFFIX##_args_t *args =                                       \
            (cast_##FROM_SUFFIX##_to_##TO_SUFFIX##_args_t *)arg;                                   \
        for (size_t i = args->start; i < args->end; i++) {                                         \
            FROM_TYPE x = args->input[args->offset + i];                                           \
            args->output[i] = CONVERT;                                                             \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_cast_##FROM_SUFFIX##_to_##TO_SUFFIX(const void *input, void *output,             \
                                                      const size_t *metadata) {                    \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const FROM_TYPE *in = (const FROM_TYPE *)input;                                            \
        TO_TYPE *out = (TO_TYPE *)output;                                                          \
                                                                                                   \
        const size_t *dims = metadata + 2;                                                         \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = (num_dims > 0) ? metadata[2 + 2 * num_dims] : 0;                     \
                                                                                                   \
        bool contiguous = is_contiguous(num_dims, dims, strides);                                  \
                                                                                                   \
        if (contiguous) {                                                                          \
            const size_t min_work_per_thread = 100000;                                             \
            size_t num_threads = get_optimal_threads(num_els, min_work_per_thread);                \
                                                                                                   \
            if (num_threads > 1) {                                                                 \
                thread_t threads[num_threads];                                                     \
                cast_##FROM_SUFFIX##_to_##TO_SUFFIX##_args_t args[num_threads];                    \
                                                                                                   \
                size_t chunk_size = num_els / num_threads;                                         \
                size_t remaining = num_els % num_threads;                                          \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    args[t].input = in;                                                            \
                    args[t].output = out;                                                          \
                    args[t].offset = offset;                                                       \
                    args[t].start = t * chunk_size + (t < remaining ? t : remaining);              \
                    args[t].end = args[t].start + chunk_size + (t < remaining ? 1 : 0);            \
                    thread_create(&threads[t], cast_##FROM_SUFFIX##_to_##TO_SUFFIX##_worker,       \
                                  &args[t]);                                                       \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    FROM_TYPE x = in[offset + i];                                                  \
                    out[i] = CONVERT;                                                              \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                FROM_TYPE x = in[strided_i];                                                       \
                out[i] = CONVERT;                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// BOOL conversions
// ============================================================================
IMPL_CAST_OP(uint8_t, bool, uint8_t, bool, (x != 0))
IMPL_CAST_OP(uint8_t, bool, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)(x != 0)))
IMPL_CAST_OP(uint8_t, bool, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)(x != 0)))
IMPL_CAST_OP(uint8_t, bool, bf16_t, bf16, float_to_bf16((float)(x != 0)))
IMPL_CAST_OP(uint8_t, bool, f16_t, f16, float_to_f16((float)(x != 0)))
IMPL_CAST_OP(uint8_t, bool, f32_t, f32, (f32_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, f64_t, f64, (f64_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, u8_t, u8, (u8_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, u16_t, u16, (u16_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, u32_t, u32, (u32_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, u64_t, u64, (u64_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, i8_t, i8, (i8_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, i16_t, i16, (i16_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, i32_t, i32, (i32_t)(x != 0))
IMPL_CAST_OP(uint8_t, bool, i64_t, i64, (i64_t)(x != 0))

// ============================================================================
// F8E4M3 conversions
// ============================================================================
IMPL_CAST_OP(f8e4m3_t, f8e4m3, uint8_t, bool, (f8e4m3_to_float(x) != 0.0f))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, f8e4m3_t, f8e4m3, x)
IMPL_CAST_OP(f8e4m3_t, f8e4m3, f8e5m2_t, f8e5m2, float_to_f8e5m2(f8e4m3_to_float(x)))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, bf16_t, bf16, float_to_bf16(f8e4m3_to_float(x)))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, f16_t, f16, float_to_f16(f8e4m3_to_float(x)))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, f32_t, f32, f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, f64_t, f64, (f64_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, u8_t, u8, (u8_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, u16_t, u16, (u16_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, u32_t, u32, (u32_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, u64_t, u64, (u64_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, i8_t, i8, (i8_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, i16_t, i16, (i16_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, i32_t, i32, (i32_t)f8e4m3_to_float(x))
IMPL_CAST_OP(f8e4m3_t, f8e4m3, i64_t, i64, (i64_t)f8e4m3_to_float(x))

// ============================================================================
// F8E5M2 conversions
// ============================================================================
IMPL_CAST_OP(f8e5m2_t, f8e5m2, uint8_t, bool, (f8e5m2_to_float(x) != 0.0f))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, f8e4m3_t, f8e4m3, float_to_f8e4m3(f8e5m2_to_float(x)))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, f8e5m2_t, f8e5m2, x)
IMPL_CAST_OP(f8e5m2_t, f8e5m2, bf16_t, bf16, float_to_bf16(f8e5m2_to_float(x)))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, f16_t, f16, float_to_f16(f8e5m2_to_float(x)))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, f32_t, f32, f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, f64_t, f64, (f64_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, u8_t, u8, (u8_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, u16_t, u16, (u16_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, u32_t, u32, (u32_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, u64_t, u64, (u64_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, i8_t, i8, (i8_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, i16_t, i16, (i16_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, i32_t, i32, (i32_t)f8e5m2_to_float(x))
IMPL_CAST_OP(f8e5m2_t, f8e5m2, i64_t, i64, (i64_t)f8e5m2_to_float(x))

// ============================================================================
// BF16 conversions
// ============================================================================
IMPL_CAST_OP(bf16_t, bf16, uint8_t, bool, (bf16_to_float(x) != 0.0f))
IMPL_CAST_OP(bf16_t, bf16, f8e4m3_t, f8e4m3, float_to_f8e4m3(bf16_to_float(x)))
IMPL_CAST_OP(bf16_t, bf16, f8e5m2_t, f8e5m2, float_to_f8e5m2(bf16_to_float(x)))
IMPL_CAST_OP(bf16_t, bf16, bf16_t, bf16, x)
IMPL_CAST_OP(bf16_t, bf16, f16_t, f16, float_to_f16(bf16_to_float(x)))
IMPL_CAST_OP(bf16_t, bf16, f32_t, f32, bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, f64_t, f64, (f64_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, u8_t, u8, (u8_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, u16_t, u16, (u16_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, u32_t, u32, (u32_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, u64_t, u64, (u64_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, i8_t, i8, (i8_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, i16_t, i16, (i16_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, i32_t, i32, (i32_t)bf16_to_float(x))
IMPL_CAST_OP(bf16_t, bf16, i64_t, i64, (i64_t)bf16_to_float(x))

// ============================================================================
// F16 conversions
// ============================================================================
IMPL_CAST_OP(f16_t, f16, uint8_t, bool, (f16_to_float(x) != 0.0f))
IMPL_CAST_OP(f16_t, f16, f8e4m3_t, f8e4m3, float_to_f8e4m3(f16_to_float(x)))
IMPL_CAST_OP(f16_t, f16, f8e5m2_t, f8e5m2, float_to_f8e5m2(f16_to_float(x)))
IMPL_CAST_OP(f16_t, f16, bf16_t, bf16, float_to_bf16(f16_to_float(x)))
IMPL_CAST_OP(f16_t, f16, f16_t, f16, x)
IMPL_CAST_OP(f16_t, f16, f32_t, f32, f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, f64_t, f64, (f64_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, u8_t, u8, (u8_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, u16_t, u16, (u16_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, u32_t, u32, (u32_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, u64_t, u64, (u64_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, i8_t, i8, (i8_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, i16_t, i16, (i16_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, i32_t, i32, (i32_t)f16_to_float(x))
IMPL_CAST_OP(f16_t, f16, i64_t, i64, (i64_t)f16_to_float(x))

// ============================================================================
// F32 conversions
// ============================================================================
IMPL_CAST_OP(f32_t, f32, uint8_t, bool, (x != 0.0f))
IMPL_CAST_OP(f32_t, f32, f8e4m3_t, f8e4m3, float_to_f8e4m3(x))
IMPL_CAST_OP(f32_t, f32, f8e5m2_t, f8e5m2, float_to_f8e5m2(x))
IMPL_CAST_OP(f32_t, f32, bf16_t, bf16, float_to_bf16(x))
IMPL_CAST_OP(f32_t, f32, f16_t, f16, float_to_f16(x))
IMPL_CAST_OP(f32_t, f32, f32_t, f32, x)
IMPL_CAST_OP(f32_t, f32, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(f32_t, f32, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(f32_t, f32, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(f32_t, f32, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(f32_t, f32, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(f32_t, f32, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(f32_t, f32, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(f32_t, f32, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(f32_t, f32, i64_t, i64, (i64_t)x)

// ============================================================================
// F64 conversions
// ============================================================================
IMPL_CAST_OP(f64_t, f64, uint8_t, bool, (x != 0.0))
IMPL_CAST_OP(f64_t, f64, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(f64_t, f64, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(f64_t, f64, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(f64_t, f64, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(f64_t, f64, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(f64_t, f64, f64_t, f64, x)
IMPL_CAST_OP(f64_t, f64, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(f64_t, f64, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(f64_t, f64, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(f64_t, f64, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(f64_t, f64, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(f64_t, f64, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(f64_t, f64, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(f64_t, f64, i64_t, i64, (i64_t)x)

// ============================================================================
// U8 conversions
// ============================================================================
IMPL_CAST_OP(u8_t, u8, uint8_t, bool, (x != 0))
IMPL_CAST_OP(u8_t, u8, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(u8_t, u8, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(u8_t, u8, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(u8_t, u8, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(u8_t, u8, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(u8_t, u8, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(u8_t, u8, u8_t, u8, x)
IMPL_CAST_OP(u8_t, u8, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(u8_t, u8, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(u8_t, u8, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(u8_t, u8, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(u8_t, u8, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(u8_t, u8, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(u8_t, u8, i64_t, i64, (i64_t)x)

// ============================================================================
// U16 conversions
// ============================================================================
IMPL_CAST_OP(u16_t, u16, uint8_t, bool, (x != 0))
IMPL_CAST_OP(u16_t, u16, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(u16_t, u16, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(u16_t, u16, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(u16_t, u16, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(u16_t, u16, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(u16_t, u16, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(u16_t, u16, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(u16_t, u16, u16_t, u16, x)
IMPL_CAST_OP(u16_t, u16, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(u16_t, u16, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(u16_t, u16, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(u16_t, u16, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(u16_t, u16, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(u16_t, u16, i64_t, i64, (i64_t)x)

// ============================================================================
// U32 conversions
// ============================================================================
IMPL_CAST_OP(u32_t, u32, uint8_t, bool, (x != 0))
IMPL_CAST_OP(u32_t, u32, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(u32_t, u32, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(u32_t, u32, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(u32_t, u32, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(u32_t, u32, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(u32_t, u32, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(u32_t, u32, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(u32_t, u32, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(u32_t, u32, u32_t, u32, x)
IMPL_CAST_OP(u32_t, u32, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(u32_t, u32, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(u32_t, u32, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(u32_t, u32, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(u32_t, u32, i64_t, i64, (i64_t)x)

// ============================================================================
// U64 conversions
// ============================================================================
IMPL_CAST_OP(u64_t, u64, uint8_t, bool, (x != 0))
IMPL_CAST_OP(u64_t, u64, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(u64_t, u64, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(u64_t, u64, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(u64_t, u64, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(u64_t, u64, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(u64_t, u64, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(u64_t, u64, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(u64_t, u64, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(u64_t, u64, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(u64_t, u64, u64_t, u64, x)
IMPL_CAST_OP(u64_t, u64, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(u64_t, u64, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(u64_t, u64, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(u64_t, u64, i64_t, i64, (i64_t)x)

// ============================================================================
// I8 conversions
// ============================================================================
IMPL_CAST_OP(i8_t, i8, uint8_t, bool, (x != 0))
IMPL_CAST_OP(i8_t, i8, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(i8_t, i8, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(i8_t, i8, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(i8_t, i8, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(i8_t, i8, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(i8_t, i8, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(i8_t, i8, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(i8_t, i8, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(i8_t, i8, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(i8_t, i8, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(i8_t, i8, i8_t, i8, x)
IMPL_CAST_OP(i8_t, i8, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(i8_t, i8, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(i8_t, i8, i64_t, i64, (i64_t)x)

// ============================================================================
// I16 conversions
// ============================================================================
IMPL_CAST_OP(i16_t, i16, uint8_t, bool, (x != 0))
IMPL_CAST_OP(i16_t, i16, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(i16_t, i16, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(i16_t, i16, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(i16_t, i16, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(i16_t, i16, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(i16_t, i16, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(i16_t, i16, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(i16_t, i16, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(i16_t, i16, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(i16_t, i16, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(i16_t, i16, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(i16_t, i16, i16_t, i16, x)
IMPL_CAST_OP(i16_t, i16, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(i16_t, i16, i64_t, i64, (i64_t)x)

// ============================================================================
// I32 conversions
// ============================================================================
IMPL_CAST_OP(i32_t, i32, uint8_t, bool, (x != 0))
IMPL_CAST_OP(i32_t, i32, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(i32_t, i32, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(i32_t, i32, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(i32_t, i32, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(i32_t, i32, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(i32_t, i32, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(i32_t, i32, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(i32_t, i32, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(i32_t, i32, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(i32_t, i32, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(i32_t, i32, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(i32_t, i32, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(i32_t, i32, i32_t, i32, x)
IMPL_CAST_OP(i32_t, i32, i64_t, i64, (i64_t)x)

// ============================================================================
// I64 conversions
// ============================================================================
IMPL_CAST_OP(i64_t, i64, uint8_t, bool, (x != 0))
IMPL_CAST_OP(i64_t, i64, f8e4m3_t, f8e4m3, float_to_f8e4m3((float)x))
IMPL_CAST_OP(i64_t, i64, f8e5m2_t, f8e5m2, float_to_f8e5m2((float)x))
IMPL_CAST_OP(i64_t, i64, bf16_t, bf16, float_to_bf16((float)x))
IMPL_CAST_OP(i64_t, i64, f16_t, f16, float_to_f16((float)x))
IMPL_CAST_OP(i64_t, i64, f32_t, f32, (f32_t)x)
IMPL_CAST_OP(i64_t, i64, f64_t, f64, (f64_t)x)
IMPL_CAST_OP(i64_t, i64, u8_t, u8, (u8_t)x)
IMPL_CAST_OP(i64_t, i64, u16_t, u16, (u16_t)x)
IMPL_CAST_OP(i64_t, i64, u32_t, u32, (u32_t)x)
IMPL_CAST_OP(i64_t, i64, u64_t, u64, (u64_t)x)
IMPL_CAST_OP(i64_t, i64, i8_t, i8, (i8_t)x)
IMPL_CAST_OP(i64_t, i64, i16_t, i16, (i16_t)x)
IMPL_CAST_OP(i64_t, i64, i32_t, i32, (i32_t)x)
IMPL_CAST_OP(i64_t, i64, i64_t, i64, x)
