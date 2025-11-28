#include "storage.h"
#include "thread_utils.h"
#include "types.h"
#include <string.h>

// ============================================================================
// STORAGE OPERATION IMPLEMENTATION
// ============================================================================
//
// Storage operations handle low-level tensor buffer manipulation.
// The const_set operation fills a tensor with a constant value.

/**
 * @brief Macro to implement const_set operation
 *
 * Fills tensor with a constant value. Supports both contiguous and strided access.
 */
#define IMPL_CONST_SET_OP(TYPE, TYPE_SUFFIX)                                                       \
    typedef struct {                                                                               \
        TYPE *output;                                                                              \
        TYPE value;                                                                                \
        size_t start;                                                                              \
        size_t end;                                                                                \
        size_t offset;                                                                             \
    } const_set_##TYPE_SUFFIX##_args_t;                                                            \
                                                                                                   \
    static void *const_set_##TYPE_SUFFIX##_worker(void *arg) {                                     \
        const_set_##TYPE_SUFFIX##_args_t *args = (const_set_##TYPE_SUFFIX##_args_t *)arg;          \
        for (size_t i = args->start; i < args->end; i++) {                                         \
            args->output[args->offset + i] = args->value;                                          \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_const_set_##TYPE_SUFFIX(void *output, const size_t *metadata,                    \
                                          const void *value) {                                     \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        TYPE *out = (TYPE *)output;                                                                \
        TYPE val = *(const TYPE *)value;                                                           \
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
                const_set_##TYPE_SUFFIX##_args_t args[num_threads];                                \
                                                                                                   \
                size_t chunk_size = num_els / num_threads;                                         \
                size_t remaining = num_els % num_threads;                                          \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    args[t].output = out;                                                          \
                    args[t].value = val;                                                           \
                    args[t].offset = offset;                                                       \
                    args[t].start = t * chunk_size + (t < remaining ? t : remaining);              \
                    args[t].end = args[t].start + chunk_size + (t < remaining ? 1 : 0);            \
                    thread_create(&threads[t], const_set_##TYPE_SUFFIX##_worker, &args[t]);        \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    out[offset + i] = val;                                                         \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t i = 0; i < num_els; i++) {                                                 \
                size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);         \
                out[strided_i] = val;                                                              \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// const_set operations for all types
// ============================================================================

IMPL_CONST_SET_OP(uint8_t, bool)
IMPL_CONST_SET_OP(f8e4m3_t, f8e4m3)
IMPL_CONST_SET_OP(f8e5m2_t, f8e5m2)
IMPL_CONST_SET_OP(bf16_t, bf16)
IMPL_CONST_SET_OP(f16_t, f16)
IMPL_CONST_SET_OP(f32_t, f32)
IMPL_CONST_SET_OP(f64_t, f64)
IMPL_CONST_SET_OP(u8_t, u8)
IMPL_CONST_SET_OP(u16_t, u16)
IMPL_CONST_SET_OP(u32_t, u32)
IMPL_CONST_SET_OP(u64_t, u64)
IMPL_CONST_SET_OP(i8_t, i8)
IMPL_CONST_SET_OP(i16_t, i16)
IMPL_CONST_SET_OP(i32_t, i32)
IMPL_CONST_SET_OP(i64_t, i64)
