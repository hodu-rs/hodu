#include "ops_memory.h"
#include "thread_utils.h"
#include "types.h"
#include <string.h>

// ============================================================================
// MEMORY OPERATION IMPLEMENTATION
// ============================================================================
//
// Memory operations handle tensor memory layout transformations.
// The contiguous operation copies strided tensor data to contiguous memory.

/**
 * @brief Macro to implement contiguous copy operation
 *
 * Copies tensor data from potentially strided layout to contiguous memory.
 * Optimizes for already-contiguous tensors with memcpy.
 */
#define IMPL_CONTIGUOUS_OP(TYPE, TYPE_SUFFIX)                                                      \
    typedef struct {                                                                               \
        const TYPE *input;                                                                         \
        TYPE *output;                                                                              \
        size_t start;                                                                              \
        size_t end;                                                                                \
        size_t offset;                                                                             \
    } contiguous_##TYPE_SUFFIX##_args_t;                                                           \
                                                                                                   \
    static void *contiguous_##TYPE_SUFFIX##_worker(void *arg) {                                    \
        contiguous_##TYPE_SUFFIX##_args_t *args = (contiguous_##TYPE_SUFFIX##_args_t *)arg;        \
        for (size_t i = args->start; i < args->end; i++) {                                         \
            args->output[i] = args->input[args->offset + i];                                       \
        }                                                                                          \
        return NULL;                                                                               \
    }                                                                                              \
                                                                                                   \
    void hodu_cpu_contiguous_##TYPE_SUFFIX(const void *input, void *output,                        \
                                           const size_t *metadata) {                               \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const TYPE *in = (const TYPE *)input;                                                      \
        TYPE *out = (TYPE *)output;                                                                \
                                                                                                   \
        const size_t *dims = metadata + 2;                                                         \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = (num_dims > 0) ? metadata[2 + 2 * num_dims] : 0;                     \
                                                                                                   \
        bool contiguous = is_contiguous(num_dims, dims, strides);                                  \
                                                                                                   \
        if (contiguous) {                                                                          \
            /* Fast path: already contiguous, use memcpy */                                        \
            memcpy(out, in + offset, num_els * sizeof(TYPE));                                      \
        } else {                                                                                   \
            /* Slow path: strided access */                                                        \
            const size_t min_work_per_thread = 100000;                                             \
            size_t num_threads = get_optimal_threads(num_els, min_work_per_thread);                \
                                                                                                   \
            if (num_threads > 1 && contiguous) {                                                   \
                /* Multi-threaded contiguous copy (won't reach here, but for completeness) */      \
                thread_t threads[num_threads];                                                     \
                contiguous_##TYPE_SUFFIX##_args_t args[num_threads];                               \
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
                    thread_create(&threads[t], contiguous_##TYPE_SUFFIX##_worker, &args[t]);       \
                }                                                                                  \
                                                                                                   \
                for (size_t t = 0; t < num_threads; t++) {                                         \
                    thread_join(threads[t]);                                                       \
                }                                                                                  \
            } else {                                                                               \
                /* Single-threaded strided copy */                                                 \
                for (size_t i = 0; i < num_els; i++) {                                             \
                    size_t strided_i = offset + get_strided_index(i, num_dims, dims, strides);     \
                    out[i] = in[strided_i];                                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// ============================================================================
// Contiguous operations for all types
// ============================================================================

IMPL_CONTIGUOUS_OP(uint8_t, bool)
IMPL_CONTIGUOUS_OP(f8e4m3_t, f8e4m3)
IMPL_CONTIGUOUS_OP(f8e5m2_t, f8e5m2)
IMPL_CONTIGUOUS_OP(bf16_t, bf16)
IMPL_CONTIGUOUS_OP(f16_t, f16)
IMPL_CONTIGUOUS_OP(f32_t, f32)
IMPL_CONTIGUOUS_OP(f64_t, f64)
IMPL_CONTIGUOUS_OP(u8_t, u8)
IMPL_CONTIGUOUS_OP(u16_t, u16)
IMPL_CONTIGUOUS_OP(u32_t, u32)
IMPL_CONTIGUOUS_OP(u64_t, u64)
IMPL_CONTIGUOUS_OP(i8_t, i8)
IMPL_CONTIGUOUS_OP(i16_t, i16)
IMPL_CONTIGUOUS_OP(i32_t, i32)
IMPL_CONTIGUOUS_OP(i64_t, i64)
