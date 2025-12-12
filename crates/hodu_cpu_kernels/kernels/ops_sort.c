#include "ops_sort.h"
#include "types.h"
#include <stdlib.h>

// Helper struct for sorting with indices
typedef struct {
    float value;
    i32_t index;
} ValueIndex;

// Comparison functions for qsort
static int cmp_largest(const void *a, const void *b) {
    float va = ((ValueIndex *)a)->value;
    float vb = ((ValueIndex *)b)->value;
    if (va > vb)
        return -1;
    if (va < vb)
        return 1;
    return 0;
}

static int cmp_smallest(const void *a, const void *b) {
    float va = ((ValueIndex *)a)->value;
    float vb = ((ValueIndex *)b)->value;
    if (va < vb)
        return -1;
    if (va > vb)
        return 1;
    return 0;
}

// Partition for quickselect
static size_t partition(ValueIndex *arr, size_t left, size_t right, int largest) {
    float pivot = arr[right].value;
    size_t i = left;
    for (size_t j = left; j < right; j++) {
        int should_swap = largest ? (arr[j].value > pivot) : (arr[j].value < pivot);
        if (should_swap) {
            ValueIndex tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
        }
    }
    ValueIndex tmp = arr[i];
    arr[i] = arr[right];
    arr[right] = tmp;
    return i;
}

// QuickSelect to find k-th element and partition array
static void quickselect_k(ValueIndex *arr, size_t n, size_t k, int largest) {
    if (n <= 1 || k == 0)
        return;

    size_t left = 0;
    size_t right = n - 1;

    while (left < right) {
        size_t pivot_idx = partition(arr, left, right, largest);
        if (pivot_idx == k - 1) {
            break;
        } else if (pivot_idx < k - 1) {
            left = pivot_idx + 1;
        } else {
            right = pivot_idx - 1;
        }
    }
}

// Select top-k elements using partial sort
static void topk_select(ValueIndex *arr, size_t n, size_t k, int largest, int sorted) {
    if (k >= n) {
        // If k >= n, just sort everything
        if (largest) {
            qsort(arr, n, sizeof(ValueIndex), cmp_largest);
        } else {
            qsort(arr, n, sizeof(ValueIndex), cmp_smallest);
        }
        return;
    }

    // Use quickselect to partition first k elements
    quickselect_k(arr, n, k, largest);

    // Sort the first k elements if sorted is requested
    if (sorted) {
        if (largest) {
            qsort(arr, k, sizeof(ValueIndex), cmp_largest);
        } else {
            qsort(arr, k, sizeof(ValueIndex), cmp_smallest);
        }
    }
}

#define IMPL_TOPK(TYPE, TYPE_SUFFIX, TO_FLOAT)                                                     \
    void hodu_cpu_topk_##TYPE_SUFFIX(const void *input, void *values, void *indices,               \
                                     const size_t *metadata) {                                     \
        const size_t k = metadata[1];                                                              \
        const size_t last_dim_size = metadata[2];                                                  \
        const size_t outer_size = metadata[3];                                                     \
        const int largest = (int)metadata[4];                                                      \
        const int sorted = (int)metadata[5];                                                       \
        const size_t offset = metadata[6];                                                         \
                                                                                                   \
        const TYPE *in = (const TYPE *)input + offset;                                             \
        TYPE *val_out = (TYPE *)values;                                                            \
        i32_t *idx_out = (i32_t *)indices;                                                         \
                                                                                                   \
        /* Allocate temporary buffer for sorting */                                                \
        ValueIndex *temp = (ValueIndex *)malloc(last_dim_size * sizeof(ValueIndex));               \
        if (!temp)                                                                                 \
            return;                                                                                \
                                                                                                   \
        for (size_t outer = 0; outer < outer_size; outer++) {                                      \
            const TYPE *row = in + outer * last_dim_size;                                          \
                                                                                                   \
            /* Fill temporary buffer with values and indices */                                    \
            for (size_t i = 0; i < last_dim_size; i++) {                                           \
                temp[i].value = TO_FLOAT(row[i]);                                                  \
                temp[i].index = (i32_t)i;                                                          \
            }                                                                                      \
                                                                                                   \
            /* Select top-k elements */                                                            \
            topk_select(temp, last_dim_size, k, largest, sorted);                                  \
                                                                                                   \
            /* Copy top-k results to output */                                                     \
            TYPE *val_row = val_out + outer * k;                                                   \
            i32_t *idx_row = idx_out + outer * k;                                                  \
            for (size_t i = 0; i < k; i++) {                                                       \
                val_row[i] = row[temp[i].index];                                                   \
                idx_row[i] = temp[i].index;                                                        \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        free(temp);                                                                                \
    }

#define IDENTITY(x) (x)

IMPL_TOPK(f32_t, f32, IDENTITY)
IMPL_TOPK(f64_t, f64, IDENTITY)
IMPL_TOPK(u8_t, u8, IDENTITY)
IMPL_TOPK(u16_t, u16, IDENTITY)
IMPL_TOPK(u32_t, u32, IDENTITY)
IMPL_TOPK(u64_t, u64, IDENTITY)
IMPL_TOPK(i8_t, i8, IDENTITY)
IMPL_TOPK(i16_t, i16, IDENTITY)
IMPL_TOPK(i32_t, i32, IDENTITY)
IMPL_TOPK(i64_t, i64, IDENTITY)
IMPL_TOPK(bf16_t, bf16, bf16_to_float)
IMPL_TOPK(f16_t, f16, f16_to_float)
IMPL_TOPK(f8e4m3_t, f8e4m3, f8e4m3_to_float)
IMPL_TOPK(f8e5m2_t, f8e5m2, f8e5m2_to_float)
