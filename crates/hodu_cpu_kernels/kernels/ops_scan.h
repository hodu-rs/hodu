#ifndef OPS_SCAN_H
#define OPS_SCAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Cumulative sum operations
// Metadata layout:
// - metadata[0]: num_els (total number of elements)
// - metadata[1]: num_dims (number of dimensions)
// - metadata[2..2+num_dims]: shape
// - metadata[2+num_dims..2+2*num_dims]: strides
// - metadata[2+2*num_dims]: offset
// - metadata[3+2*num_dims]: dim (dimension to scan along)

void hodu_cpu_cumsum_f8e4m3(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_f8e5m2(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_bf16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_f16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_f32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_f64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_u8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_u16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_u32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_u64(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_i8(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_i16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_i32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_cumsum_i64(const void *input, void *output, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif
