#ifndef OPS_SORT_H
#define OPS_SORT_H

#include <stddef.h>

void hodu_cpu_topk_f32(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_f64(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_u8(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_u16(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_u32(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_u64(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_i8(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_i16(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_i32(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_i64(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_bf16(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_f16(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_f8e4m3(const void *input, void *values, void *indices, const size_t *metadata);
void hodu_cpu_topk_f8e5m2(const void *input, void *values, void *indices, const size_t *metadata);

#endif // OPS_SORT_H
