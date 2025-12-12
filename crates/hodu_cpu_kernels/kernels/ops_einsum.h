#ifndef EINSUM_H
#define EINSUM_H

#include <stddef.h>

void hodu_cpu_einsum_f32(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_f64(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_u8(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_u16(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_u32(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_u64(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_i8(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_i16(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_i32(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_i64(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_bf16(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_f16(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_f8e4m3(const void **inputs, void *output, const size_t *metadata);
void hodu_cpu_einsum_f8e5m2(const void **inputs, void *output, const size_t *metadata);

#endif
