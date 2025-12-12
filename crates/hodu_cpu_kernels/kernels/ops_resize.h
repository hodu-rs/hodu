#ifndef HODU_CPU_OPS_RESIZE_H
#define HODU_CPU_OPS_RESIZE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Resize mode
#define RESIZE_MODE_NEAREST 0
#define RESIZE_MODE_LINEAR 1
#define RESIZE_MODE_CUBIC 2

// Coordinate transformation mode
#define RESIZE_COORD_HALF_PIXEL 0
#define RESIZE_COORD_ASYMMETRIC 1
#define RESIZE_COORD_ALIGN_CORNERS 2
#define RESIZE_COORD_PYTORCH_HALF_PIXEL 3

// Nearest mode (rounding)
#define RESIZE_NEAREST_FLOOR 0
#define RESIZE_NEAREST_CEIL 1
#define RESIZE_NEAREST_ROUND_PREFER_FLOOR 2
#define RESIZE_NEAREST_ROUND_PREFER_CEIL 3

void hodu_cpu_resize_f8e4m3(const void *input, void *output, const size_t *metadata);
void hodu_cpu_resize_f8e5m2(const void *input, void *output, const size_t *metadata);
void hodu_cpu_resize_bf16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_resize_f16(const void *input, void *output, const size_t *metadata);
void hodu_cpu_resize_f32(const void *input, void *output, const size_t *metadata);
void hodu_cpu_resize_f64(const void *input, void *output, const size_t *metadata);

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_OPS_RESIZE_H
