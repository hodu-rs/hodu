#ifndef HODU_CPU_KERNELS_ATOMIC_H
#define HODU_CPU_KERNELS_ATOMIC_H

#include "types.h"
#include <stdatomic.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline void atomic_add_f32(float *addr, float val) {
    union {
        float f;
        uint32_t i;
    } old_val, new_val;

    uint32_t *addr_as_int = (uint32_t *)addr;
    old_val.i = atomic_load_explicit((_Atomic uint32_t *)addr_as_int, memory_order_relaxed);

    do {
        new_val.f = old_val.f + val;
    } while (!atomic_compare_exchange_weak_explicit((_Atomic uint32_t *)addr_as_int, &old_val.i,
                                                    new_val.i, memory_order_relaxed,
                                                    memory_order_relaxed));
}

static inline void atomic_add_f64(double *addr, double val) {
    union {
        double d;
        uint64_t i;
    } old_val, new_val;

    uint64_t *addr_as_int = (uint64_t *)addr;
    old_val.i = atomic_load_explicit((_Atomic uint64_t *)addr_as_int, memory_order_relaxed);

    do {
        new_val.d = old_val.d + val;
    } while (!atomic_compare_exchange_weak_explicit((_Atomic uint64_t *)addr_as_int, &old_val.i,
                                                    new_val.i, memory_order_relaxed,
                                                    memory_order_relaxed));
}

static inline void atomic_add_f16(f16_t *addr, f16_t val) {
    union {
        uint16_t i;
    } old_val, new_val;

    old_val.i = atomic_load_explicit((_Atomic uint16_t *)addr, memory_order_relaxed);

    do {
        float old_f = fp16_to_float(old_val.i);
        float val_f = fp16_to_float(val);
        new_val.i = float_to_fp16(old_f + val_f);
    } while (!atomic_compare_exchange_weak_explicit((_Atomic uint16_t *)addr, &old_val.i, new_val.i,
                                                    memory_order_relaxed, memory_order_relaxed));
}

static inline void atomic_add_bf16(bf16_t *addr, bf16_t val) {
    union {
        uint16_t i;
    } old_val, new_val;

    old_val.i = atomic_load_explicit((_Atomic uint16_t *)addr, memory_order_relaxed);

    do {
        float old_f = bf16_to_float(old_val.i);
        float val_f = bf16_to_float(val);
        new_val.i = float_to_bf16(old_f + val_f);
    } while (!atomic_compare_exchange_weak_explicit((_Atomic uint16_t *)addr, &old_val.i, new_val.i,
                                                    memory_order_relaxed, memory_order_relaxed));
}

static inline void atomic_add_f8e4m3(f8e4m3_t *addr, f8e4m3_t val) {
    union {
        uint8_t i;
    } old_val, new_val;

    old_val.i = atomic_load_explicit((_Atomic uint8_t *)addr, memory_order_relaxed);

    do {
        float old_f = fp8_e4m3_to_float(old_val.i);
        float val_f = fp8_e4m3_to_float(val);
        new_val.i = float_to_fp8_e4m3(old_f + val_f);
    } while (!atomic_compare_exchange_weak_explicit((_Atomic uint8_t *)addr, &old_val.i, new_val.i,
                                                    memory_order_relaxed, memory_order_relaxed));
}

static inline void atomic_add_f8e5m2(f8e5m2_t *addr, f8e5m2_t val) {
    union {
        uint8_t i;
    } old_val, new_val;

    old_val.i = atomic_load_explicit((_Atomic uint8_t *)addr, memory_order_relaxed);

    do {
        float old_f = fp8_e5m2_to_float(old_val.i);
        float val_f = fp8_e5m2_to_float(val);
        new_val.i = float_to_fp8_e5m2(old_f + val_f);
    } while (!atomic_compare_exchange_weak_explicit((_Atomic uint8_t *)addr, &old_val.i, new_val.i,
                                                    memory_order_relaxed, memory_order_relaxed));
}

#ifdef __cplusplus
}
#endif

#endif // HODU_CPU_KERNELS_ATOMIC_H
