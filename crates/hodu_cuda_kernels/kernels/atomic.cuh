#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ void atomic_add_f8e4m3(__nv_fp8_e4m3 *addr, __nv_fp8_e4m3 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_fp8_e4m3 current = *(__nv_fp8_e4m3 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_fp8_e4m3 updated = __nv_fp8_e4m3((float)current + (float)value);
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_f8e5m2(__nv_fp8_e5m2 *addr, __nv_fp8_e5m2 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_fp8_e5m2 current = *(__nv_fp8_e5m2 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_fp8_e5m2 updated = __nv_fp8_e5m2((float)current + (float)value);
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16 *addr, __nv_bfloat16 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_bfloat16 current = *(__nv_bfloat16 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_bfloat16 updated = __nv_bfloat16((float)current + (float)value);
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_f16(__half *addr, __half value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __half current = *((__half *)((uint8_t *)&assumed + ((size_t)addr & 3)));
        __half updated = __hadd(current, value);
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_f32(float *addr, float value) { atomicAdd(addr, value); }

__device__ __forceinline__ void atomic_add_f64(double *addr, double value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        old_val = atomicCAS(addr_as_ull, assumed,
                            __double_as_longlong(__longlong_as_double(assumed) + value));
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_i8(int8_t *addr, int8_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        int8_t current = (int8_t)((assumed >> shift) & 0xFF);
        int8_t updated = current + value;
        new_val = (assumed & ~mask) | (((uint32_t)(uint8_t)updated) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_i16(int16_t *addr, int16_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        int16_t current = (int16_t)((assumed >> offset) & 0xFFFF);
        int16_t updated = current + value;
        new_val = (assumed & ~mask) | (((uint32_t)(uint16_t)updated) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_i32(int32_t *addr, int32_t value) {
    atomicAdd(addr, value);
}

__device__ __forceinline__ void atomic_add_i64(int64_t *addr, int64_t value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        old_val = atomicCAS(addr_as_ull, assumed, assumed + (unsigned long long)value);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_u8(uint8_t *addr, uint8_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        uint8_t current = (uint8_t)((assumed >> shift) & 0xFF);
        uint8_t updated = current + value;
        new_val = (assumed & ~mask) | (((uint32_t)updated) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_u16(uint16_t *addr, uint16_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        uint16_t current = (uint16_t)((assumed >> offset) & 0xFFFF);
        uint16_t updated = current + value;
        new_val = (assumed & ~mask) | (((uint32_t)updated) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_add_u32(uint32_t *addr, uint32_t value) {
    atomicAdd(addr, value);
}

__device__ __forceinline__ void atomic_add_u64(uint64_t *addr, uint64_t value) {
    atomicAdd((unsigned long long *)addr, (unsigned long long)value);
}

__device__ __forceinline__ void atomic_max_f8e4m3(__nv_fp8_e4m3 *addr, __nv_fp8_e4m3 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_fp8_e4m3 current = *(__nv_fp8_e4m3 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_fp8_e4m3 updated = ((float)current > (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_f8e5m2(__nv_fp8_e5m2 *addr, __nv_fp8_e5m2 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_fp8_e5m2 current = *(__nv_fp8_e5m2 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_fp8_e5m2 updated = ((float)current > (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_bf16(__nv_bfloat16 *addr, __nv_bfloat16 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_bfloat16 current = *(__nv_bfloat16 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_bfloat16 updated = ((float)current > (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_f16(__half *addr, __half value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __half current = *((__half *)((uint8_t *)&assumed + ((size_t)addr & 3)));
        __half updated = ((float)current > (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_f32(float *addr, float value) {
    uint32_t *addr_as_uint = (uint32_t *)addr;
    uint32_t old_val = *addr_as_uint, assumed;
    do {
        assumed = old_val;
        float current = __uint_as_float(assumed);
        float updated = fmaxf(current, value);
        old_val = atomicCAS(addr_as_uint, assumed, __float_as_uint(updated));
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_f64(double *addr, double value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        double current = __longlong_as_double(assumed);
        double updated = fmax(current, value);
        old_val = atomicCAS(addr_as_ull, assumed, __double_as_longlong(updated));
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_i8(int8_t *addr, int8_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        int8_t current = (int8_t)((assumed >> shift) & 0xFF);
        int8_t updated = (current > value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)(uint8_t)updated) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_i16(int16_t *addr, int16_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        int16_t current = (int16_t)((assumed >> offset) & 0xFFFF);
        int16_t updated = (current > value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)(uint16_t)updated) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_i32(int32_t *addr, int32_t value) {
    atomicMax(addr, value);
}

__device__ __forceinline__ void atomic_max_i64(int64_t *addr, int64_t value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        int64_t current = (int64_t)assumed;
        int64_t updated = (current > value) ? current : value;
        old_val = atomicCAS(addr_as_ull, assumed, (unsigned long long)updated);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_u8(uint8_t *addr, uint8_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        uint8_t current = (uint8_t)((assumed >> shift) & 0xFF);
        uint8_t updated = (current > value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)updated) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_u16(uint16_t *addr, uint16_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        uint16_t current = (uint16_t)((assumed >> offset) & 0xFFFF);
        uint16_t updated = (current > value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)updated) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_max_u32(uint32_t *addr, uint32_t value) {
    atomicMax(addr, value);
}

__device__ __forceinline__ void atomic_max_u64(uint64_t *addr, uint64_t value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        unsigned long long updated =
            (assumed > (unsigned long long)value) ? assumed : (unsigned long long)value;
        old_val = atomicCAS(addr_as_ull, assumed, updated);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_f8e4m3(__nv_fp8_e4m3 *addr, __nv_fp8_e4m3 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_fp8_e4m3 current = *(__nv_fp8_e4m3 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_fp8_e4m3 updated = ((float)current < (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_f8e5m2(__nv_fp8_e5m2 *addr, __nv_fp8_e5m2 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_fp8_e5m2 current = *(__nv_fp8_e5m2 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_fp8_e5m2 updated = ((float)current < (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_bf16(__nv_bfloat16 *addr, __nv_bfloat16 value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __nv_bfloat16 current = *(__nv_bfloat16 *)((uint8_t *)&assumed + ((size_t)addr & 3));
        __nv_bfloat16 updated = ((float)current < (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_f16(__half *addr, __half value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        __half current = *((__half *)((uint8_t *)&assumed + ((size_t)addr & 3)));
        __half updated = ((float)current < (float)value) ? current : value;
        new_val = (assumed & ~mask) | ((*((uint16_t *)&updated)) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_f32(float *addr, float value) {
    uint32_t *addr_as_uint = (uint32_t *)addr;
    uint32_t old_val = *addr_as_uint, assumed;
    do {
        assumed = old_val;
        float current = __uint_as_float(assumed);
        float updated = fminf(current, value);
        old_val = atomicCAS(addr_as_uint, assumed, __float_as_uint(updated));
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_f64(double *addr, double value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        double current = __longlong_as_double(assumed);
        double updated = fmin(current, value);
        old_val = atomicCAS(addr_as_ull, assumed, __double_as_longlong(updated));
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_i8(int8_t *addr, int8_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        int8_t current = (int8_t)((assumed >> shift) & 0xFF);
        int8_t updated = (current < value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)(uint8_t)updated) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_i16(int16_t *addr, int16_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        int16_t current = (int16_t)((assumed >> offset) & 0xFFFF);
        int16_t updated = (current < value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)(uint16_t)updated) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_i32(int32_t *addr, int32_t value) {
    atomicMin(addr, value);
}

__device__ __forceinline__ void atomic_min_i64(int64_t *addr, int64_t value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        int64_t current = (int64_t)assumed;
        int64_t updated = (current < value) ? current : value;
        old_val = atomicCAS(addr_as_ull, assumed, (unsigned long long)updated);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_u8(uint8_t *addr, uint8_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        uint8_t current = (uint8_t)((assumed >> shift) & 0xFF);
        uint8_t updated = (current < value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)updated) << shift);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_u16(uint16_t *addr, uint16_t value) {
    uint32_t *base = (uint32_t *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;
    uint32_t old_val, new_val, assumed;
    old_val = *base;
    do {
        assumed = old_val;
        uint16_t current = (uint16_t)((assumed >> offset) & 0xFFFF);
        uint16_t updated = (current < value) ? current : value;
        new_val = (assumed & ~mask) | (((uint32_t)updated) << offset);
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
}

__device__ __forceinline__ void atomic_min_u32(uint32_t *addr, uint32_t value) {
    atomicMin(addr, value);
}

__device__ __forceinline__ void atomic_min_u64(uint64_t *addr, uint64_t value) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long old_val = *addr_as_ull, assumed;
    do {
        assumed = old_val;
        unsigned long long updated =
            (assumed < (unsigned long long)value) ? assumed : (unsigned long long)value;
        old_val = atomicCAS(addr_as_ull, assumed, updated);
    } while (assumed != old_val);
}
