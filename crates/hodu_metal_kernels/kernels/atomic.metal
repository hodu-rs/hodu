#pragma once
#include <metal_stdlib>

using namespace metal;

// Atomic operations wrapper for Metal
// Provides atomic add/max/min operations for types that support them
// For types without hardware atomic support, falls back to non-atomic operations or CAS-based
// atomics

// ============================================================================
// ATOMIC ADD OPERATIONS
// ============================================================================

// Atomic add for bfloat using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_add_wrapper(device bfloat *addr, bfloat value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        bfloat current = as_type<bfloat>((uint16_t)((old_val >> offset) & 0xFFFF));
        bfloat updated = current + value;
        uint16_t updated_bits = as_type<uint16_t>(updated);
        new_val = (old_val & ~mask) | (((uint32_t)updated_bits) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic add for half using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_add_wrapper(device half *addr, half value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        half current = as_type<half>((uint16_t)((old_val >> offset) & 0xFFFF));
        half updated = current + value;
        uint16_t updated_bits = as_type<uint16_t>(updated);
        new_val = (old_val & ~mask) | (((uint32_t)updated_bits) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic add for float using compare-and-swap
// Thread-safe implementation for concurrent access
inline void atomic_add_wrapper(device float *addr, float value) {
    uint32_t old_val, new_val;
    do {
        old_val = as_type<uint32_t>(*addr);
        new_val = as_type<uint32_t>(as_type<float>(old_val) + value);
    } while (!atomic_compare_exchange_weak_explicit((device atomic_uint *)addr, &old_val, new_val,
                                                    memory_order_relaxed, memory_order_relaxed));
}

// Atomic add for uint8_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_add_wrapper(device uint8_t *addr, uint8_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        uint8_t current = (uint8_t)((old_val >> shift) & 0xFF);
        uint8_t updated = current + value;
        new_val = (old_val & ~mask) | (((uint32_t)updated) << shift);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic add for uint16_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_add_wrapper(device uint16_t *addr, uint16_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        uint16_t current = (uint16_t)((old_val >> offset) & 0xFFFF);
        uint16_t updated = current + value;
        new_val = (old_val & ~mask) | (((uint32_t)updated) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic add for uint32_t (natively supported)
inline void atomic_add_wrapper(device uint32_t *addr, uint32_t value) {
    atomic_fetch_add_explicit((device atomic_uint *)addr, value, memory_order_relaxed);
}

// Atomic add for uint64_t - fallback to non-atomic
// WARNING: Metal doesn't support 64-bit atomics on all devices
inline void atomic_add_wrapper(device uint64_t *addr, uint64_t value) { *addr = *addr + value; }

// Atomic add for int8_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_add_wrapper(device int8_t *addr, int8_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        int8_t current = (int8_t)((old_val >> shift) & 0xFF);
        int8_t updated = current + value;
        new_val = (old_val & ~mask) | (((uint32_t)(uint8_t)updated) << shift);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic add for int16_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_add_wrapper(device int16_t *addr, int16_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        int16_t current = (int16_t)((old_val >> offset) & 0xFFFF);
        int16_t updated = current + value;
        new_val = (old_val & ~mask) | (((uint32_t)(uint16_t)updated) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic add for int32_t (natively supported)
inline void atomic_add_wrapper(device int32_t *addr, int32_t value) {
    atomic_fetch_add_explicit((device atomic_int *)addr, value, memory_order_relaxed);
}

// Atomic add for int64_t - fallback to non-atomic
// WARNING: Metal doesn't support 64-bit atomics on all devices
inline void atomic_add_wrapper(device int64_t *addr, int64_t value) { *addr = *addr + value; }

// ============================================================================
// ATOMIC MAX OPERATIONS
// ============================================================================

// Atomic max for bfloat using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_max_wrapper(device bfloat *addr, bfloat value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        bfloat current = as_type<bfloat>((uint16_t)((old_val >> offset) & 0xFFFF));
        bfloat updated = (current > value) ? current : value;
        uint16_t updated_bits = as_type<uint16_t>(updated);
        new_val = (old_val & ~mask) | (((uint32_t)updated_bits) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic max for half using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_max_wrapper(device half *addr, half value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        half current = as_type<half>((uint16_t)((old_val >> offset) & 0xFFFF));
        half updated = metal::max(current, value);
        uint16_t updated_bits = as_type<uint16_t>(updated);
        new_val = (old_val & ~mask) | (((uint32_t)updated_bits) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic max for float using compare-and-swap
// Thread-safe implementation for concurrent access
inline void atomic_max_wrapper(device float *addr, float value) {
    uint32_t old_val, new_val;
    do {
        old_val = as_type<uint32_t>(*addr);
        float current = as_type<float>(old_val);
        float updated = metal::max(current, value);
        new_val = as_type<uint32_t>(updated);
    } while (!atomic_compare_exchange_weak_explicit((device atomic_uint *)addr, &old_val, new_val,
                                                    memory_order_relaxed, memory_order_relaxed));
}

// Atomic max for uint8_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_max_wrapper(device uint8_t *addr, uint8_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        uint8_t current = (uint8_t)((old_val >> shift) & 0xFF);
        uint8_t updated = metal::max(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)updated) << shift);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic max for uint16_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_max_wrapper(device uint16_t *addr, uint16_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        uint16_t current = (uint16_t)((old_val >> offset) & 0xFFFF);
        uint16_t updated = metal::max(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)updated) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic max for uint32_t (natively supported)
inline void atomic_max_wrapper(device uint32_t *addr, uint32_t value) {
    atomic_fetch_max_explicit((device atomic_uint *)addr, value, memory_order_relaxed);
}

// Atomic max for uint64_t - fallback to non-atomic
// WARNING: Metal doesn't support 64-bit atomics on all devices
inline void atomic_max_wrapper(device uint64_t *addr, uint64_t value) {
    *addr = metal::max(*addr, value);
}

// Atomic max for int8_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_max_wrapper(device int8_t *addr, int8_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        int8_t current = (int8_t)((old_val >> shift) & 0xFF);
        int8_t updated = metal::max(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)(uint8_t)updated) << shift);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic max for int16_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_max_wrapper(device int16_t *addr, int16_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        int16_t current = (int16_t)((old_val >> offset) & 0xFFFF);
        int16_t updated = metal::max(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)(uint16_t)updated) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic max for int32_t (natively supported)
inline void atomic_max_wrapper(device int32_t *addr, int32_t value) {
    atomic_fetch_max_explicit((device atomic_int *)addr, value, memory_order_relaxed);
}

// Atomic max for int64_t - fallback to non-atomic
// WARNING: Metal doesn't support 64-bit atomics on all devices
inline void atomic_max_wrapper(device int64_t *addr, int64_t value) {
    *addr = metal::max(*addr, value);
}

// ====================================//
// ============================================================================ ATOMIC MIN
// OPERATIONS
// ============================================================================

// Atomic min for bfloat using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_min_wrapper(device bfloat *addr, bfloat value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        bfloat current = as_type<bfloat>((uint16_t)((old_val >> offset) & 0xFFFF));
        bfloat updated = (current < value) ? current : value;
        uint16_t updated_bits = as_type<uint16_t>(updated);
        new_val = (old_val & ~mask) | (((uint32_t)updated_bits) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic min for half using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_min_wrapper(device half *addr, half value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        half current = as_type<half>((uint16_t)((old_val >> offset) & 0xFFFF));
        half updated = metal::min(current, value);
        uint16_t updated_bits = as_type<uint16_t>(updated);
        new_val = (old_val & ~mask) | (((uint32_t)updated_bits) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic min for float using compare-and-swap
// Thread-safe implementation for concurrent access
inline void atomic_min_wrapper(device float *addr, float value) {
    uint32_t old_val, new_val;
    do {
        old_val = as_type<uint32_t>(*addr);
        float current = as_type<float>(old_val);
        float updated = metal::min(current, value);
        new_val = as_type<uint32_t>(updated);
    } while (!atomic_compare_exchange_weak_explicit((device atomic_uint *)addr, &old_val, new_val,
                                                    memory_order_relaxed, memory_order_relaxed));
}

// Atomic min for uint8_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_min_wrapper(device uint8_t *addr, uint8_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        uint8_t current = (uint8_t)((old_val >> shift) & 0xFF);
        uint8_t updated = metal::min(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)updated) << shift);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic min for uint16_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_min_wrapper(device uint16_t *addr, uint16_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        uint16_t current = (uint16_t)((old_val >> offset) & 0xFFFF);
        uint16_t updated = metal::min(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)updated) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic min for uint32_t (natively supported)
inline void atomic_min_wrapper(device uint32_t *addr, uint32_t value) {
    atomic_fetch_min_explicit((device atomic_uint *)addr, value, memory_order_relaxed);
}

// Atomic min for uint64_t - fallback to non-atomic
// WARNING: Metal doesn't support 64-bit atomics on all devices
inline void atomic_min_wrapper(device uint64_t *addr, uint64_t value) {
    *addr = metal::min(*addr, value);
}

// Atomic min for int8_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_min_wrapper(device int8_t *addr, int8_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = (size_t)addr & 3;
    uint32_t shift = offset * 8;
    uint32_t mask = 0xFF << shift;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        int8_t current = (int8_t)((old_val >> shift) & 0xFF);
        int8_t updated = metal::min(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)(uint8_t)updated) << shift);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic min for int16_t using compare-and-swap on aligned uint32_t
// Thread-safe implementation for concurrent access
inline void atomic_min_wrapper(device int16_t *addr, int16_t value) {
    device atomic_uint *base = (device atomic_uint *)((size_t)addr & ~3);
    uint32_t offset = ((size_t)addr & 2) ? 16 : 0;
    uint32_t mask = 0xFFFF << offset;

    uint32_t old_val, new_val;
    do {
        old_val = atomic_load_explicit(base, memory_order_relaxed);
        int16_t current = (int16_t)((old_val >> offset) & 0xFFFF);
        int16_t updated = metal::min(current, value);
        new_val = (old_val & ~mask) | (((uint32_t)(uint16_t)updated) << offset);
    } while (!atomic_compare_exchange_weak_explicit(base, &old_val, new_val, memory_order_relaxed,
                                                    memory_order_relaxed));
}

// Atomic min for int32_t (natively supported)
inline void atomic_min_wrapper(device int32_t *addr, int32_t value) {
    atomic_fetch_min_explicit((device atomic_int *)addr, value, memory_order_relaxed);
}

// Atomic min for int64_t - fallback to non-atomic
// WARNING: Metal doesn't support 64-bit atomics on all devices
inline void atomic_min_wrapper(device int64_t *addr, int64_t value) {
    *addr = metal::min(*addr, value);
}
