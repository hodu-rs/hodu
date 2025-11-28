// Hodu Metal Kernels (placeholder)
// TODO: Bundle actual hodu_metal_kernels source

#include <metal_stdlib>
using namespace metal;

// Placeholder kernel - actual kernels will be bundled from hodu_metal_kernels
kernel void hodu_metal_placeholder(device float *output [[buffer(0)]],
                                   uint id [[thread_position_in_grid]]) {
    output[id] = 0.0f;
}
