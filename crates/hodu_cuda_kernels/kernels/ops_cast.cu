#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CAST_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, CAST_EXPR)                                     \
    extern "C" __global__ void FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *out,                \
                                       const size_t *metadata) {                                   \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *dims = metadata + 2;                                                         \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        bool cont = is_contiguous(num_dims, dims, strides);                                        \
        if (cont) {                                                                                \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                IN_TYPENAME x = input[offset + i];                                                 \
                out[i] = CAST_EXPR;                                                                \
            }                                                                                      \
        } else {                                                                                   \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t idx = offset + get_strided_index(i, num_dims, dims, strides);             \
                IN_TYPENAME x = input[idx];                                                        \
                out[i] = CAST_EXPR;                                                                \
            }                                                                                      \
        }                                                                                          \
    }

CAST_OP(bool, __nv_fp8_e4m3, cast_bool_to_f8e4m3, x ? __nv_fp8_e4m3(1.0f) : __nv_fp8_e4m3(0.0f))
CAST_OP(bool, __nv_fp8_e5m2, cast_bool_to_f8e5m2, x ? __nv_fp8_e5m2(1.0f) : __nv_fp8_e5m2(0.0f))
CAST_OP(bool, __nv_bfloat16, cast_bool_to_bf16,
        x ? from_float<__nv_bfloat16>(1.0f) : from_float<__nv_bfloat16>(0.0f))
CAST_OP(bool, __half, cast_bool_to_f16, x ? from_float<__half>(1.0f) : from_float<__half>(0.0f))
CAST_OP(bool, float, cast_bool_to_f32, x ? 1.0f : 0.0f)
CAST_OP(bool, double, cast_bool_to_f64, x ? 1.0 : 0.0)
CAST_OP(bool, uint8_t, cast_bool_to_u8, x ? (uint8_t)1 : (uint8_t)0)
CAST_OP(bool, uint16_t, cast_bool_to_u16, x ? (uint16_t)1 : (uint16_t)0)
CAST_OP(bool, uint32_t, cast_bool_to_u32, x ? (uint32_t)1 : (uint32_t)0)
CAST_OP(bool, uint64_t, cast_bool_to_u64, x ? (uint64_t)1 : (uint64_t)0)
CAST_OP(bool, int8_t, cast_bool_to_i8, x ? (int8_t)1 : (int8_t)0)
CAST_OP(bool, int16_t, cast_bool_to_i16, x ? (int16_t)1 : (int16_t)0)
CAST_OP(bool, int32_t, cast_bool_to_i32, x ? (int32_t)1 : (int32_t)0)
CAST_OP(bool, int64_t, cast_bool_to_i64, x ? (int64_t)1 : (int64_t)0)

CAST_OP(__nv_fp8_e4m3, bool, cast_f8e4m3_to_bool, (float)x != 0.0f)
CAST_OP(__nv_fp8_e4m3, __nv_fp8_e5m2, cast_f8e4m3_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(__nv_fp8_e4m3, __nv_bfloat16, cast_f8e4m3_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(__nv_fp8_e4m3, __half, cast_f8e4m3_to_f16, from_float<__half>((float)x))
CAST_OP(__nv_fp8_e4m3, float, cast_f8e4m3_to_f32, (float)x)
CAST_OP(__nv_fp8_e4m3, double, cast_f8e4m3_to_f64, (double)(float)x)
CAST_OP(__nv_fp8_e4m3, uint8_t, cast_f8e4m3_to_u8, (uint8_t)fminf(fmaxf((float)x, 0.0f), 255.0f))
CAST_OP(__nv_fp8_e4m3, uint16_t, cast_f8e4m3_to_u16,
        (uint16_t)fminf(fmaxf((float)x, 0.0f), 65535.0f))
CAST_OP(__nv_fp8_e4m3, uint32_t, cast_f8e4m3_to_u32, (uint32_t)fmaxf((float)x, 0.0f))
CAST_OP(__nv_fp8_e4m3, uint64_t, cast_f8e4m3_to_u64, (uint64_t)fmaxf((float)x, 0.0f))
CAST_OP(__nv_fp8_e4m3, int8_t, cast_f8e4m3_to_i8, (int8_t)fminf(fmaxf((float)x, -128.0f), 127.0f))
CAST_OP(__nv_fp8_e4m3, int16_t, cast_f8e4m3_to_i16,
        (int16_t)fminf(fmaxf((float)x, -32768.0f), 32767.0f))
CAST_OP(__nv_fp8_e4m3, int32_t, cast_f8e4m3_to_i32, (int32_t)(float)x)
CAST_OP(__nv_fp8_e4m3, int64_t, cast_f8e4m3_to_i64, (int64_t)(float)x)

CAST_OP(__nv_fp8_e5m2, bool, cast_f8e5m2_to_bool, (float)x != 0.0f)
CAST_OP(__nv_fp8_e5m2, __nv_fp8_e4m3, cast_f8e5m2_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(__nv_fp8_e5m2, __nv_bfloat16, cast_f8e5m2_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(__nv_fp8_e5m2, __half, cast_f8e5m2_to_f16, from_float<__half>((float)x))
CAST_OP(__nv_fp8_e5m2, float, cast_f8e5m2_to_f32, (float)x)
CAST_OP(__nv_fp8_e5m2, double, cast_f8e5m2_to_f64, (double)(float)x)
CAST_OP(__nv_fp8_e5m2, uint8_t, cast_f8e5m2_to_u8, (uint8_t)fminf(fmaxf((float)x, 0.0f), 255.0f))
CAST_OP(__nv_fp8_e5m2, uint16_t, cast_f8e5m2_to_u16,
        (uint16_t)fminf(fmaxf((float)x, 0.0f), 65535.0f))
CAST_OP(__nv_fp8_e5m2, uint32_t, cast_f8e5m2_to_u32, (uint32_t)fmaxf((float)x, 0.0f))
CAST_OP(__nv_fp8_e5m2, uint64_t, cast_f8e5m2_to_u64, (uint64_t)fmaxf((float)x, 0.0f))
CAST_OP(__nv_fp8_e5m2, int8_t, cast_f8e5m2_to_i8, (int8_t)fminf(fmaxf((float)x, -128.0f), 127.0f))
CAST_OP(__nv_fp8_e5m2, int16_t, cast_f8e5m2_to_i16,
        (int16_t)fminf(fmaxf((float)x, -32768.0f), 32767.0f))
CAST_OP(__nv_fp8_e5m2, int32_t, cast_f8e5m2_to_i32, (int32_t)(float)x)
CAST_OP(__nv_fp8_e5m2, int64_t, cast_f8e5m2_to_i64, (int64_t)(float)x)

CAST_OP(__nv_bfloat16, bool, cast_bf16_to_bool, to_float(x) != 0.0f)
CAST_OP(__nv_bfloat16, __nv_fp8_e4m3, cast_bf16_to_f8e4m3, __nv_fp8_e4m3(to_float(x)))
CAST_OP(__nv_bfloat16, __nv_fp8_e5m2, cast_bf16_to_f8e5m2, __nv_fp8_e5m2(to_float(x)))
CAST_OP(__nv_bfloat16, __half, cast_bf16_to_f16, from_float<__half>(to_float(x)))
CAST_OP(__nv_bfloat16, float, cast_bf16_to_f32, to_float(x))
CAST_OP(__nv_bfloat16, double, cast_bf16_to_f64, (double)to_float(x))
CAST_OP(__nv_bfloat16, uint8_t, cast_bf16_to_u8, (uint8_t)fminf(fmaxf(to_float(x), 0.0f), 255.0f))
CAST_OP(__nv_bfloat16, uint16_t, cast_bf16_to_u16,
        (uint16_t)fminf(fmaxf(to_float(x), 0.0f), 65535.0f))
CAST_OP(__nv_bfloat16, uint32_t, cast_bf16_to_u32, (uint32_t)fmaxf(to_float(x), 0.0f))
CAST_OP(__nv_bfloat16, uint64_t, cast_bf16_to_u64, (uint64_t)fmaxf(to_float(x), 0.0f))
CAST_OP(__nv_bfloat16, int8_t, cast_bf16_to_i8, (int8_t)fminf(fmaxf(to_float(x), -128.0f), 127.0f))
CAST_OP(__nv_bfloat16, int16_t, cast_bf16_to_i16,
        (int16_t)fminf(fmaxf(to_float(x), -32768.0f), 32767.0f))
CAST_OP(__nv_bfloat16, int32_t, cast_bf16_to_i32, (int32_t)to_float(x))
CAST_OP(__nv_bfloat16, int64_t, cast_bf16_to_i64, (int64_t)to_float(x))

CAST_OP(__half, bool, cast_f16_to_bool, to_float(x) != 0.0f)
CAST_OP(__half, __nv_fp8_e4m3, cast_f16_to_f8e4m3, __nv_fp8_e4m3(to_float(x)))
CAST_OP(__half, __nv_fp8_e5m2, cast_f16_to_f8e5m2, __nv_fp8_e5m2(to_float(x)))
CAST_OP(__half, __nv_bfloat16, cast_f16_to_bf16, from_float<__nv_bfloat16>(to_float(x)))
CAST_OP(__half, float, cast_f16_to_f32, to_float(x))
CAST_OP(__half, double, cast_f16_to_f64, (double)to_float(x))
CAST_OP(__half, uint8_t, cast_f16_to_u8, (uint8_t)fminf(fmaxf(to_float(x), 0.0f), 255.0f))
CAST_OP(__half, uint16_t, cast_f16_to_u16, (uint16_t)fminf(fmaxf(to_float(x), 0.0f), 65504.0f))
CAST_OP(__half, uint32_t, cast_f16_to_u32, (uint32_t)fmaxf(to_float(x), 0.0f))
CAST_OP(__half, uint64_t, cast_f16_to_u64, (uint64_t)fmaxf(to_float(x), 0.0f))
CAST_OP(__half, int8_t, cast_f16_to_i8, (int8_t)fminf(fmaxf(to_float(x), -128.0f), 127.0f))
CAST_OP(__half, int16_t, cast_f16_to_i16, (int16_t)fminf(fmaxf(to_float(x), -32768.0f), 32767.0f))
CAST_OP(__half, int32_t, cast_f16_to_i32, (int32_t)to_float(x))
CAST_OP(__half, int64_t, cast_f16_to_i64, (int64_t)to_float(x))

CAST_OP(float, bool, cast_f32_to_bool, x != 0.0f)
CAST_OP(float, __nv_fp8_e4m3, cast_f32_to_f8e4m3, __nv_fp8_e4m3(x))
CAST_OP(float, __nv_fp8_e5m2, cast_f32_to_f8e5m2, __nv_fp8_e5m2(x))
CAST_OP(float, __nv_bfloat16, cast_f32_to_bf16, from_float<__nv_bfloat16>(x))
CAST_OP(float, __half, cast_f32_to_f16, from_float<__half>(x))
CAST_OP(float, double, cast_f32_to_f64, (double)x)
CAST_OP(float, uint8_t, cast_f32_to_u8, (uint8_t)fminf(fmaxf(x, 0.0f), 255.0f))
CAST_OP(float, uint16_t, cast_f32_to_u16, (uint16_t)fminf(fmaxf(x, 0.0f), 65535.0f))
CAST_OP(float, uint32_t, cast_f32_to_u32, (uint32_t)fmaxf(x, 0.0f))
CAST_OP(float, uint64_t, cast_f32_to_u64, (uint64_t)fmaxf(x, 0.0f))
CAST_OP(float, int8_t, cast_f32_to_i8, (int8_t)fminf(fmaxf(x, -128.0f), 127.0f))
CAST_OP(float, int16_t, cast_f32_to_i16, (int16_t)fminf(fmaxf(x, -32768.0f), 32767.0f))
CAST_OP(float, int32_t, cast_f32_to_i32, (int32_t)x)
CAST_OP(float, int64_t, cast_f32_to_i64, (int64_t)x)

CAST_OP(double, bool, cast_f64_to_bool, x != 0.0)
CAST_OP(double, __nv_fp8_e4m3, cast_f64_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(double, __nv_fp8_e5m2, cast_f64_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(double, __nv_bfloat16, cast_f64_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(double, __half, cast_f64_to_f16, from_float<__half>((float)x))
CAST_OP(double, float, cast_f64_to_f32, (float)x)
CAST_OP(double, uint8_t, cast_f64_to_u8, (uint8_t)fmin(fmax(x, 0.0), 255.0))
CAST_OP(double, uint16_t, cast_f64_to_u16, (uint16_t)fmin(fmax(x, 0.0), 65535.0))
CAST_OP(double, uint32_t, cast_f64_to_u32, (uint32_t)fmax(x, 0.0))
CAST_OP(double, uint64_t, cast_f64_to_u64, (uint64_t)fmax(x, 0.0))
CAST_OP(double, int8_t, cast_f64_to_i8, (int8_t)fmin(fmax(x, -128.0), 127.0))
CAST_OP(double, int16_t, cast_f64_to_i16, (int16_t)fmin(fmax(x, -32768.0), 32767.0))
CAST_OP(double, int32_t, cast_f64_to_i32, (int32_t)x)
CAST_OP(double, int64_t, cast_f64_to_i64, (int64_t)x)

CAST_OP(uint8_t, bool, cast_u8_to_bool, x != 0)
CAST_OP(uint8_t, __nv_fp8_e4m3, cast_u8_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(uint8_t, __nv_fp8_e5m2, cast_u8_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(uint8_t, __nv_bfloat16, cast_u8_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(uint8_t, __half, cast_u8_to_f16, from_float<__half>((float)x))
CAST_OP(uint8_t, float, cast_u8_to_f32, (float)x)
CAST_OP(uint8_t, double, cast_u8_to_f64, (double)x)
CAST_OP(uint8_t, uint16_t, cast_u8_to_u16, (uint16_t)x)
CAST_OP(uint8_t, uint32_t, cast_u8_to_u32, (uint32_t)x)
CAST_OP(uint8_t, uint64_t, cast_u8_to_u64, (uint64_t)x)
CAST_OP(uint8_t, int8_t, cast_u8_to_i8, (int8_t)minimum((uint8_t)127, x))
CAST_OP(uint8_t, int16_t, cast_u8_to_i16, (int16_t)x)
CAST_OP(uint8_t, int32_t, cast_u8_to_i32, (int32_t)x)
CAST_OP(uint8_t, int64_t, cast_u8_to_i64, (int64_t)x)

CAST_OP(uint16_t, bool, cast_u16_to_bool, x != 0)
CAST_OP(uint16_t, __nv_fp8_e4m3, cast_u16_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(uint16_t, __nv_fp8_e5m2, cast_u16_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(uint16_t, __nv_bfloat16, cast_u16_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(uint16_t, __half, cast_u16_to_f16, from_float<__half>((float)x))
CAST_OP(uint16_t, float, cast_u16_to_f32, (float)x)
CAST_OP(uint16_t, double, cast_u16_to_f64, (double)x)
CAST_OP(uint16_t, uint8_t, cast_u16_to_u8, (uint8_t)minimum((uint16_t)255, x))
CAST_OP(uint16_t, uint32_t, cast_u16_to_u32, (uint32_t)x)
CAST_OP(uint16_t, uint64_t, cast_u16_to_u64, (uint64_t)x)
CAST_OP(uint16_t, int8_t, cast_u16_to_i8, (int8_t)minimum((uint16_t)127, x))
CAST_OP(uint16_t, int16_t, cast_u16_to_i16, (int16_t)minimum((uint16_t)32767, x))
CAST_OP(uint16_t, int32_t, cast_u16_to_i32, (int32_t)x)
CAST_OP(uint16_t, int64_t, cast_u16_to_i64, (int64_t)x)

CAST_OP(uint32_t, bool, cast_u32_to_bool, x != 0)
CAST_OP(uint32_t, __nv_fp8_e4m3, cast_u32_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(uint32_t, __nv_fp8_e5m2, cast_u32_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(uint32_t, __nv_bfloat16, cast_u32_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(uint32_t, __half, cast_u32_to_f16, from_float<__half>((float)x))
CAST_OP(uint32_t, float, cast_u32_to_f32, (float)x)
CAST_OP(uint32_t, double, cast_u32_to_f64, (double)x)
CAST_OP(uint32_t, uint8_t, cast_u32_to_u8, (uint8_t)minimum((uint32_t)255, x))
CAST_OP(uint32_t, uint16_t, cast_u32_to_u16, (uint16_t)minimum((uint32_t)65535, x))
CAST_OP(uint32_t, uint64_t, cast_u32_to_u64, (uint64_t)x)
CAST_OP(uint32_t, int8_t, cast_u32_to_i8, (int8_t)minimum((uint32_t)127, x))
CAST_OP(uint32_t, int16_t, cast_u32_to_i16, (int16_t)minimum((uint32_t)32767, x))
CAST_OP(uint32_t, int32_t, cast_u32_to_i32, (int32_t)minimum((uint32_t)2147483647, x))
CAST_OP(uint32_t, int64_t, cast_u32_to_i64, (int64_t)x)

CAST_OP(uint64_t, bool, cast_u64_to_bool, x != 0)
CAST_OP(uint64_t, __nv_fp8_e4m3, cast_u64_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(uint64_t, __nv_fp8_e5m2, cast_u64_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(uint64_t, __nv_bfloat16, cast_u64_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(uint64_t, __half, cast_u64_to_f16, from_float<__half>((float)x))
CAST_OP(uint64_t, float, cast_u64_to_f32, (float)x)
CAST_OP(uint64_t, double, cast_u64_to_f64, (double)x)
CAST_OP(uint64_t, uint8_t, cast_u64_to_u8, (uint8_t)minimum((uint64_t)255, x))
CAST_OP(uint64_t, uint16_t, cast_u64_to_u16, (uint16_t)minimum((uint64_t)65535, x))
CAST_OP(uint64_t, uint32_t, cast_u64_to_u32, (uint32_t)minimum((uint64_t)4294967295ULL, x))
CAST_OP(uint64_t, int8_t, cast_u64_to_i8, (int8_t)minimum((uint64_t)127, x))
CAST_OP(uint64_t, int16_t, cast_u64_to_i16, (int16_t)minimum((uint64_t)32767, x))
CAST_OP(uint64_t, int32_t, cast_u64_to_i32, (int32_t)minimum((uint64_t)2147483647, x))
CAST_OP(uint64_t, int64_t, cast_u64_to_i64, (int64_t)minimum((uint64_t)9223372036854775807ULL, x))

CAST_OP(int8_t, bool, cast_i8_to_bool, x != 0)
CAST_OP(int8_t, __nv_fp8_e4m3, cast_i8_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(int8_t, __nv_fp8_e5m2, cast_i8_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(int8_t, __nv_bfloat16, cast_i8_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(int8_t, __half, cast_i8_to_f16, from_float<__half>((float)x))
CAST_OP(int8_t, float, cast_i8_to_f32, (float)x)
CAST_OP(int8_t, double, cast_i8_to_f64, (double)x)
CAST_OP(int8_t, uint8_t, cast_i8_to_u8, (uint8_t)maximum((int8_t)0, x))
CAST_OP(int8_t, uint16_t, cast_i8_to_u16, (uint16_t)maximum((int8_t)0, x))
CAST_OP(int8_t, uint32_t, cast_i8_to_u32, (uint32_t)maximum((int8_t)0, x))
CAST_OP(int8_t, uint64_t, cast_i8_to_u64, (uint64_t)maximum((int8_t)0, x))
CAST_OP(int8_t, int16_t, cast_i8_to_i16, (int16_t)x)
CAST_OP(int8_t, int32_t, cast_i8_to_i32, (int32_t)x)
CAST_OP(int8_t, int64_t, cast_i8_to_i64, (int64_t)x)

CAST_OP(int16_t, bool, cast_i16_to_bool, x != 0)
CAST_OP(int16_t, __nv_fp8_e4m3, cast_i16_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(int16_t, __nv_fp8_e5m2, cast_i16_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(int16_t, __nv_bfloat16, cast_i16_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(int16_t, __half, cast_i16_to_f16, from_float<__half>((float)x))
CAST_OP(int16_t, float, cast_i16_to_f32, (float)x)
CAST_OP(int16_t, double, cast_i16_to_f64, (double)x)
CAST_OP(int16_t, uint8_t, cast_i16_to_u8, (uint8_t)maximum((int16_t)0, minimum((int16_t)255, x)))
CAST_OP(int16_t, uint16_t, cast_i16_to_u16, (uint16_t)maximum((int16_t)0, x))
CAST_OP(int16_t, uint32_t, cast_i16_to_u32, (uint32_t)maximum((int16_t)0, x))
CAST_OP(int16_t, uint64_t, cast_i16_to_u64, (uint64_t)maximum((int16_t)0, x))
CAST_OP(int16_t, int8_t, cast_i16_to_i8, (int8_t)maximum((int16_t)-128, minimum((int16_t)127, x)))
CAST_OP(int16_t, int32_t, cast_i16_to_i32, (int32_t)x)
CAST_OP(int16_t, int64_t, cast_i16_to_i64, (int64_t)x)

CAST_OP(int32_t, bool, cast_i32_to_bool, x != 0)
CAST_OP(int32_t, __nv_fp8_e4m3, cast_i32_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(int32_t, __nv_fp8_e5m2, cast_i32_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(int32_t, __nv_bfloat16, cast_i32_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(int32_t, __half, cast_i32_to_f16, from_float<__half>((float)x))
CAST_OP(int32_t, float, cast_i32_to_f32, (float)x)
CAST_OP(int32_t, double, cast_i32_to_f64, (double)x)
CAST_OP(int32_t, uint8_t, cast_i32_to_u8, (uint8_t)maximum(0, minimum(255, x)))
CAST_OP(int32_t, uint16_t, cast_i32_to_u16, (uint16_t)maximum(0, minimum(65535, x)))
CAST_OP(int32_t, uint32_t, cast_i32_to_u32, (uint32_t)maximum(0, x))
CAST_OP(int32_t, uint64_t, cast_i32_to_u64, (uint64_t)maximum(0, x))
CAST_OP(int32_t, int8_t, cast_i32_to_i8, (int8_t)maximum(-128, minimum(127, x)))
CAST_OP(int32_t, int16_t, cast_i32_to_i16, (int16_t)maximum(-32768, minimum(32767, x)))
CAST_OP(int32_t, int64_t, cast_i32_to_i64, (int64_t)x)

CAST_OP(int64_t, bool, cast_i64_to_bool, x != 0)
CAST_OP(int64_t, __nv_fp8_e4m3, cast_i64_to_f8e4m3, __nv_fp8_e4m3((float)x))
CAST_OP(int64_t, __nv_fp8_e5m2, cast_i64_to_f8e5m2, __nv_fp8_e5m2((float)x))
CAST_OP(int64_t, __nv_bfloat16, cast_i64_to_bf16, from_float<__nv_bfloat16>((float)x))
CAST_OP(int64_t, __half, cast_i64_to_f16, from_float<__half>((float)x))
CAST_OP(int64_t, float, cast_i64_to_f32, (float)x)
CAST_OP(int64_t, double, cast_i64_to_f64, (double)x)
CAST_OP(int64_t, uint8_t, cast_i64_to_u8, (uint8_t)maximum((int64_t)0, minimum((int64_t)255, x)))
CAST_OP(int64_t, uint16_t, cast_i64_to_u16,
        (uint16_t)maximum((int64_t)0, minimum((int64_t)65535, x)))
CAST_OP(int64_t, uint32_t, cast_i64_to_u32,
        (uint32_t)maximum((int64_t)0, minimum((int64_t)4294967295LL, x)))
CAST_OP(int64_t, uint64_t, cast_i64_to_u64, (uint64_t)maximum((int64_t)0, x))
CAST_OP(int64_t, int8_t, cast_i64_to_i8, (int8_t)maximum((int64_t)-128, minimum((int64_t)127, x)))
CAST_OP(int64_t, int16_t, cast_i64_to_i16,
        (int16_t)maximum((int64_t)-32768, minimum((int64_t)32767, x)))
CAST_OP(int64_t, int32_t, cast_i64_to_i32,
        (int32_t)maximum((int64_t)-2147483648LL, minimum((int64_t)2147483647LL, x)))
