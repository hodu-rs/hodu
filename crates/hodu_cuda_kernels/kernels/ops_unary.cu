#include "math.cuh"
#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define UNARY_OP(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)                                         \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *out,    \
                                                   const size_t *metadata) {                       \
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
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else {                                                                                   \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t idx = offset + get_strided_index(i, num_dims, dims, strides);             \
                IN_TYPENAME x = input[idx];                                                        \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

#define UNARY_OP_WITH_SCALAR(IN_TYPENAME, OUT_TYPENAME, FN_NAME, FUNC)                             \
    extern "C" __global__ void hodu_cuda_##FN_NAME(const IN_TYPENAME *input, OUT_TYPENAME *out,    \
                                                   const size_t *metadata,                         \
                                                   const IN_TYPENAME *scalar) {                    \
        const size_t num_els = metadata[0];                                                        \
        const size_t num_dims = metadata[1];                                                       \
        const size_t *dims = metadata + 2;                                                         \
        const size_t *strides = metadata + 2 + num_dims;                                           \
        const size_t offset = metadata[2 + 2 * num_dims];                                          \
        IN_TYPENAME const_val = *scalar;                                                           \
        bool cont = is_contiguous(num_dims, dims, strides);                                        \
        if (cont) {                                                                                \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                IN_TYPENAME x = input[offset + i];                                                 \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        } else {                                                                                   \
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_els;                  \
                 i += blockDim.x * gridDim.x) {                                                    \
                uint32_t idx = offset + get_strided_index(i, num_dims, dims, strides);             \
                IN_TYPENAME x = input[idx];                                                        \
                out[i] = FUNC;                                                                     \
            }                                                                                      \
        }                                                                                          \
    }

UNARY_OP(bool, bool, neg_bool, !x)
UNARY_OP(bool, bool, abs_bool, x)
UNARY_OP(bool, bool, sign_bool, x ? 1 : 0)
UNARY_OP(bool, bool, softsign_bool, x)
UNARY_OP(bool, bool, square_bool, x)
UNARY_OP(bool, bool, sqrt_bool, x)
UNARY_OP(bool, bool, relu_bool, x)
UNARY_OP(bool, bool, sigmoid_bool, x)
UNARY_OP(bool, bool, selu_bool, x)
UNARY_OP(bool, bool, celu_bool, x)
UNARY_OP(bool, bool, logical_not_bool, !x)
UNARY_OP(bool, bool, tanh_bool, x)

UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, neg_f8e4m3, from_float<__nv_fp8_e4m3>(-to_float(x)))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, abs_f8e4m3, from_float<__nv_fp8_e4m3>(fabsf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, sign_f8e4m3, sign(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, softsign_f8e4m3, softsign(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, square_f8e4m3,
         from_float<__nv_fp8_e4m3>(to_float(x) * to_float(x)))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, sqrt_f8e4m3, from_float<__nv_fp8_e4m3>(sqrtf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, recip_f8e4m3, from_float<__nv_fp8_e4m3>(1.0f / to_float(x)))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, relu_f8e4m3, relu(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, sigmoid_f8e4m3, sigmoid(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, hardsigmoid_f8e4m3, hardsigmoid(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, gelu_f8e4m3, gelu(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, softplus_f8e4m3, softplus(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, silu_f8e4m3, silu(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, hardsilu_f8e4m3, hardsilu(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, mish_f8e4m3, mish(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, selu_f8e4m3, selu(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, celu_f8e4m3, celu(x))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, sin_f8e4m3, from_float<__nv_fp8_e4m3>(sinf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, cos_f8e4m3, from_float<__nv_fp8_e4m3>(cosf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, tan_f8e4m3, from_float<__nv_fp8_e4m3>(m_tan(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, asin_f8e4m3, from_float<__nv_fp8_e4m3>(asinf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, acos_f8e4m3, from_float<__nv_fp8_e4m3>(acosf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, atan_f8e4m3, from_float<__nv_fp8_e4m3>(atanf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, sinh_f8e4m3, from_float<__nv_fp8_e4m3>(sinhf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, cosh_f8e4m3, from_float<__nv_fp8_e4m3>(coshf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, tanh_f8e4m3, from_float<__nv_fp8_e4m3>(tanhf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, asinh_f8e4m3, from_float<__nv_fp8_e4m3>(asinhf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, acosh_f8e4m3, from_float<__nv_fp8_e4m3>(acoshf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, atanh_f8e4m3, from_float<__nv_fp8_e4m3>(atanhf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, exp_f8e4m3, from_float<__nv_fp8_e4m3>(expf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, exp2_f8e4m3, from_float<__nv_fp8_e4m3>(exp2f(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, exp10_f8e4m3,
         from_float<__nv_fp8_e4m3>(m_exp10(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, ln_f8e4m3, from_float<__nv_fp8_e4m3>(logf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, log2_f8e4m3, from_float<__nv_fp8_e4m3>(log2f(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, log10_f8e4m3, from_float<__nv_fp8_e4m3>(log10f(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, ceil_f8e4m3, from_float<__nv_fp8_e4m3>(ceilf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, floor_f8e4m3, from_float<__nv_fp8_e4m3>(floorf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, round_f8e4m3, from_float<__nv_fp8_e4m3>(roundf(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, erf_f8e4m3, from_float<__nv_fp8_e4m3>(erff(to_float(x))))
UNARY_OP(__nv_fp8_e4m3, bool, logical_not_f8e4m3, !is_nonzero(x))

UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, neg_f8e5m2, from_float<__nv_fp8_e5m2>(-to_float(x)))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, abs_f8e5m2, from_float<__nv_fp8_e5m2>(fabsf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, sign_f8e5m2, sign(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, softsign_f8e5m2, softsign(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, square_f8e5m2,
         from_float<__nv_fp8_e5m2>(to_float(x) * to_float(x)))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, sqrt_f8e5m2, from_float<__nv_fp8_e5m2>(sqrtf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, recip_f8e5m2, from_float<__nv_fp8_e5m2>(1.0f / to_float(x)))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, relu_f8e5m2, relu(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, sigmoid_f8e5m2, sigmoid(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, hardsigmoid_f8e5m2, hardsigmoid(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, gelu_f8e5m2, gelu(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, softplus_f8e5m2, softplus(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, silu_f8e5m2, silu(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, hardsilu_f8e5m2, hardsilu(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, mish_f8e5m2, mish(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, selu_f8e5m2, selu(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, celu_f8e5m2, celu(x))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, sin_f8e5m2, from_float<__nv_fp8_e5m2>(sinf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, cos_f8e5m2, from_float<__nv_fp8_e5m2>(cosf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, tan_f8e5m2, from_float<__nv_fp8_e5m2>(m_tan(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, asin_f8e5m2, from_float<__nv_fp8_e5m2>(asinf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, acos_f8e5m2, from_float<__nv_fp8_e5m2>(acosf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, atan_f8e5m2, from_float<__nv_fp8_e5m2>(atanf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, sinh_f8e5m2, from_float<__nv_fp8_e5m2>(sinhf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, cosh_f8e5m2, from_float<__nv_fp8_e5m2>(coshf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, tanh_f8e5m2, from_float<__nv_fp8_e5m2>(tanhf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, asinh_f8e5m2, from_float<__nv_fp8_e5m2>(asinhf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, acosh_f8e5m2, from_float<__nv_fp8_e5m2>(acoshf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, atanh_f8e5m2, from_float<__nv_fp8_e5m2>(atanhf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, exp_f8e5m2, from_float<__nv_fp8_e5m2>(expf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, exp2_f8e5m2, from_float<__nv_fp8_e5m2>(exp2f(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, exp10_f8e5m2,
         from_float<__nv_fp8_e5m2>(m_exp10(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, ln_f8e5m2, from_float<__nv_fp8_e5m2>(logf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, log2_f8e5m2, from_float<__nv_fp8_e5m2>(log2f(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, log10_f8e5m2, from_float<__nv_fp8_e5m2>(log10f(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, ceil_f8e5m2, from_float<__nv_fp8_e5m2>(ceilf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, floor_f8e5m2, from_float<__nv_fp8_e5m2>(floorf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, round_f8e5m2, from_float<__nv_fp8_e5m2>(roundf(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, __nv_fp8_e5m2, erf_f8e5m2, from_float<__nv_fp8_e5m2>(erff(to_float(x))))
UNARY_OP(__nv_fp8_e5m2, bool, logical_not_f8e5m2, !is_nonzero(x))

UNARY_OP(__nv_bfloat16, __nv_bfloat16, neg_bf16, from_float<__nv_bfloat16>(-to_float(x)))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, abs_bf16, from_float<__nv_bfloat16>(fabsf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, sign_bf16, sign(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, softsign_bf16, softsign(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, square_bf16,
         from_float<__nv_bfloat16>(to_float(x) * to_float(x)))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, sqrt_bf16, from_float<__nv_bfloat16>(sqrtf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, recip_bf16, from_float<__nv_bfloat16>(1.0f / to_float(x)))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, relu_bf16, relu(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, sigmoid_bf16, sigmoid(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, hardsigmoid_bf16, hardsigmoid(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, gelu_bf16, gelu(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, softplus_bf16, softplus(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, silu_bf16, silu(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, hardsilu_bf16, hardsilu(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, mish_bf16, mish(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, selu_bf16, selu(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, celu_bf16, celu(x))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, sin_bf16, from_float<__nv_bfloat16>(sinf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, cos_bf16, from_float<__nv_bfloat16>(cosf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, tan_bf16, from_float<__nv_bfloat16>(m_tan(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, asin_bf16, from_float<__nv_bfloat16>(asinf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, acos_bf16, from_float<__nv_bfloat16>(acosf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, atan_bf16, from_float<__nv_bfloat16>(atanf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, sinh_bf16, from_float<__nv_bfloat16>(sinhf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, cosh_bf16, from_float<__nv_bfloat16>(coshf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, tanh_bf16, from_float<__nv_bfloat16>(tanhf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, asinh_bf16, from_float<__nv_bfloat16>(asinhf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, acosh_bf16, from_float<__nv_bfloat16>(acoshf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, atanh_bf16, from_float<__nv_bfloat16>(atanhf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, exp_bf16, from_float<__nv_bfloat16>(expf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, exp2_bf16, from_float<__nv_bfloat16>(exp2f(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, exp10_bf16, from_float<__nv_bfloat16>(m_exp10(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, ln_bf16, from_float<__nv_bfloat16>(logf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, log2_bf16, from_float<__nv_bfloat16>(log2f(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, log10_bf16, from_float<__nv_bfloat16>(log10f(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, ceil_bf16, from_float<__nv_bfloat16>(ceilf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, floor_bf16, from_float<__nv_bfloat16>(floorf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, round_bf16, from_float<__nv_bfloat16>(roundf(to_float(x))))
UNARY_OP(__nv_bfloat16, __nv_bfloat16, erf_bf16, __float2bfloat16(erff(__bfloat162float(x))))
UNARY_OP(__nv_bfloat16, bool, logical_not_bf16, !is_nonzero(x))

UNARY_OP(__half, __half, neg_f16, from_float<__half>(-to_float(x)))
UNARY_OP(__half, __half, abs_f16, from_float<__half>(fabsf(to_float(x))))
UNARY_OP(__half, __half, sign_f16, sign(x))
UNARY_OP(__half, __half, softsign_f16, softsign(x))
UNARY_OP(__half, __half, square_f16, from_float<__half>(to_float(x) * to_float(x)))
UNARY_OP(__half, __half, sqrt_f16, from_float<__half>(sqrtf(to_float(x))))
UNARY_OP(__half, __half, recip_f16, from_float<__half>(1.0f / to_float(x)))
UNARY_OP(__half, __half, relu_f16, relu(x))
UNARY_OP(__half, __half, sigmoid_f16, sigmoid(x))
UNARY_OP(__half, __half, hardsigmoid_f16, hardsigmoid(x))
UNARY_OP(__half, __half, gelu_f16, gelu(x))
UNARY_OP(__half, __half, softplus_f16, softplus(x))
UNARY_OP(__half, __half, silu_f16, silu(x))
UNARY_OP(__half, __half, hardsilu_f16, hardsilu(x))
UNARY_OP(__half, __half, mish_f16, mish(x))
UNARY_OP(__half, __half, selu_f16, selu(x))
UNARY_OP(__half, __half, celu_f16, celu(x))
UNARY_OP(__half, __half, sin_f16, from_float<__half>(sinf(to_float(x))))
UNARY_OP(__half, __half, cos_f16, from_float<__half>(cosf(to_float(x))))
UNARY_OP(__half, __half, tan_f16, from_float<__half>(m_tan(to_float(x))))
UNARY_OP(__half, __half, asin_f16, from_float<__half>(asinf(to_float(x))))
UNARY_OP(__half, __half, acos_f16, from_float<__half>(acosf(to_float(x))))
UNARY_OP(__half, __half, atan_f16, from_float<__half>(atanf(to_float(x))))
UNARY_OP(__half, __half, sinh_f16, from_float<__half>(sinhf(to_float(x))))
UNARY_OP(__half, __half, cosh_f16, from_float<__half>(coshf(to_float(x))))
UNARY_OP(__half, __half, tanh_f16, from_float<__half>(tanhf(to_float(x))))
UNARY_OP(__half, __half, asinh_f16, from_float<__half>(asinhf(to_float(x))))
UNARY_OP(__half, __half, acosh_f16, from_float<__half>(acoshf(to_float(x))))
UNARY_OP(__half, __half, atanh_f16, from_float<__half>(atanhf(to_float(x))))
UNARY_OP(__half, __half, exp_f16, from_float<__half>(expf(to_float(x))))
UNARY_OP(__half, __half, exp2_f16, from_float<__half>(exp2f(to_float(x))))
UNARY_OP(__half, __half, exp10_f16, from_float<__half>(m_exp10(to_float(x))))
UNARY_OP(__half, __half, ln_f16, from_float<__half>(logf(to_float(x))))
UNARY_OP(__half, __half, log2_f16, from_float<__half>(log2f(to_float(x))))
UNARY_OP(__half, __half, log10_f16, from_float<__half>(log10f(to_float(x))))
UNARY_OP(__half, __half, ceil_f16, from_float<__half>(ceilf(to_float(x))))
UNARY_OP(__half, __half, floor_f16, from_float<__half>(floorf(to_float(x))))
UNARY_OP(__half, __half, round_f16, from_float<__half>(roundf(to_float(x))))
UNARY_OP(__half, __half, erf_f16, __float2half(erff(__half2float(x))))
UNARY_OP(__half, bool, logical_not_f16, !is_nonzero(x))

UNARY_OP(float, float, neg_f32, -x)
UNARY_OP(float, float, abs_f32, fabsf(x))
UNARY_OP(float, float, sign_f32, sign(x))
UNARY_OP(float, float, softsign_f32, x / (1.0f + fabsf(x)))
UNARY_OP(float, float, square_f32, x *x)
UNARY_OP(float, float, sqrt_f32, sqrtf(x))
UNARY_OP(float, float, recip_f32, 1.0f / x)
UNARY_OP(float, float, relu_f32, relu(x))
UNARY_OP(float, float, sigmoid_f32, sigmoid(x))
UNARY_OP(float, float, hardsigmoid_f32, fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f)))
UNARY_OP(float, float, gelu_f32, gelu(x))
UNARY_OP(float, float, softplus_f32, softplus(x))
UNARY_OP(float, float, silu_f32, silu(x))
UNARY_OP(float, float, hardsilu_f32, x *fmaxf(0.0f, fminf(1.0f, (x + 3.0f) / 6.0f)))
UNARY_OP(float, float, mish_f32, mish(x))
UNARY_OP(float, float, selu_f32, SELU_SCALE *(x > 0.0f ? x : SELU_ALPHA * (expf(x) - 1.0f)))
UNARY_OP(float, float, celu_f32, fmaxf(0.0f, x) + fminf(0.0f, expf(x) - 1.0f))
UNARY_OP(float, float, sin_f32, sinf(x))
UNARY_OP(float, float, cos_f32, cosf(x))
UNARY_OP(float, float, tan_f32, m_tan(x))
UNARY_OP(float, float, asin_f32, asinf(x))
UNARY_OP(float, float, acos_f32, acosf(x))
UNARY_OP(float, float, atan_f32, atanf(x))
UNARY_OP(float, float, sinh_f32, sinhf(x))
UNARY_OP(float, float, cosh_f32, coshf(x))
UNARY_OP(float, float, tanh_f32, tanhf(x))
UNARY_OP(float, float, asinh_f32, asinhf(x))
UNARY_OP(float, float, acosh_f32, acoshf(x))
UNARY_OP(float, float, atanh_f32, atanhf(x))
UNARY_OP(float, float, exp_f32, expf(x))
UNARY_OP(float, float, exp2_f32, exp2f(x))
UNARY_OP(float, float, exp10_f32, m_exp10(x))
UNARY_OP(float, float, ln_f32, logf(x))
UNARY_OP(float, float, log2_f32, log2f(x))
UNARY_OP(float, float, log10_f32, log10f(x))
UNARY_OP(float, float, ceil_f32, ceilf(x))
UNARY_OP(float, float, floor_f32, floorf(x))
UNARY_OP(float, float, round_f32, roundf(x))
UNARY_OP(float, float, erf_f32, erff(x))
UNARY_OP(float, bool, logical_not_f32, x == 0.0f)

UNARY_OP(double, double, neg_f64, -x)
UNARY_OP(double, double, abs_f64, fabs(x))
UNARY_OP(double, double, sign_f64, sign(x))
UNARY_OP(double, double, softsign_f64, x / (1.0 + fabs(x)))
UNARY_OP(double, double, square_f64, x *x)
UNARY_OP(double, double, sqrt_f64, sqrt(x))
UNARY_OP(double, double, recip_f64, 1.0 / x)
UNARY_OP(double, double, relu_f64, (x > 0.0) ? x : 0.0)
UNARY_OP(double, double, sigmoid_f64, 1.0 / (1.0 + exp(-x)))
UNARY_OP(double, double, hardsigmoid_f64, fmax(0.0, fmin(1.0, (x + 3.0) / 6.0)))
UNARY_OP(double, double, gelu_f64,
         0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))))
UNARY_OP(double, double, softplus_f64, log(1.0 + exp(x)))
UNARY_OP(double, double, silu_f64, x / (1.0 + exp(-x)))
UNARY_OP(double, double, hardsilu_f64, x *fmax(0.0, fmin(1.0, (x + 3.0) / 6.0)))
UNARY_OP(double, double, mish_f64, x *tanh(log(1.0 + exp(x))))
UNARY_OP(double, double, selu_f64,
         (double)SELU_SCALE *(x > 0.0 ? x : (double)SELU_ALPHA * (exp(x) - 1.0)))
UNARY_OP(double, double, celu_f64, fmax(0.0, x) + fmin(0.0, exp(x) - 1.0))
UNARY_OP(double, double, sin_f64, sin(x))
UNARY_OP(double, double, cos_f64, cos(x))
UNARY_OP(double, double, tan_f64, tan(x))
UNARY_OP(double, double, asin_f64, asin(x))
UNARY_OP(double, double, acos_f64, acos(x))
UNARY_OP(double, double, atan_f64, atan(x))
UNARY_OP(double, double, sinh_f64, sinh(x))
UNARY_OP(double, double, cosh_f64, cosh(x))
UNARY_OP(double, double, tanh_f64, tanh(x))
UNARY_OP(double, double, asinh_f64, asinh(x))
UNARY_OP(double, double, acosh_f64, acosh(x))
UNARY_OP(double, double, atanh_f64, atanh(x))
UNARY_OP(double, double, exp_f64, exp(x))
UNARY_OP(double, double, exp2_f64, exp2(x))
UNARY_OP(double, double, exp10_f64, exp10(x))
UNARY_OP(double, double, ln_f64, log(x))
UNARY_OP(double, double, log2_f64, log2(x))
UNARY_OP(double, double, log10_f64, log10(x))
UNARY_OP(double, double, ceil_f64, ceil(x))
UNARY_OP(double, double, floor_f64, floor(x))
UNARY_OP(double, double, round_f64, round(x))
UNARY_OP(double, double, erf_f64, erf(x))
UNARY_OP(double, bool, logical_not_f64, x == 0.0)

UNARY_OP(uint8_t, uint8_t, neg_u8, -x)
UNARY_OP(uint8_t, uint8_t, abs_u8, x)
UNARY_OP(uint8_t, uint8_t, sign_u8, x > 0 ? 1 : 0)
UNARY_OP(uint8_t, uint8_t, softsign_u8, (uint8_t)((float)x / (1.0f + (float)x)))
UNARY_OP(uint8_t, uint8_t, square_u8, x *x)
UNARY_OP(uint8_t, uint8_t, sqrt_u8, sqrtf(x))
UNARY_OP(uint8_t, bool, logical_not_u8, x == 0)

UNARY_OP(uint16_t, uint16_t, neg_u16, -x)
UNARY_OP(uint16_t, uint16_t, abs_u16, x)
UNARY_OP(uint16_t, uint16_t, sign_u16, x > 0 ? 1 : 0)
UNARY_OP(uint16_t, uint16_t, softsign_u16, (uint16_t)((float)x / (1.0f + (float)x)))
UNARY_OP(uint16_t, uint16_t, square_u16, x *x)
UNARY_OP(uint16_t, uint16_t, sqrt_u16, sqrtf(x))
UNARY_OP(uint16_t, bool, logical_not_u16, x == 0)

UNARY_OP(uint32_t, uint32_t, neg_u32, -x)
UNARY_OP(uint32_t, uint32_t, abs_u32, x)
UNARY_OP(uint32_t, uint32_t, sign_u32, x > 0 ? 1 : 0)
UNARY_OP(uint32_t, uint32_t, softsign_u32, (uint32_t)((float)x / (1.0f + (float)x)))
UNARY_OP(uint32_t, uint32_t, square_u32, x *x)
UNARY_OP(uint32_t, uint32_t, sqrt_u32, sqrtf(x))
UNARY_OP(uint32_t, bool, logical_not_u32, x == 0)

UNARY_OP(uint64_t, uint64_t, neg_u64, -x)
UNARY_OP(uint64_t, uint64_t, abs_u64, x)
UNARY_OP(uint64_t, uint64_t, sign_u64, x > 0 ? 1 : 0)
UNARY_OP(uint64_t, uint64_t, softsign_u64, (uint64_t)((double)x / (1.0 + (double)x)))
UNARY_OP(uint64_t, uint64_t, square_u64, x *x)
UNARY_OP(uint64_t, uint64_t, sqrt_u64, (uint64_t)sqrt((double)x))
UNARY_OP(uint64_t, bool, logical_not_u64, x == 0)

UNARY_OP(int8_t, int8_t, neg_i8, -x)
UNARY_OP(int8_t, int8_t, abs_i8, x < 0 ? -x : x)
UNARY_OP(int8_t, int8_t, sign_i8, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
UNARY_OP(int8_t, int8_t, softsign_i8, (int8_t)((float)x / (1.0f + fabsf((float)x))))
UNARY_OP(int8_t, int8_t, square_i8, x *x)
UNARY_OP(int8_t, int8_t, sqrt_i8, sqrtf(x))
UNARY_OP(int8_t, bool, logical_not_i8, x == 0)

UNARY_OP(int16_t, int16_t, neg_i16, -x)
UNARY_OP(int16_t, int16_t, abs_i16, x < 0 ? -x : x)
UNARY_OP(int16_t, int16_t, sign_i16, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
UNARY_OP(int16_t, int16_t, softsign_i16, (int16_t)((float)x / (1.0f + fabsf((float)x))))
UNARY_OP(int16_t, int16_t, square_i16, x *x)
UNARY_OP(int16_t, int16_t, sqrt_i16, sqrtf(x))
UNARY_OP(int16_t, bool, logical_not_i16, x == 0)

UNARY_OP(int32_t, int32_t, neg_i32, -x)
UNARY_OP(int32_t, int32_t, abs_i32, x < 0 ? -x : x)
UNARY_OP(int32_t, int32_t, sign_i32, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
UNARY_OP(int32_t, int32_t, softsign_i32, (int32_t)((float)x / (1.0f + fabsf((float)x))))
UNARY_OP(int32_t, int32_t, square_i32, x *x)
UNARY_OP(int32_t, int32_t, sqrt_i32, sqrtf(x))
UNARY_OP(int32_t, bool, logical_not_i32, x == 0)

UNARY_OP(int64_t, int64_t, neg_i64, -x)
UNARY_OP(int64_t, int64_t, abs_i64, x < 0 ? -x : x)
UNARY_OP(int64_t, int64_t, sign_i64, (x > 0) ? 1 : ((x < 0) ? -1 : 0))
UNARY_OP(int64_t, int64_t, softsign_i64, (int64_t)((double)x / (1.0 + fabs((double)x))))
UNARY_OP(int64_t, int64_t, square_i64, x *x)
UNARY_OP(int64_t, int64_t, sqrt_i64, (int64_t)sqrt((double)x))
UNARY_OP(int64_t, bool, logical_not_i64, x == 0)

UNARY_OP_WITH_SCALAR(float, float, add_scalar_f32, x + const_val)
UNARY_OP_WITH_SCALAR(float, float, sub_scalar_f32, x - const_val)
UNARY_OP_WITH_SCALAR(float, float, mul_scalar_f32, x *const_val)
UNARY_OP_WITH_SCALAR(float, float, div_scalar_f32, x / const_val)
UNARY_OP_WITH_SCALAR(float, float, rem_scalar_f32, fmodf(x, const_val))
UNARY_OP_WITH_SCALAR(float, float, pow_scalar_f32, m_pow_float(x, const_val))
UNARY_OP_WITH_SCALAR(float, float, maximum_scalar_f32, (x > const_val) ? x : const_val)
UNARY_OP_WITH_SCALAR(float, float, minimum_scalar_f32, (x < const_val) ? x : const_val)
UNARY_OP_WITH_SCALAR(float, float, leaky_relu_f32, (x > 0.0f) ? x : const_val *x)
UNARY_OP_WITH_SCALAR(float, float, elu_f32, (x > 0.0f) ? x : const_val *(expf(x) - 1.0f))
UNARY_OP_WITH_SCALAR(float, float, prelu_f32, (x > 0.0f) ? x : const_val *x)
UNARY_OP_WITH_SCALAR(float, bool, eq_scalar_f32, x == const_val)
UNARY_OP_WITH_SCALAR(float, bool, ne_scalar_f32, x != const_val)
UNARY_OP_WITH_SCALAR(float, bool, lt_scalar_f32, x < const_val)
UNARY_OP_WITH_SCALAR(float, bool, le_scalar_f32, x <= const_val)
UNARY_OP_WITH_SCALAR(float, bool, gt_scalar_f32, x > const_val)
UNARY_OP_WITH_SCALAR(float, bool, ge_scalar_f32, x >= const_val)

UNARY_OP_WITH_SCALAR(double, double, add_scalar_f64, x + const_val)
UNARY_OP_WITH_SCALAR(double, double, sub_scalar_f64, x - const_val)
UNARY_OP_WITH_SCALAR(double, double, mul_scalar_f64, x *const_val)
UNARY_OP_WITH_SCALAR(double, double, div_scalar_f64, x / const_val)
UNARY_OP_WITH_SCALAR(double, double, rem_scalar_f64, fmod(x, const_val))
UNARY_OP_WITH_SCALAR(double, double, pow_scalar_f64, pow(x, const_val))
UNARY_OP_WITH_SCALAR(double, double, maximum_scalar_f64, (x > const_val) ? x : const_val)
UNARY_OP_WITH_SCALAR(double, double, minimum_scalar_f64, (x < const_val) ? x : const_val)
UNARY_OP_WITH_SCALAR(double, double, leaky_relu_f64, (x > 0.0) ? x : const_val *x)
UNARY_OP_WITH_SCALAR(double, double, elu_f64, (x > 0.0) ? x : const_val *(exp(x) - 1.0))
UNARY_OP_WITH_SCALAR(double, double, prelu_f64, (x > 0.0) ? x : const_val *x)
UNARY_OP_WITH_SCALAR(double, bool, eq_scalar_f64, x == const_val)
UNARY_OP_WITH_SCALAR(double, bool, ne_scalar_f64, x != const_val)
UNARY_OP_WITH_SCALAR(double, bool, lt_scalar_f64, x < const_val)
UNARY_OP_WITH_SCALAR(double, bool, le_scalar_f64, x <= const_val)
UNARY_OP_WITH_SCALAR(double, bool, gt_scalar_f64, x > const_val)
UNARY_OP_WITH_SCALAR(double, bool, ge_scalar_f64, x >= const_val)

UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, add_scalar_bf16,
                     from_float<__nv_bfloat16>(to_float(x) + to_float(const_val)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, sub_scalar_bf16,
                     from_float<__nv_bfloat16>(to_float(x) - to_float(const_val)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, mul_scalar_bf16,
                     from_float<__nv_bfloat16>(to_float(x) * to_float(const_val)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, div_scalar_bf16,
                     from_float<__nv_bfloat16>(to_float(x) / to_float(const_val)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, rem_scalar_bf16,
                     from_float<__nv_bfloat16>(fmodf(to_float(x), to_float(const_val))))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, pow_scalar_bf16,
                     from_float<__nv_bfloat16>(m_pow_float(to_float(x), to_float(const_val))))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, maximum_scalar_bf16,
                     from_float<__nv_bfloat16>((to_float(x) > to_float(const_val))
                                                   ? to_float(x)
                                                   : to_float(const_val)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, minimum_scalar_bf16,
                     from_float<__nv_bfloat16>((to_float(x) < to_float(const_val))
                                                   ? to_float(x)
                                                   : to_float(const_val)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, leaky_relu_bf16,
                     from_float<__nv_bfloat16>((to_float(x) > 0.0f)
                                                   ? to_float(x)
                                                   : to_float(const_val) * to_float(x)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, elu_bf16,
                     from_float<__nv_bfloat16>((to_float(x) > 0.0f)
                                                   ? to_float(x)
                                                   : to_float(const_val) *
                                                         (expf(to_float(x)) - 1.0f)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, __nv_bfloat16, prelu_bf16,
                     from_float<__nv_bfloat16>((to_float(x) > 0.0f)
                                                   ? to_float(x)
                                                   : to_float(const_val) * to_float(x)))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, bool, eq_scalar_bf16, to_float(x) == to_float(const_val))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, bool, ne_scalar_bf16, to_float(x) != to_float(const_val))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, bool, lt_scalar_bf16, to_float(x) < to_float(const_val))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, bool, le_scalar_bf16, to_float(x) <= to_float(const_val))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, bool, gt_scalar_bf16, to_float(x) > to_float(const_val))
UNARY_OP_WITH_SCALAR(__nv_bfloat16, bool, ge_scalar_bf16, to_float(x) >= to_float(const_val))

UNARY_OP_WITH_SCALAR(__half, __half, add_scalar_f16,
                     from_float<__half>(to_float(x) + to_float(const_val)))
UNARY_OP_WITH_SCALAR(__half, __half, sub_scalar_f16,
                     from_float<__half>(to_float(x) - to_float(const_val)))
UNARY_OP_WITH_SCALAR(__half, __half, mul_scalar_f16,
                     from_float<__half>(to_float(x) * to_float(const_val)))
UNARY_OP_WITH_SCALAR(__half, __half, div_scalar_f16,
                     from_float<__half>(to_float(x) / to_float(const_val)))
UNARY_OP_WITH_SCALAR(__half, __half, rem_scalar_f16,
                     from_float<__half>(fmodf(to_float(x), to_float(const_val))))
UNARY_OP_WITH_SCALAR(__half, __half, pow_scalar_f16,
                     from_float<__half>(m_pow_float(to_float(x), to_float(const_val))))
UNARY_OP_WITH_SCALAR(__half, __half, maximum_scalar_f16,
                     from_float<__half>((to_float(x) > to_float(const_val)) ? to_float(x)
                                                                            : to_float(const_val)))
UNARY_OP_WITH_SCALAR(__half, __half, minimum_scalar_f16,
                     from_float<__half>((to_float(x) < to_float(const_val)) ? to_float(x)
                                                                            : to_float(const_val)))
UNARY_OP_WITH_SCALAR(__half, __half, leaky_relu_f16,
                     from_float<__half>((to_float(x) > 0.0f) ? to_float(x)
                                                             : to_float(const_val) * to_float(x)))
UNARY_OP_WITH_SCALAR(__half, __half, elu_f16,
                     from_float<__half>((to_float(x) > 0.0f)
                                            ? to_float(x)
                                            : to_float(const_val) * (expf(to_float(x)) - 1.0f)))
UNARY_OP_WITH_SCALAR(__half, __half, prelu_f16,
                     from_float<__half>((to_float(x) > 0.0f) ? to_float(x)
                                                             : to_float(const_val) * to_float(x)))
UNARY_OP_WITH_SCALAR(__half, bool, eq_scalar_f16, to_float(x) == to_float(const_val))
UNARY_OP_WITH_SCALAR(__half, bool, ne_scalar_f16, to_float(x) != to_float(const_val))
UNARY_OP_WITH_SCALAR(__half, bool, lt_scalar_f16, to_float(x) < to_float(const_val))
UNARY_OP_WITH_SCALAR(__half, bool, le_scalar_f16, to_float(x) <= to_float(const_val))
UNARY_OP_WITH_SCALAR(__half, bool, gt_scalar_f16, to_float(x) > to_float(const_val))
UNARY_OP_WITH_SCALAR(__half, bool, ge_scalar_f16, to_float(x) >= to_float(const_val))

UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, add_scalar_f8e4m3,
                     __nv_fp8_e4m3((float)x + (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, sub_scalar_f8e4m3,
                     __nv_fp8_e4m3((float)x - (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, mul_scalar_f8e4m3,
                     __nv_fp8_e4m3((float)x *(float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, div_scalar_f8e4m3,
                     __nv_fp8_e4m3((float)x / (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, rem_scalar_f8e4m3,
                     __nv_fp8_e4m3(fmodf((float)x, (float)const_val)))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, pow_scalar_f8e4m3,
                     __nv_fp8_e4m3(m_pow_float((float)x, (float)const_val)))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, maximum_scalar_f8e4m3,
                     __nv_fp8_e4m3(((float)x > (float)const_val) ? (float)x : (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, minimum_scalar_f8e4m3,
                     __nv_fp8_e4m3(((float)x < (float)const_val) ? (float)x : (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, leaky_relu_f8e4m3,
                     __nv_fp8_e4m3(((float)x > 0.0f) ? (float)x : (float)const_val *(float)x))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, elu_f8e4m3,
                     __nv_fp8_e4m3(((float)x > 0.0f) ? (float)x
                                                     : (float)const_val *(expf((float)x) - 1.0f)))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, __nv_fp8_e4m3, prelu_f8e4m3,
                     __nv_fp8_e4m3(((float)x > 0.0f) ? (float)x : (float)const_val *(float)x))
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, bool, eq_scalar_f8e4m3, (float)x == (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, bool, ne_scalar_f8e4m3, (float)x != (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, bool, lt_scalar_f8e4m3, (float)x < (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, bool, le_scalar_f8e4m3, (float)x <= (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, bool, gt_scalar_f8e4m3, (float)x > (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e4m3, bool, ge_scalar_f8e4m3, (float)x >= (float)const_val)

UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, add_scalar_f8e5m2,
                     __nv_fp8_e5m2((float)x + (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, sub_scalar_f8e5m2,
                     __nv_fp8_e5m2((float)x - (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, mul_scalar_f8e5m2,
                     __nv_fp8_e5m2((float)x *(float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, div_scalar_f8e5m2,
                     __nv_fp8_e5m2((float)x / (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, rem_scalar_f8e5m2,
                     __nv_fp8_e5m2(fmodf((float)x, (float)const_val)))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, pow_scalar_f8e5m2,
                     __nv_fp8_e5m2(m_pow_float((float)x, (float)const_val)))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, maximum_scalar_f8e5m2,
                     __nv_fp8_e5m2(((float)x > (float)const_val) ? (float)x : (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, minimum_scalar_f8e5m2,
                     __nv_fp8_e5m2(((float)x < (float)const_val) ? (float)x : (float)const_val))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, leaky_relu_f8e5m2,
                     __nv_fp8_e5m2(((float)x > 0.0f) ? (float)x : (float)const_val *(float)x))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, elu_f8e5m2,
                     __nv_fp8_e5m2(((float)x > 0.0f) ? (float)x
                                                     : (float)const_val *(expf((float)x) - 1.0f)))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, __nv_fp8_e5m2, prelu_f8e5m2,
                     __nv_fp8_e5m2(((float)x > 0.0f) ? (float)x : (float)const_val *(float)x))
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, bool, eq_scalar_f8e5m2, (float)x == (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, bool, ne_scalar_f8e5m2, (float)x != (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, bool, lt_scalar_f8e5m2, (float)x < (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, bool, le_scalar_f8e5m2, (float)x <= (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, bool, gt_scalar_f8e5m2, (float)x > (float)const_val)
UNARY_OP_WITH_SCALAR(__nv_fp8_e5m2, bool, ge_scalar_f8e5m2, (float)x >= (float)const_val)

UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, add_scalar_u8, x + const_val)
UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, sub_scalar_u8, (x > const_val) ? (x - const_val) : 0)
UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, mul_scalar_u8, x *const_val)
UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, div_scalar_u8, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, rem_scalar_u8, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, pow_scalar_u8, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, maximum_scalar_u8, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(uint8_t, uint8_t, minimum_scalar_u8, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(uint8_t, bool, eq_scalar_u8, x == const_val)
UNARY_OP_WITH_SCALAR(uint8_t, bool, ne_scalar_u8, x != const_val)
UNARY_OP_WITH_SCALAR(uint8_t, bool, lt_scalar_u8, x < const_val)
UNARY_OP_WITH_SCALAR(uint8_t, bool, le_scalar_u8, x <= const_val)
UNARY_OP_WITH_SCALAR(uint8_t, bool, gt_scalar_u8, x > const_val)
UNARY_OP_WITH_SCALAR(uint8_t, bool, ge_scalar_u8, x >= const_val)

UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, add_scalar_u16, x + const_val)
UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, sub_scalar_u16, (x > const_val) ? (x - const_val) : 0)
UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, mul_scalar_u16, x *const_val)
UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, div_scalar_u16, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, rem_scalar_u16, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, pow_scalar_u16, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, maximum_scalar_u16, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(uint16_t, uint16_t, minimum_scalar_u16, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(uint16_t, bool, eq_scalar_u16, x == const_val)
UNARY_OP_WITH_SCALAR(uint16_t, bool, ne_scalar_u16, x != const_val)
UNARY_OP_WITH_SCALAR(uint16_t, bool, lt_scalar_u16, x < const_val)
UNARY_OP_WITH_SCALAR(uint16_t, bool, le_scalar_u16, x <= const_val)
UNARY_OP_WITH_SCALAR(uint16_t, bool, gt_scalar_u16, x > const_val)
UNARY_OP_WITH_SCALAR(uint16_t, bool, ge_scalar_u16, x >= const_val)

UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, add_scalar_u32, x + const_val)
UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, sub_scalar_u32, (x > const_val) ? (x - const_val) : 0)
UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, mul_scalar_u32, x *const_val)
UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, div_scalar_u32, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, rem_scalar_u32, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, pow_scalar_u32, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, maximum_scalar_u32, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(uint32_t, uint32_t, minimum_scalar_u32, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(uint32_t, bool, eq_scalar_u32, x == const_val)
UNARY_OP_WITH_SCALAR(uint32_t, bool, ne_scalar_u32, x != const_val)
UNARY_OP_WITH_SCALAR(uint32_t, bool, lt_scalar_u32, x < const_val)
UNARY_OP_WITH_SCALAR(uint32_t, bool, le_scalar_u32, x <= const_val)
UNARY_OP_WITH_SCALAR(uint32_t, bool, gt_scalar_u32, x > const_val)
UNARY_OP_WITH_SCALAR(uint32_t, bool, ge_scalar_u32, x >= const_val)

UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, add_scalar_u64, x + const_val)
UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, sub_scalar_u64, (x > const_val) ? (x - const_val) : 0)
UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, mul_scalar_u64, x *const_val)
UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, div_scalar_u64, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, rem_scalar_u64, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, pow_scalar_u64, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, maximum_scalar_u64, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(uint64_t, uint64_t, minimum_scalar_u64, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(uint64_t, bool, eq_scalar_u64, x == const_val)
UNARY_OP_WITH_SCALAR(uint64_t, bool, ne_scalar_u64, x != const_val)
UNARY_OP_WITH_SCALAR(uint64_t, bool, lt_scalar_u64, x < const_val)
UNARY_OP_WITH_SCALAR(uint64_t, bool, le_scalar_u64, x <= const_val)
UNARY_OP_WITH_SCALAR(uint64_t, bool, gt_scalar_u64, x > const_val)
UNARY_OP_WITH_SCALAR(uint64_t, bool, ge_scalar_u64, x >= const_val)

UNARY_OP_WITH_SCALAR(int8_t, int8_t, add_scalar_i8, x + const_val)
UNARY_OP_WITH_SCALAR(int8_t, int8_t, sub_scalar_i8, x - const_val)
UNARY_OP_WITH_SCALAR(int8_t, int8_t, mul_scalar_i8, x *const_val)
UNARY_OP_WITH_SCALAR(int8_t, int8_t, div_scalar_i8, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(int8_t, int8_t, rem_scalar_i8, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(int8_t, int8_t, pow_scalar_i8, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(int8_t, int8_t, maximum_scalar_i8, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(int8_t, int8_t, minimum_scalar_i8, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(int8_t, bool, eq_scalar_i8, x == const_val)
UNARY_OP_WITH_SCALAR(int8_t, bool, ne_scalar_i8, x != const_val)
UNARY_OP_WITH_SCALAR(int8_t, bool, lt_scalar_i8, x < const_val)
UNARY_OP_WITH_SCALAR(int8_t, bool, le_scalar_i8, x <= const_val)
UNARY_OP_WITH_SCALAR(int8_t, bool, gt_scalar_i8, x > const_val)
UNARY_OP_WITH_SCALAR(int8_t, bool, ge_scalar_i8, x >= const_val)

UNARY_OP_WITH_SCALAR(int16_t, int16_t, add_scalar_i16, x + const_val)
UNARY_OP_WITH_SCALAR(int16_t, int16_t, sub_scalar_i16, x - const_val)
UNARY_OP_WITH_SCALAR(int16_t, int16_t, mul_scalar_i16, x *const_val)
UNARY_OP_WITH_SCALAR(int16_t, int16_t, div_scalar_i16, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(int16_t, int16_t, rem_scalar_i16, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(int16_t, int16_t, pow_scalar_i16, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(int16_t, int16_t, maximum_scalar_i16, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(int16_t, int16_t, minimum_scalar_i16, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(int16_t, bool, eq_scalar_i16, x == const_val)
UNARY_OP_WITH_SCALAR(int16_t, bool, ne_scalar_i16, x != const_val)
UNARY_OP_WITH_SCALAR(int16_t, bool, lt_scalar_i16, x < const_val)
UNARY_OP_WITH_SCALAR(int16_t, bool, le_scalar_i16, x <= const_val)
UNARY_OP_WITH_SCALAR(int16_t, bool, gt_scalar_i16, x > const_val)
UNARY_OP_WITH_SCALAR(int16_t, bool, ge_scalar_i16, x >= const_val)

UNARY_OP_WITH_SCALAR(int32_t, int32_t, add_scalar_i32, x + const_val)
UNARY_OP_WITH_SCALAR(int32_t, int32_t, sub_scalar_i32, x - const_val)
UNARY_OP_WITH_SCALAR(int32_t, int32_t, mul_scalar_i32, x *const_val)
UNARY_OP_WITH_SCALAR(int32_t, int32_t, div_scalar_i32, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(int32_t, int32_t, rem_scalar_i32, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(int32_t, int32_t, pow_scalar_i32, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(int32_t, int32_t, maximum_scalar_i32, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(int32_t, int32_t, minimum_scalar_i32, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(int32_t, bool, eq_scalar_i32, x == const_val)
UNARY_OP_WITH_SCALAR(int32_t, bool, ne_scalar_i32, x != const_val)
UNARY_OP_WITH_SCALAR(int32_t, bool, lt_scalar_i32, x < const_val)
UNARY_OP_WITH_SCALAR(int32_t, bool, le_scalar_i32, x <= const_val)
UNARY_OP_WITH_SCALAR(int32_t, bool, gt_scalar_i32, x > const_val)
UNARY_OP_WITH_SCALAR(int32_t, bool, ge_scalar_i32, x >= const_val)

UNARY_OP_WITH_SCALAR(int64_t, int64_t, add_scalar_i64, x + const_val)
UNARY_OP_WITH_SCALAR(int64_t, int64_t, sub_scalar_i64, x - const_val)
UNARY_OP_WITH_SCALAR(int64_t, int64_t, mul_scalar_i64, x *const_val)
UNARY_OP_WITH_SCALAR(int64_t, int64_t, div_scalar_i64, (const_val != 0) ? (x / const_val) : 0)
UNARY_OP_WITH_SCALAR(int64_t, int64_t, rem_scalar_i64, (const_val != 0) ? (x % const_val) : 0)
UNARY_OP_WITH_SCALAR(int64_t, int64_t, pow_scalar_i64, ipow(x, const_val))
UNARY_OP_WITH_SCALAR(int64_t, int64_t, maximum_scalar_i64, maximum(x, const_val))
UNARY_OP_WITH_SCALAR(int64_t, int64_t, minimum_scalar_i64, minimum(x, const_val))
UNARY_OP_WITH_SCALAR(int64_t, bool, eq_scalar_i64, x == const_val)
UNARY_OP_WITH_SCALAR(int64_t, bool, ne_scalar_i64, x != const_val)
UNARY_OP_WITH_SCALAR(int64_t, bool, lt_scalar_i64, x < const_val)
UNARY_OP_WITH_SCALAR(int64_t, bool, le_scalar_i64, x <= const_val)
UNARY_OP_WITH_SCALAR(int64_t, bool, gt_scalar_i64, x > const_val)
UNARY_OP_WITH_SCALAR(int64_t, bool, ge_scalar_i64, x >= const_val)
