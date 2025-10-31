#ifndef HODU_CPU_KERNELS_CONSTANTS_H
#define HODU_CPU_KERNELS_CONSTANTS_H

#include <stdint.h>

// Integer type limits
#ifndef INT8_MIN
#define INT8_MIN (-128)
#endif
#ifndef INT8_MAX
#define INT8_MAX 127
#endif
#ifndef INT16_MIN
#define INT16_MIN (-32768)
#endif
#ifndef INT16_MAX
#define INT16_MAX 32767
#endif
#ifndef INT32_MIN
#define INT32_MIN (-2147483648)
#endif
#ifndef INT32_MAX
#define INT32_MAX 2147483647
#endif
#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807LL - 1)
#endif
#ifndef INT64_MAX
#define INT64_MAX 9223372036854775807LL
#endif
#ifndef UINT8_MAX
#define UINT8_MAX 255
#endif
#ifndef UINT16_MAX
#define UINT16_MAX 65535
#endif
#ifndef UINT32_MAX
#define UINT32_MAX 4294967295U
#endif
#ifndef UINT64_MAX
#define UINT64_MAX 18446744073709551615ULL
#endif

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif

// Platform-specific compatibility for exp10
#include <math.h>

#ifndef exp10f
static inline float exp10f(float x) {
    return expf(x * 2.302585092994046f); // ln(10)
}
#endif

#ifndef exp10
static inline double exp10(double x) {
    return exp(x * 2.302585092994045); // ln(10)
}
#endif

#endif // HODU_CPU_KERNELS_CONSTANTS_H
