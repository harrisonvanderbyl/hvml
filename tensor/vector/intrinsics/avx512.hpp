#ifndef AVX_HVEC_HPP
#define AVX_HVEC_HPP
#include "vector/Hvec.hpp"
#if defined(__NEON__)
#include "arm_neon.h"
#else
#include "immintrin.h"
#endif

// ============================================================================
// FLOAT32 OPERATIONS - SIZE 4
// ============================================================================

__weak Hvec<float, 4> __device__ __host__ operator+(Hvec<float, 4> a, Hvec<float, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    float32x2& vec_a_low = *(float32x2*)&a.data[0];
    float32x2& vec_a_high = *(float32x2*)&a.data[2];
    float32x2& vec_b_low = *(float32x2*)&b.data[0];
    float32x2& vec_b_high = *(float32x2*)&b.data[2];
    float32x4 vec_result;
    float32x2& vec_result_low = *(float32x2*)&vec_result.data[0];
    float32x2& vec_result_high = *(float32x2*)&vec_result.data[2];
    vec_result_low = vec_a_low + vec_b_low;
    vec_result_high = vec_a_high + vec_b_high;
    return vec_result;
#elif defined(__AVX512F__)
    __m512 vec_a = _mm512_loadu_ps(a.data);
    __m512 vec_b = _mm512_loadu_ps(b.data);
    __m512 vec_result = _mm512_add_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_a = _mm256_loadu_ps(a.data);
    __m256 vec_b = _mm256_loadu_ps(b.data);
    __m256 vec_result = _mm256_add_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_add_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__NEON__)
    float32x4_t vec_a = vld1q_f32(a.data);
    float32x4_t vec_b = vld1q_f32(b.data);
    float32x4_t vec_result = vaddq_f32(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#else  
    Hvec<float, 4> result;
    for (int i = 0; i < 4; i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
#endif
}

__weak Hvec<float, 4> __device__ __host__ operator-(Hvec<float, 4> a, Hvec<float, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    float32x2& vec_a_low = *(float32x2*)&a.data[0];
    float32x2& vec_a_high = *(float32x2*)&a.data[2];
    float32x2& vec_b_low = *(float32x2*)&b.data[0];
    float32x2& vec_b_high = *(float32x2*)&b.data[2];
    float32x4 vec_result;
    float32x2& vec_result_low = *(float32x2*)&vec_result.data[0];
    float32x2& vec_result_high = *(float32x2*)&vec_result.data[2];
    vec_result_low = vec_a_low - vec_b_low;
    vec_result_high = vec_a_high - vec_b_high;
    return vec_result;
#elif defined(__AVX512F__)
    __m512 vec_a = _mm512_loadu_ps(a.data);
    __m512 vec_b = _mm512_loadu_ps(b.data);
    __m512 vec_result = _mm512_sub_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_a = _mm256_loadu_ps(a.data);
    __m256 vec_b = _mm256_loadu_ps(b.data);
    __m256 vec_result = _mm256_sub_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_sub_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__NEON__)
    float32x4_t vec_a = vld1q_f32(a.data);
    float32x4_t vec_b = vld1q_f32(b.data);
    float32x4_t vec_result = vsubq_f32(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#else  
    Hvec<float, 4> result;
    for (int i = 0; i < 4; i++) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
#endif
}

__weak Hvec<float, 4> __device__ __host__ operator*(Hvec<float, 4> a, Hvec<float, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    float32x2& vec_a_low = *(float32x2*)&a.data[0];
    float32x2& vec_a_high = *(float32x2*)&a.data[2];
    float32x2& vec_b_low = *(float32x2*)&b.data[0];
    float32x2& vec_b_high = *(float32x2*)&b.data[2];
    float32x4 vec_result;
    float32x2& vec_result_low = *(float32x2*)&vec_result.data[0];
    float32x2& vec_result_high = *(float32x2*)&vec_result.data[2];
    vec_result_low = vec_a_low * vec_b_low;
    vec_result_high = vec_a_high * vec_b_high;
    return vec_result;
#elif defined(__AVX512F__)
    __m512 vec_a = _mm512_loadu_ps(a.data);
    __m512 vec_b = _mm512_loadu_ps(b.data);
    __m512 vec_result = _mm512_mul_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_a = _mm256_loadu_ps(a.data);
    __m256 vec_b = _mm256_loadu_ps(b.data);
    __m256 vec_result = _mm256_mul_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_mul_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__NEON__)
    float32x4_t vec_a = vld1q_f32(a.data);
    float32x4_t vec_b = vld1q_f32(b.data);
    float32x4_t vec_result = vmulq_f32(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#else  
    Hvec<float, 4> result;
    for (int i = 0; i < 4; i++) {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
#endif
}

__weak Hvec<float, 4> __device__ __host__ operator/(Hvec<float, 4> a, Hvec<float, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    float32x2& vec_a_low = *(float32x2*)&a.data[0];
    float32x2& vec_a_high = *(float32x2*)&a.data[2];
    float32x2& vec_b_low = *(float32x2*)&b.data[0];
    float32x2& vec_b_high = *(float32x2*)&b.data[2];
    float32x4 vec_result;
    float32x2& vec_result_low = *(float32x2*)&vec_result.data[0];
    float32x2& vec_result_high = *(float32x2*)&vec_result.data[2];
    vec_result_low = vec_a_low / vec_b_low;
    vec_result_high = vec_a_high / vec_b_high;
    return vec_result;
#elif defined(__AVX512F__)
    __m512 vec_a = _mm512_loadu_ps(a.data);
    __m512 vec_b = _mm512_loadu_ps(b.data);
    __m512 vec_result = _mm512_div_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_a = _mm256_loadu_ps(a.data);
    __m256 vec_b = _mm256_loadu_ps(b.data);
    __m256 vec_result = _mm256_div_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_div_ps(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#elif defined(__NEON__)
    float32x4_t vec_a = vld1q_f32(a.data);
    float32x4_t vec_b = vld1q_f32(b.data);
    float32x4_t vec_result = vdivq_f32(vec_a, vec_b);
    return *(Hvec<float, 4>*)&vec_result;
#else  
    Hvec<float, 4> result;
    for (int i = 0; i < 4; i++) {
        result.data[i] = a.data[i] / b.data[i];
    }
    return result;
#endif
}

__weak float __device__ __host__ dot(Hvec<float, 4> a, Hvec<float, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    float32x2& vec_a_low = *(float32x2*)&a.data[0];
    float32x2& vec_a_high = *(float32x2*)&a.data[2];
    float32x2& vec_b_low = *(float32x2*)&b.data[0];
    float32x2& vec_b_high = *(float32x2*)&b.data[2];
    float32x2 prod_low = vec_a_low * vec_b_low;
    float32x2 prod_high = vec_a_high * vec_b_high;
    return prod_low.x() + prod_low.y() + prod_high.x() + prod_high.y();
#elif defined(__AVX512F__)
    __m512 vec_a = _mm512_loadu_ps(a.data);
    __m512 vec_b = _mm512_loadu_ps(b.data);
    __m512 prod = _mm512_mul_ps(vec_a, vec_b);
    return _mm512_reduce_add_ps(prod);
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_a = _mm256_loadu_ps(a.data);
    __m256 vec_b = _mm256_loadu_ps(b.data);
    __m256 prod = _mm256_mul_ps(vec_a, vec_b);
    __m128 sum_high = _mm256_extractf128_ps(prod, 1);
    __m128 sum_low = _mm256_castps256_ps128(prod);
    __m128 sum = _mm_add_ps(sum_low, sum_high);
    __m128 shuf = _mm_movehdup_ps(sum);
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#elif defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 prod = _mm_mul_ps(vec_a, vec_b);
    __m128 shuf = _mm_movehdup_ps(prod);
    __m128 sums = _mm_add_ps(prod, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#elif defined(__NEON__)
    float32x4_t vec_a = vld1q_f32(a.data);
    float32x4_t vec_b = vld1q_f32(b.data);
    float32x4_t prod = vmulq_f32(vec_a, vec_b);
    return vaddvq_f32(prod);
#else  
    float result = 0.0f;
    for (int i = 0; i < 4; i++) {
        result += a.data[i] * b.data[i];
    }
    return result;
#endif
}

// ============================================================================
// FLOAT32 OPERATIONS - SIZE 2
// ============================================================================

__weak Hvec<float, 2> __device__ __host__ operator+(Hvec<float, 2> a, Hvec<float, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
#elif defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_add_ps(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#elif defined(__NEON__)
    float32x2_t vec_a = vld1_f32(a.data);
    float32x2_t vec_b = vld1_f32(b.data);
    float32x2_t vec_result = vadd_f32(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#else
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
#endif
}

__weak Hvec<float, 2> __device__ __host__ operator-(Hvec<float, 2> a, Hvec<float, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
#elif defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_sub_ps(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#elif defined(__NEON__)
    float32x2_t vec_a = vld1_f32(a.data);
    float32x2_t vec_b = vld1_f32(b.data);
    float32x2_t vec_result = vsub_f32(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#else
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
#endif
}

__weak Hvec<float, 2> __device__ __host__ operator*(Hvec<float, 2> a, Hvec<float, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
#elif defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_mul_ps(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#elif defined(__NEON__)
    float32x2_t vec_a = vld1_f32(a.data);
    float32x2_t vec_b = vld1_f32(b.data);
    float32x2_t vec_result = vmul_f32(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#else
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
#endif
}

__weak Hvec<float, 2> __device__ __host__ operator/(Hvec<float, 2> a, Hvec<float, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] / b.data[i];
    }
    return result;
#elif defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 vec_result = _mm_div_ps(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#elif defined(__NEON__)
    float32x2_t vec_a = vld1_f32(a.data);
    float32x2_t vec_b = vld1_f32(b.data);
    float32x2_t vec_result = vdiv_f32(vec_a, vec_b);
    return *(Hvec<float, 2>*)&vec_result;
#else
    Hvec<float, 2> result;
    for (int i = 0; i < 2; i++) {
        result.data[i] = a.data[i] / b.data[i];
    }
    return result;
#endif
}

__weak float __device__ __host__ dot(Hvec<float, 2> a, Hvec<float, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return a.data[0] * b.data[0] + a.data[1] * b.data[1];
#elif defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
    __m128 vec_a = _mm_loadu_ps(a.data);
    __m128 vec_b = _mm_loadu_ps(b.data);
    __m128 prod = _mm_mul_ps(vec_a, vec_b);
    __m128 shuf = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(prod, shuf);
    return _mm_cvtss_f32(sums);
#elif defined(__NEON__)
    float32x2_t vec_a = vld1_f32(a.data);
    float32x2_t vec_b = vld1_f32(b.data);
    float32x2_t prod = vmul_f32(vec_a, vec_b);
    return vaddv_f32(prod);
#else
    return a.data[0] * b.data[0] + a.data[1] * b.data[1];
#endif
}

// ============================================================================
// FLOAT16 (HALF) OPERATIONS - SIZE 4
// ============================================================================



__weak Hvec<float16, 4> __device__ __host__ operator+(Hvec<float16, 4> a, Hvec<float16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __half2& vec_a_low = *(__half2*)&a.data[0];
    __half2& vec_a_high = *(__half2*)&a.data[2];
    __half2& vec_b_low = *(__half2*)&b.data[0];
    __half2& vec_b_high = *(__half2*)&b.data[2];
    Hvec<float16, 4> vec_result;
    __half2& vec_result_low = *(__half2*)&vec_result.data[0];
    __half2& vec_result_high = *(__half2*)&vec_result.data[2];
    vec_result_low = __hadd2(vec_a_low, vec_b_low);
    vec_result_high = __hadd2(vec_a_high, vec_b_high);
    return vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_add_ph(vec_a, vec_b);
    return *(Hvec<float16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vadd_f16(vec_a, vec_b);
    Hvec<float16, 4> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 4> result;
    for (int i = 0; i < 4; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa + fb);
    }
    return result;
#endif
}

__weak Hvec<float16, 4> __device__ __host__ operator-(Hvec<float16, 4> a, Hvec<float16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __half2& vec_a_low = *(__half2*)&a.data[0];
    __half2& vec_a_high = *(__half2*)&a.data[2];
    __half2& vec_b_low = *(__half2*)&b.data[0];
    __half2& vec_b_high = *(__half2*)&b.data[2];
    Hvec<float16, 4> vec_result;
    __half2& vec_result_low = *(__half2*)&vec_result.data[0];
    __half2& vec_result_high = *(__half2*)&vec_result.data[2];
    vec_result_low = __hsub2(vec_a_low, vec_b_low);
    vec_result_high = __hsub2(vec_a_high, vec_b_high);
    return vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_sub_ph(vec_a, vec_b);
    return *(Hvec<float16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vsub_f16(vec_a, vec_b);
    Hvec<float16, 4> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 4> result;
    for (int i = 0; i < 4; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa - fb);
    }
    return result;
#endif
}

__weak Hvec<float16, 4> __device__ __host__ operator*(Hvec<float16, 4> a, Hvec<float16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a_low = *(half2*)&a.data[0];
    half2& vec_a_high = *(half2*)&a.data[2];
    half2& vec_b_low = *(half2*)&b.data[0];
    half2& vec_b_high = *(half2*)&b.data[2];
    Hvec<float16, 4> vec_result;
    half2& vec_result_low = *(half2*)&vec_result.data[0];
    half2& vec_result_high = *(half2*)&vec_result.data[2];
    vec_result_low = __hmul2(vec_a_low, vec_b_low);
    vec_result_high = __hmul2(vec_a_high, vec_b_high);
    return vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_mul_ph(vec_a, vec_b);
    return *(Hvec<float16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vmul_f16(vec_a, vec_b);
    Hvec<float16, 4> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 4> result;
    for (int i = 0; i < 4; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa * fb);
    }
    return result;
#endif
}

__weak Hvec<float16, 4> __device__ __host__ operator/(Hvec<float16, 4> a, Hvec<float16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a_low = *(half2*)&a.data[0];
    half2& vec_a_high = *(half2*)&a.data[2];
    half2& vec_b_low = *(half2*)&b.data[0];
    half2& vec_b_high = *(half2*)&b.data[2];
    Hvec<float16, 4> vec_result;
    half2& vec_result_low = *(half2*)&vec_result.data[0];
    half2& vec_result_high = *(half2*)&vec_result.data[2];
    vec_result_low = __h2div(vec_a_low, vec_b_low);
    vec_result_high = __h2div(vec_a_high, vec_b_high);
    return vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_div_ph(vec_a, vec_b);
    return *(Hvec<float16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vdiv_f16(vec_a, vec_b);
    Hvec<float16, 4> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 4> result;
    for (int i = 0; i < 4; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa / fb);
    }
    return result;
#endif
}

__weak float __device__ __host__ dot(Hvec<float16, 4> a, Hvec<float16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a_low = *(half2*)&a.data[0];
    half2& vec_a_high = *(half2*)&a.data[2];
    half2& vec_b_low = *(half2*)&b.data[0];
    half2& vec_b_high = *(half2*)&b.data[2];
    half2 prod_low = __hmul2(vec_a_low, vec_b_low);
    half2 prod_high = __hmul2(vec_a_high, vec_b_high);
    float sum = __half2float(prod_low.x) + __half2float(prod_low.y) + 
                __half2float(prod_high.x) + __half2float(prod_high.y);
    return sum;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h prod = _mm_mul_ph(vec_a, vec_b);
    // Convert to float and sum
    __m128 prod_f32 = _mm_cvtph_ps(_mm_castph_si128(prod));
    __m128 shuf = _mm_movehdup_ps(prod_f32);
    __m128 sums = _mm_add_ps(prod_f32, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t prod = vmul_f16(vec_a, vec_b);
    float32x4_t prod_f32 = vcvt_f32_f16(prod);
    return vaddvq_f32(prod_f32);
#else
    float result = 0.0f;
    for (int i = 0; i < 4; i++) {
        result += (float)a.data[i] * (float)b.data[i];
    }
    return result;
#endif
}

// ============================================================================
// FLOAT16 (HALF) OPERATIONS - SIZE 2
// ============================================================================

__weak Hvec<float16, 2> __device__ __host__ operator+(Hvec<float16, 2> a, Hvec<float16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a = *(half2*)&a.data[0];
    half2& vec_b = *(half2*)&b.data[0];
    half2 vec_result = __hadd2(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_add_ph(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vadd_f16(vec_a, vec_b);
    Hvec<float16, 2> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa + fb);
    }
    return result;
#endif
}

__weak Hvec<float16, 2> __device__ __host__ operator-(Hvec<float16, 2> a, Hvec<float16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a = *(half2*)&a.data[0];
    half2& vec_b = *(half2*)&b.data[0];
    half2 vec_result = __hsub2(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_sub_ph(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vsub_f16(vec_a, vec_b);
    Hvec<float16, 2> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa - fb);
    }
    return result;
#endif
}

__weak Hvec<float16, 2> __device__ __host__ operator*(Hvec<float16, 2> a, Hvec<float16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a = *(half2*)&a.data[0];
    half2& vec_b = *(half2*)&b.data[0];
    half2 vec_result = __hmul2(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_mul_ph(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vmul_f16(vec_a, vec_b);
    Hvec<float16, 2> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa * fb);
    }
    return result;
#endif
}

__weak Hvec<float16, 2> __device__ __host__ operator/(Hvec<float16, 2> a, Hvec<float16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a = *(half2*)&a.data[0];
    half2& vec_b = *(half2*)&b.data[0];
    half2 vec_result = __h2div(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h vec_result = _mm_div_ph(vec_a, vec_b);
    return *(Hvec<float16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t vec_result = vdiv_f16(vec_a, vec_b);
    Hvec<float16, 2> result;
    vst1_f16((__fp16*)result.data, vec_result);
    return result;
#else
    Hvec<float16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = (float)a.data[i];
        float fb = (float)b.data[i];
        result.data[i] = (float16)(fa / fb);
    }
    return result;
#endif
}

__weak float __device__ __host__ dot(Hvec<float16, 2> a, Hvec<float16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    half2& vec_a = *(half2*)&a.data[0];
    half2& vec_b = *(half2*)&b.data[0];
    half2 prod = __hmul2(vec_a, vec_b);
    return __half2float(prod.x) + __half2float(prod.y);
#elif defined(__AVX512FP16__)
    __m128h vec_a = _mm_loadu_ph(a.data);
    __m128h vec_b = _mm_loadu_ph(b.data);
    __m128h prod = _mm_mul_ph(vec_a, vec_b);
    __m128 prod_f32 = _mm_cvtph_ps(_mm_castph_si128(prod));
    __m128 shuf = _mm_shuffle_ps(prod_f32, prod_f32, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(prod_f32, shuf);
    return _mm_cvtss_f32(sums);
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x4_t vec_a = vld1_f16((const __fp16*)a.data);
    float16x4_t vec_b = vld1_f16((const __fp16*)b.data);
    float16x4_t prod = vmul_f16(vec_a, vec_b);
    float32x2_t prod_f32 = vcvt_f32_f16(vget_low_f16(prod));
    return vaddv_f32(prod_f32);
#else
    return (float)a.data[0] * (float)b.data[0] + (float)a.data[1] * (float)b.data[1];
#endif
}

// ============================================================================
// BFLOAT16 OPERATIONS - SIZE 4
// ============================================================================



__weak Hvec<bfloat16, 4> __device__ __host__ operator+(Hvec<bfloat16, 4> a, Hvec<bfloat16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a_low = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_a_high = *(__nv_bfloat162*)&a.data[2];
    __nv_bfloat162& vec_b_low = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162& vec_b_high = *(__nv_bfloat162*)&b.data[2];
    Hvec<bfloat16, 4> vec_result;
    __nv_bfloat162& vec_result_low = *(__nv_bfloat162*)&vec_result.data[0];
    __nv_bfloat162& vec_result_high = *(__nv_bfloat162*)&vec_result.data[2];
    vec_result_low = __hadd2(vec_a_low, vec_b_low);
    vec_result_high = __hadd2(vec_a_high, vec_b_high);
    return vec_result;
// #elif defined(__AVX512BF16__)
    // __m128bh vec_a = _mm_loadu_pbh(a.data);
    // __m128bh vec_b = _mm_loadu_pbh(b.data);
    // __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
    // __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
    // __m256 vec_result_f32 = _mm256_add_ps(vec_a_f32, vec_b_f32);
    // __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
    // return *(Hvec<bfloat16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vaddq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 4> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 4> result;
    for (int i = 0; i < 4; i++) {
        // BF16 to float conversion (shift left by 16 bits)
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float sum = fa + fb;
        // Float to BF16 conversion (take upper 16 bits)
        result.data[i] = (bfloat16)(*((unsigned int*)&sum) >> 16);
    }
    return result;
#endif
}

__weak Hvec<bfloat16, 4> __device__ __host__ operator-(Hvec<bfloat16, 4> a, Hvec<bfloat16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a_low = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_a_high = *(__nv_bfloat162*)&a.data[2];
    __nv_bfloat162& vec_b_low = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162& vec_b_high = *(__nv_bfloat162*)&b.data[2];
    Hvec<bfloat16, 4> vec_result;
    __nv_bfloat162& vec_result_low = *(__nv_bfloat162*)&vec_result.data[0];
    __nv_bfloat162& vec_result_high = *(__nv_bfloat162*)&vec_result.data[2];
    vec_result_low = __hsub2(vec_a_low, vec_b_low);
    vec_result_high = __hsub2(vec_a_high, vec_b_high);
    return vec_result;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 vec_result_f32 = _mm256_sub_ps(vec_a_f32, vec_b_f32);
//     __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
//     return *(Hvec<bfloat16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vsubq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 4> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 4> result;
    for (int i = 0; i < 4; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float diff = fa - fb;
        result.data[i] = (bfloat16)(*((unsigned int*)&diff) >> 16);
    }
    return result;
#endif
}

__weak Hvec<bfloat16, 4> __device__ __host__ operator*(Hvec<bfloat16, 4> a, Hvec<bfloat16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a_low = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_a_high = *(__nv_bfloat162*)&a.data[2];
    __nv_bfloat162& vec_b_low = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162& vec_b_high = *(__nv_bfloat162*)&b.data[2];
    Hvec<bfloat16, 4> vec_result;
    __nv_bfloat162& vec_result_low = *(__nv_bfloat162*)&vec_result.data[0];
    __nv_bfloat162& vec_result_high = *(__nv_bfloat162*)&vec_result.data[2];
    vec_result_low = __hmul2(vec_a_low, vec_b_low);
    vec_result_high = __hmul2(vec_a_high, vec_b_high);
    return vec_result;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 vec_result_f32 = _mm256_mul_ps(vec_a_f32, vec_b_f32);
//     __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
//     return *(Hvec<bfloat16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vmulq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 4> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 4> result;
    for (int i = 0; i < 4; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float prod = fa * fb;
        result.data[i] = (bfloat16)(*((unsigned int*)&prod) >> 16);
    }
    return result;
#endif
}

__weak Hvec<bfloat16, 4> __device__ __host__ operator/(Hvec<bfloat16, 4> a, Hvec<bfloat16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a_low = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_a_high = *(__nv_bfloat162*)&a.data[2];
    __nv_bfloat162& vec_b_low = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162& vec_b_high = *(__nv_bfloat162*)&b.data[2];
    Hvec<bfloat16, 4> vec_result;
    __nv_bfloat162& vec_result_low = *(__nv_bfloat162*)&vec_result.data[0];
    __nv_bfloat162& vec_result_high = *(__nv_bfloat162*)&vec_result.data[2];
    vec_result_low = __h2div(vec_a_low, vec_b_low);
    vec_result_high = __h2div(vec_a_high, vec_b_high);
    return vec_result;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 vec_result_f32 = _mm256_div_ps(vec_a_f32, vec_b_f32);
//     __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
//     return *(Hvec<bfloat16, 4>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vdivq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 4> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 4> result;
    for (int i = 0; i < 4; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float quot = fa / fb;
        result.data[i] = quot;
    }
    return result;
#endif
}

__weak float __device__ __host__ dot(Hvec<bfloat16, 4> a, Hvec<bfloat16, 4> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a_low = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_a_high = *(__nv_bfloat162*)&a.data[2];
    __nv_bfloat162& vec_b_low = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162& vec_b_high = *(__nv_bfloat162*)&b.data[2];
    __nv_bfloat162 prod_low = __hmul2(vec_a_low, vec_b_low);
    __nv_bfloat162 prod_high = __hmul2(vec_a_high, vec_b_high);
    float sum = __bfloat162float(prod_low.x) + __bfloat162float(prod_low.y) + 
                __bfloat162float(prod_high.x) + __bfloat162float(prod_high.y);
    return sum;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 prod = _mm256_mul_ps(vec_a_f32, vec_b_f32);
//     __m128 sum_high = _mm256_extractf128_ps(prod, 1);
//     __m128 sum_low = _mm256_castps256_ps128(prod);
//     __m128 sum = _mm_add_ps(sum_low, sum_high);
//     __m128 shuf = _mm_movehdup_ps(sum);
//     __m128 sums = _mm_add_ps(sum, shuf);
//     shuf = _mm_movehl_ps(shuf, sums);
//     sums = _mm_add_ss(sums, shuf);
//     return _mm_cvtss_f32(sums);
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t prod = vmulq_f32(vec_a_f32, vec_b_f32);
    return vaddvq_f32(prod);
#else
    float result = 0.0f;
    for (int i = 0; i < 4; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        result += fa * fb;
    }
    return result;
#endif
}

// ============================================================================
// BFLOAT16 OPERATIONS - SIZE 2
// ============================================================================

__weak Hvec<bfloat16, 2> __device__ __host__ operator+(Hvec<bfloat16, 2> a, Hvec<bfloat16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_b = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162 vec_result = __hadd2(vec_a, vec_b);
    return *(Hvec<bfloat16, 2>*)&vec_result;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 vec_result_f32 = _mm256_add_ps(vec_a_f32, vec_b_f32);
//     __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
//     return *(Hvec<bfloat16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vaddq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 2> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float sum = fa + fb;
        result.data[i] = (bfloat16)(*((unsigned int*)&sum) >> 16);
    }
    return result;
#endif
}

__weak Hvec<bfloat16, 2> __device__ __host__ operator-(Hvec<bfloat16, 2> a, Hvec<bfloat16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_b = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162 vec_result = __hsub2(vec_a, vec_b);
    return *(Hvec<bfloat16, 2>*)&vec_result;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 vec_result_f32 = _mm256_sub_ps(vec_a_f32, vec_b_f32);
//     __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
//     return *(Hvec<bfloat16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vsubq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 2> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float diff = fa - fb;
        result.data[i] = (bfloat16)(*((unsigned int*)&diff) >> 16);
    }
    return result;
#endif
}

__weak Hvec<bfloat16, 2> __device__ __host__ operator*(Hvec<bfloat16, 2> a, Hvec<bfloat16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_b = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162 vec_result = __hmul2(vec_a, vec_b);
    return *(Hvec<bfloat16, 2>*)&vec_result;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 vec_result_f32 = _mm256_mul_ps(vec_a_f32, vec_b_f32);
//     __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
//     return *(Hvec<bfloat16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vmulq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 2> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float prod = fa * fb;
        result.data[i] = (bfloat16)(*((unsigned int*)&prod) >> 16);
    }
    return result;
#endif
}

__weak Hvec<bfloat16, 2> __device__ __host__ operator/(Hvec<bfloat16, 2> a, Hvec<bfloat16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_b = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162 vec_result = __h2div(vec_a, vec_b);
    return *(Hvec<bfloat16, 2>*)&vec_result;
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 vec_result_f32 = _mm256_div_ps(vec_a_f32, vec_b_f32);
//     __m128bh vec_result = _mm256_cvtneps_pbh(vec_result_f32);
//     return *(Hvec<bfloat16, 2>*)&vec_result;
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t vec_result_f32 = vdivq_f32(vec_a_f32, vec_b_f32);
    bfloat16x4_t vec_result = vcvt_bf16_f32(vec_result_f32);
    Hvec<bfloat16, 2> result;
    vst1_bf16((bfloat16*)result.data, vec_result);
    return result;
#else
    Hvec<bfloat16, 2> result;
    for (int i = 0; i < 2; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        float quot = fa / fb;
        result.data[i] = quot;
    }
    return result;
#endif
}

__weak float __device__ __host__ dot(Hvec<bfloat16, 2> a, Hvec<bfloat16, 2> b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __nv_bfloat162& vec_a = *(__nv_bfloat162*)&a.data[0];
    __nv_bfloat162& vec_b = *(__nv_bfloat162*)&b.data[0];
    __nv_bfloat162 prod = __hmul2(vec_a, vec_b);
    return __bfloat162float(prod.x) + __bfloat162float(prod.y);
// #elif defined(__AVX512BF16__)
//     __m128bh vec_a = _mm_loadu_pbh(a.data);
//     __m128bh vec_b = _mm_loadu_pbh(b.data);
//     __m256 vec_a_f32 = _mm256_cvtpbh_ps(vec_a);
//     __m256 vec_b_f32 = _mm256_cvtpbh_ps(vec_b);
//     __m256 prod = _mm256_mul_ps(vec_a_f32, vec_b_f32);
//     __m128 prod_low = _mm256_castps256_ps128(prod);
//     __m128 shuf = _mm_shuffle_ps(prod_low, prod_low, _MM_SHUFFLE(2, 3, 0, 1));
//     __m128 sums = _mm_add_ps(prod_low, shuf);
//     return _mm_cvtss_f32(sums);
#elif defined(__ARM_FEATURE_BF16)
    bfloat16x4_t vec_a = vld1_bf16((const bfloat16*)a.data);
    bfloat16x4_t vec_b = vld1_bf16((const bfloat16*)b.data);
    float32x4_t vec_a_f32 = vcvt_f32_bf16(vec_a);
    float32x4_t vec_b_f32 = vcvt_f32_bf16(vec_b);
    float32x4_t prod = vmulq_f32(vec_a_f32, vec_b_f32);
    float32x2_t prod_low = vget_low_f32(prod);
    return vaddv_f32(prod_low);
#else
    float result = 0.0f;
    for (int i = 0; i < 2; i++) {
        float fa = float(a.data[i]);
        float fb = float(b.data[i]);
        result += fa * fb;
    }
    return result;
#endif
}

#endif // AVX_HVEC_HPP