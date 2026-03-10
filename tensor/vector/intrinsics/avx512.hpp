#ifndef AVX_HVEC_HPP
#define AVX_HVEC_HPP

#include "vector/Hvec.hpp"

// ============================================================================
// PLATFORM DETECTION & INTRINSIC HEADERS
// ============================================================================

#if defined(__NEON__) || defined(__ARM_NEON)
    #include "arm_neon.h"
    // For bf16 on ARM: requires -march=armv8.6-a+bf16 or equivalent
    #if defined(__ARM_FEATURE_BF16)
        #include "arm_bf16.h"
    #endif
#else
    #include "immintrin.h"
    // F16C: _mm256_cvtph_ps / _mm256_cvtps_ph  (requires __F16C__)
    // AVX-512 FP16: native float16 arithmetic   (requires __AVX512FP16__)
    // AVX-512 BF16: native bfloat16 arithmetic  (requires __AVX512BF16__)
#endif

// ============================================================================
// HALF-PRECISION TYPE STUBS
//
// Where a native compiler type exists (_Float16, __bf16) we alias it.
// Otherwise we define storage-only wrappers that hold the raw bit pattern.
// Arithmetic always upcasts to float32 on paths without native HW support.
// ============================================================================

// ----- float16 -----


// ============================================================================
// INTRINSICS ENUM  (must match Hvec.hpp)
// ============================================================================

__device__ __host__ static inline constexpr Intrinsics get_available_intrinsics()
{
#if defined(__CUDA_ARCH__)
    return Intrinsics::CUDA;
#elif defined(__HIP_DEVICE_COMPILE__)
    return Intrinsics::HIP;
#elif defined(__AVX512F__)
    return Intrinsics::AVX512;
#elif defined(__AVX__)
    return Intrinsics::AVX;
#elif defined(__SSE2__)
    return Intrinsics::SSE2;
#elif defined(__NEON__) || defined(__ARM_NEON)
    return Intrinsics::NEON;
#else
    return Intrinsics::None;
#endif
}

// ============================================================================
// GENERIC (SCALAR FALLBACK) OPERATION SET
// ============================================================================

template <Intrinsics intrin, typename T, int N>
struct vectorOperationSet
{
    static Hvec<T, N> add(const Hvec<T, N>& a, const Hvec<T, N>& b)
    {
        Hvec<T, N> r;
        for (int i = 0; i < N; i++) r.data[i] = a.data[i] + b.data[i];
        return r;
    }
    static Hvec<T, N> sub(const Hvec<T, N>& a, const Hvec<T, N>& b)
    {
        Hvec<T, N> r;
        for (int i = 0; i < N; i++) r.data[i] = a.data[i] - b.data[i];
        return r;
    }
    static Hvec<T, N> mul(const Hvec<T, N>& a, const Hvec<T, N>& b)
    {
        Hvec<T, N> r;
        for (int i = 0; i < N; i++) r.data[i] = a.data[i] * b.data[i];
        return r;
    }
    static Hvec<T, N> div(const Hvec<T, N>& a, const Hvec<T, N>& b)
    {
        Hvec<T, N> r;
        for (int i = 0; i < N; i++) r.data[i] = a.data[i] / b.data[i];
        return r;
    }
    // fma: (a * b) + c  — scalar fallback, not fused
    static Hvec<T, N> fma(const Hvec<T, N>& a, const Hvec<T, N>& b, const Hvec<T, N>& c)
    {
        Hvec<T, N> r;
        for (int i = 0; i < N; i++) r.data[i] = a.data[i] * b.data[i] + c.data[i];
        return r;
    }
    static T hsum(const Hvec<T, N>& a)
    {
        T acc = a.data[0];
        for (int i = 1; i < N; i++) acc += a.data[i];
        return acc;
    }
};

// ============================================================================
// SCALAR FALLBACK FOR float16  (upcast -> f32 op -> downcast)
// Covers all (intrin, float16, N) combos not explicitly specialised below.
// ============================================================================

template <Intrinsics intrin, int N>
struct vectorOperationSet<intrin, float16, N>
{
    using F16Vec = Hvec<float16, N>;
    using F32Vec = Hvec<float,   N>;

    static F32Vec upcast(const F16Vec& v)
    {
        F32Vec r;
        for (int i = 0; i < N; i++) r.data[i] = float(v.data[i]);
        return r;
    }
    static F16Vec downcast(const F32Vec& v)
    {
        F16Vec r;
        for (int i = 0; i < N; i++) r.data[i] = float16(v.data[i]);
        return r;
    }
    static F16Vec   add(const F16Vec& a, const F16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::add(upcast(a), upcast(b))); }
    static F16Vec   sub(const F16Vec& a, const F16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::sub(upcast(a), upcast(b))); }
    static F16Vec   mul(const F16Vec& a, const F16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::mul(upcast(a), upcast(b))); }
    static F16Vec   div(const F16Vec& a, const F16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::div(upcast(a), upcast(b))); }
    static F16Vec   fma(const F16Vec& a, const F16Vec& b, const F16Vec& c) { return downcast(vectorOperationSet<intrin,float,N>::fma(upcast(a), upcast(b), upcast(c))); }
    static float16 hsum(const F16Vec& a) { return float16(vectorOperationSet<intrin,float,N>::hsum(upcast(a))); }
};

// ============================================================================
// SCALAR FALLBACK FOR bfloat16  (upcast -> f32 op -> downcast)
// ============================================================================

template <Intrinsics intrin, int N>
struct vectorOperationSet<intrin, bfloat16, N>
{
    using BF16Vec = Hvec<bfloat16, N>;
    using F32Vec  = Hvec<float,    N>;

    static F32Vec upcast(const BF16Vec& v)
    {
        F32Vec r;
        for (int i = 0; i < N; i++) r.data[i] = float(v.data[i]);
        return r;
    }
    static BF16Vec downcast(const F32Vec& v)
    {
        BF16Vec r;
        for (int i = 0; i < N; i++) r.data[i] = bfloat16(v.data[i]);
        return r;
    }
    static BF16Vec   add(const BF16Vec& a, const BF16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::add(upcast(a), upcast(b))); }
    static BF16Vec   sub(const BF16Vec& a, const BF16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::sub(upcast(a), upcast(b))); }
    static BF16Vec   mul(const BF16Vec& a, const BF16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::mul(upcast(a), upcast(b))); }
    static BF16Vec   div(const BF16Vec& a, const BF16Vec& b) { return downcast(vectorOperationSet<intrin,float,N>::div(upcast(a), upcast(b))); }
    static BF16Vec   fma(const BF16Vec& a, const BF16Vec& b, const BF16Vec& c) { return downcast(vectorOperationSet<intrin,float,N>::fma(upcast(a), upcast(b), upcast(c))); }
    static bfloat16 hsum(const BF16Vec& a) { return bfloat16(vectorOperationSet<intrin,float,N>::hsum(upcast(a))); }
};

// ============================================================================
// SSE2  — float32 x4
// ============================================================================

#if defined(__SSE2__)
template <>
struct vectorOperationSet<Intrinsics::SSE2, float, 4>
{
    static Hvec<float, 4> add(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        __m128 va = _mm_loadu_ps(a.data), vb = _mm_loadu_ps(b.data);
        __m128 vr = _mm_add_ps(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    static Hvec<float, 4> sub(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        __m128 va = _mm_loadu_ps(a.data), vb = _mm_loadu_ps(b.data);
        __m128 vr = _mm_sub_ps(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    static Hvec<float, 4> mul(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        __m128 va = _mm_loadu_ps(a.data), vb = _mm_loadu_ps(b.data);
        __m128 vr = _mm_mul_ps(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    static Hvec<float, 4> div(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        __m128 va = _mm_loadu_ps(a.data), vb = _mm_loadu_ps(b.data);
        __m128 vr = _mm_div_ps(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    static Hvec<float, 4> fma(const Hvec<float, 4>& a, const Hvec<float, 4>& b, const Hvec<float, 4>& c)
    {
        __m128 va = _mm_loadu_ps(a.data), vb = _mm_loadu_ps(b.data), vc = _mm_loadu_ps(c.data);
#if defined(__FMA__)
        __m128 vr = _mm_fmadd_ps(va, vb, vc);
#else
        __m128 vr = _mm_add_ps(_mm_mul_ps(va, vb), vc);
#endif
        return *(Hvec<float, 4>*)&vr;
    }
    // Classic 4-lane horizontal sum via two shuffles + two adds
    static float hsum(const Hvec<float, 4>& a)
    {
        __m128 v  = _mm_loadu_ps(a.data);
        __m128 s1 = _mm_add_ps(v,  _mm_shuffle_ps(v,  v,  _MM_SHUFFLE(2, 3, 0, 1)));
        __m128 s2 = _mm_add_ps(s1, _mm_shuffle_ps(s1, s1, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(s2);
    }
};
#endif // __SSE2__

// ============================================================================
// AVX  — float32 x8
// ============================================================================

#if defined(__AVX__)
template <>
struct vectorOperationSet<Intrinsics::AVX, float, 8>
{
    static Hvec<float, 8> add(const Hvec<float, 8>& a, const Hvec<float, 8>& b)
    {
        __m256 va = _mm256_loadu_ps(a.data), vb = _mm256_loadu_ps(b.data);
        __m256 vr = _mm256_add_ps(va, vb);
        return *(Hvec<float, 8>*)&vr;
    }
    static Hvec<float, 8> sub(const Hvec<float, 8>& a, const Hvec<float, 8>& b)
    {
        __m256 va = _mm256_loadu_ps(a.data), vb = _mm256_loadu_ps(b.data);
        __m256 vr = _mm256_sub_ps(va, vb);
        return *(Hvec<float, 8>*)&vr;
    }
    static Hvec<float, 8> mul(const Hvec<float, 8>& a, const Hvec<float, 8>& b)
    {
        __m256 va = _mm256_loadu_ps(a.data), vb = _mm256_loadu_ps(b.data);
        __m256 vr = _mm256_mul_ps(va, vb);
        return *(Hvec<float, 8>*)&vr;
    }
    static Hvec<float, 8> div(const Hvec<float, 8>& a, const Hvec<float, 8>& b)
    {
        __m256 va = _mm256_loadu_ps(a.data), vb = _mm256_loadu_ps(b.data);
        __m256 vr = _mm256_div_ps(va, vb);
        return *(Hvec<float, 8>*)&vr;
    }
    static Hvec<float, 8> fma(const Hvec<float, 8>& a, const Hvec<float, 8>& b, const Hvec<float, 8>& c)
    {
        __m256 va = _mm256_loadu_ps(a.data), vb = _mm256_loadu_ps(b.data), vc = _mm256_loadu_ps(c.data);
#if defined(__FMA__)
        __m256 vr = _mm256_fmadd_ps(va, vb, vc);
#else
        __m256 vr = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
#endif
        return *(Hvec<float, 8>*)&vr;
    }
    // Fold 256->128 then use the SSE2 4-lane hsum
    static float hsum(const Hvec<float, 8>& a)
    {
        __m256 v  = _mm256_loadu_ps(a.data);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        __m128 s1 = _mm_add_ps(s,  _mm_shuffle_ps(s,  s,  _MM_SHUFFLE(2, 3, 0, 1)));
        __m128 s2 = _mm_add_ps(s1, _mm_shuffle_ps(s1, s1, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(s2);
    }
};

// AVX CPUs can still do 128-bit ops — forward to SSE2 specialisation
template <>
struct vectorOperationSet<Intrinsics::AVX, float, 4>
    : vectorOperationSet<Intrinsics::SSE2, float, 4> {};

// ---- float16 x8: upcast pairs via F16C (__F16C__ + __AVX__) ----
#if defined(__F16C__)
template <>
struct vectorOperationSet<Intrinsics::AVX, float16, 8>
{
    // Pack 8 float16 bit-patterns into a __m128i then widen with F16C
    static __m256 load_to_f32(const Hvec<float16, 8>& v)
    {
        __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v.data));
        return _mm256_cvtph_ps(bits);
    }
    static Hvec<float16, 8> store_from_f32(const __m256& vr)
    {
        __m128i half = _mm256_cvtps_ph(vr, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        Hvec<float16, 8> result;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result.data), half);
        return result;
    }

    static Hvec<float16, 8> add(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store_from_f32(_mm256_add_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<float16, 8> sub(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store_from_f32(_mm256_sub_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<float16, 8> mul(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store_from_f32(_mm256_mul_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<float16, 8> div(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store_from_f32(_mm256_div_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<float16, 8> fma(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b, const Hvec<float16, 8>& c)
    {
        __m256 va = load_to_f32(a), vb = load_to_f32(b), vc = load_to_f32(c);
#if defined(__FMA__)
        return store_from_f32(_mm256_fmadd_ps(va, vb, vc));
#else
        return store_from_f32(_mm256_add_ps(_mm256_mul_ps(va, vb), vc));
#endif
    }
    static float16 hsum(const Hvec<float16, 8>& a)
    {
        // Reuse the f32 hsum after upcasting
        __m256 v = load_to_f32(a);
        Hvec<float, 8> tmp; _mm256_storeu_ps(tmp.data, v);
        return float16(vectorOperationSet<Intrinsics::AVX, float, 8>::hsum(tmp));
    }
};
#endif // __F16C__

// ---- bfloat16 x8 via AVX: upcast with zero-extend + left-shift trick ----
template <>
struct vectorOperationSet<Intrinsics::AVX, bfloat16, 8>
{
    // bf16 bits sit in the upper halfword of f32 — zero-extend and shift
    static __m256 load_to_f32(const Hvec<bfloat16, 8>& v)
    {
        __m128i bits16  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v.data));
        __m256i bits32  = _mm256_cvtepu16_epi32(bits16);          // u16 -> u32 zero-extend
        __m256i shifted = _mm256_slli_epi32(bits32, 16);          // bf16 into upper half
        return _mm256_castsi256_ps(shifted);
    }
    // Truncate f32 -> bf16: take upper 16 bits of each lane.
    // Note: this truncates (round-towards-zero). For round-to-nearest-even
    // you'd add a rounding bias of 0x7FFF + (bit16 & 1) before shifting.
    static Hvec<bfloat16, 8> store_from_f32(const __m256& vr)
    {
        __m256i bits32  = _mm256_castps_si256(vr);
        __m256i shifted = _mm256_srli_epi32(bits32, 16);
        // _mm256_packus_epi32 packs within 128-bit lanes; needs a final permute
        __m128i lo      = _mm256_castsi256_si128(shifted);
        __m128i hi      = _mm256_extracti128_si256(shifted, 1);
        __m128i packed  = _mm_packus_epi32(lo, hi);           // 8x u16 in order
        Hvec<bfloat16, 8> result;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result.data), packed);
        return result;
    }

    static Hvec<bfloat16, 8> add(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    { return store_from_f32(_mm256_add_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 8> sub(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    { return store_from_f32(_mm256_sub_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 8> mul(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    { return store_from_f32(_mm256_mul_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 8> div(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    { return store_from_f32(_mm256_div_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 8> fma(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b, const Hvec<bfloat16, 8>& c)
    {
        __m256 va = load_to_f32(a), vb = load_to_f32(b), vc = load_to_f32(c);
#if defined(__FMA__)
        return store_from_f32(_mm256_fmadd_ps(va, vb, vc));
#else
        return store_from_f32(_mm256_add_ps(_mm256_mul_ps(va, vb), vc));
#endif
    }
    static bfloat16 hsum(const Hvec<bfloat16, 8>& a)
    {
        __m256 v = load_to_f32(a);
        Hvec<float, 8> tmp; _mm256_storeu_ps(tmp.data, v);
        return bfloat16(vectorOperationSet<Intrinsics::AVX, float, 8>::hsum(tmp));
    }
};

#endif // __AVX__

// ============================================================================
// AVX-512  — float32 x16
// ============================================================================

#if defined(__AVX512F__)
template <>
struct vectorOperationSet<Intrinsics::AVX512, float, 16>
{
    static Hvec<float, 16> add(const Hvec<float, 16>& a, const Hvec<float, 16>& b)
    {
        __m512 va = _mm512_loadu_ps(a.data), vb = _mm512_loadu_ps(b.data);
        __m512 vr = _mm512_add_ps(va, vb);
        return *(Hvec<float, 16>*)&vr;
    }
    static Hvec<float, 16> sub(const Hvec<float, 16>& a, const Hvec<float, 16>& b)
    {
        __m512 va = _mm512_loadu_ps(a.data), vb = _mm512_loadu_ps(b.data);
        __m512 vr = _mm512_sub_ps(va, vb);
        return *(Hvec<float, 16>*)&vr;
    }
    static Hvec<float, 16> mul(const Hvec<float, 16>& a, const Hvec<float, 16>& b)
    {
        __m512 va = _mm512_loadu_ps(a.data), vb = _mm512_loadu_ps(b.data);
        __m512 vr = _mm512_mul_ps(va, vb);
        return *(Hvec<float, 16>*)&vr;
    }
    static Hvec<float, 16> div(const Hvec<float, 16>& a, const Hvec<float, 16>& b)
    {
        __m512 va = _mm512_loadu_ps(a.data), vb = _mm512_loadu_ps(b.data);
        __m512 vr = _mm512_div_ps(va, vb);
        return *(Hvec<float, 16>*)&vr;
    }
    // AVX-512F always includes FMA — no fallback needed
    static Hvec<float, 16> fma(const Hvec<float, 16>& a, const Hvec<float, 16>& b, const Hvec<float, 16>& c)
    {
        __m512 va = _mm512_loadu_ps(a.data), vb = _mm512_loadu_ps(b.data), vc = _mm512_loadu_ps(c.data);
        __m512 vr = _mm512_fmadd_ps(va, vb, vc);
        return *(Hvec<float, 16>*)&vr;
    }
    static float hsum(const Hvec<float, 16>& a)
    {
        return _mm512_reduce_add_ps(_mm512_loadu_ps(a.data));
    }
};

// Forward smaller widths to AVX/SSE2 specialisations on AVX-512 hosts
template <>
struct vectorOperationSet<Intrinsics::AVX512, float, 8>
    : vectorOperationSet<Intrinsics::AVX, float, 8> {};

template <>
struct vectorOperationSet<Intrinsics::AVX512, float, 4>
    : vectorOperationSet<Intrinsics::SSE2, float, 4> {};

// ---- float16 x16: native with AVX-512 FP16 ----
// Without __AVX512FP16__ the generic upcast fallback is used automatically.
#if defined(__AVX512FP16__)
template <>
struct vectorOperationSet<Intrinsics::AVX512, float16, 16>
{
    // 16 float16 values = 256 bits = __m256h
    static __m256h load(const Hvec<float16, 16>& v)
    {
        return _mm256_castsi256_ph(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(v.data)));
    }
    static Hvec<float16, 16> store(const __m256h& vr)
    {
        Hvec<float16, 16> result;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data), _mm256_castph_si256(vr));
        return result;
    }

    static Hvec<float16, 16> add(const Hvec<float16, 16>& a, const Hvec<float16, 16>& b)
    { return store(_mm256_add_ph(load(a), load(b))); }
    static Hvec<float16, 16> sub(const Hvec<float16, 16>& a, const Hvec<float16, 16>& b)
    { return store(_mm256_sub_ph(load(a), load(b))); }
    static Hvec<float16, 16> mul(const Hvec<float16, 16>& a, const Hvec<float16, 16>& b)
    { return store(_mm256_mul_ph(load(a), load(b))); }
    static Hvec<float16, 16> div(const Hvec<float16, 16>& a, const Hvec<float16, 16>& b)
    { return store(_mm256_div_ph(load(a), load(b))); }
    static Hvec<float16, 16> fma(const Hvec<float16, 16>& a, const Hvec<float16, 16>& b, const Hvec<float16, 16>& c)
    { return store(_mm256_fmadd_ph(load(a), load(b), load(c))); }
    static float16 hsum(const Hvec<float16, 16>& a)
    { return static_cast<float16>(_mm256_reduce_add_ph(load(a))); }
};
#endif // __AVX512FP16__

// ---- bfloat16 x16: AVX-512 BF16 provides vcvtneps2bf16 for the downcast,
//      but still no native bf16 arithmetic — must go through f32. ----
#if defined(__AVX512BF16__)
template <>
struct vectorOperationSet<Intrinsics::AVX512, bfloat16, 16>
{
    static __m512 load_to_f32(const Hvec<bfloat16, 16>& v)
    {
        __m256i bits16  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v.data));
        __m512i bits32  = _mm512_cvtepu16_epi32(bits16);
        __m512i shifted = _mm512_slli_epi32(bits32, 16);
        return _mm512_castsi512_ps(shifted);
    }
    static Hvec<bfloat16, 16> store_from_f32(const __m512& vr)
    {
        // _mm512_cvtneps_pbh: 16x f32 -> 16x bf16 (round-to-nearest-even)
        __m256i packed = (__m256i)_mm512_cvtneps_pbh(vr);
        Hvec<bfloat16, 16> result;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data), packed);
        return result;
    }

    static Hvec<bfloat16, 16> add(const Hvec<bfloat16, 16>& a, const Hvec<bfloat16, 16>& b)
    { return store_from_f32(_mm512_add_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 16> sub(const Hvec<bfloat16, 16>& a, const Hvec<bfloat16, 16>& b)
    { return store_from_f32(_mm512_sub_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 16> mul(const Hvec<bfloat16, 16>& a, const Hvec<bfloat16, 16>& b)
    { return store_from_f32(_mm512_mul_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 16> div(const Hvec<bfloat16, 16>& a, const Hvec<bfloat16, 16>& b)
    { return store_from_f32(_mm512_div_ps(load_to_f32(a), load_to_f32(b))); }
    static Hvec<bfloat16, 16> fma(const Hvec<bfloat16, 16>& a, const Hvec<bfloat16, 16>& b, const Hvec<bfloat16, 16>& c)
    { return store_from_f32(_mm512_fmadd_ps(load_to_f32(a), load_to_f32(b), load_to_f32(c))); }
    static bfloat16 hsum(const Hvec<bfloat16, 16>& a)
    { return bfloat16(_mm512_reduce_add_ps(load_to_f32(a))); }
};
#endif // __AVX512BF16__

#endif // __AVX512F__

// ============================================================================
// NEON  — float32 x4  (AArch64)
// ============================================================================

#if defined(__NEON__) || defined(__ARM_NEON)
template <>
struct vectorOperationSet<Intrinsics::NEON, float, 4>
{
    static Hvec<float, 4> add(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        float32x4_t va = vld1q_f32(a.data), vb = vld1q_f32(b.data);
        float32x4_t vr = vaddq_f32(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    static Hvec<float, 4> sub(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        float32x4_t va = vld1q_f32(a.data), vb = vld1q_f32(b.data);
        float32x4_t vr = vsubq_f32(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    static Hvec<float, 4> mul(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        float32x4_t va = vld1q_f32(a.data), vb = vld1q_f32(b.data);
        float32x4_t vr = vmulq_f32(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    static Hvec<float, 4> div(const Hvec<float, 4>& a, const Hvec<float, 4>& b)
    {
        float32x4_t va = vld1q_f32(a.data), vb = vld1q_f32(b.data);
        float32x4_t vr = vdivq_f32(va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    // vfmaq_f32(c, a, b) = c + a*b  (fused, AArch64)
    static Hvec<float, 4> fma(const Hvec<float, 4>& a, const Hvec<float, 4>& b, const Hvec<float, 4>& c)
    {
        float32x4_t va = vld1q_f32(a.data), vb = vld1q_f32(b.data), vc = vld1q_f32(c.data);
        float32x4_t vr = vfmaq_f32(vc, va, vb);
        return *(Hvec<float, 4>*)&vr;
    }
    // vaddvq_f32: horizontal add across all 4 lanes (AArch64 only)
    static float hsum(const Hvec<float, 4>& a)
    {
        return vaddvq_f32(vld1q_f32(a.data));
    }
};

// ---- float16 x8: native AArch64 half-precision (__ARM_FP16_FORMAT_IEEE) ----
// Without the feature flag the generic upcast fallback handles this.
#if defined(__ARM_FP16_FORMAT_IEEE)
template <>
struct vectorOperationSet<Intrinsics::NEON, float16, 8>
{
    static float16x8_t load(const Hvec<float16, 8>& v)
    { return vld1q_f16(reinterpret_cast<const __fp16*>(v.data)); }

    static Hvec<float16, 8> store(float16x8_t vr)
    {
        Hvec<float16, 8> result;
        vst1q_f16(reinterpret_cast<__fp16*>(result.data), vr);
        return result;
    }

    static Hvec<float16, 8> add(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store(vaddq_f16(load(a), load(b))); }
    static Hvec<float16, 8> sub(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store(vsubq_f16(load(a), load(b))); }
    static Hvec<float16, 8> mul(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store(vmulq_f16(load(a), load(b))); }
    static Hvec<float16, 8> div(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b)
    { return store(vdivq_f16(load(a), load(b))); }
    static Hvec<float16, 8> fma(const Hvec<float16, 8>& a, const Hvec<float16, 8>& b, const Hvec<float16, 8>& c)
    { return store(vfmaq_f16(load(c), load(a), load(b))); }   // c + a*b
    static float16 hsum(const Hvec<float16, 8>& a)
    { return static_cast<float16>(vaddvq_f16(load(a))); }
};
#endif // __ARM_FP16_FORMAT_IEEE

// ---- bfloat16 x8: ARMv8.6+ (__ARM_FEATURE_BF16) ----
// ARM BF16 provides cvt intrinsics but not elementwise bf16 arithmetic,
// so we upcast to float32x4x2 (two 128-bit registers), operate, then downcast.
#if defined(__ARM_FEATURE_BF16)
template <>
struct vectorOperationSet<Intrinsics::NEON, bfloat16, 8>
{
    static float32x4x2_t upcast(const Hvec<bfloat16, 8>& v)
    {
        bfloat16x8_t vb = vld1q_bf16(reinterpret_cast<const __bf16*>(v.data));
        float32x4x2_t r;
        r.val[0] = vcvt_f32_bf16(vget_low_bf16(vb));
        r.val[1] = vcvt_f32_bf16(vget_high_bf16(vb));
        return r;
    }
    static Hvec<bfloat16, 8> downcast(float32x4x2_t f)
    {
        bfloat16x8_t combined = vcombine_bf16(vcvt_bf16_f32(f.val[0]), vcvt_bf16_f32(f.val[1]));
        Hvec<bfloat16, 8> result;
        vst1q_bf16(reinterpret_cast<__bf16*>(result.data), combined);
        return result;
    }

    static Hvec<bfloat16, 8> add(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    {
        auto va = upcast(a), vb = upcast(b);
        return downcast({ vaddq_f32(va.val[0], vb.val[0]), vaddq_f32(va.val[1], vb.val[1]) });
    }
    static Hvec<bfloat16, 8> sub(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    {
        auto va = upcast(a), vb = upcast(b);
        return downcast({ vsubq_f32(va.val[0], vb.val[0]), vsubq_f32(va.val[1], vb.val[1]) });
    }
    static Hvec<bfloat16, 8> mul(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    {
        auto va = upcast(a), vb = upcast(b);
        return downcast({ vmulq_f32(va.val[0], vb.val[0]), vmulq_f32(va.val[1], vb.val[1]) });
    }
    static Hvec<bfloat16, 8> div(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b)
    {
        auto va = upcast(a), vb = upcast(b);
        return downcast({ vdivq_f32(va.val[0], vb.val[0]), vdivq_f32(va.val[1], vb.val[1]) });
    }
    static Hvec<bfloat16, 8> fma(const Hvec<bfloat16, 8>& a, const Hvec<bfloat16, 8>& b, const Hvec<bfloat16, 8>& c)
    {
        auto va = upcast(a), vb = upcast(b), vc = upcast(c);
        return downcast({ vfmaq_f32(vc.val[0], va.val[0], vb.val[0]),
                          vfmaq_f32(vc.val[1], va.val[1], vb.val[1]) });
    }
    static bfloat16 hsum(const Hvec<bfloat16, 8>& a)
    {
        auto va = upcast(a);
        return bfloat16(vaddvq_f32(va.val[0]) + vaddvq_f32(va.val[1]));
    }
};
#endif // __ARM_FEATURE_BF16

#endif // __NEON__ / __ARM_NEON

// ============================================================================
// OPERATOR OVERLOADS
// A pair of macros generates +, -, *, /, fma(), and hsum() for each type/width.
// The __weak attribute from the original is dropped in favour of plain inline
// to stay portable; if ODR issues arise in your link model, add
// __attribute__((weak)) explicitly.
// ============================================================================

#define HVEC_OPS(T, N)                                                                          \
    inline Hvec<T, N> operator+(const Hvec<T, N>& A, const Hvec<T, N>& B)                      \
    { return vectorOperationSet<get_available_intrinsics(), T, N>::add(A, B); }                 \
    inline Hvec<T, N> operator-(const Hvec<T, N>& A, const Hvec<T, N>& B)                      \
    { return vectorOperationSet<get_available_intrinsics(), T, N>::sub(A, B); }                 \
    inline Hvec<T, N> operator*(const Hvec<T, N>& A, const Hvec<T, N>& B)                      \
    { return vectorOperationSet<get_available_intrinsics(), T, N>::mul(A, B); }                 \
    inline Hvec<T, N> operator/(const Hvec<T, N>& A, const Hvec<T, N>& B)                      \
    { return vectorOperationSet<get_available_intrinsics(), T, N>::div(A, B); }                 \
    inline Hvec<T, N> fma(const Hvec<T, N>& A, const Hvec<T, N>& B, const Hvec<T, N>& C)      \
    { return vectorOperationSet<get_available_intrinsics(), T, N>::fma(A, B, C); }              \
    inline T hsum(const Hvec<T, N>& A)                                                          \
    { return vectorOperationSet<get_available_intrinsics(), T, N>::hsum(A); }

// float32
HVEC_OPS(float,    4)
HVEC_OPS(float,    8)
HVEC_OPS(float,   16)

// float16
HVEC_OPS(float16,  4)
HVEC_OPS(float16,  8)
HVEC_OPS(float16, 16)

// bfloat16
HVEC_OPS(bfloat16,  4)
HVEC_OPS(bfloat16,  8)
HVEC_OPS(bfloat16, 16)

#undef HVEC_OPS

#endif // AVX_HVEC_HPP