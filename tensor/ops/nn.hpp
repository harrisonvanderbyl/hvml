#ifndef OPS_NN_HPP
#define OPS_NN_HPP

//
//  ops/nn.hpp — neural-network building blocks.
//
//  Everything here is an element-wise HardamardOperation or a reduction
//  (DotProduct, ReduceSum) applied to *views*: slices, unsqueeze, transpose,
//  view and tensor_index decide which elements meet, the ops only do the
//  arithmetic.  A matmul, for example, is
//
//      DotProduct<-1>::run(x.unsqueeze(1), w.unsqueeze(0))     // [M,1,K]·[1,N,K] → [M,N]
//
//  and runs on whatever device the tensors are on.
//

#include <cmath>
#include <vector>
#include <cstring>
#include "tensor.hpp"
#include "ops/ops.hpp"

// ===========================================================================
//  Element-wise operations
// ===========================================================================

// GELU (erf form)
struct OpGelu : public HardamardOperation<OpGelu> {
    template <typename A>
    __host__ __device__ static inline float apply(const A& x) {
        float v = float(x);
        return 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    }
};

// silu(gate) * up
struct OpSiluMul : public HardamardOperation<OpSiluMul> {
    __host__ __device__ static inline float apply(const float& gate, const float& up) {
        return gate / (1.0f + expf(-gate)) * up;
    }
};

// a = max(a, b)
struct OpMaxEq : public HardamardOperation<OpMaxEq> {
    __host__ __device__ static inline void apply(float& a, const float& b) {
        if (b > a) a = b;
    }
};

// s = exp(s - m)
struct OpExpSubEq : public HardamardOperation<OpExpSubEq> {
    __host__ __device__ static inline void apply(float& s, const float& m) { s = expf(s - m); }
};

// 1 / sqrt(sum · inv_n + eps)
struct OpRsqrtMean : public HardamardOperation<OpRsqrtMean> {
    __host__ __device__ static inline float apply(const float& sum, const float& inv_n, const float& eps) {
        return 1.0f / sqrtf(sum * inv_n + eps);
    }
};

// (x - mean) · rstd · w + b
struct OpLayerNormApply : public HardamardOperation<OpLayerNormApply> {
    template <typename W, typename B>
    __host__ __device__ static inline float apply(const float& x, const float& mean, const float& rstd, const W& w,
                                                  const B& b) {
        return (x - mean) * rstd * float(w) + float(b);
    }
};

// x · rstd · w
struct OpRMSNormApply : public HardamardOperation<OpRMSNormApply> {
    template <typename W>
    __host__ __device__ static inline float apply(const float& x, const float& rstd, const W& w) {
        return x * rstd * float(w);
    }
};

// Rotates the pair (first half, second half) of a head in place.
struct OpRotateHalf : public HardamardOperation<OpRotateHalf> {
    __host__ __device__ static inline void apply(float& x1, float& x2, const float& c, const float& s) {
        float a = x1, b = x2;
        x1 = a * c - b * s;
        x2 = b * c + a * s;
    }
};

// Causal mask: keys after the query position get -inf.
struct OpCausalMask : public HardamardOperation<OpCausalMask> {
    __host__ __device__ static inline void apply(float& score, const long& q_pos, const long& k_pos) {
        if (k_pos > q_pos) score = -INFINITY;
    }
};

// float cos / sin (OperationCos / OperationSin return double)
struct OpCos : public HardamardOperation<OpCos> {
    __host__ __device__ static inline float apply(const float& a) { return cosf(a); }
};
struct OpSin : public HardamardOperation<OpSin> {
    __host__ __device__ static inline float apply(const float& a) { return sinf(a); }
};

// re² + im²
struct OpPower : public HardamardOperation<OpPower> {
    __host__ __device__ static inline float apply(const float& re, const float& im) { return re * re + im * im; }
};

// log10(max(x, 1e-10))
struct OpLog10Clamp : public HardamardOperation<OpLog10Clamp> {
    __host__ __device__ static inline float apply(const float& x) { return log10f(x > 1e-10f ? x : 1e-10f); }
};

// Whisper normalisation: (max(x, top - 8) + 4) / 4
struct OpWhisperNorm : public HardamardOperation<OpWhisperNorm> {
    __host__ __device__ static inline float apply(const float& x, const float& top) {
        float floor_value = top - 8.0f;
        return ((x > floor_value ? x : floor_value) + 4.0f) / 4.0f;
    }
};

// ===========================================================================
//  Where to put new activations for weights living on `device`.  Weights
//  still mmapped from a checkpoint (kDISK) compute in host memory; new
//  tensors must not be allocated on the disk device, which is backed by that
//  checkpoint file.
// ===========================================================================

inline MemoryLocation working_location(AllocationMap* device) {
    if (device->this_device_type == MemoryType::kDISK) return MemoryLocation(MemoryType::kDDR);
    return MemoryLocation(*device);
}

// ===========================================================================
//  Host transfers
// ===========================================================================

// Copy host data into a new tensor on `loc`.
template <typename T, int R>
inline Tensor<T, R> tensor_from_host(Shape<R> shape, const T* src, MemoryLocation loc) {
    Tensor<T, R> host(shape, MemoryLocation(MemoryType::kDDR), ComputeType::kCPU);
    memcpy((void*)host.data.data, (const void*)src, shape.total_size() * sizeof(T));
    if (loc.memory_type == MemoryType::kDDR) return host;
    return host.to(loc);
}

// Copy a tensor (any device, any strides) to a host vector.
template <typename T, int R>
inline std::vector<T> tensor_to_host(const Tensor<T, R>& t) {
    Tensor<T, R> h = t.contiguous().to(MemoryLocation(MemoryType::kDDR), ComputeType::kCPU);
    h.device->synchronize_function();
    std::vector<T> out(t.shape.total_size());
    memcpy((void*)out.data(), (const void*)h.data.data, out.size() * sizeof(T));
    return out;
}

// ===========================================================================
//  Composite functions
// ===========================================================================

// x [M, K] · w [N, K]ᵀ → [M, N]
template <typename W>
inline Tensor<float, 2> linear(const Tensor<float, 2>& x, const Tensor<W, 2>& w) {
    return DotProduct<-1>::run(x.unsqueeze(1), w.unsqueeze(0));
}

template <int R>
inline Tensor<float, R> gelu(const Tensor<float, R>& x) {
    return OpGelu::run(x);
}

// Largest value of each row of x [M, S] → [M, 1], by folding the upper half
// of the row onto the lower half until one column is left.
inline Tensor<float, 2> row_max(const Tensor<float, 2>& x) {
    Tensor<float, 2> m = x.contiguous();
    long n = m.shape[1];
    while (n > 1) {
        long h = n / 2;
        OpMaxEq::run(m[{{}, {0, h}}], m[{{}, {n - h, n}}]);
        n -= h;
    }
    return m[{{}, {0, 1}}];
}

// Softmax over each row of x [M, S], in place.
inline void softmax_rows(Tensor<float, 2> x) {
    OpExpSubEq::run(x, row_max(x));
    x /= ReduceSum<-1>::run(x).unsqueeze(1);
}

// RMSNorm over rows: x [M, K], w [K]
template <typename W>
inline Tensor<float, 2> rms_norm(const Tensor<float, 2>& x, const Tensor<W, 1>& w, float eps) {
    auto rstd = OpRsqrtMean::run(DotProduct<-1>::run(x, x), 1.0f / x.shape[1], eps);
    return OpRMSNormApply::run(x, rstd.unsqueeze(1), w.unsqueeze(0));
}

// LayerNorm over rows: x [M, K], w and b [K]
template <typename W, typename B>
inline Tensor<float, 2> layer_norm(const Tensor<float, 2>& x, const Tensor<W, 1>& w, const Tensor<B, 1>& b,
                                   float eps) {
    float inv_k = 1.0f / x.shape[1];
    auto mean = (ReduceSum<-1>::run(x) * inv_k).unsqueeze(1);
    auto centered = x - mean;
    auto rstd = OpRsqrtMean::run(DotProduct<-1>::run(centered, centered), inv_k, eps);
    return OpLayerNormApply::run(x, mean, rstd.unsqueeze(1), w.unsqueeze(0), b.unsqueeze(0));
}

// cos / sin tables for rotary embeddings at `positions`: [T, dim/2] each
struct RopeTables {
    Tensor<float, 2> cos, sin;
};

inline RopeTables rope_tables(const std::vector<long>& positions, long dim, float theta, MemoryLocation loc) {
    long half = dim / 2;
    std::vector<float> inv_freq(half), pos(positions.begin(), positions.end());
    for (long i = 0; i < half; i++) inv_freq[i] = 1.0f / powf(theta, (float)(2 * i) / (float)dim);
    auto p = tensor_from_host(Shape<1>{(long)pos.size()}, pos.data(), loc);
    auto f = tensor_from_host(Shape<1>{half}, inv_freq.data(), loc);
    Tensor<float, 2> angles = p.unsqueeze(1) * f.unsqueeze(0);
    return {OpCos::run(angles), OpSin::run(angles)};
}

// Rotary embedding (rotate-half) of x [T, H, D], in place.
inline void rope(const Tensor<float, 3>& x, const RopeTables& t) {
    long half = x.shape[2] / 2;
    OpRotateHalf::run(x[{{}, {}, {0, half}}], x[{{}, {}, {half, 2 * half}}], t.cos.unsqueeze(1), t.sin.unsqueeze(1));
}

// Grouped-query attention over one sequence.
//   q [T, Hq, D] (contiguous), k and v [Hk, S, D] (any strides)
//   causal: query t is at position q_offset + t and sees keys 0 .. q_offset + t
//   → [T, Hq * D]
inline Tensor<float, 2> attention(Tensor<float, 3> q, const Tensor<float, 3>& k, const Tensor<float, 3>& v,
                                  float scale, long q_offset, bool causal) {
    long T = q.shape[0], Hq = q.shape[1], D = q.shape[2];
    long Hk = k.shape[0], S = k.shape[1], G = Hq / Hk;
    MemoryLocation loc(*q.device);

    // scores[hk, g, t, s] = q[t, hk·G + g] · k[hk, s]
    Tensor<float, 4> qg = q.view(Shape<4>{T, Hk, G, D});
    Tensor<float, 4> scores = DotProduct<-1>::run(qg.transpose(0, 1).transpose(1, 2).unsqueeze(3),   // [Hk,G,T,1,D]
                                                  k.unsqueeze(1).unsqueeze(2));                          // [Hk,1,1,S,D]
    scores *= scale;

    if (causal) {
        std::vector<long> qp(T), kp(S);
        for (long t = 0; t < T; t++) qp[t] = q_offset + t;
        for (long s = 0; s < S; s++) kp[s] = s;
        auto qpos = tensor_from_host(Shape<4>{1, 1, T, 1}, qp.data(), loc);
        auto kpos = tensor_from_host(Shape<4>{1, 1, 1, S}, kp.data(), loc);
        OpCausalMask::run(scores, qpos, kpos);
    }
    softmax_rows(scores.view(Shape<2>{Hk * G * T, S}));

    // out[t, hk, g, d] = Σ_s p[hk, g, t, s] · v[hk, s, d]
    Tensor<float, 4> out = DotProduct<-1>::run(scores.transpose(1, 2).transpose(0, 1).unsqueeze(3),   // [T,Hk,G,1,S]
                                               v.transpose(1, 2).unsqueeze(0).unsqueeze(2));            // [1,Hk,1,D,S]
    return out.view(Shape<2>{T, Hq * D});
}

#endif // OPS_NN_HPP
