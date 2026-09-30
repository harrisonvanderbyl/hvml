// vulkan_ops_test.cpp — runs hvml operations on a Vulkan device and checks
// them against the CPU.
//
//   ./vulkcc examples/vulkan_ops_test.cpp -o vulkan_ops_test -I./tensor -std=c++20 -O2
//   DEVICE_PLUGIN_DIR=tensor/device/plugins ./vulkan_ops_test
//
// Every operation runs the kernels in ops/vulkan/ops.vk, compiled by vulkcc.

#include "ops/nn.hpp"
#include "module/conv/conv2d.hpp"
#include "bfloat16/bf16.hpp"
#include <cstdio>
#include <random>

static MemoryLocation cpu() { return MemoryLocation(MemoryType::kDDR); }
static MemoryLocation gpu() { return MemoryLocation(MemoryType::kUnknown_MEM, 0); }   // the plugin's Vulkan device

static int failures = 0;

template <typename T, int R>
static void check(const char* name, const Tensor<T, R>& vk, const Tensor<T, R>& ref, double tol = 1e-5) {
    auto a = tensor_to_host(vk);
    auto b = tensor_to_host(ref);
    double worst = 0, scale = 0;
    for (size_t i = 0; i < b.size(); i++) {
        worst = std::max(worst, std::fabs((double)a[i] - (double)b[i]));
        scale = std::max(scale, std::fabs((double)b[i]));
    }
    bool ok = a.size() == b.size() && worst <= tol * std::max(1.0, scale);
    if (!ok) failures++;
    printf("%s %-28s %6zu values  max|diff| %.3g\n", ok ? "[PASS]" : "[FAIL]", name, b.size(), worst);
}

template <typename T, int R>
static Tensor<T, R> random(Shape<R> shape, MemoryLocation loc, int seed, float lo = -1, float hi = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<T> v(shape.total_size());
    for (auto& x : v) x = T(dist(rng));
    return tensor_from_host(shape, v.data(), loc);
}

template <typename T, int R>
static Tensor<T, R> on(const Tensor<T, R>& t, MemoryLocation loc) {
    return tensor_from_host(t.shape, tensor_to_host(t).data(), loc);
}

int main() {
    auto x_c = random<float, 2>(Shape<2>{37, 64}, cpu(), 1);
    auto y_c = random<float, 2>(Shape<2>{37, 64}, cpu(), 2);
    auto b_c = random<float, 1>(Shape<1>{64}, cpu(), 3);
    auto x = on(x_c, gpu());
    auto y = on(y_c, gpu());
    auto b = on(b_c, gpu());

    // element-wise, broadcast, scalars, in place
    { Tensor<float, 2> r = x + y; Tensor<float, 2> rc = x_c + y_c; check("x + y", r, rc); }
    { Tensor<float, 2> r = x * b.unsqueeze(0); Tensor<float, 2> rc = x_c * b_c.unsqueeze(0); check("broadcast x * b", r, rc); }
    { Tensor<float, 2> r = x.contiguous(); Tensor<float, 2> rc = x_c.contiguous(); r *= 2.5f; rc *= 2.5f; check("in place *= scalar", r, rc); }
    { Tensor<float, 2> r = OpGelu::run(x); Tensor<float, 2> rc = OpGelu::run(x_c); check("gelu (erf)", r, rc, 1e-5); }
    { Tensor<float, 2> r = ChainOperations<OperationAdd, OpGelu>::run(x, b.unsqueeze(0));
      Tensor<float, 2> rc = ChainOperations<OperationAdd, OpGelu>::run(x_c, b_c.unsqueeze(0));
      check("ChainOperations<Add, Gelu>", r, rc); }
    { Tensor<float, 2> r = OpSiluMul::run(x, y); Tensor<float, 2> rc = OpSiluMul::run(x_c, y_c); check("silu(x) * y", r, rc); }

    // slices and transposes (views with offsets and strides)
    { Tensor<float, 2> r = x[{{3, 20}, {5, 40, 3}}]; Tensor<float, 2> rc = x_c[{{3, 20}, {5, 40, 3}}];
      check("strided slice copy", r.contiguous(), rc.contiguous()); }
    { Tensor<float, 2> r = x.transpose(0, 1).contiguous(); Tensor<float, 2> rc = x_c.transpose(0, 1).contiguous();
      check("transpose copy", r, rc); }
    { Tensor<float, 2> r = x.contiguous(); Tensor<float, 2> rc = x_c.contiguous();
      r[{{2, 5}}] = y[{{10, 13}}]; rc[{{2, 5}}] = y_c[{{10, 13}}];
      check("assign into slice", r, rc); }

    // reductions
    { Tensor<float, 1> r = ReduceSum<-1>::run(x); Tensor<float, 1> rc = ReduceSum<-1>::run(x_c); check("ReduceSum rows", r, rc); }
    { Tensor<float, 1> r = ReduceMax<-1>::run(x); Tensor<float, 1> rc = ReduceMax<-1>::run(x_c); check("ReduceMax rows", r, rc); }
    { Tensor<float, 1> r = ReduceSum<0>::run(x); Tensor<float, 1> rc = ReduceSum<0>::run(x_c); check("ReduceSum columns", r, rc); }
    { Tensor<float, 2> r = x.contiguous(); Tensor<float, 2> rc = x_c.contiguous();
      softmax_rows(r, 0.7f); softmax_rows(rc, 0.7f); check("softmax_rows", r, rc); }

    // matmul with bfloat16 weights
    auto w_c = random<bfloat16, 2>(Shape<2>{48, 64}, cpu(), 4);
    auto w = on(w_c, gpu());
    { Tensor<float, 2> r = linear(x, w); Tensor<float, 2> rc = linear(x_c, w_c); check("linear, bf16 weights", r, rc); }

    // norms
    auto g_c = random<bfloat16, 1>(Shape<1>{64}, cpu(), 5, 0.5f, 1.5f);
    auto g = on(g_c, gpu());
    { Tensor<float, 2> r = rms_norm(x, g, 1e-6f); Tensor<float, 2> rc = rms_norm(x_c, g_c, 1e-6f); check("rms_norm", r, rc); }
    { Tensor<float, 2> r = layer_norm(x, g, g, 1e-5f); Tensor<float, 2> rc = layer_norm(x_c, g_c, g_c, 1e-5f); check("layer_norm", r, rc); }

    // tensor_index (embedding rows) + bf16 → float conversion
    {
        std::vector<long> ids = {5, 0, 47, 12, 12};
        auto i_c = tensor_from_host(Shape<1>{5}, ids.data(), cpu());
        auto i_g = tensor_from_host(Shape<1>{5}, ids.data(), gpu());
        Tensor<float, 2> r(Shape<2>{5, 64}, gpu()); r = w.tensor_index(i_g);
        Tensor<float, 2> rc(Shape<2>{5, 64}, cpu()); rc = w_c.tensor_index(i_c);
        check("tensor_index rows", r, rc);
    }

    // rotary embedding (in-place op on two slices of one tensor)
    {
        auto q_c = random<float, 3>(Shape<3>{7, 4, 16}, cpu(), 6);
        auto q = on(q_c, gpu());
        std::vector<long> pos = {0, 1, 2, 3, 4, 5, 6};
        rope(q, rope_tables(pos, 16, 10000.0f, gpu()));
        rope(q_c, rope_tables(pos, 16, 10000.0f, cpu()));
        check("rope", q, q_c);
    }

    // attention (GQA, causal) — transposed views into DotProduct
    {
        auto q_c = random<float, 3>(Shape<3>{6, 4, 16}, cpu(), 7);
        auto k_c = random<float, 3>(Shape<3>{2, 9, 16}, cpu(), 8);
        auto v_c = random<float, 3>(Shape<3>{2, 9, 16}, cpu(), 9);
        auto q = on(q_c, gpu());
        auto k = on(k_c, gpu());
        auto v = on(v_c, gpu());
        check("attention (GQA, causal)", attention(q, k, v, 0.25f, 3, true), attention(q_c, k_c, v_c, 0.25f, 3, true));
    }

    // convolution: im2col from strided slices + DotProduct
    {
        Conv2d<float> conv_c(2, 1), conv(2, 1);
        conv_c.weight = random<float, 4>(Shape<4>{8, 3, 3, 3}, cpu(), 10);
        conv_c.bias = random<float, 1>(Shape<1>{8}, cpu(), 11);
        conv.weight = on(conv_c.weight, gpu());
        conv.bias = on(conv_c.bias, gpu());
        auto img_c = random<float, 4>(Shape<4>{2, 3, 12, 10}, cpu(), 12);
        auto img = on(img_c, gpu());
        check("conv2d stride 2 + gelu", conv.forward(img, true), conv_c.forward(img_c, true), 2e-5);
    }

    printf("%s\n", failures ? "SOME CHECKS FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
