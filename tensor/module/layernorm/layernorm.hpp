#ifndef MODULE_LAYERNORM_HPP
#define MODULE_LAYERNORM_HPP
#include "module/base/module.hpp"
#include "ops/nn.hpp"

//  LayerNorm over the last dimension:  (x - mean) / sqrt(var + eps) * w + b

template <typename W = float>
struct LayerNorm : public Module<Tensor<W, 1>, Tensor<W, 1>>
{
    Tensor<W, 1> weight;
    Tensor<W, 1> bias;
    float eps = 1e-5f;

    LayerNorm(float eps = 1e-5f) : Module<Tensor<W, 1>, Tensor<W, 1>>({weight, "weight"}, {bias, "bias"}), eps(eps) {}

    LayerNorm(size_t features, float eps = 1e-5f, MemoryLocation loc = MemoryType::kDDR)
        : Module<Tensor<W, 1>, Tensor<W, 1>>({weight, "weight"}, {bias, "bias"}),
          weight(Shape<1>{(long)features}, loc), bias(Shape<1>{(long)features}, loc), eps(eps) {}

    // x [rows, features]
    Tensor<float, 2> forward(const Tensor<float, 2>& x) const { return layer_norm(x, weight, bias, eps); }
    Tensor<float, 2> operator()(const Tensor<float, 2>& x) const { return forward(x); }
};

#endif //MODULE_LAYERNORM_HPP
