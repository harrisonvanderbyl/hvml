#ifndef MODULE_RMSNORM_HPP
#define MODULE_RMSNORM_HPP
#include "module/base/module.hpp"
#include "ops/nn.hpp"

//  RMSNorm over the last dimension:  x / sqrt(mean(x²) + eps) * w

template <typename W = float>
struct RMSNorm : public Module<Tensor<W, 1>>
{
    Tensor<W, 1> weight;
    float eps = 1e-6f;

    RMSNorm(float eps = 1e-6f) : Module<Tensor<W, 1>>({weight, "weight"}), eps(eps) {}

    RMSNorm(size_t features, float eps = 1e-6f, MemoryLocation loc = MemoryType::kDDR)
        : Module<Tensor<W, 1>>({weight, "weight"}), weight(Shape<1>{(long)features}, loc), eps(eps) {}

    // x [rows, features]
    Tensor<float, 2> forward(const Tensor<float, 2>& x) const { return rms_norm(x, weight, eps); }
    Tensor<float, 2> operator()(const Tensor<float, 2>& x) const { return forward(x); }
};

#endif //MODULE_RMSNORM_HPP
