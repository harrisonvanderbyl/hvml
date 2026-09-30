#ifndef MODULE_LINEAR_HPP
#define MODULE_LINEAR_HPP
#include "module/base/module.hpp"
#include "ops/nn.hpp"

//  y = x · Wᵀ (+ b).   Weight [out, in] in the checkpoint's dtype (e.g.
//  bfloat16); activations are float.  Runs on the device the weights are on.

template <typename W = float, bool HasBias = false>
struct Linear;

template <typename W>
struct Linear<W, false> : public Module<Tensor<W, 2>>
{
    Tensor<W, 2> weight;

    Linear() : Module<Tensor<W, 2>>({weight, "weight"}) {}

    Linear(size_t in_features, size_t out_features, MemoryLocation loc = MemoryType::kDDR)
        : Module<Tensor<W, 2>>({weight, "weight"}), weight(Shape<2>{(long)out_features, (long)in_features}, loc) {}

    // x [rows, in] → [rows, out], optionally followed by GELU
    Tensor<float, 2> forward(const Tensor<float, 2>& x, bool with_gelu = false) const {
        Tensor<float, 2> y = linear(x, weight);
        return with_gelu ? gelu(y) : y;
    }
    Tensor<float, 2> operator()(const Tensor<float, 2>& x) const { return forward(x); }
};

template <typename W>
struct Linear<W, true> : public Module<Tensor<W, 2>, Tensor<W, 1>>
{
    Tensor<W, 2> weight;
    Tensor<W, 1> bias;

    Linear() : Module<Tensor<W, 2>, Tensor<W, 1>>({weight, "weight"}, {bias, "bias"}) {}

    Linear(size_t in_features, size_t out_features, MemoryLocation loc = MemoryType::kDDR)
        : Module<Tensor<W, 2>, Tensor<W, 1>>({weight, "weight"}, {bias, "bias"}),
          weight(Shape<2>{(long)out_features, (long)in_features}, loc),
          bias(Shape<1>{(long)out_features}, loc) {}

    // x [rows, in] → [rows, out], optionally followed by GELU (fused with
    // the bias add into one kernel)
    Tensor<float, 2> forward(const Tensor<float, 2>& x, bool with_gelu = false) const {
        Tensor<float, 2> y = linear(x, weight);
        if (with_gelu) return ChainOperations<OperationAdd, OpGelu>::run(y, bias.unsqueeze(0));
        y += bias.unsqueeze(0);
        return y;
    }
    Tensor<float, 2> operator()(const Tensor<float, 2>& x) const { return forward(x); }
};

#endif //MODULE_LINEAR_HPP
