#ifndef RWKV7_BLOCK_HPP
#define RWKV7_BLOCK_HPP

#include "module/base/module.hpp"
#include "module/normalization/normalization.hpp"
#include "models/rwkv7/timemix.hpp"
#include "models/rwkv7/channelmix.hpp"

template <typename T = float>
struct RWKV7_Block : public Module<LayerNorm<T>, LayerNorm<T>, RWKV_TimeMix<T>, RWKV_ChannelMix<T>>
{
    public:
    size_t layer_id;
    LayerNorm<T> ln1;
    LayerNorm<T> ln2;
    RWKV_TimeMix<T> att;
    RWKV_ChannelMix<T> ffn;
    
    RWKV7_Block(
        size_t layer_id,
        size_t n_layer,
        size_t n_embd,
        size_t n_head,
        size_t head_size,
        size_t dim_att,
        size_t dim_ffn,
        size_t decay_lora,
        size_t aaa_lora,
        size_t mv_lora,
        size_t gate_lora,
        DeviceType device_type = DeviceType::kCPU
    ) : layer_id(layer_id),
        ln1(n_embd, 1e-5, device_type),
        ln2(n_embd, 1e-5, device_type),
        att(layer_id, n_layer, n_embd, n_head, head_size, dim_att, 
            decay_lora, aaa_lora, mv_lora, gate_lora, device_type),
        ffn(layer_id, n_layer, n_embd, dim_ffn, device_type),
        Module<LayerNorm<T>, LayerNorm<T>, RWKV_TimeMix<T>, RWKV_ChannelMix<T>>(
            Submodule<LayerNorm<T>>(ln1, "ln1"), Submodule<LayerNorm<T>>(ln2, "ln2"), 
            Submodule<RWKV_TimeMix<T>>(att, "att"), Submodule<RWKV_ChannelMix<T>>(ffn, "ffn")
        )
    {
    }
    
    std::pair<Tensor<T, 3>, Tensor<T, 3>> forward(const std::pair<Tensor<T, 3>, Tensor<T, 3>>& input) {
        auto x = input.first;
        auto v_first = input.second;
        
        // Attention block with residual connection
        auto x_norm1 = ln1.forward(x);
        auto [xa, v_first_out] = att.forward(x_norm1, v_first);
        x = x + xa;
        
        // Feed-forward block with residual connection
        auto x_norm2 = ln2.forward(x);
        auto xf = ffn.forward(x_norm2);
        x = x + xf;
        
        return std::make_pair(x, v_first_out);
    }
};

#endif // RWKV7_BLOCK_HPP
