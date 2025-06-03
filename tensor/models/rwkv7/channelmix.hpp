#ifndef RWKV7_CHANNELMIX_HPP
#define RWKV7_CHANNELMIX_HPP

#include "module/base/module.hpp"
#include "module/linear/linear.hpp"
#include "models/rwkv7/timeshift/kernels.hpp"
#include "ops/activations.hpp"

template <typename T = float>
struct RWKV_ChannelMix : public Module<Tensor<T, 3>, Linear<T>, Linear<T>>
{
    public:
    Tensor<T, 3> x_k;  // [1, 1, n_embd]
    Linear<T> key;
    Linear<T> value;
    Tensor<T, 2> shift_state;
    bool lastlayer;
    
    RWKV_ChannelMix(
        size_t layer_id,
        size_t n_layer,
        size_t n_embd,
        size_t dim_ffn,
        DeviceType device_type = DeviceType::kCPU
    ) : x_k({1, 1, n_embd}, device_type),
        key(n_embd, dim_ffn, false, device_type),  // no bias
        value(dim_ffn, n_embd, false, device_type), // no bias
        shift_state({1, n_embd}, device_type),
        lastlayer(layer_id == n_layer - 1),
        Module<Tensor<T, 3>, Linear<T>, Linear<T>>(Submodule<Tensor<T, 3>>(x_k, "x_k"), Submodule<Linear<T>>(key, "key"), Submodule<Linear<T>>(value, "value"))
    {
        // Initialize x_k to zeros
        for (size_t i = 0; i < x_k.total_size; i++) {
            x_k.flatget(i) = static_cast<T>(0.0);
        }
    }
    
    Tensor<T, 3> forward(const Tensor<T, 3>& x) {
        size_t B = x.shape[0];
        size_t seq_len = x.shape[1];
        size_t C = x.shape[2];
        
        // Perform time shift manually
        Tensor<T, 3> xx({B, seq_len, C}, x.device_type);
        
        // Expand shift state for batch size if needed
        if (shift_state.shape[0] != B) {
            Shape<2> new_shift_shape(B, C);
            Tensor<T, 2> new_shift_state(new_shift_shape, x.device_type);
            
            // Broadcast the single batch state to all batches
            for (size_t b = 0; b < B; b++) {
                for (size_t c = 0; c < C; c++) {
                    new_shift_state[SliceList<1>(b)][SliceList<1>(c)] = 
                        shift_state[SliceList<1>(0)][SliceList<1>(c)];
                }
            }
            shift_state = new_shift_state;
        }
        
        // Apply time shift
        for (size_t b = 0; b < B; b++) {
            for (size_t t = 0; t < seq_len; t++) {
                for (size_t c = 0; c < C; c++) {
                    if (t == 0) {
                        xx[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(c)] = 
                            shift_state[SliceList<1>(b)][SliceList<1>(c)];
                    } else {
                        xx[SliceList<1>(b)][SliceList<1>(t)][SliceList<1>(c)] = 
                            x[SliceList<1>(b)][SliceList<1>(t-1)][SliceList<1>(c)];
                    }
                }
            }
            // Update shift state with last token
            for (size_t c = 0; c < C; c++) {
                shift_state[SliceList<1>(b)][SliceList<1>(c)] = 
                    x[SliceList<1>(b)][SliceList<1>(seq_len-1)][SliceList<1>(c)];
            }
        }
        
        // Compute k = x + xx * x_k
        auto k = x + (xx * x_k);
        
        // Apply key transformation and ReLU^2
        auto k_transformed = key.forward(k);
        auto k_relu = relu(k_transformed);
        auto k_squared = power(k_relu, static_cast<T>(2.0));
        
        // Apply value transformation
        auto output = value.forward(k_squared);
        
        return output;
    }
};

#endif // RWKV7_CHANNELMIX_HPP
