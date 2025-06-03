#ifndef MODULE_LINEAR_HPP
#define MODULE_LINEAR_HPP
#include "module/base/module.hpp"
#include "ops/matmul.hpp"

template <typename T = float>
struct Linear : public Module<Tensor<T, 2>, Tensor<T, 1>>
{
    public:
    Tensor<T, 2> weight;
    Tensor<T, 1> bias;
    size_t in_features;
    size_t out_features;
    bool use_bias;
    
    Linear(
        size_t in_features,
        size_t out_features,
        bool use_bias = true,
        DeviceType device_type = DeviceType::kCPU
    ) : use_bias(use_bias), in_features(in_features), out_features(out_features),
        weight({in_features, out_features}, device_type),
        bias({out_features}, device_type),
        Module<Tensor<T, 2>, Tensor<T, 1>>(Submodule<Tensor<T, 2>>(weight, "weight"), 
                                           use_bias ? Submodule<Tensor<T, 1>>(bias, "bias") : Submodule<Tensor<T, 1>>(bias, ""))
    {
        // Initialize weights with Xavier/Glorot initialization
        T scale = std::sqrt(static_cast<T>(2.0) / (in_features + out_features));
        for (size_t i = 0; i < weight.total_size; i++) {
            weight.flatget(i) = scale * (static_cast<T>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        }
        
        // Initialize bias to zero
        if (use_bias) {
            for (size_t i = 0; i < out_features; i++) {
                this->bias.flatget(i) = static_cast<T>(0.0);
            }
        }
    }
    
    // Forward pass for 2D input (batch_size, in_features)
    Tensor<T, 2> forward(const Tensor<T, 2>& input) {
        assert(input.shape[1] == in_features);
        
        // Compute input @ weight
        auto output = matmul(input, weight);
        
        // Add bias if enabled
        if (use_bias) {
            for (size_t b = 0; b < output.shape[0]; b++) {
                for (size_t f = 0; f < out_features; f++) {
                    output[SliceList<1>(b)][SliceList<1>(f)] += bias.flatget(f);
                }
            }
        }
        
        return output;
    }
    
    // Forward pass for 3D input (batch_size, seq_len, in_features)
    Tensor<T, 3> forward(const Tensor<T, 3>& input) {
        assert(input.shape[2] == in_features);
        
        size_t batch_size = input.shape[0];
        size_t seq_len = input.shape[1];
        
        Shape<3> output_shape(batch_size, seq_len, out_features);
        Tensor<T, 3> output(output_shape, input.device_type);
        
        // Process each batch and sequence position
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t out_f = 0; out_f < out_features; out_f++) {
                    T sum = static_cast<T>(0);
                    for (size_t in_f = 0; in_f < in_features; in_f++) {
                        sum += input[SliceList<1>(b)][SliceList<1>(s)][SliceList<1>(in_f)] * 
                               weight[SliceList<1>(in_f)][SliceList<1>(out_f)];
                    }
                    if (use_bias) {
                        sum += bias.flatget(out_f);
                    }
                    output[SliceList<1>(b)][SliceList<1>(s)][SliceList<1>(out_f)] = sum;
                }
            }
        }
        
        return output;
    }
};

template <typename T = float>
struct FFN : public Module<Linear<T>,Linear<T>>
{
    public:
    Linear<T> key;
    Linear<T> value;
    
    FFN(
        size_t n_dim,
        size_t ffn_dim,
        DeviceType device_type = DeviceType::kCPU
    ):key(n_dim, ffn_dim, device_type), value(ffn_dim, n_dim, device_type), Module<Linear<T>,Linear<T>>(Submodule<Linear<T>>(key, "key"), Submodule<Linear<T>>(value, "value"))
    {
        
    }
     
    Tensor <T> forward(Tensor<T> input)
    {
        return value(key(input));
    }
    
};

#endif //MODULE_LINEAR_HPP
