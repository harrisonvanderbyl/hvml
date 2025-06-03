#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP

#include "module/base/module.hpp"
#include <cmath>

template <typename T = float>
struct LayerNorm : public Module<Tensor<T, 1>, Tensor<T, 1>>
{
    public:
    Tensor<T, 1> weight;
    Tensor<T, 1> bias;
    T eps;
    size_t normalized_shape;
    
    LayerNorm(
        size_t normalized_shape,
        T eps = 1e-5,
        DeviceType device_type = DeviceType::kCPU
    ) : normalized_shape(normalized_shape), eps(eps),
        weight({normalized_shape}, device_type),
        bias({normalized_shape}, device_type),
        Module<Tensor<T, 1>, Tensor<T, 1>>(Submodule<Tensor<T, 1>>(weight, "weight"), Submodule<Tensor<T, 1>>(bias, "bias"))
    {
        // Initialize weight to 1 and bias to 0
        for (size_t i = 0; i < normalized_shape; i++) {
            weight.flatget(i) = static_cast<T>(1.0);
            bias.flatget(i) = static_cast<T>(0.0);
        }
    }
    
    template <int D>
    Tensor<T, D> forward(const Tensor<T, D>& input) {
        Tensor<T, D> output(input.shape, input.device_type);
        
        // Normalize along the last dimension
        size_t outer_size = input.total_size / normalized_shape;
        
        for (size_t i = 0; i < outer_size; i++) {
            size_t offset = i * normalized_shape;
            
            // Compute mean
            T mean = static_cast<T>(0);
            for (size_t j = 0; j < normalized_shape; j++) {
                mean += input.flatget(offset + j);
            }
            mean /= static_cast<T>(normalized_shape);
            
            // Compute variance
            T variance = static_cast<T>(0);
            for (size_t j = 0; j < normalized_shape; j++) {
                T diff = input.flatget(offset + j) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<T>(normalized_shape);
            
            // Normalize and apply affine transformation
            T inv_std = static_cast<T>(1.0) / std::sqrt(variance + eps);
            for (size_t j = 0; j < normalized_shape; j++) {
                T normalized = (input.flatget(offset + j) - mean) * inv_std;
                output.flatget(offset + j) = normalized * weight.flatget(j) + bias.flatget(j);
            }
        }
        
        return output;
    }
};

template <typename T = float>
struct GroupNorm : public Module<Tensor<T, 1>, Tensor<T, 1>>
{
    public:
    Tensor<T, 1> weight;
    Tensor<T, 1> bias;
    T eps;
    size_t num_groups;
    size_t num_channels;
    
    GroupNorm(
        size_t num_groups,
        size_t num_channels,
        T eps = 1e-5,
        DeviceType device_type = DeviceType::kCPU
    ) : num_groups(num_groups), num_channels(num_channels), eps(eps),
        weight({num_channels}, device_type),
        bias({num_channels}, device_type),
        Module<Tensor<T, 1>, Tensor<T, 1>>(Submodule<Tensor<T, 1>>(weight, "weight"), Submodule<Tensor<T, 1>>(bias, "bias"))
    {
        assert(num_channels % num_groups == 0);
        
        // Initialize weight to 1 and bias to 0
        for (size_t i = 0; i < num_channels; i++) {
            weight.flatget(i) = static_cast<T>(1.0);
            bias.flatget(i) = static_cast<T>(0.0);
        }
    }
    
    template <int D>
    Tensor<T, D> forward(const Tensor<T, D>& input) {
        static_assert(D >= 2, "GroupNorm requires at least 2D input (batch, channels, ...)");
        
        Tensor<T, D> output(input.shape, input.device_type);
        
        size_t batch_size = input.shape[0];
        size_t channels_per_group = num_channels / num_groups;
        
        // Calculate spatial dimensions
        size_t spatial_size = 1;
        for (int i = 2; i < D; i++) {
            spatial_size *= input.shape[i];
        }
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t g = 0; g < num_groups; g++) {
                // Compute mean and variance for this group
                T mean = static_cast<T>(0);
                size_t group_size = channels_per_group * spatial_size;
                
                for (size_t c = 0; c < channels_per_group; c++) {
                    size_t channel_idx = g * channels_per_group + c;
                    for (size_t s = 0; s < spatial_size; s++) {
                        size_t idx = b * num_channels * spatial_size + 
                                   channel_idx * spatial_size + s;
                        mean += input.flatget(idx);
                    }
                }
                mean /= static_cast<T>(group_size);
                
                T variance = static_cast<T>(0);
                for (size_t c = 0; c < channels_per_group; c++) {
                    size_t channel_idx = g * channels_per_group + c;
                    for (size_t s = 0; s < spatial_size; s++) {
                        size_t idx = b * num_channels * spatial_size + 
                                   channel_idx * spatial_size + s;
                        T diff = input.flatget(idx) - mean;
                        variance += diff * diff;
                    }
                }
                variance /= static_cast<T>(group_size);
                
                // Normalize and apply affine transformation
                T inv_std = static_cast<T>(1.0) / std::sqrt(variance + eps);
                for (size_t c = 0; c < channels_per_group; c++) {
                    size_t channel_idx = g * channels_per_group + c;
                    for (size_t s = 0; s < spatial_size; s++) {
                        size_t idx = b * num_channels * spatial_size + 
                                   channel_idx * spatial_size + s;
                        T normalized = (input.flatget(idx) - mean) * inv_std;
                        output.flatget(idx) = normalized * weight.flatget(channel_idx) + 
                                            bias.flatget(channel_idx);
                    }
                }
            }
        }
        
        return output;
    }
};

#endif // NORMALIZATION_HPP
