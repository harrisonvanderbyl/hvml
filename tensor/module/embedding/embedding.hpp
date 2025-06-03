#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "module/base/module.hpp"
#include <cassert>

template <typename T = float>
struct Embedding : public Module<Tensor<T, 2>>
{
    public:
    Tensor<T, 2> weight;
    size_t num_embeddings;
    size_t embedding_dim;
    
    Embedding(
        size_t num_embeddings,
        size_t embedding_dim,
        DeviceType device_type = DeviceType::kCPU
    ) : num_embeddings(num_embeddings), embedding_dim(embedding_dim),
        weight({num_embeddings, embedding_dim}, device_type),
        Module<Tensor<T, 2>>(Submodule<Tensor<T, 2>>(weight, "weight"))
    {
        // Initialize with small random values (simplified initialization)
        T scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(embedding_dim));
        for (size_t i = 0; i < weight.total_size; i++) {
            // Simple pseudo-random initialization
            weight.flatget(i) = scale * (static_cast<T>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        }
    }
    
    // Forward pass for 1D input (sequence of token indices)
    template <int D>
    Tensor<T, D+1> forward(const Tensor<int, D>& input) {
        // Create output shape: input_shape + [embedding_dim]
        Shape<D+1> output_shape;
        for (int i = 0; i < D; i++) {
            output_shape[i] = input.shape[i];
        }
        output_shape[D] = embedding_dim;
        
        Tensor<T, D+1> output(output_shape, input.device_type);
        
        // Lookup embeddings
        for (size_t i = 0; i < input.total_size; i++) {
            int token_id = input.flatget(i);
            assert(token_id >= 0 && token_id < static_cast<int>(num_embeddings));
            
            size_t output_offset = i * embedding_dim;
            size_t weight_offset = token_id * embedding_dim;
            
            for (size_t j = 0; j < embedding_dim; j++) {
                output.flatget(output_offset + j) = weight.flatget(weight_offset + j);
            }
        }
        
        return output;
    }
    
    // Specialized forward for common case: 2D input -> 3D output
    Tensor<T, 3> forward(const Tensor<int, 2>& input) {
        Shape<3> output_shape(input.shape[0], input.shape[1], embedding_dim);
        Tensor<T, 3> output(output_shape, input.device_type);
        
        size_t batch_size = input.shape[0];
        size_t seq_len = input.shape[1];
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                int token_id = input[SliceList<1>(b)][SliceList<1>(s)];
                assert(token_id >= 0 && token_id < static_cast<int>(num_embeddings));
                
                for (size_t e = 0; e < embedding_dim; e++) {
                    output[SliceList<1>(b)][SliceList<1>(s)][SliceList<1>(e)] = 
                        weight[SliceList<1>(token_id)][SliceList<1>(e)];
                }
            }
        }
        
        return output;
    }
    
    // Specialized forward for 1D input -> 2D output
    Tensor<T, 2> forward(const Tensor<int, 1>& input) {
        Shape<2> output_shape(input.shape[0], embedding_dim);
        Tensor<T, 2> output(output_shape, input.device_type);
        
        size_t seq_len = input.shape[0];
        
        for (size_t s = 0; s < seq_len; s++) {
            int token_id = input[SliceList<1>(s)];
            assert(token_id >= 0 && token_id < static_cast<int>(num_embeddings));
            
            for (size_t e = 0; e < embedding_dim; e++) {
                output[SliceList<1>(s)][SliceList<1>(e)] = 
                    weight[SliceList<1>(token_id)][SliceList<1>(e)];
            }
        }
        
        return output;
    }
};

#endif // EMBEDDING_HPP
