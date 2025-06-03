#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "tensor.hpp"
#include <cmath>
#include <algorithm>

// ReLU activation function
template <typename T, int D>
Tensor<T, D> relu(const Tensor<T, D>& input) {
    Tensor<T, D> output(input.shape, input.device_type);
    for (size_t i = 0; i < input.total_size; i++) {
        output.flatget(i) = std::max(static_cast<T>(0), input.flatget(i));
    }
    return output;
}

// Sigmoid activation function
template <typename T, int D>
Tensor<T, D> sigmoid(const Tensor<T, D>& input) {
    Tensor<T, D> output(input.shape, input.device_type);
    for (size_t i = 0; i < input.total_size; i++) {
        T x = input.flatget(i);
        output.flatget(i) = static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
    }
    return output;
}

// Tanh activation function
template <typename T, int D>
Tensor<T, D> tanh_activation(const Tensor<T, D>& input) {
    Tensor<T, D> output(input.shape, input.device_type);
    for (size_t i = 0; i < input.total_size; i++) {
        output.flatget(i) = std::tanh(input.flatget(i));
    }
    return output;
}

// Softmax activation function
template <typename T, int D>
Tensor<T, D> softmax(const Tensor<T, D>& input, int dim = -1) {
    Tensor<T, D> output(input.shape, input.device_type);
    
    if (dim < 0) {
        dim = input.shape.ndim() + dim;
    }
    
    // For simplicity, implement softmax along the last dimension
    if (dim == input.shape.ndim() - 1) {
        size_t outer_size = input.total_size / input.shape[-1];
        size_t inner_size = input.shape[-1];
        
        for (size_t i = 0; i < outer_size; i++) {
            size_t offset = i * inner_size;
            
            // Find max for numerical stability
            T max_val = input.flatget(offset);
            for (size_t j = 1; j < inner_size; j++) {
                max_val = std::max(max_val, input.flatget(offset + j));
            }
            
            // Compute exp and sum
            T sum = static_cast<T>(0);
            for (size_t j = 0; j < inner_size; j++) {
                T exp_val = std::exp(input.flatget(offset + j) - max_val);
                output.flatget(offset + j) = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (size_t j = 0; j < inner_size; j++) {
                output.flatget(offset + j) /= sum;
            }
        }
    }
    
    return output;
}

// Power function
template <typename T, int D>
Tensor<T, D> power(const Tensor<T, D>& input, T exponent) {
    Tensor<T, D> output(input.shape, input.device_type);
    for (size_t i = 0; i < input.total_size; i++) {
        output.flatget(i) = std::pow(input.flatget(i), exponent);
    }
    return output;
}

// L2 Normalize function
template <typename T, int D>
Tensor<T, D> normalize(const Tensor<T, D>& input, int dim = -1, T eps = 1e-12) {
    Tensor<T, D> output(input.shape, input.device_type);
    
    if (dim < 0) {
        dim = input.shape.ndim() + dim;
    }
    
    // For simplicity, implement normalization along the last dimension
    if (dim == input.shape.ndim() - 1) {
        size_t outer_size = input.total_size / input.shape[-1];
        size_t inner_size = input.shape[-1];
        
        for (size_t i = 0; i < outer_size; i++) {
            size_t offset = i * inner_size;
            
            // Compute L2 norm
            T norm_sq = static_cast<T>(0);
            for (size_t j = 0; j < inner_size; j++) {
                T val = input.flatget(offset + j);
                norm_sq += val * val;
            }
            T norm = std::sqrt(norm_sq + eps);
            
            // Normalize
            for (size_t j = 0; j < inner_size; j++) {
                output.flatget(offset + j) = input.flatget(offset + j) / norm;
            }
        }
    }
    
    return output;
}

#endif // ACTIVATIONS_HPP
