#ifndef MATMUL_HPP
#define MATMUL_HPP

#include "tensor.hpp"
#include <cassert>

// Matrix multiplication for 2D tensors
template <typename T>
Tensor<T, 2> matmul_2d(const Tensor<T, 2>& a, const Tensor<T, 2>& b) {
    assert(a.shape[1] == b.shape[0]);
    
    Shape<2> output_shape(a.shape[0], b.shape[1]);
    Tensor<T, 2> output(output_shape, a.device_type);
    
    // Simple CPU implementation
    for (size_t i = 0; i < a.shape[0]; i++) {
        for (size_t j = 0; j < b.shape[1]; j++) {
            T sum = static_cast<T>(0);
            for (size_t k = 0; k < a.shape[1]; k++) {
                sum += a[SliceList<1>(i)][SliceList<1>(k)] * b[SliceList<1>(k)][SliceList<1>(j)];
            }
            output[SliceList<1>(i)][SliceList<1>(j)] = sum;
        }
    }
    
    return output;
}

// Batched matrix multiplication for 3D tensors
template <typename T>
Tensor<T, 3> matmul_3d(const Tensor<T, 3>& a, const Tensor<T, 3>& b) {
    assert(a.shape[0] == b.shape[0]); // Same batch size
    assert(a.shape[2] == b.shape[1]); // Compatible inner dimensions
    
    Shape<3> output_shape({a.shape[0], a.shape[1], b.shape[2]});
    Tensor<T, 3> output(output_shape, a.device_type);
    
    // Batch matrix multiplication
    for (size_t batch = 0; batch < a.shape[0]; batch++) {
        for (size_t i = 0; i < a.shape[1]; i++) {
            for (size_t j = 0; j < b.shape[2]; j++) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < a.shape[2]; k++) {
                    sum += a[SliceList<1>(batch)][SliceList<1>(i)][SliceList<1>(k)] * 
                           b[SliceList<1>(batch)][SliceList<1>(k)][SliceList<1>(j)];
                }
                output[SliceList<1>(batch)][SliceList<1>(i)][SliceList<1>(j)] = sum;
            }
        }
    }
    
    return output;
}

// General matrix multiplication with broadcasting
template <typename T, int D1, int D2>
auto matmul(const Tensor<T, D1>& a, const Tensor<T, D2>& b) {
    static_assert(D1 >= 2 && D2 >= 2, "Tensors must be at least 2D for matrix multiplication");
    
    if constexpr (D1 == 2 && D2 == 2) {
        return matmul_2d(a, b);
    } else if constexpr (D1 == 3 && D2 == 3) {
        return matmul_3d(a, b);
    } else {
        // For higher dimensions, treat as batched operations
        // This is a simplified implementation
        static_assert(D1 == D2, "Higher dimensional matmul requires same number of dimensions");
        
        // Check that all batch dimensions match
        for (int i = 0; i < D1 - 2; i++) {
            assert(a.shape[i] == b.shape[i]);
        }
        
        // Check matrix dimensions are compatible
        assert(a.shape[D1-1] == b.shape[D2-2]);
        
        // Create output shape
        Shape<D1> output_shape;
        for (int i = 0; i < D1 - 2; i++) {
            output_shape[i] = a.shape[i];
        }
        output_shape[D1-2] = a.shape[D1-2];
        output_shape[D1-1] = b.shape[D2-1];
        
        Tensor<T, D1> output(output_shape, a.device_type);
        
        // Compute total batch size
        size_t batch_size = 1;
        for (int i = 0; i < D1 - 2; i++) {
            batch_size *= a.shape[i];
        }
        
        size_t M = a.shape[D1-2];
        size_t K = a.shape[D1-1];
        size_t N = b.shape[D2-1];
        
        // Perform batched matrix multiplication
        for (size_t batch = 0; batch < batch_size; batch++) {
            size_t a_offset = batch * M * K;
            size_t b_offset = batch * K * N;
            size_t out_offset = batch * M * N;
            
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    T sum = static_cast<T>(0);
                    for (size_t k = 0; k < K; k++) {
                        sum += a.flatget(a_offset + i * K + k) * 
                               b.flatget(b_offset + k * N + j);
                    }
                    output.flatget(out_offset + i * N + j) = sum;
                }
            }
        }
        
        return output;
    }
}

#endif // MATMUL_HPP
