#include <iostream>
#include "tensor/tensor.hpp"
#include "module/linear/linear.hpp"

int main() {
    Linear<float> linear(3, 2, true);  // 3 input features, 2 output features, with bias
    
    std::cout << "Weight shape: " << linear.weight.shape[0] << " x " << linear.weight.shape[1] << std::endl;
    
    // Test 2D input: [[1, 2, 3]]
    Tensor<float, 2> input({1, 3});
    std::cout << "Input shape: " << input.shape[0] << " x " << input.shape[1] << std::endl;
    
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f; input.flatget(2) = 3.0f;
    
    std::cout << "About to call matmul with shapes: " << input.shape[0] << "x" << input.shape[1] 
              << " @ " << linear.weight.shape[0] << "x" << linear.weight.shape[1] << std::endl;
    
    return 0;
}
