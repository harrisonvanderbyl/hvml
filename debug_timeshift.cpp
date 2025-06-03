#include <iostream>
#include "tensor/tensor.hpp"
#include "models/rwkv7/rwkv7.hpp"

int main() {
    TimeShift<float> shift(1, 4);  // batch=1, dim=4
    
    // Initialize state to zeros
    for (size_t i = 0; i < shift.state.total_size; i++) {
        shift.state.flatget(i) = 0.0f;
    }
    
    // Test input: [[[1, 2, 3, 4], [5, 6, 7, 8]]]
    Tensor<float, 3> input({1, 2, 4});
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f; input.flatget(2) = 3.0f; input.flatget(3) = 4.0f;
    input.flatget(4) = 5.0f; input.flatget(5) = 6.0f; input.flatget(6) = 7.0f; input.flatget(7) = 8.0f;
    
    std::cout << "Input before timeshift:" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << input.flatget(i) << " ";
    }
    std::cout << std::endl;
    
    auto output = shift.forward(input);
    
    std::cout << "Output after timeshift:" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << output.flatget(i) << " ";
    }
    std::cout << std::endl;
    
    std::cout << "State after timeshift:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << shift.state.flatget(i) << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
