#include <iostream>
#include "tensor/tensor.hpp"
#include "models/rwkv7/rwkv7.hpp"

int main() {
    TimeShift<float> shift(1, 4);
    
    // Initialize state to zeros
    for (size_t i = 0; i < shift.state.total_size; i++) {
        shift.state.flatget(i) = 0.0f;
    }
    
    // Test input: [[[1, 2, 3, 4], [5, 6, 7, 8]]]
    Tensor<float, 3> input({1, 2, 4});
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f; input.flatget(2) = 3.0f; input.flatget(3) = 4.0f;
    input.flatget(4) = 5.0f; input.flatget(5) = 6.0f; input.flatget(6) = 7.0f; input.flatget(7) = 8.0f;
    
    auto output = shift.forward(input);
    
    std::cout << "output.flatget(0) = " << output.flatget(0) << std::endl;
    std::cout << "output.flatget(4) = " << output.flatget(4) << std::endl;
    
    // Test the assertions
    if (std::abs(output.flatget(0) - 0.0f) <= 1e-6f) {
        std::cout << "First assertion would PASS" << std::endl;
    } else {
        std::cout << "First assertion would FAIL: expected 0.0, got " << output.flatget(0) << std::endl;
    }
    
    if (std::abs(output.flatget(4) - 1.0f) <= 1e-6f) {
        std::cout << "Second assertion would PASS" << std::endl;
    } else {
        std::cout << "Second assertion would FAIL: expected 1.0, got " << output.flatget(4) << std::endl;
    }
    
    return 0;
}
