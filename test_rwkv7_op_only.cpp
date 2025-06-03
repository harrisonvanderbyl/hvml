#include "test_framework.hpp"
#include "models/rwkv7/rwkv7_op.hpp"

int main() {
    size_t B = 1, T = 2, C = 4, head_size = 2;
    
    // Create input tensors
    Tensor<float, 3> r({B, T, C});
    Tensor<float, 3> w({B, T, C});
    Tensor<float, 3> k({B, T, C});
    Tensor<float, 3> v({B, T, C});
    Tensor<float, 3> u({B, T, C});
    Tensor<float, 3> s({B, T, C});
    
    // Initialize with simple values
    for (size_t i = 0; i < r.total_size; i++) {
        r.flatget(i) = 0.1f;
        w.flatget(i) = 0.1f;
        k.flatget(i) = 0.1f;
        v.flatget(i) = 0.1f;
        u.flatget(i) = 0.1f;
        s.flatget(i) = 0.1f;
    }
    
    auto [output, state] = RWKV7_OP(r, w, k, v, u, s, head_size);
    
    std::cout << "Output shape: [" << output.shape[0] << ", " << output.shape[1] << ", " << output.shape[2] << "]" << std::endl;
    std::cout << "State shape: [" << state.shape[0] << ", " << state.shape[1] << ", " << state.shape[2] << ", " << state.shape[3] << "]" << std::endl;
    
    // Check for finite values
    bool output_finite = true;
    bool state_finite = true;
    
    for (size_t i = 0; i < output.total_size; i++) {
        if (!std::isfinite(output.flatget(i))) {
            output_finite = false;
            std::cout << "Non-finite output at index " << i << ": " << output.flatget(i) << std::endl;
            break;
        }
    }
    
    for (size_t i = 0; i < state.total_size; i++) {
        if (!std::isfinite(state.flatget(i))) {
            state_finite = false;
            std::cout << "Non-finite state at index " << i << ": " << state.flatget(i) << std::endl;
            break;
        }
    }
    
    std::cout << "Output finite: " << (output_finite ? "YES" : "NO") << std::endl;
    std::cout << "State finite: " << (state_finite ? "YES" : "NO") << std::endl;
    
    // Print some sample values
    std::cout << "Sample output values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(8), output.total_size); i++) {
        std::cout << "  output[" << i << "] = " << output.flatget(i) << std::endl;
    }
    
    return 0;
}
