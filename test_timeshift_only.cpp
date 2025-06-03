#include "test_framework.hpp"
#include "models/rwkv7/rwkv7.hpp"

REGISTER_TEST("TimeShift Forward", test_timeshift_forward) {
    TimeShift<float> shift(1, 4);  // batch=1, dim=4
    
    // Initialize state
    for (size_t i = 0; i < shift.state.total_size; i++) {
        shift.state.flatget(i) = 0.0f;
    }
    
    // Test input: [[[1, 2, 3, 4], [5, 6, 7, 8]]]
    Tensor<float, 3> input({1, 2, 4});
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f; input.flatget(2) = 3.0f; input.flatget(3) = 4.0f;
    input.flatget(4) = 5.0f; input.flatget(5) = 6.0f; input.flatget(6) = 7.0f; input.flatget(7) = 8.0f;
    
    auto output = shift.forward(input);
    TestFramework::assert_tensor_shape(output, {1, 2, 4}, "Output shape");
    
    std::cout << "Debug: output.flatget(0) = " << output.flatget(0) << std::endl;
    std::cout << "Debug: output.flatget(4) = " << output.flatget(4) << std::endl;
    
    TestFramework::assert_near(output.flatget(0), 0.0f, 1e-6f, "First timestep should use zero state");
    TestFramework::assert_near(output.flatget(4), 1.0f, 1e-6f, "Second timestep should use first timestep");
}

int main() {
    TestFramework::run_all_tests();
    return 0;
}
