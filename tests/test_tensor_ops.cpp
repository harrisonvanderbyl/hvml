#include "test_framework.hpp"
#include "tensor.hpp"
#include "ops/ops.h"
#include "ops/activations.hpp"
#include "ops/matmul.hpp"

REGISTER_TEST("Tensor Creation and Basic Operations", test_tensor_creation) {
    // Test tensor creation
    Tensor<float, 2> tensor({3, 4});
    TestFramework::assert_tensor_shape(tensor, {3, 4}, "Tensor shape");
    TestFramework::assert_equal(tensor.total_size, size_t(12), "Total size");
    
    // Test tensor initialization
    for (size_t i = 0; i < tensor.total_size; i++) {
        tensor.flatget(i) = static_cast<float>(i);
    }
    
    // Test tensor access
    TestFramework::assert_near(tensor.flatget(0), 0.0f, 1e-6f, "First element");
    TestFramework::assert_near(tensor.flatget(11), 11.0f, 1e-6f, "Last element");
}

REGISTER_TEST("Tensor Addition", test_tensor_addition) {
    Tensor<float, 2> a({2, 3});
    Tensor<float, 2> b({2, 3});
    
    // Initialize tensors
    for (size_t i = 0; i < a.total_size; i++) {
        a.flatget(i) = static_cast<float>(i + 1);
        b.flatget(i) = static_cast<float>(i + 2);
    }
    
    auto c = a + b;
    TestFramework::assert_tensor_shape(c, {2, 3}, "Result shape");
    
    // Check values
    for (size_t i = 0; i < c.total_size; i++) {
        float expected = static_cast<float>(2 * i + 3);
        TestFramework::assert_near(c.flatget(i), expected, 1e-6f, "Addition result");
    }
}

REGISTER_TEST("Tensor Multiplication", test_tensor_multiplication) {
    Tensor<float, 2> a({2, 3});
    Tensor<float, 2> b({2, 3});
    
    // Initialize tensors
    for (size_t i = 0; i < a.total_size; i++) {
        a.flatget(i) = static_cast<float>(i + 1);
        b.flatget(i) = static_cast<float>(2);
    }
    
    auto c = a * b;
    TestFramework::assert_tensor_shape(c, {2, 3}, "Result shape");
    
    // Check values
    for (size_t i = 0; i < c.total_size; i++) {
        float expected = static_cast<float>((i + 1) * 2);
        TestFramework::assert_near(c.flatget(i), expected, 1e-6f, "Multiplication result");
    }
}

REGISTER_TEST("Matrix Multiplication 2D", test_matmul_2d) {
    Tensor<float, 2> a({2, 3});
    Tensor<float, 2> b({3, 2});
    
    // Initialize a = [[1, 2, 3], [4, 5, 6]]
    a.flatget(0) = 1; a.flatget(1) = 2; a.flatget(2) = 3;
    a.flatget(3) = 4; a.flatget(4) = 5; a.flatget(5) = 6;
    
    // Initialize b = [[1, 2], [3, 4], [5, 6]]
    b.flatget(0) = 1; b.flatget(1) = 2;
    b.flatget(2) = 3; b.flatget(3) = 4;
    b.flatget(4) = 5; b.flatget(5) = 6;
    
    auto c = matmul(a, b);
    TestFramework::assert_tensor_shape(c, {2, 2}, "Result shape");
    
    // Expected result: [[22, 28], [49, 64]]
    TestFramework::assert_near(c.flatget(0), 22.0f, 1e-6f, "c[0,0]");
    TestFramework::assert_near(c.flatget(1), 28.0f, 1e-6f, "c[0,1]");
    TestFramework::assert_near(c.flatget(2), 49.0f, 1e-6f, "c[1,0]");
    TestFramework::assert_near(c.flatget(3), 64.0f, 1e-6f, "c[1,1]");
}

REGISTER_TEST("ReLU Activation", test_relu) {
    Tensor<float, 2> input({2, 3});
    
    // Initialize with some negative and positive values
    input.flatget(0) = -2.0f; input.flatget(1) = -1.0f; input.flatget(2) = 0.0f;
    input.flatget(3) = 1.0f;  input.flatget(4) = 2.0f;  input.flatget(5) = 3.0f;
    
    auto output = relu(input);
    TestFramework::assert_tensor_shape(output, {2, 3}, "Output shape");
    
    // Check ReLU results
    TestFramework::assert_near(output.flatget(0), 0.0f, 1e-6f, "ReLU(-2)");
    TestFramework::assert_near(output.flatget(1), 0.0f, 1e-6f, "ReLU(-1)");
    TestFramework::assert_near(output.flatget(2), 0.0f, 1e-6f, "ReLU(0)");
    TestFramework::assert_near(output.flatget(3), 1.0f, 1e-6f, "ReLU(1)");
    TestFramework::assert_near(output.flatget(4), 2.0f, 1e-6f, "ReLU(2)");
    TestFramework::assert_near(output.flatget(5), 3.0f, 1e-6f, "ReLU(3)");
}

REGISTER_TEST("Sigmoid Activation", test_sigmoid) {
    Tensor<float, 2> input({1, 3});
    
    // Initialize with known values
    input.flatget(0) = 0.0f;   // sigmoid(0) = 0.5
    input.flatget(1) = 1.0f;   // sigmoid(1) ≈ 0.731
    input.flatget(2) = -1.0f;  // sigmoid(-1) ≈ 0.269
    
    auto output = sigmoid(input);
    TestFramework::assert_tensor_shape(output, {1, 3}, "Output shape");
    
    TestFramework::assert_near(output.flatget(0), 0.5f, 1e-6f, "sigmoid(0)");
    TestFramework::assert_near(output.flatget(1), 0.7310586f, 1e-6f, "sigmoid(1)");
    TestFramework::assert_near(output.flatget(2), 0.2689414f, 1e-6f, "sigmoid(-1)");
}

REGISTER_TEST("Power Function", test_power) {
    Tensor<float, 2> input({2, 2});
    
    // Initialize with values [1, 2, 3, 4]
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f;
    input.flatget(2) = 3.0f; input.flatget(3) = 4.0f;
    
    auto output = power(input, 2.0f);
    TestFramework::assert_tensor_shape(output, {2, 2}, "Output shape");
    
    // Check squared values
    TestFramework::assert_near(output.flatget(0), 1.0f, 1e-6f, "1^2");
    TestFramework::assert_near(output.flatget(1), 4.0f, 1e-6f, "2^2");
    TestFramework::assert_near(output.flatget(2), 9.0f, 1e-6f, "3^2");
    TestFramework::assert_near(output.flatget(3), 16.0f, 1e-6f, "4^2");
}

REGISTER_TEST("L2 Normalization", test_normalize) {
    Tensor<float, 2> input({2, 3});
    
    // Initialize first row: [3, 4, 0] (norm = 5)
    input.flatget(0) = 3.0f; input.flatget(1) = 4.0f; input.flatget(2) = 0.0f;
    // Initialize second row: [1, 1, 1] (norm = sqrt(3))
    input.flatget(3) = 1.0f; input.flatget(4) = 1.0f; input.flatget(5) = 1.0f;
    
    auto output = normalize(input, -1);
    TestFramework::assert_tensor_shape(output, {2, 3}, "Output shape");
    
    // Check normalized values
    TestFramework::assert_near(output.flatget(0), 0.6f, 1e-6f, "3/5");
    TestFramework::assert_near(output.flatget(1), 0.8f, 1e-6f, "4/5");
    TestFramework::assert_near(output.flatget(2), 0.0f, 1e-6f, "0/5");
    
    float sqrt3 = std::sqrt(3.0f);
    TestFramework::assert_near(output.flatget(3), 1.0f/sqrt3, 1e-6f, "1/sqrt(3)");
    TestFramework::assert_near(output.flatget(4), 1.0f/sqrt3, 1e-6f, "1/sqrt(3)");
    TestFramework::assert_near(output.flatget(5), 1.0f/sqrt3, 1e-6f, "1/sqrt(3)");
}
