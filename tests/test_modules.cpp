#include "test_framework.hpp"
#include "module/linear/linear.hpp"
#include "module/embedding/embedding.hpp"
#include "module/normalization/normalization.hpp"

REGISTER_TEST("Linear Module Forward Pass", test_linear_forward) {
    Linear<float> linear(3, 2, true);  // 3 input features, 2 output features, with bias
    
    // Set known weights and bias for testing
    // Weight matrix: [[1, 2], [3, 4], [5, 6]] (3x2: 3 input features, 2 output features)
    // Row-major storage: weight[i][j] = flatget(i * out_features + j)
    linear.weight.flatget(0) = 1.0f; linear.weight.flatget(1) = 2.0f;  // Row 0: [1, 2]
    linear.weight.flatget(2) = 3.0f; linear.weight.flatget(3) = 4.0f;  // Row 1: [3, 4]
    linear.weight.flatget(4) = 5.0f; linear.weight.flatget(5) = 6.0f;  // Row 2: [5, 6]
    
    // Bias: [0.1, 0.2]
    linear.bias.flatget(0) = 0.1f;
    linear.bias.flatget(1) = 0.2f;
    
    // Test 2D input: [[1, 2, 3]]
    Tensor<float, 2> input({1, 3});
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f; input.flatget(2) = 3.0f;
    
    auto output = linear.forward(input);
    TestFramework::assert_tensor_shape(output, {1, 2}, "Output shape");
    
    // Expected: [1*1 + 2*3 + 3*5 + 0.1, 1*2 + 2*4 + 3*6 + 0.2] = [22.1, 28.2]
    TestFramework::assert_near(output.flatget(0), 22.1f, 1e-5f, "First output");
    TestFramework::assert_near(output.flatget(1), 28.2f, 1e-5f, "Second output");
}

REGISTER_TEST("Linear Module 3D Input", test_linear_3d) {
    Linear<float> linear(2, 3, false);  // 2 input features, 3 output features, no bias
    
    // Set weights: [[1, 2, 3], [4, 5, 6]]
    linear.weight.flatget(0) = 1.0f; linear.weight.flatget(1) = 2.0f; linear.weight.flatget(2) = 3.0f;
    linear.weight.flatget(3) = 4.0f; linear.weight.flatget(4) = 5.0f; linear.weight.flatget(5) = 6.0f;
    
    // Test 3D input: [[[1, 2], [3, 4]]] (batch=1, seq=2, features=2)
    Tensor<float, 3> input({1, 2, 2});
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f;
    input.flatget(2) = 3.0f; input.flatget(3) = 4.0f;
    
    auto output = linear.forward(input);
    TestFramework::assert_tensor_shape(output, {1, 2, 3}, "Output shape");
    
    // First position: [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]
    TestFramework::assert_near(output.flatget(0), 9.0f, 1e-5f, "First position, first output");
    TestFramework::assert_near(output.flatget(1), 12.0f, 1e-5f, "First position, second output");
    TestFramework::assert_near(output.flatget(2), 15.0f, 1e-5f, "First position, third output");
    
    // Second position: [3*1 + 4*4, 3*2 + 4*5, 3*3 + 4*6] = [19, 26, 33]
    TestFramework::assert_near(output.flatget(3), 19.0f, 1e-5f, "Second position, first output");
    TestFramework::assert_near(output.flatget(4), 26.0f, 1e-5f, "Second position, second output");
    TestFramework::assert_near(output.flatget(5), 33.0f, 1e-5f, "Second position, third output");
}

REGISTER_TEST("Embedding Module", test_embedding) {
    Embedding<float> emb(5, 3);  // 5 tokens, 3 embedding dimensions
    
    // Set known embeddings
    // Token 0: [1, 2, 3]
    emb.weight.flatget(0) = 1.0f; emb.weight.flatget(1) = 2.0f; emb.weight.flatget(2) = 3.0f;
    // Token 1: [4, 5, 6]
    emb.weight.flatget(3) = 4.0f; emb.weight.flatget(4) = 5.0f; emb.weight.flatget(5) = 6.0f;
    // Token 2: [7, 8, 9]
    emb.weight.flatget(6) = 7.0f; emb.weight.flatget(7) = 8.0f; emb.weight.flatget(8) = 9.0f;
    
    // Test 1D input: [0, 1, 2]
    Tensor<int, 1> input(Shape<1>{3});
    input.flatget(0) = 0; input.flatget(1) = 1; input.flatget(2) = 2;
    
    auto output = emb.forward(input);
    TestFramework::assert_tensor_shape(output, {3, 3}, "Output shape");
    
    // Check embeddings
    TestFramework::assert_near(output.flatget(0), 1.0f, 1e-5f, "Token 0, dim 0");
    TestFramework::assert_near(output.flatget(1), 2.0f, 1e-5f, "Token 0, dim 1");
    TestFramework::assert_near(output.flatget(2), 3.0f, 1e-5f, "Token 0, dim 2");
    
    TestFramework::assert_near(output.flatget(3), 4.0f, 1e-5f, "Token 1, dim 0");
    TestFramework::assert_near(output.flatget(4), 5.0f, 1e-5f, "Token 1, dim 1");
    TestFramework::assert_near(output.flatget(5), 6.0f, 1e-5f, "Token 1, dim 2");
}

REGISTER_TEST("Embedding Module 2D Input", test_embedding_2d) {
    Embedding<float> emb(3, 2);  // 3 tokens, 2 embedding dimensions
    
    // Set embeddings
    emb.weight.flatget(0) = 1.0f; emb.weight.flatget(1) = 2.0f;  // Token 0
    emb.weight.flatget(2) = 3.0f; emb.weight.flatget(3) = 4.0f;  // Token 1
    emb.weight.flatget(4) = 5.0f; emb.weight.flatget(5) = 6.0f;  // Token 2
    
    // Test 2D input: [[0, 1], [2, 0]] (batch=2, seq=2)
    Tensor<int, 2> input({2, 2});
    input[SliceList<1>(0)][SliceList<1>(0)] = 0;
    input[SliceList<1>(0)][SliceList<1>(1)] = 1;
    input[SliceList<1>(1)][SliceList<1>(0)] = 2;
    input[SliceList<1>(1)][SliceList<1>(1)] = 0;
    
    auto output = emb.forward(input);
    TestFramework::assert_tensor_shape(output, {2, 2, 2}, "Output shape");
    
    // Check first batch, first position (token 0): [1, 2]
    TestFramework::assert_near(output[SliceList<1>(0)][SliceList<1>(0)][SliceList<1>(0)], 1.0f, 1e-5f, "Batch 0, pos 0, dim 0");
    TestFramework::assert_near(output[SliceList<1>(0)][SliceList<1>(0)][SliceList<1>(1)], 2.0f, 1e-5f, "Batch 0, pos 0, dim 1");
    
    // Check first batch, second position (token 1): [3, 4]
    TestFramework::assert_near(output[SliceList<1>(0)][SliceList<1>(1)][SliceList<1>(0)], 3.0f, 1e-5f, "Batch 0, pos 1, dim 0");
    TestFramework::assert_near(output[SliceList<1>(0)][SliceList<1>(1)][SliceList<1>(1)], 4.0f, 1e-5f, "Batch 0, pos 1, dim 1");
}

REGISTER_TEST("LayerNorm Module", test_layernorm) {
    LayerNorm<float> ln(3);  // Normalize over 3 features
    
    // Set weight and bias
    ln.weight.flatget(0) = 1.0f; ln.weight.flatget(1) = 2.0f; ln.weight.flatget(2) = 3.0f;
    ln.bias.flatget(0) = 0.1f; ln.bias.flatget(1) = 0.2f; ln.bias.flatget(2) = 0.3f;
    
    // Test input: [[1, 2, 3], [4, 5, 6]]
    Tensor<float, 2> input({2, 3});
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f; input.flatget(2) = 3.0f;
    input.flatget(3) = 4.0f; input.flatget(4) = 5.0f; input.flatget(5) = 6.0f;
    
    auto output = ln.forward(input);
    TestFramework::assert_tensor_shape(output, {2, 3}, "Output shape");
    
    // For first row [1, 2, 3]: mean = 2, var = 2/3, std = sqrt(2/3 + eps)
    // Normalized: [(-1)/std, 0/std, 1/std] * weight + bias
    float mean1 = 2.0f;
    float var1 = 2.0f/3.0f;
    float std1 = std::sqrt(var1 + 1e-5f);
    
    float expected_0 = (-1.0f / std1) * 1.0f + 0.1f;
    float expected_1 = (0.0f / std1) * 2.0f + 0.2f;
    float expected_2 = (1.0f / std1) * 3.0f + 0.3f;
    
    TestFramework::assert_near(output.flatget(0), expected_0, 1e-5f, "First row, first element");
    TestFramework::assert_near(output.flatget(1), expected_1, 1e-5f, "First row, second element");
    TestFramework::assert_near(output.flatget(2), expected_2, 1e-5f, "First row, third element");
}

REGISTER_TEST("GroupNorm Module", test_groupnorm) {
    GroupNorm<float> gn(2, 4);  // 2 groups, 4 channels
    
    // Set weight and bias
    for (size_t i = 0; i < 4; i++) {
        gn.weight.flatget(i) = 1.0f;
        gn.bias.flatget(i) = 0.0f;
    }
    
    // Test input: [[[1, 2, 3, 4]]] (batch=1, channels=4, spatial=1)
    Tensor<float, 3> input({1, 4, 1});
    input.flatget(0) = 1.0f; input.flatget(1) = 2.0f;
    input.flatget(2) = 3.0f; input.flatget(3) = 4.0f;
    
    auto output = gn.forward(input);
    TestFramework::assert_tensor_shape(output, {1, 4, 1}, "Output shape");
    
    // Group 0 (channels 0,1): values [1, 2], mean = 1.5, var = 0.25
    // Group 1 (channels 2,3): values [3, 4], mean = 3.5, var = 0.25
    float std_val = std::sqrt(0.25f + 1e-5f);
    
    TestFramework::assert_near(output.flatget(0), (1.0f - 1.5f) / std_val, 1e-5f, "Group 0, channel 0");
    TestFramework::assert_near(output.flatget(1), (2.0f - 1.5f) / std_val, 1e-5f, "Group 0, channel 1");
    TestFramework::assert_near(output.flatget(2), (3.0f - 3.5f) / std_val, 1e-5f, "Group 1, channel 2");
    TestFramework::assert_near(output.flatget(3), (4.0f - 3.5f) / std_val, 1e-5f, "Group 1, channel 3");
}
