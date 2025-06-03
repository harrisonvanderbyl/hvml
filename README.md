# RWKV7 Implementation in C++

A complete C++ implementation of the RWKV7 (Receptance Weighted Key Value) neural network architecture with comprehensive testing suite.

## Overview

This project provides a full implementation of RWKV7, including:

- **Tensor Operations**: Custom tensor library with support for various data types and operations
- **Neural Network Modules**: Linear layers, embeddings, normalization layers
- **RWKV7 Components**: TimeMix, ChannelMix, and complete model implementation
- **Comprehensive Tests**: Full test suite ensuring correctness of all components

## Architecture

### Core Components

#### 1. Tensor Library (`tensor/`)
- **tensor.hpp**: Core tensor class with multi-dimensional support
- **ops/**: Mathematical operations (activations, matrix multiplication, etc.)
- **enums/**: Device types and data type enumerations
- **vector/**: Vector operations for SIMD support

#### 2. Neural Network Modules (`tensor/module/`)
- **linear/**: Linear transformation layers
- **embedding/**: Token embedding layers
- **normalization/**: Layer normalization and group normalization
- **base/**: Base module class and parameter management

#### 3. RWKV7 Model (`tensor/models/rwkv7/`)
- **rwkv7_op.hpp**: Core RWKV7 operation kernel
- **timemix.hpp**: Time-mixing attention mechanism
- **channelmix.hpp**: Channel-mixing feed-forward network
- **block.hpp**: Complete RWKV7 transformer block
- **rwkv7.hpp**: Full model implementation with parameter loading

#### 4. Test Suite (`tests/`)
- **test_framework.hpp**: Custom testing framework
- **test_tensor_ops.cpp**: Tests for tensor operations
- **test_modules.cpp**: Tests for neural network modules
- **test_rwkv7.cpp**: Tests for RWKV7 components
- **run_tests.cpp**: Main test runner

## Key Features

### RWKV7 Architecture
- **Linear Attention**: O(n) complexity instead of O(n²) for transformers
- **Time Mixing**: Novel attention mechanism with receptance, key, value, and decay
- **Channel Mixing**: Enhanced feed-forward networks with squared ReLU
- **State Management**: Efficient recurrent state handling
- **LoRA Integration**: Low-rank adaptation for parameter efficiency

### Implementation Highlights
- **Template-based Design**: Type-safe and efficient tensor operations
- **Memory Management**: Automatic memory handling with device support
- **Modular Architecture**: Clean separation of concerns
- **Comprehensive Testing**: 20+ test cases covering all components

## Building and Testing

### Prerequisites
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- Make utility

### Build and Run Tests
```bash
# Build and run all tests
make test

# Or step by step
make run_tests
./run_tests

# Clean build artifacts
make clean
```

### Test Coverage
The test suite includes:

1. **Tensor Operations**
   - Tensor creation and basic operations
   - Element-wise operations (addition, multiplication)
   - Matrix multiplication
   - Activation functions (ReLU, Sigmoid, Tanh)
   - Normalization operations

2. **Neural Network Modules**
   - Linear layer forward pass (2D and 3D inputs)
   - Embedding lookup (1D and 2D inputs)
   - Layer normalization
   - Group normalization

3. **RWKV7 Components**
   - Core RWKV7 operation
   - ChannelMix forward pass
   - TimeShift mechanism
   - Model parameter structure
   - Full model construction and forward pass
   - Block-level operations

## Usage Example

```cpp
#include "models/rwkv7/rwkv7.hpp"

// Define model parameters
ModelParams params;
params.n_layer = 12;
params.n_embd = 768;
params.n_head = 12;
params.head_size = 64;
params.dim_ffn = 3072;
params.vocab_size = 50257;
params.decay_lora = 32;
params.aaa_lora = 16;
params.mv_lora = 24;
params.gate_lora = 20;

// Create model
RWKV7_Model<float> model(params);

// Prepare input (batch_size=1, sequence_length=10)
Tensor<int, 2> input({1, 10});
for (int i = 0; i < 10; i++) {
    input.flatget(i) = i;  // Token IDs
}

// Forward pass
auto logits = model.forward(input);
// logits shape: [1, 10, 50257]
```

## Model Loading

The implementation supports loading from safetensors format:

```cpp
// Load model from file
RWKV7_Model<float> model("path/to/model.safetensors");

// The model automatically identifies parameters and loads weights
auto output = model.forward(input_tokens);
```

## Performance Considerations

- **Memory Efficiency**: Tensors use contiguous memory layout
- **CPU Optimization**: Optimized for CPU execution with potential for SIMD
- **State Management**: Efficient recurrent state handling for inference
- **Template Specialization**: Compile-time optimizations for different data types

## File Structure

```
├── tensor/
│   ├── tensor.hpp              # Core tensor implementation
│   ├── shape.hpp               # Shape handling
│   ├── ops/                    # Mathematical operations
│   │   ├── ops.h              # Operation declarations
│   │   ├── activations.hpp    # Activation functions
│   │   └── matmul.hpp         # Matrix multiplication
│   ├── module/                # Neural network modules
│   │   ├── base/              # Base classes
│   │   ├── linear/            # Linear layers
│   │   ├── embedding/         # Embedding layers
│   │   └── normalization/     # Normalization layers
│   ├── models/rwkv7/          # RWKV7 implementation
│   │   ├── rwkv7.hpp          # Main model
│   │   ├── rwkv7_op.hpp       # Core operation
│   │   ├── timemix.hpp        # Time mixing
│   │   ├── channelmix.hpp     # Channel mixing
│   │   ├── block.hpp          # Transformer block
│   │   └── timeshift/         # Time shift kernels
│   ├── enums/                 # Enumerations
│   ├── vector/                # Vector operations
│   ├── dtypes/                # Data types
│   └── file_loaders/          # Model loading utilities
├── tests/                     # Test suite
│   ├── test_framework.hpp     # Testing framework
│   ├── test_tensor_ops.cpp    # Tensor operation tests
│   ├── test_modules.cpp       # Module tests
│   ├── test_rwkv7.cpp         # RWKV7 tests
│   └── run_tests.cpp          # Test runner
├── Makefile                   # Build configuration
└── README.md                  # This file
```

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Ensure all tests pass before submitting
4. Document new features and APIs

## License

This implementation is provided for educational and research purposes. Please refer to the original RWKV paper and implementation for licensing details.

## References

- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- [Original RWKV Implementation](https://github.com/BlinkDL/RWKV-LM)

## Testing Results

Run `make test` to see detailed test results. All tests should pass for a correct implementation:

```
=== RWKV7 Implementation Test Suite ===

Running 20+ tests...

✓ Tensor Creation and Basic Operations
✓ Tensor Addition
✓ Matrix Multiplication 2D
✓ ReLU Activation
✓ Linear Module Forward Pass
✓ Embedding Module
✓ LayerNorm Module
✓ RWKV7 Operation Basic
✓ RWKV7 Model Construction
... and more

=== Test Results ===
Passed: XX
Failed: 0
Total:  XX

All tests passed! ✓
