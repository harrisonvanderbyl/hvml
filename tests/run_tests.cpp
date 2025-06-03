#include "test_framework.hpp"

// Include all test files
#include "test_tensor_ops.cpp"
#include "test_modules.cpp"
#include "test_rwkv7.cpp"

int main() {
    std::cout << "=== RWKV7 Implementation Test Suite ===\n\n";
    
    TestFramework::run_all_tests();
    
    return 0;
}
