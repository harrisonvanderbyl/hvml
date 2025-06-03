#ifndef TEST_FRAMEWORK_HPP
#define TEST_FRAMEWORK_HPP

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <cassert>
#include "tensor.hpp"

class TestFramework {
private:
    static std::vector<std::function<void()>> tests;
    static std::vector<std::string> test_names;
    static int passed_tests;
    static int failed_tests;

public:
    static void add_test(const std::string& name, std::function<void()> test) {
        test_names.push_back(name);
        tests.push_back(test);
    }

    static void run_all_tests() {
        std::cout << "Running " << tests.size() << " tests...\n\n";
        
        for (size_t i = 0; i < tests.size(); i++) {
            std::cout << "Running test: " << test_names[i] << " ... ";
            try {
                tests[i]();
                std::cout << "PASSED\n";
                passed_tests++;
            } catch (const std::exception& e) {
                std::cout << "FAILED: " << e.what() << "\n";
                failed_tests++;
            } catch (...) {
                std::cout << "FAILED: Unknown error\n";
                failed_tests++;
            }
        }
        
        std::cout << "\n=== Test Results ===\n";
        std::cout << "Passed: " << passed_tests << "\n";
        std::cout << "Failed: " << failed_tests << "\n";
        std::cout << "Total:  " << (passed_tests + failed_tests) << "\n";
        
        if (failed_tests == 0) {
            std::cout << "All tests passed! ✓\n";
        } else {
            std::cout << "Some tests failed! ✗\n";
        }
    }

    // Helper functions for assertions
    template<typename T>
    static void assert_equal(T expected, T actual, const std::string& message = "") {
        if (expected != actual) {
            throw std::runtime_error("Assertion failed: " + message + 
                " (expected: " + std::to_string(expected) + 
                ", actual: " + std::to_string(actual) + ")");
        }
    }

    template<typename T>
    static void assert_near(T expected, T actual, T tolerance = 1e-6, const std::string& message = "") {
        if (std::abs(expected - actual) > tolerance) {
            throw std::runtime_error("Assertion failed: " + message + 
                " (expected: " + std::to_string(expected) + 
                ", actual: " + std::to_string(actual) + 
                ", tolerance: " + std::to_string(tolerance) + ")");
        }
    }

    static void assert_true(bool condition, const std::string& message = "") {
        if (!condition) {
            throw std::runtime_error("Assertion failed: " + message);
        }
    }

    template<typename T, int D>
    static void assert_tensor_shape(const Tensor<T, D>& tensor, const std::vector<size_t>& expected_shape, const std::string& message = "") {
        if (tensor.shape.ndim() != expected_shape.size()) {
            throw std::runtime_error("Shape dimension mismatch: " + message);
        }
        for (size_t i = 0; i < expected_shape.size(); i++) {
            if (tensor.shape[i] != expected_shape[i]) {
                throw std::runtime_error("Shape mismatch at dimension " + std::to_string(i) + ": " + message);
            }
        }
    }

    template<typename T, int D>
    static void assert_tensor_near(const Tensor<T, D>& expected, const Tensor<T, D>& actual, T tolerance = 1e-6, const std::string& message = "") {
        assert_tensor_shape(expected, {expected.shape.data, expected.shape.data + expected.shape.ndim()}, message + " (shape mismatch)");
        assert_tensor_shape(actual, {expected.shape.data, expected.shape.data + expected.shape.ndim()}, message + " (shape mismatch)");
        
        for (size_t i = 0; i < expected.total_size; i++) {
            T exp_val = expected.flatget(i);
            T act_val = actual.flatget(i);
            if (std::abs(exp_val - act_val) > tolerance) {
                throw std::runtime_error("Tensor value mismatch at index " + std::to_string(i) + ": " + message +
                    " (expected: " + std::to_string(exp_val) + ", actual: " + std::to_string(act_val) + ")");
            }
        }
    }
};

// Static member definitions
std::vector<std::function<void()>> TestFramework::tests;
std::vector<std::string> TestFramework::test_names;
int TestFramework::passed_tests = 0;
int TestFramework::failed_tests = 0;

// Macro for easy test registration
#define REGISTER_TEST(name, test_func) \
    static void test_func(); \
    static bool registered_##test_func = []() { \
        TestFramework::add_test(name, test_func); \
        return true; \
    }(); \
    static void test_func()

#endif // TEST_FRAMEWORK_HPP
