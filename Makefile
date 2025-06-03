# Makefile for RWKV7 Implementation

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I. -I./tensor -I./tests
LDFLAGS = 

# Source directories
TENSOR_DIR = tensor
TESTS_DIR = tests

# Find all source files
TENSOR_SOURCES = $(shell find $(TENSOR_DIR) -name "*.cpp" 2>/dev/null || true)
TEST_SOURCES = $(TESTS_DIR)/run_tests.cpp

# Object files
TENSOR_OBJECTS = $(TENSOR_SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# Targets
.PHONY: all clean test help

all: test

# Build and run tests
test: run_tests
	@echo "Running RWKV7 tests..."
	./run_tests

# Build test executable
run_tests: $(TEST_OBJECTS) $(TENSOR_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -name "*.o" -delete
	rm -f run_tests

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build and run tests (default)"
	@echo "  test    - Build and run tests"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"

# Dependencies (simplified - in a real project you'd use automatic dependency generation)
$(TEST_OBJECTS): $(wildcard $(TESTS_DIR)/*.hpp) $(wildcard $(TENSOR_DIR)/**/*.hpp)
$(TENSOR_OBJECTS): $(wildcard $(TENSOR_DIR)/**/*.hpp)
