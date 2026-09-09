# deviceManager Makefile — builds backend .so plugins + test binary
#
# The main binary is linked with -rdynamic so that dlopen'd plugins can
# resolve the `global_device_manager` symbol back into the main executable.
#
# Usage:
#   make            # build everything available
#   make cpu        # build only the CPU plugin + test
#   make cuda       # build CUDA plugin (requires nvcc)
#   make hip        # build HIP plugin (requires hipcc)
#   make opengl     # build OpenGL plugin
#   make test       # build test binary
#   make clean

CXX      ?= g++
NVCC     ?= nvcc
HIPCC    ?= hipcc

CXXFLAGS := -std=c++20 -I. -I../ -ggdb -Wall
LDFLAGS  := -ldl

# Plugin output directory
PLUGIN_DIR := plugins

# ---- Test binary ----
TEST_SRC := test/device_test.cpp
TEST_BIN := device_test

# ============================================================
#  CPU plugin (always built)
# ============================================================
CPU_SO := $(PLUGIN_DIR)/cpu/libcpu_plugin.so

$(CPU_SO): plugins/cpu/cpu_plugin.cpp common.hpp plugin.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -o $@ $(LDFLAGS)

# ============================================================
#  CUDA plugin (requires nvcc + CUDA toolkit)
# ============================================================
CUDA_SO := $(PLUGIN_DIR)/cuda/libcuda_plugin.so

$(CUDA_SO): plugins/cuda/cuda_plugin.cpp common.hpp plugin.hpp
	@mkdir -p $(@D)
	$(NVCC) -std=c++20 -Xcompiler -fPIC -shared \
		-I. -I../ \
		$< -o $@ -lcudart

# ============================================================
#  HIP plugin (requires hipcc + ROCm)
# ============================================================
HIP_SO := $(PLUGIN_DIR)/hip/libhip_plugin.so

$(HIP_SO): plugins/hip/hip_plugin.cpp common.hpp plugin.hpp
	@mkdir -p $(@D)
	$(HIPCC) -std=c++20 -fPIC -shared \
		-I. -I../ \
		$< -o $@

# ============================================================
#  Vulkan plugin (requires Vulkan SDK / libvulkan)
# ============================================================
VK_SO := $(PLUGIN_DIR)/vulkan/libvulkan_plugin.so

$(VK_SO): plugins/vulkan/vulkan_plugin.cpp common.hpp plugin.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -fPIC -shared \
		$< -o $@ -lvulkan

# ============================================================
#  OpenGL plugin (requires GL, GLEW, SDL3)
# ============================================================
OGL_SO := $(PLUGIN_DIR)/opengl/libopengl_plugin.so

$(OGL_SO): plugins/opengl/opengl_plugin.cpp common.hpp plugin.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -fPIC -shared \
		$< -o $@ -lGL -lGLEW -lSDL3

# ============================================================
#  Test binary — linked with -rdynamic so plugins see global_device_manager
# ============================================================
$(TEST_BIN): $(TEST_SRC) common.hpp device.hpp
	$(CXX) $(CXXFLAGS) -rdynamic $< -o $@ $(LDFLAGS)

# ============================================================
#  Phony targets
# ============================================================

.PHONY: all cpu cuda hip vulkan opengl test clean

# Build all plugins + test
all: cpu cuda hip vulkan opengl test

cpu: $(CPU_SO)
cuda: $(CUDA_SO)
hip: $(HIP_SO)
vulkan: $(VK_SO)
opengl: $(OGL_SO)
test: $(TEST_BIN)

clean:
	rm -f $(CPU_SO) $(CUDA_SO) $(HIP_SO) $(VK_SO) $(OGL_SO) $(TEST_BIN) test_swap.bin
