# ========= Config =========

BASEFILE ?=
OUTPUT   ?= a.out

CXX      := g++-14
NVCC     := nvcc
HIPCC    := hipcc

SHADER_BUILD_DIR := shader-compiler/build
SHADER_TOOL := $(SHADER_BUILD_DIR)/shader_tool

CXXFLAGS := -std=c++20 -I./tensor -ggdb
LDFLAGS  := -lSDL3 -lGL -lGLEW -lamdhip64 -lcudart -lvulkan \
            -L/usr/local/cuda/lib64

NVCCFLAGS := -std=c++20 -I./tensor -Xcompiler -fPIC -Xcompiler -ggdb
HIPFLAGS  := -std=c++20 -I./tensor -ggdb

# ========= Files =========

SHAD_CPP := shad.cpp
TMP_CU   := tmp.cu
HIP_OBJ  := hip.o
NV_OBJ   := nvidia.o

# ========= Default =========

.PHONY: all
all: $(OUTPUT)

# ========= Shader compiler =========

$(SHADER_TOOL):
	cd shader-compiler/build && cmake .. .. && $(MAKE) -j8

# ========= Shader generation =========

$(SHAD_CPP): $(BASEFILE) | $(SHADER_TOOL)
	./$(SHADER_TOOL) $(BASEFILE) -- \
	    -std=c++20 -I./tensor \
	    --gcc-toolchain=/usr \
	    -isystem /usr/lib/gcc/x86_64-redhat-linux/16/include \
	    -fsyntax-only -DSTBI_NO_SIMD

$(TMP_CU): $(SHAD_CPP)
	cp $(SHAD_CPP) $(TMP_CU)

# ========= GPU compilation =========

$(HIP_OBJ): $(SHAD_CPP)
	$(HIPCC) $(SHAD_CPP) -c -o $@ $(HIPFLAGS)

$(NV_OBJ): $(TMP_CU)
	$(NVCC) -ccbin=$(CXX) $(TMP_CU) -c -o $@ $(NVCCFLAGS)

# ========= Linking =========

$(OUTPUT): $(HIP_OBJ) $(NV_OBJ)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

# ========= Cleanup =========

.PHONY: clean
clean:
	rm -f $(HIP_OBJ) $(NV_OBJ) $(TMP_CU) $(OUTPUT)
