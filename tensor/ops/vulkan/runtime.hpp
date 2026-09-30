#ifndef VULKCC_RUNTIME_HPP
#define VULKCC_RUNTIME_HPP

//
//  vulkan/runtime.hpp — vulkcc's counterpart of cuda_runtime.h.
//
//  vulkcc force-includes this into every file it compiles, in two modes:
//
//   * __VULKCC_ANALYSIS__  clang parses the program as CUDA (host side) so
//     __global__ / __device__ / __shared__, threadIdx... and <<<>>> have
//     their CUDA meaning; the shader tool translates the device code.
//   * otherwise            the host compiler (g++ / clang++) builds the
//     program: the qualifiers expand to nothing, kernels are plain
//     functions, and launches go through vulkcc::launch<kernel>(...).
//
//  Device intrinsics (__syncthreads, atomics, shuffles, fast math) are
//  defined here with host bodies; the translator recognises them by name and
//  emits their Vulkan equivalents.
//

#define __VULKCC__ 1

#include <cstdint>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include "device/vulkan_resource.hpp"

struct uint3 {
    unsigned int x = 0, y = 0, z = 0;
};

struct dim3 {
    unsigned int x = 1, y = 1, z = 1;
    constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
    constexpr dim3(const uint3& v) : x(v.x), y(v.y), z(v.z) {}
};

#if defined(__VULKCC_ANALYSIS__)
// Device code being compiled for Vulkan — the vulkcc counterpart of
// __CUDA_ARCH__ / __HIP_DEVICE_COMPILE__.  Kernels are translated from this
// parse only (the host build does not define it), so code that picks a
// device path with `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`
// adds `|| defined(__VULKCC_DEVICE__)` to use it for Vulkan too.
#define __VULKCC_DEVICE__ 1
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#include "__clang_cuda_builtin_vars.h"
typedef struct CUstream_st* cudaStream_t;
extern "C" unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, unsigned long shmem = 0, void* stream = 0);
#else
#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__
// Host builds see the builtin variables as ordinary values (kernels are
// compiled as host functions but only run on the GPU).
inline thread_local uint3 threadIdx, blockIdx;
inline thread_local dim3 blockDim, gridDim;
constexpr int warpSize = 32;
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __restrict__
#define __restrict__ __restrict
#endif

// ---------------------------------------------------------------------------
//  Device intrinsics (host bodies; translated by name in kernels)
// ---------------------------------------------------------------------------

#if !defined(__VULKCC_ANALYSIS__)   // a clang builtin when parsing as CUDA
__host__ __device__ inline void __syncthreads() {}
#endif
__host__ __device__ inline void __syncwarp(unsigned int = 0xffffffffu) {}
__host__ __device__ inline void __threadfence() {}
__host__ __device__ inline void __threadfence_block() {}

// The type comes from the pointer only; values convert to it, as with CUDA's
// overloads (atomicExch((unsigned long long*)p, 0)).
template <typename T> struct vulkcc_same { using type = T; };
template <typename T> using vulkcc_value = typename vulkcc_same<T>::type;

template <typename T> __host__ __device__ inline T atomicAdd(T* p, vulkcc_value<T> v) { T o = *p; *p = o + v; return o; }
template <typename T> __host__ __device__ inline T atomicSub(T* p, vulkcc_value<T> v) { T o = *p; *p = o - v; return o; }
template <typename T> __host__ __device__ inline T atomicExch(T* p, vulkcc_value<T> v) { T o = *p; *p = v; return o; }
template <typename T> __host__ __device__ inline T atomicMin(T* p, vulkcc_value<T> v) { T o = *p; *p = v < o ? v : o; return o; }
template <typename T> __host__ __device__ inline T atomicMax(T* p, vulkcc_value<T> v) { T o = *p; *p = v > o ? v : o; return o; }
template <typename T> __host__ __device__ inline T atomicAnd(T* p, vulkcc_value<T> v) { T o = *p; *p = o & v; return o; }
template <typename T> __host__ __device__ inline T atomicOr(T* p, vulkcc_value<T> v) { T o = *p; *p = o | v; return o; }
template <typename T> __host__ __device__ inline T atomicXor(T* p, vulkcc_value<T> v) { T o = *p; *p = o ^ v; return o; }
template <typename T> __host__ __device__ inline T atomicCAS(T* p, vulkcc_value<T> compare, vulkcc_value<T> v) { T o = *p; if (o == compare) *p = v; return o; }

template <typename T> __host__ __device__ inline T __shfl_sync(unsigned int, T v, int, int = 32) { return v; }
template <typename T> __host__ __device__ inline T __shfl_up_sync(unsigned int, T v, unsigned int, int = 32) { return v; }
template <typename T> __host__ __device__ inline T __shfl_down_sync(unsigned int, T v, unsigned int, int = 32) { return v; }
template <typename T> __host__ __device__ inline T __shfl_xor_sync(unsigned int, T v, int, int = 32) { return v; }
template <typename T> __host__ __device__ inline T __shfl(T v, int, int = 32) { return v; }
template <typename T> __host__ __device__ inline T __shfl_up(T v, unsigned int, int = 32) { return v; }
template <typename T> __host__ __device__ inline T __shfl_down(T v, unsigned int, int = 32) { return v; }
template <typename T> __host__ __device__ inline T __shfl_xor(T v, int, int = 32) { return v; }
__host__ __device__ inline int __any_sync(unsigned int, int p) { return p; }
__host__ __device__ inline int __all_sync(unsigned int, int p) { return p; }
__host__ __device__ inline unsigned int __ballot_sync(unsigned int, int p) { return p ? 1u : 0u; }

template <typename T> __host__ __device__ inline T __ldg(const T* p) { return *p; }

// __expf, __logf, __sinf, __cosf, __powf come from <math.h>
__host__ __device__ inline float __fdividef(float x, float y) { return x / y; }
__host__ __device__ inline float __frsqrt_rn(float x) { return 1.0f / sqrtf(x); }
__host__ __device__ inline float __saturatef(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
__host__ __device__ inline float __fmaf_rn(float a, float b, float c) { return fmaf(a, b, c); }
__host__ __device__ inline int __float_as_int(float x) { int i; memcpy(&i, &x, 4); return i; }
__host__ __device__ inline unsigned int __float_as_uint(float x) { unsigned int i; memcpy(&i, &x, 4); return i; }
__host__ __device__ inline float __int_as_float(int i) { float x; memcpy(&x, &i, 4); return x; }
__host__ __device__ inline float __uint_as_float(unsigned int i) { float x; memcpy(&x, &i, 4); return x; }
__host__ __device__ inline int __popc(unsigned int v) { return __builtin_popcount(v); }
__host__ __device__ inline int __clz(int v) { return v ? __builtin_clz((unsigned)v) : 32; }
__host__ __device__ inline int __ffs(int v) { return __builtin_ffs(v); }

// ---------------------------------------------------------------------------
//  Launching kernels
// ---------------------------------------------------------------------------

namespace vulkcc {

struct KernelBinary {
    const uint32_t* spirv = nullptr;
    unsigned long words = 0;
};

// Filled at start-up by the code vulkcc generates, keyed by
// typeid(KernelTag<kernel>).name().
inline std::map<std::string, KernelBinary>& registry() {
    static std::map<std::string, KernelBinary> r;
    return r;
}

inline int register_kernel(const char* key, const uint32_t* spirv, unsigned long words) {
    registry()[key] = KernelBinary{spirv, words};
    return 0;
}

template <auto Kernel>
struct KernelTag {};

// Pointer on the device the next launches should run on (like cudaSetDevice,
// but by data).  Without it the first pointer argument decides.
inline thread_local const void* current_device_hint = nullptr;
inline void set_device(const void* any_pointer_on_the_device) { current_device_hint = any_pointer_on_the_device; }

template <auto Kernel>
const KernelBinary& kernel_binary() {
    static const KernelBinary* found = [] {
        auto it = registry().find(typeid(KernelTag<Kernel>).name());
        return it == registry().end() ? nullptr : &it->second;
    }();
    if (!found) {
        throw std::runtime_error(std::string("[vulkcc] no Vulkan kernel was compiled for ") +
                                 typeid(KernelTag<Kernel>).name() + " (build with vulkcc)");
    }
    return *found;
}

inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

template <typename... P, typename... A>
std::vector<uint8_t> pack_arguments(void (*)(P...), const void*& hint, A&&... args) {
    static_assert(sizeof...(P) == sizeof...(A), "wrong number of kernel arguments");
    std::tuple<std::decay_t<P>...> values(std::forward<A>(args)...);
    std::vector<uint8_t> bytes;
    std::apply([&](auto&... v) {
        size_t off = 0;
        auto put = [&](auto& value) {
            using V = std::decay_t<decltype(value)>;
            off = align_up(off, alignof(V));
            if (bytes.size() < off + sizeof(V)) bytes.resize(off + sizeof(V));
            memcpy(bytes.data() + off, &value, sizeof(V));
            if constexpr (std::is_pointer<V>::value) {
                if (!hint) hint = (const void*)value;
            }
            off += sizeof(V);
        };
        (put(v), ...);
    }, values);
    return bytes;
}

// kernel<<<grid, block, shmem, stream>>>(args...) becomes
// vulkcc::launch<kernel>(grid, block, shmem, stream, args...).
template <auto Kernel, typename... A>
void launch(dim3 grid, dim3 block, size_t shmem, const void* stream, A&&... args) {
    (void)shmem;
    (void)stream;
    const KernelBinary& k = kernel_binary<Kernel>();
    static auto launch_fn = (hvml_vk_launch_fn)dlsym(RTLD_DEFAULT, "hvml_vk_launch");
    if (!launch_fn) throw std::runtime_error("[vulkcc] hvml_vk_launch not found (is the vulkan plugin loaded?)");
    const void* hint = current_device_hint;
    std::vector<uint8_t> bytes = pack_arguments(Kernel, hint, std::forward<A>(args)...);
    HvmlVkLaunch l;
    l.spirv = k.spirv;
    l.spirv_words = k.words;
    l.grid[0] = grid.x; l.grid[1] = grid.y; l.grid[2] = grid.z;
    l.block[0] = block.x; l.block[1] = block.y; l.block[2] = block.z;
    l.args = bytes.data();
    l.args_bytes = bytes.size();
    l.device_hint = hint;
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
    if (launch_fn(&l) != 0) {
        throw std::runtime_error(std::string("[vulkcc] launch failed: ") + typeid(KernelTag<Kernel>).name());
    }
}

} // namespace vulkcc

#endif // VULKCC_RUNTIME_HPP
