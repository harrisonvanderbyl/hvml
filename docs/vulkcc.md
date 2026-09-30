# vulkcc — CUDA-style hvml programs on Vulkan

`vulkcc` does for Vulkan what `nvcc` and `hipcc` do for CUDA and HIP. It
compiles a single-source program with `__global__` kernels and
`kernel<<<grid, block>>>(args)` launches, turning every launched kernel into a
SPIR-V compute shader. It runs on any GPU with a Vulkan driver (NVIDIA, AMD,
Intel, integrated) and needs no CUDA or ROCm install. hvml operations on
Vulkan tensors go through the same machinery, using kernels in
`tensor/ops/vulkan/ops.vk` that mirror `ops.cuh`.

```bash
# executable (host code + material shaders + Vulkan kernels)
./vulkcc main.cpp -o app -I./tensor -std=c++20 -O3 -fopenmp

# object with the program and its kernels, like `nvcc -c` (what hvcc links)
./vulkcc main.cpp -ccbin=g++ -c -o vulkan.o -I./tensor -std=c++20 -O3
```

- **Flags:** `-I`, `-D`, `-std`, `-O`, `-march` and `-f…` go to the
  compilers; `-l`, `-L` and `-Wl` go to the link.
- **Host compiler:** `-ccbin=<compiler>` picks it (default `g++`). Use the
  compiler the device plugins were built with: g++ and clang export
  `global_device_manager` differently, and a mix ends up with two device
  managers.
- **Keeping generated files:** `VULKCC_KEEP=1` keeps the rewritten source,
  the GLSL and the registration code.

## Writing kernels

Kernels are ordinary CUDA:

```cpp
struct Particle {
    float x, y, vx, vy;
    __host__ __device__ void step(float dt) { x += vx * dt; y += vy * dt; }
};

__global__ void simulate(Particle* ps, int n, float dt, float* energy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ps[i].step(dt);
    atomicAdd(energy, ps[i].vx * ps[i].vx);
}

MemoryLocation vk(global_device_manager.get_compute_device(ComputeType::kVULKAN, 0));   // Vulkan device 0
Tensor<uint8_t, 1> ps = ...;                             // allocated on vk
simulate<<<(n + 255) / 256, 256>>>((Particle*)ps.data.data, n, 0.01f, energy.data);
```

## Where Vulkan tensors live

A Vulkan device allocates in the memory map of the GPU it runs on:

- an NVIDIA GPU uses `kCUDA_VRAM`, an AMD GPU uses `kHIP_VRAM` (the same GPU's
  CUDA / HIP index, matched by PCI address);
- integrated and software GPUs use host memory, `kDDR`;
- a device with no such map (no CUDA / HIP plugin for it) gets its own
  `kUnknown_MEM` map.

That map's `compute_device_allocators[kVULKAN]` creates buffers on the device.
`to()` and the shape constructor allocate with the compute type you ask for
and return its view, so these are the same:

```cpp
auto& vk = global_device_manager.get_compute_device(ComputeType::kVULKAN, i);
auto a1 = a.to(MemoryLocation(vk.default_memory_type, vk.default_memory_device_id), ComputeType::kVULKAN);
auto a2 = a.to(MemoryLocation(vk));        // the location carries kVULKAN
Tensor<float, 1> c = a1 + b1;              // vulkcc kernels; c is a Vulkan tensor too
```

`MemoryLocation` can carry a compute type. `Tensor::location()` returns a
tensor's own (memory map and compute type), so tensors made there work with
it in kernels. With a window open, `VulkanContext::getRenderingMemoryLocation()`
is the rendering GPU's memory.

### Zero-copy views (`to_compute`)

Rows are the compute type an allocation was made with; columns are the views
`to_compute(col)` returns in place. Everything else copies with `to()`.

| allocated ↓ / view → | kCPU | kCUDA | kHIP | kVULKAN | kVULKANTEXTURE |
|---|---|---|---|---|---|
| **kCPU** | ✓ | host memory ¹ | host memory ¹ | – | – |
| **kCUDA** | managed (host) only | ✓ | – | – | – |
| **kHIP** | managed (host) only | – | ✓ | – | – |
| **kVULKAN** | host memory ² | `kCUDA_VRAM` ³ | `kHIP_VRAM` ³ | ✓ device address | ✓ linear image alias |
| **kVULKANTEXTURE** | host, `kLINEAR` ² | `kCUDA_VRAM` ³ ⁴ | `kHIP_VRAM` ³ ⁴ | `kLINEAR` only | ✓ other image views |

1. `cudaHostRegister` / `hipHostRegister`, on GPUs that can map host memory.
2. Host-visible buffers are persistently mapped.
3. The Vulkan memory is exported as an opaque fd (`VK_KHR_external_memory_fd`,
   enabled on every Vulkan device that has it) and imported by the CUDA / HIP
   plugin, like their OpenGL interop. The imports are released with the
   allocation.
4. `kLINEAR` textures give a device pointer to their pixels. Optimal-tiled
   images give a `cudaArray_t` / `hipArray_t` (level 0) for surface and
   texture objects. They are exported as dedicated allocations. HIP image
   import needs `hipExternalMemoryGetMappedMipmappedArray`, which Linux ROCm
   (up to at least 7.0) declares but does not export; the plugin looks it up
   at run time, and without it an optimal image has no HIP view (use
   `kLINEAR`).

OpenGL has the same pattern: kOPENGL buffers view as CPU (host), CUDA and HIP,
and kOPENGLTEXTURE views as a `cudaArray_t`.

A kVULKAN tensor's `data` is its kernel view, `to_compute(kVULKAN)`: the
buffer's **device address**. So tensors, slices and `Parameter<T>` hand
kernels real pointers, as on CUDA and HIP. Image and texel-buffer views
(`kTEXTURE`, `kSURFACE`, `kTEXELBUFFER`, …) are still `VulkanResource`s for
the display layer.

What translates:

| CUDA / C++ | Vulkan |
|---|---|
| structs, classes, templates, methods, constructors, operators | GLSL structs with the C++ byte layout (checked), free functions |
| `T*`, `p[i]`, `*p`, `p->f`, pointer arithmetic, pointers in structs | 64-bit device addresses (`GL_EXT_buffer_reference`) |
| references | to memory: addresses; to locals: `inout`; `const &`: by value |
| `threadIdx` / `blockIdx` / `blockDim` / `gridDim` | invocation / workgroup IDs, sizes (block size = specialization constants) |
| `__shared__` arrays, `__syncthreads()` | `shared`, `barrier()` |
| `atomicAdd/Sub/Exch/Min/Max/And/Or/Xor/CAS` | GLSL atomics (`float` needs `VK_EXT_shader_atomic_float`) |
| `__shfl_sync/_up/_down/_xor` (and HIP's `__shfl*`) | subgroup shuffles; through shared memory when the subgroup is narrower than the width |
| `__any_sync`, `__all_sync`, `__ballot_sync`, `__syncwarp`, `__ldg` | subgroup vote / ballot / barrier, a load |
| `<cmath>`, `__expf`, `rsqrtf`, `__float_as_int`, `__popc`, … | GLSL built-ins |
| `*(float*)&bits` type punning of locals | bit casts |

Not supported (the kernel is skipped with a message, and launching it
throws):
- unions, virtual functions and recursion;
- function pointers and pointers to locals or `__shared__` memory;
- dynamic shared memory, and globals other than `__shared__`.

Transcendental math on `double` is computed in `float`.

## How it works

1. **Parse.** The shader tool (`shader-compiler/`) parses the program as CUDA
   (clang, host side, no CUDA install) with `tensor/ops/vulkan/runtime.hpp`
   force-included. That header is vulkcc's `cuda_runtime.h`.
2. **Rewrite launches.** `kernel<<<g, b, s, st>>>(args)` in the main file
   becomes `vulkcc::launch<kernel>(g, b, s, st, args)`. Headers call
   `vulkcc::launch` directly, as `ops.vk` does.
3. **Translate kernels.** Every kernel passed to `vulkcc::launch` is
   translated from the instantiated AST to GLSL
   (`shader-compiler/vulkcc_translate.hpp`) and compiled with
   glslangValidator. The SPIR-V is registered under
   `typeid(vulkcc::KernelTag<kernel>).name()`.
4. **Compile the host side.** The host compiler builds the rewritten program
   (qualifiers expand to nothing) together with the registrations.
5. **Launch.** `vulkcc::launch` copies the arguments with their C++ layout
   and calls `hvml_vk_launch` in the Vulkan plugin. The plugin passes the
   argument buffer's address as a push constant, caches one pipeline per
   (device, kernel, block size), and runs the kernel on the device the
   pointers belong to.

The plugin and the display's `VulkanContext` enable the features kernels use
wherever the device has them: `bufferDeviceAddress`, `shaderInt64`/`Int16`/
`Int8`, 8/16-bit storage, `scalarBlockLayout`, `shaderFloat64` and float
atomics (`tensor/device/vulkan_compute_features.hpp`).

## Tests

- `examples/vulkcc_kernels_test.cpp`: hand-written CUDA kernels. Covers
  structs, pointers, `__shared__` tiles, atomics, warp shuffles and 2-D
  blocks.
- `examples/vulkan_ops_test.cpp`: hvml operations against the CPU.
- `examples/device_interface_test.cpp`: `a + b` on the CPU, CUDA, HIP and
  every Vulkan device, picking devices the way user code does.
- `examples/qwen3asr.cpp --device vulkan`: the whole model.

## Limits

- **Synchronous launches:** each launch waits for the GPU to finish, which
  keeps tensor lifetimes simple. Batching launches into one command buffer is
  the next step for speed.
- **Launch syntax in headers:** `<<<>>>` is rewritten in the main file only;
  headers use `vulkcc::launch<kernel>(...)`.
