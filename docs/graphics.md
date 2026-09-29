# Graphics (Vulkan)

The display layer is built on one idea: **every GPU object is a tensor**.
Vertex and index buffers, textures, render targets, texel buffers, depth
buffers, material uniform blocks and the window's own swapchain images are all
allocated (or wrapped) by the vulkan plugin, and one allocation can be
*viewed in place* as any of them with `Tensor::to_compute(type, flags)`. The
display code never allocates Vulkan memory itself.

```
tensor/enums/device.hpp           AllocationFlags (kSURFACE, kTEXTURE, kTEXELBUFFER, ...)
tensor/device/vulkan_resource.hpp VulkanResource — the handle behind every Vulkan tensor
tensor/device/plugins/vulkan/     allocator, in-place views, upload/readback
tensor/display/vulkan_context.hpp VulkanContext — device, render passes, render-target stack
tensor/display/displaytensor.hpp  DisplayTensor<T> — texture / render texture / buffer texture
tensor/display/display.hpp        Window — SDL window + swapchain + input + frame loop
```

## Allocation flags

Pass these in `AllocationMetadata::rwstatus` (or to `DisplayTensor`):

| flag           | Vulkan object                                          | GLSL                     |
|----------------|--------------------------------------------------------|--------------------------|
| `kSURFACE`     | colour attachment (depth attachment for depth formats) | render into it           |
| `kTEXTURE`     | sampled image                                          | `sampler2D`              |
| `kTEXELBUFFER` | `VkBufferView` over a buffer                           | `samplerBuffer`          |
| `kSTORAGE`     | storage image / storage texel buffer                   | `image2D`, `imageBuffer` |
| `kDEPTH`       | `D32_SFLOAT` format                                    |                          |
| `kLINEAR`      | require buffer-backed (row-major) image memory         |                          |
| `kOPTIMAL`     | opt out: GPU-only tiled image                          |                          |

What gets created depends on the compute type too:

| compute type     | flags                         | result                                                 |
|------------------|-------------------------------|--------------------------------------------------------|
| `kVULKAN`        | any                           | `VkBuffer` (vertex/index/uniform/storage/texel usage)  |
| `kVULKAN`        | `+kTEXELBUFFER`               | ... plus a `VkBufferView`                              |
| `kVULKAN`        | `+kTEXTURE/kSURFACE/kSTORAGE` | ... plus a linear `VkImage` on the same memory         |
| `kVULKANTEXTURE` | (default, colour formats)     | buffer-backed linear image — viewable as anything      |
| `kVULKANTEXTURE` | `+kOPTIMAL`, or depth formats | optimal-tiled `VkImage` (not buffer-viewable)          |

Images always get **every usage their format supports**: a tensor allocated
only as a surface can be sampled, and a texture can be rendered into. The
flags say what you need at minimum (and fail loudly if the device can't), not
what you're limited to later. If a colour format can't be linear on the device
(or its rows would need padding), the plugin silently falls back to an optimal
image unless you asked for `kLINEAR` explicitly.

The image format comes from `AllocationMetadata::format`: `0` = infer from the
element size, `1` = depth, anything else is a `VkFormat`. `DisplayTensor<T>`
fills it in from `VkFormatOf<T>` (`uint84` → `R8G8B8A8_UNORM`,
`float16x4` → `R16G16B16A16_SFLOAT`, `float` → `R32_SFLOAT`, ...).

Shape is `{width, height}` and pixel memory is row-major with `width` texels
per row — the same convention `load_texture` uses.

## In-place views

```cpp
DisplayTensor<uint84> buf({512, 512}, kVULKAN);   // one VkDeviceMemory

auto texel = buf.texel_buffer();   // to_compute(kVULKAN,        kTEXELBUFFER)  samplerBuffer
auto tex   = buf.texture();        // to_compute(kVULKANTEXTURE, kTEXTURE)      sampler2D
auto rt    = buf.surface();        // to_compute(kVULKANTEXTURE, kSURFACE)      render target
auto ptr   = buf.compute();        // to_compute(kHIP / kCUDA / kCPU)           kernel pointer
```

No data is copied: every view shares the allocation's memory. Views are made
once, cached on the allocation, and destroyed with it. `vk_resource(tensor)`
gives the `VulkanResource*` (image, image view, buffer, buffer view, format,
layout) behind any of them.

| from \ to                    | buffer | texel buffer | texture / surface / storage image    | HIP/CUDA ptr  |
|------------------------------|:------:|:------------:|:------------------------------------:|:-------------:|
| `kVULKAN` buffer             | ✓      | ✓            | ✓ (linear alias)                     | ✓ (fd import) |
| `kVULKANTEXTURE` (default)   | ✓      | ✓            | ✓                                    | ✓ (fd import) |
| `kOPTIMAL` / depth           | ✗      | ✗            | ✓ (whatever the format supports)     | ✗             |

A view the allocation can't give throws an exception saying why.

```cpp
DisplayTensor<uint84> surf({w, h}, kVULKANTEXTURE, kSURFACE);
surf.render(cmd, [&]{ ... });                          // draw into it
material->setTexture("tex", surf);                     // sample it (converted in place)
auto tex = surf.to_compute(kVULKANTEXTURE, kTEXTURE);  // or take the view explicitly
auto buf = surf.to_compute(kVULKAN);                   // same memory as a VkBuffer
auto ptr = surf.compute();                             // same memory for HIP/CUDA/CPU kernels
std::cout << surf.read();                              // back on the CPU
```

Every image has a *resting layout* (`SHADER_READ_ONLY_OPTIMAL` for sampled
images, `GENERAL` for linear and storage images, attachment layouts
otherwise). Render passes, uploads and readbacks all return the image to it,
so you never track layouts yourself.

## Windows, render textures and buffer textures

```cpp
#include "display/display.hpp"

Window window({1280, 720}, WP_RESIZABLE, "demo");          // alias: VulkanDisplay

DisplayTensor<uint84>    canvas({640, 360});               // kVULKANTEXTURE, kTEXTURE|kSURFACE
DisplayTensor<float16x4> field({640, 360}, kVULKAN);       // kVULKAN, kTEXELBUFFER
DisplayTensor<uint84>    fast({640, 360}, kVULKANTEXTURE,
                              kSURFACE | kOPTIMAL);              // GPU-only tiled

window.add_on_update([&](CurrentScreenInputInfo& in, VkCommandBuffer cmd) {
    // draw a mesh into the render texture (depth buffer created automatically)
    canvas.render(cmd, [&] { mesh.bind(cmd); mesh.draw(cmd); });

    // then onto the window
    canvas.draw(cmd, 0, 0, 640, 720);                      // any rectangle
    field.draw(cmd, 640, 0, 640, 720);                     // buffer shown via samplerBuffer
});
window.displayLoop();
```

- Inside a frame callback the window is the current render target.
  `begin_render(cmd)` / `render(cmd, fn)` switch to a texture and switch back,
  keeping what was already drawn to the window. `window.activateBackBuffer(cmd)`
  also switches back.
- `draw(cmd[, x, y, w, h])` stretches the tensor over the current target (or a
  rectangle of it); `present(cmd)` draws it 1:1 at the top-left. Both use
  `OverLayShader` from `materials/overlay.hpp`: `<true>` for buffer tensors,
  `<false>` for textures.
- `read()` copies pixels back to a CPU tensor; `write(host)` uploads them.
- `window.backbuffer()` is this frame's swapchain image as a
  `DisplayTensor<uint84>`; the window is rendered to exactly like any other
  render texture (it shares one depth tensor across its images).
- Several windows can exist at once; they share one `VulkanContext`.
- No window works too — `VulkanContext::get()` creates a headless device, and
  `ctx.submit([&](VkCommandBuffer cmd){ ... })` records, submits and waits.

## Materials

Assign any Vulkan tensor to a material by its GLSL name:

```cpp
material->setTexture("texture1", rock);      // sampler2D
material->setTexture("particles", buf);      // samplerBuffer → texel-buffer view made in place
```

The descriptor type comes from the shader's `layout(binding = N)` declaration
(`sampler2D`, `samplerBuffer`, `image2D`, `imageBuffer`), and the bound tensor
is converted to the matching view in place — a plain `kVULKAN` buffer can be
bound to a `sampler2D`. (`textures_ids[name] = tensor` still works; use
`setTexture` to change a texture after the material has been drawn.) The
material's uniform block is itself a `kVULKAN` tensor in host-visible memory
(`material->uniforms`), written directly by the uniform setters.

Pipelines follow the current render target and the `RenderStruct`'s
`primitive_type`, so the same material can be drawn into the window and into
render textures of other formats. `Material` also has `depth_test`,
`depth_write`, `transparent`, `double_sided`, `sampler_nearest` and
`topology`.

## Environment

| variable               | effect                                                          |
|------------------------|-----------------------------------------------------------------|
| `DEVICE_PLUGIN_DIR`    | where the plugin `.so`s are (default `tensor/device/plugins`)   |
| `HVML_VK_DEVICE=n`     | use GPU `n` instead of the first discrete GPU                   |
| `HVML_VK_VALIDATION=0` | don't enable the validation layer even if it is installed       |

## Example

`examples/vulkan_views.cpp` checks every view conversion with pixel readback,
then opens a window showing a render texture next to a buffer texture. Build
it through the shader-compiler like any other program
(`make BASEFILE=examples/vulkan_views.cpp`); `--headless` runs only the checks.
