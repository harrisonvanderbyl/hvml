#ifndef DISPLAYTENSOR_HPP
#define DISPLAYTENSOR_HPP

//
//  displaytensor.hpp — DisplayTensor<T>: a 2-D tensor that is also a GPU
//  image / render target / buffer texture.
//
//  Shape is {width, height}; pixel data is row-major with `width` texels per
//  row (the same convention load_texture uses).
//
//  What it is backed by is chosen with the compute type and AllocationFlags:
//
//      DisplayTensor<uint84> a({w,h});                         // render texture:
//                                                              //   kVULKANTEXTURE, kTEXTURE|kSURFACE
//      DisplayTensor<uint84> b({w,h}, kVULKAN);                // buffer texture:
//                                                              //   kVULKAN, kTEXELBUFFER
//      DisplayTensor<uint84> c({w,h}, kVULKANTEXTURE,
//                              kSURFACE | kOPTIMAL);           // GPU-only tiled image
//
//  Colour textures are buffer-backed by default, so any of them can be
//  viewed as a buffer / HIP / CUDA / CPU pointer; every image can be both
//  sampled and rendered into.
//      DisplayTensor<float>  d({w,h}, kVULKANTEXTURE,
//                              kSURFACE | kDEPTH);             // depth buffer
//
//  In-place views (no copies — Tensor::to_compute under the hood):
//
//      a.texture()       sampler2D view            (kVULKANTEXTURE | kTEXTURE)
//      a.surface()       render-target view        (kVULKANTEXTURE | kSURFACE)
//      a.storage_image() image2D view              (kVULKANTEXTURE | kSTORAGE)
//      b.buffer()        plain VkBuffer view       (kVULKAN)
//      b.texel_buffer()  samplerBuffer view        (kVULKAN | kTEXELBUFFER)
//      b.compute()       HIP / CUDA / CPU pointer to the same memory
//
//  Rendering into it and drawing it:
//
//      a.render(cmd, [&](VkCommandBuffer cmd){ mesh.bind(cmd); mesh.draw(cmd); });
//      a.draw(cmd);          // blit over the whole current target (OverLayShader)
//      a.present(cmd);       // blit 1:1 at the top-left of the current target
//      material->setTexture("tex", a);   // any view type; converted in place
//

#include <memory>
#include <type_traits>

#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "display/vulkan_context.hpp"
#include "display/materials/shader.hpp"
#include "display/materials/overlay.hpp"
#include "display/drawable/drawable.hpp"

// ---------------------------------------------------------------------------
//  Element type → VkFormat
// ---------------------------------------------------------------------------

template <typename T> struct VkFormatOf                { static constexpr VkFormat value = VK_FORMAT_UNDEFINED; };
template <> struct VkFormatOf<uint84>                  { static constexpr VkFormat value = VK_FORMAT_R8G8B8A8_UNORM; };
template <> struct VkFormatOf<Hvec<uint8_t, 4>>        { static constexpr VkFormat value = VK_FORMAT_R8G8B8A8_UNORM; };
template <> struct VkFormatOf<Hvec<float16, 4>>        { static constexpr VkFormat value = VK_FORMAT_R16G16B16A16_SFLOAT; };
template <> struct VkFormatOf<Hvec<float, 4>>          { static constexpr VkFormat value = VK_FORMAT_R32G32B32A32_SFLOAT; };
template <> struct VkFormatOf<Hvec<float, 2>>          { static constexpr VkFormat value = VK_FORMAT_R32G32_SFLOAT; };
template <> struct VkFormatOf<float>                   { static constexpr VkFormat value = VK_FORMAT_R32_SFLOAT; };
template <> struct VkFormatOf<float16>                 { static constexpr VkFormat value = VK_FORMAT_R16_SFLOAT; };
template <> struct VkFormatOf<uint8_t>                 { static constexpr VkFormat value = VK_FORMAT_R8_UNORM; };
template <> struct VkFormatOf<int>                     { static constexpr VkFormat value = VK_FORMAT_R32_SINT; };
template <> struct VkFormatOf<uint32_t>                { static constexpr VkFormat value = VK_FORMAT_R32_UINT; };

// ---------------------------------------------------------------------------
//  Blit quad — drawn with OverLayShader (materials/overlay.hpp, compiled by
//  the shader-compiler).  Vertices are (position, uv), triangle strip.
//  OverLayShader<false> samples a sampler2D at (u, 1 - v); OverLayShader<true>
//  texelFetches a samplerBuffer at uv * dimensions — so the two variants get
//  uvs with opposite v, and both show row 0 of the tensor at the top.
// ---------------------------------------------------------------------------

inline RenderStruct<float32x2, float32x2>* make_blit_quad(bool texelBuffer) {
    auto* quad = new RenderStruct<float32x2, float32x2>(Shape<1>{4});
    float top = texelBuffer ? 0.0f : 1.0f;
    float bottom = 1.0f - top;
    float verts[16] = {-1, -1, 0, top,
                        1, -1, 1, top,
                       -1,  1, 0, bottom,
                        1,  1, 1, bottom};
    static auto upload = (hvml_vk_upload_fn)dlsym(RTLD_DEFAULT, "hvml_vk_upload");
    VulkanResource* r = vk_resource(*quad);
    if (!r || !upload || upload(r, verts, sizeof(verts)) != 0) {
        throw std::runtime_error("[display] could not upload the blit quad");
    }
    quad->primitive_type = VK_TOPOLOGY_TRIANGLE_STRIP;
    if (texelBuffer) quad->material = new Shader<OverLayShader<true>>();
    else             quad->material = new Shader<OverLayShader<false>>();
    quad->material->transparent = true;
    quad->material->depth_test = false;
    quad->material->depth_write = false;
    quad->material->double_sided = true;
    quad->material->sampler_nearest = true;
    return quad;
}

// ---------------------------------------------------------------------------
//  DisplayTensor
// ---------------------------------------------------------------------------

template <typename T>
class DisplayTensor : public Tensor<T, 2>
{
public:
    using Base = Tensor<T, 2>;

    // Default flags for each backing type.
    static AllocationFlags default_flags(ComputeType ct) {
        if (ct == ComputeType::kVULKAN) return AllocationFlags::kRW | AllocationFlags::kTEXELBUFFER;
        return AllocationFlags::kRW | AllocationFlags::kTEXTURE | AllocationFlags::kSURFACE;
    }

    // Allocation metadata for a display tensor on the rendering GPU.
    static AllocationMetadata metadata(Shape<2> shape, ComputeType ct = ComputeType::kVULKANTEXTURE,
                                       AllocationFlags flags = (AllocationFlags)0) {
        if (ct == ComputeType::kCPU || ct == ComputeType::kUnknown) {
            return AllocationMetadata::create<T>(shape, MemoryType::kDDR, ComputeType::kCPU, 0, AllocationFlags::kRW, 0);
        }
        VulkanContext& ctx = VulkanContext::get();
        if (((int)flags & kVIEW_FLAGS) == 0) flags = flags | default_flags(ct);
        flags = flags | AllocationFlags::kRW;
        int format = has_flag(flags, AllocationFlags::kDEPTH) ? (int)VK_FORMAT_D32_SFLOAT
                                                              : (int)VkFormatOf<T>::value;
        return AllocationMetadata::create<T>(shape, ctx.getRenderingMemoryType(), ct, format, flags, 0);
    }

    // A display tensor over an image made elsewhere (e.g. a swapchain image).
    // The tensor owns the view the plugin creates, not the image.
    static DisplayTensor adopt(void* image, VkFormat format, uint32_t w, uint32_t h,
                               VkImageLayout restingLayout, VkImageUsageFlags usage) {
        static auto wrap = (hvml_vk_wrap_image_fn)dlsym(RTLD_DEFAULT, "hvml_vk_wrap_image");
        VulkanResource* r = wrap ? wrap(image, (int)format, w, h, (int)restingLayout, usage) : nullptr;
        if (!r) throw std::runtime_error("[display] could not wrap image as a tensor (vulkan plugin missing?)");

        MemoryType mem = VulkanContext::get().getRenderingMemoryType();
        Shape<2> shape{(long)w, (long)h};
        AllocationMetadata meta = AllocationMetadata::create<T>(
            shape, mem, ComputeType::kVULKANTEXTURE, (int)format,
            AllocationFlags::kRW | (AllocationFlags)r->flags, 0);
        auto* alloc = new BaseMemoryAllocation(meta, (void*)r);
        DisplayTensor t(Base(shape, MassagedMemory<T>(meta, (T*)(void*)r, alloc), MemoryLocation(mem, 0), alloc));
        alloc->dealloc();   // drop the reference `new` started with; `t` holds its own
        return t;
    }

    DisplayTensor() : Base() {}
    DisplayTensor(const Base& other) : Base(other) {}
    DisplayTensor(AllocationMetadata m) : Base(m) {}
    DisplayTensor(Shape<2> shape, ComputeType ct = ComputeType::kVULKANTEXTURE,
                  AllocationFlags flags = (AllocationFlags)0)
        : Base(metadata(shape, ct, flags)) {}

    using Base::operator=;

    uint32_t width()  const { return (uint32_t)this->shape[0]; }
    uint32_t height() const { return (uint32_t)this->shape[1]; }

    // ---- in-place views --------------------------------------------------

    Base as(ComputeType ct, AllocationFlags flags = AllocationFlags::kRW) const {
        return this->to_compute(ct, flags | AllocationFlags::kRW);
    }
    Base texture()       const { return as(ComputeType::kVULKANTEXTURE, AllocationFlags::kTEXTURE); }
    Base surface()       const { return as(ComputeType::kVULKANTEXTURE, AllocationFlags::kSURFACE); }
    Base storage_image() const { return as(ComputeType::kVULKANTEXTURE, AllocationFlags::kSTORAGE); }
    Base buffer()        const { return as(ComputeType::kVULKAN); }
    Base texel_buffer()  const { return as(ComputeType::kVULKAN, AllocationFlags::kTEXELBUFFER); }
    // Pointer for compute kernels (HIP / CUDA / CPU, depending on the GPU).
    Base compute()       const { return this->to_compute(VulkanContext::get().interopComputeType()); }

    // The allocation's own resource (not a view).
    VulkanResource* resource() const {
        return this->storage_pointer ? vk_find_resource(this->storage_pointer->data) : nullptr;
    }

    // ---- host transfer -----------------------------------------------------

    // Copy the pixels back to a CPU tensor (waits for the GPU).
    Tensor<T, 2> read() const {
        Tensor<T, 2> out(this->shape, MemoryType::kDDR, ComputeType::kCPU);
        static auto download = (hvml_vk_download_fn)dlsym(RTLD_DEFAULT, "hvml_vk_download");
        VulkanResource* r = resource();
        if (!r || !download || download(r, (void*)out.data.data, out.total_bytes) != 0) {
            throw std::runtime_error("[display] read(): not a Vulkan display tensor");
        }
        return out;
    }

    // Replace the pixels from CPU memory (contiguous, same shape).
    void write(const Tensor<T, 2>& host) {
        static auto upload = (hvml_vk_upload_fn)dlsym(RTLD_DEFAULT, "hvml_vk_upload");
        VulkanResource* r = resource();
        if (!r || !upload || upload(r, (const void*)host.data.data, std::min(host.total_bytes, this->total_bytes)) != 0) {
            throw std::runtime_error("[display] write(): not a Vulkan display tensor");
        }
    }

    // ---- rendering into it -------------------------------------------------

    // Depth attachment used when rendering into this tensor (created on
    // first use unless one is attached or use_depth is false).
    bool use_depth = true;

    void attach_depth_buffer(const Tensor<float, 2>& depthBuffer) {
        state().depth = std::make_unique<Tensor<float, 2>>(depthBuffer);
        state().target.release(VulkanContext::get().device);
    }

    void set_clear_color(float r, float g, float b, float a = 1.0f) {
        state().target.clearColor = {{r, g, b, a}};
    }

    RenderTarget& render_target() {
        State& s = state();
        if (!s.target.color) {
            Base view = surface();   // cached on the allocation
            VulkanResource* r = vk_resource(view);
            if (!r || !r->image_view || !(r->image_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
                throw std::runtime_error("[display] this tensor cannot be rendered into — allocate it with "
                                         "a format the device can render to as a colour attachment");
            }
            s.target.setColor(r);
            if (use_depth) {
                if (!s.depth) {
                    s.depth = std::make_unique<Tensor<float, 2>>(DisplayTensor<float>::metadata(
                        this->shape, ComputeType::kVULKANTEXTURE, AllocationFlags::kSURFACE | AllocationFlags::kDEPTH));
                }
                s.target.setDepth(vk_resource(*s.depth));
            }
        }
        return s.target;
    }

    void begin_render(VkCommandBuffer cmd, bool clear = true) {
        VulkanContext::get().beginTarget(cmd, render_target(), clear);
    }

    void end_render(VkCommandBuffer cmd) {
        VulkanContext::get().endTarget(cmd);
    }

    // begin_render + fn + end_render.  fn may take the command buffer or nothing.
    template <typename F>
    void render(VkCommandBuffer cmd, F&& fn, bool clear = true) {
        begin_render(cmd, clear);
        if constexpr (std::is_invocable_v<F, VkCommandBuffer>) fn(cmd);
        else fn();
        end_render(cmd);
    }

    // Old names
    void bind_as_render_target(VkCommandBuffer cmd) { begin_render(cmd); }
    void unbind_render_target(VkCommandBuffer cmd) { end_render(cmd); }

    // ---- drawing it --------------------------------------------------------

    // Blit into the rectangle (x, y, w, h) of the current render target.
    void draw(VkCommandBuffer cmd, float x, float y, float w, float h) {
        if (cmd == VK_NULL_HANDLE) return;
        VulkanContext& ctx = VulkanContext::get();
        if (!ctx.insideRenderPass()) {
            std::cerr << "[display] draw() needs an active render target (window frame or begin_render)" << std::endl;
            return;
        }

        // Buffers are shown through a texel buffer, images through a sampler.
        bool texel = this->storage_pointer &&
                     this->storage_pointer->metadata.compute_device == ComputeType::kVULKAN;
        State& s = state();
        if (!s.blit || s.blitTexel != texel) {
            s.blit.reset(make_blit_quad(texel));
            s.blitTexel = texel;
            s.blit->material->setTexture("bufferTex", (const Base&)*this);
        }
        s.blit->material->uniform_setters["dimensions"] = int32x2((int)this->shape[0], (int)this->shape[1]);
        s.blit->bind(cmd);

        VkViewport viewport{x, y, w, h, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        s.blit->draw(cmd);
        ctx.setViewport(cmd, ctx.currentExtent.width, ctx.currentExtent.height);
    }

    // Stretch over the whole current target.
    void draw(VkCommandBuffer cmd) {
        VkExtent2D e = VulkanContext::get().currentExtent;
        draw(cmd, 0.0f, 0.0f, (float)e.width, (float)e.height);
    }

    // 1:1 pixels at the top-left of the current target.
    void present(VkCommandBuffer cmd = VK_NULL_HANDLE) {
        draw(cmd, 0.0f, 0.0f, (float)width(), (float)height());
    }

private:
    // Render state shared between copies of the same DisplayTensor.
    struct State {
        RenderTarget target;
        std::unique_ptr<Tensor<float, 2>> depth;
        std::unique_ptr<RenderStruct<float32x2, float32x2>> blit;
        bool blitTexel = false;
        ~State() {
            if (blit) delete blit->material;
            if (g_vk_ctx && g_vk_ctx->device && target.framebuffer) {
                vkDeviceWaitIdle(g_vk_ctx->device);
                target.release(g_vk_ctx->device);
            }
        }
    };
    std::shared_ptr<State> state_;

    State& state() {
        if (!state_) state_ = std::make_shared<State>();
        return *state_;
    }
};

// Old name
template <typename T>
using VectorDisplay = DisplayTensor<T>;

#endif // DISPLAYTENSOR_HPP
