
#include "tensor.hpp"
#include "vector/vectors.hpp"
#include "display/materials/shader.hpp"
#include "display/materials/overlay.hpp"
#include "display/drawable/drawable.hpp"
#ifndef DISPLAYTENSOR_HPP
#define DISPLAYTENSOR_HPP

__weak RenderStruct<float32x2,float32x2>* cached_rect = nullptr;

__weak RenderStruct<float32x2,float32x2>* get_overlay_rect(){
    if (cached_rect == nullptr){
        cached_rect = new RenderStruct<float32x2,float32x2>(Shape{4});
        cached_rect->view<Hvec<float,16>,1>({1}) = Hvec<float,16>{-1, -1, 0, 1, 
                                                        1, -1, 1, 1, 
                                                        -1, 1, 0, 0,
                                                        1, 1, 1, 0};
        cached_rect->primitive_type = VK_TOPOLOGY_TRIANGLE_FAN;
    }

    return cached_rect;
}


template <typename bufftype>
class VectorDisplay: public Tensor<bufftype,2>
{
    using Tensor<bufftype,2>::Tensor; // inherit constructors
    public:
    RenderStruct<float32x2,float32x2> rect;
    bool setup = false;

    // Vulkan offscreen rendering resources
    VkImage        colorImage       = VK_NULL_HANDLE;
    VkDeviceMemory colorImageMemory = VK_NULL_HANDLE;
    VkImageView    colorView        = VK_NULL_HANDLE;
    VkImage        depthImage       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView    depthView        = VK_NULL_HANDLE;
    VkFramebuffer  framebuffer      = VK_NULL_HANDLE;
    VkRenderPass   offscreenRP      = VK_NULL_HANDLE;
    VkFormat       colorFormat      = VK_FORMAT_R8G8B8A8_UNORM;
    bool has_depth_buffer = false;

    VectorDisplay() : Tensor<bufftype,2>() {}

    // Construct from a Tensor (e.g. after to() conversion)
    VectorDisplay(const Tensor<bufftype,2>& other) : Tensor<bufftype,2>(other) {}

    VectorDisplay(Shape<2> shape, ComputeType compute_type = kVULKANTEXTURE)
        : Tensor<bufftype,2>(shape,
            (compute_type == kCPU || compute_type == kUnknown) ? MemoryType::kDDR
                : (g_vk_ctx ? g_vk_ctx->getRenderingMemoryType() : MemoryType::kDDR),
            compute_type)
    {
    }

    VectorDisplay(AllocationMetadata m):Tensor<bufftype,2>(m){};

    VkFormat getColorFormat() const {
        size_t ts = this->storage_pointer->metadata.type_size;
        if (ts == 8) return VK_FORMAT_R16G16B16A16_SFLOAT;
        if (ts == 4) return VK_FORMAT_R8G8B8A8_UNORM;
        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    void initialize_textures() {
        rect = *get_overlay_rect();

        // Check if this is a buffer (kVULKAN) or a texture (kVULKANTEXTURE)
        bool isBuffer = (this->storage_pointer &&
                         this->storage_pointer->metadata.compute_device == ComputeType::kVULKAN);

        if (isBuffer) {
            // Buffer-backed display — use texelFetch (samplerBuffer)
            rect.material = new Shader<OverLayShader<true>>();
        } else {
            // Texture-backed display — use texture() (sampler2D)
            rect.material = new Shader<OverLayShader<false>>();
        }
        rect.material->transparent = true;
        setup = true;
    }

    void attach_depth_buffer(Tensor<float, 2> depthbuffer) {
        // Mark that a depth buffer is attached — the actual VkImage for depth
        // is created in initialize_framebuffer() via g_vk_ctx.
        has_depth_buffer = true;
    }

    void initialize_framebuffer() {
        if (!setup) initialize_textures();
        if (!g_vk_ctx) return;

        uint32_t w = (uint32_t)this->shape[0];
        uint32_t h = (uint32_t)this->shape[1];
        colorFormat = getColorFormat();

        // When allocated with kVULKANTEXTURE, storage_pointer->data is already
        // a VkImageView created by the vulkan plugin on the rendering device.
        if (this->storage_pointer && this->storage_pointer->data) {
            colorView = (VkImageView)this->storage_pointer->data;
        }

        if (depthView == VK_NULL_HANDLE) {
            g_vk_ctx->createImage(w, h, g_vk_ctx->depthFormat, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                depthImage, depthImageMemory);
            depthView = g_vk_ctx->createImageView(depthImage, g_vk_ctx->depthFormat,
                VK_IMAGE_ASPECT_DEPTH_BIT);
        }
        offscreenRP = g_vk_ctx->createOffscreenRenderPass(colorFormat);
        framebuffer = g_vk_ctx->createOffscreenFramebuffer(offscreenRP, colorView, depthView, w, h);

        // Set the color image view as the texture for the overlay shader
        if (rect.material) {
            // The tensor's storage_pointer->data is the VkImageView — wrap it
            // in a Tensor<void,-1> for the Material's textures_ids map.
            Tensor<void, -1> texHandle;
            texHandle.storage_pointer = new BaseMemoryAllocation(
                AllocationMetadata::create<uint8_t>(Shape<1>{1}, MemoryType::kDDR, ComputeType::kCPU, 0),
                (void*)(size_t)colorView);
            rect.material->textures_ids["bufferTex"] = texHandle;
            rect.material->texture_types["bufferTex"] = 0;
        }
    }

    void present(VkCommandBuffer cmd = VK_NULL_HANDLE) {
        if (!setup) initialize_textures();
        if (cmd == VK_NULL_HANDLE) return;

        // Set up the bufferTex binding if not already done
        if (rect.material && rect.material->textures_ids.find("bufferTex") == rect.material->textures_ids.end()) {
            bool isBuffer = (this->storage_pointer &&
                             this->storage_pointer->metadata.compute_device == ComputeType::kVULKAN);

            if (isBuffer && this->storage_pointer && this->storage_pointer->data) {
                // For kVULKAN buffers: create a VkBufferView from the VkBuffer
                // (VulkanBufferHandle::buffer is the first field)
                VkBuffer buffer = *(VkBuffer*)this->storage_pointer->data;
                if (buffer != VK_NULL_HANDLE && g_vk_ctx) {
                    VkBufferViewCreateInfo viewCI{};
                    viewCI.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
                    viewCI.buffer = buffer;
                    viewCI.format = getColorFormat();
                    viewCI.offset = 0;
                    viewCI.range = VK_WHOLE_SIZE;
                    VkBufferView bufView;
                    VkResult bvRes = vkCreateBufferView(g_vk_ctx->device, &viewCI, nullptr, &bufView);
                    if (bvRes == VK_SUCCESS) {
                        Tensor<void, -1> texHandle;
                        texHandle.storage_pointer = new BaseMemoryAllocation(
                            AllocationMetadata::create<uint8_t>(Shape<1>{1}, MemoryType::kDDR, ComputeType::kCPU, 0),
                            (void*)(size_t)bufView);
                        rect.material->textures_ids["bufferTex"] = texHandle;
                        rect.material->texture_types["bufferTex"] = 1; // 1 = buffer
                    } else {
                        std::cerr << "[display] vkCreateBufferView failed: " << bvRes
                                  << " format=" << getColorFormat()
                                  << " buffer=" << buffer << std::endl;
                    }
                }
            } else if (this->storage_pointer && this->storage_pointer->data) {
                // For kVULKANTEXTURE: storage_pointer->data is the VkImageView
                Tensor<void, -1> texHandle;
                texHandle.storage_pointer = new BaseMemoryAllocation(
                    AllocationMetadata::create<uint8_t>(Shape<1>{1}, MemoryType::kDDR, ComputeType::kCPU, 0),
                    this->storage_pointer->data);
                rect.material->textures_ids["bufferTex"] = texHandle;
                rect.material->texture_types["bufferTex"] = 0; // 0 = texture2D
            }
        }

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)this->shape[0];
        viewport.height = (float)this->shape[1];
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = {(uint32_t)this->shape[0], (uint32_t)this->shape[1]};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        rect.bind(cmd);
        rect.material->uniform_setters["dimensions"] = this->shape;
        rect.draw(cmd);
    }

    void bind_as_render_target(VkCommandBuffer cmd) {
        if (framebuffer == VK_NULL_HANDLE) initialize_framebuffer();
        if (!g_vk_ctx) return;

        g_vk_ctx->renderingOffscreen = true;
        g_vk_ctx->activeOffscreenFramebuffer = framebuffer;
        g_vk_ctx->activeOffscreenRenderPass = offscreenRP;
        g_vk_ctx->offscreenWidth = (uint32_t)this->shape[0];
        g_vk_ctx->offscreenHeight = (uint32_t)this->shape[1];
    }

    void unbind_render_target() {
        if (g_vk_ctx) {
            g_vk_ctx->renderingOffscreen = false;
            g_vk_ctx->activeOffscreenFramebuffer = VK_NULL_HANDLE;
            g_vk_ctx->activeOffscreenRenderPass = VK_NULL_HANDLE;
        }
    }
};

#endif