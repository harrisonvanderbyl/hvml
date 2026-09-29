#ifndef DEVICE_VULKAN_RESOURCE_HPP
#define DEVICE_VULKAN_RESOURCE_HPP

//
//  vulkan_resource.hpp — the one handle type every Vulkan allocation uses.
//
//  Shared by the vulkan plugin (which creates and destroys these), the
//  HIP / CUDA plugins (which import `memory` through `fd`) and the display
//  layer (which binds the handles into descriptor sets and framebuffers).
//  Vulkan handles are stored as void* / int so this header does not need
//  <vulkan/vulkan.h>.
//
//  BaseMemoryAllocation::data for a kVULKAN or kVULKANTEXTURE allocation is
//  always a VulkanResource*.  In-place views created with
//  Tensor::to_compute(kVULKAN / kVULKANTEXTURE, flags) are VulkanResource*
//  too; a view has `parent` set, shares the parent's memory, and only owns
//  the extra objects it created (an aliasing image, an image view, a buffer
//  view...).
//
//  Layout rules
//  ------------
//  * `memory` is one VkDeviceMemory.  Buffer-backed resources (every kVULKAN
//    allocation, and kVULKANTEXTURE allocations made with kLINEAR) also have
//    a `buffer` spanning the whole allocation, and their pixel data is plain
//    row-major (`row_pitch` bytes per row), which is what makes the in-place
//    views possible.
//  * `layout` is the image's resting VkImageLayout.  Every operation that
//    changes the layout (render pass, upload, readback) returns the image to
//    this layout when it finishes, so users never have to track it.
//

#include <cstdint>

struct VulkanBufferHandle {
    // ---- fields read by the HIP / CUDA interop (keep first) -----------------
    void*              buffer     = nullptr;  // VkBuffer over the whole memory (null for optimal images)
    void*              memory     = nullptr;  // VkDeviceMemory
    int                fd         = -1;       // exported opaque fd, -1 = not exported
    unsigned long long alloc_size = 0;        // size of `memory` in bytes

    // ---- image / view handles --------------------------------------------
    void*    image        = nullptr;  // VkImage (optimal, or linear alias of `memory`)
    void*    image_view   = nullptr;  // VkImageView
    void*    buffer_view  = nullptr;  // VkBufferView (texel buffer)
    void*    device       = nullptr;  // VkDevice that owns everything above

    // ---- description ---------------------------------------------------------
    int32_t  format       = 0;        // VkFormat of image / buffer view
    uint32_t width        = 0;
    uint32_t height       = 0;
    uint32_t texel_size   = 0;        // bytes per texel of `format`
    uint32_t row_pitch    = 0;        // bytes per row of buffer-backed data
    unsigned long long data_size = 0; // bytes of tensor data (alloc_size may be padded)
    int32_t  flags        = 0;        // AllocationFlags this resource supports
    int32_t  layout       = 0;        // resting VkImageLayout of `image`
    int32_t  aspect       = 0;        // VkImageAspectFlags of `image`
    uint32_t image_usage  = 0;        // VkImageUsageFlags
    uint32_t memory_type  = 0;        // memory type index of `memory`
    int32_t  device_id    = -1;       // plugin device index, -1 = rendering device
    void*    mapped       = nullptr;  // persistent host mapping (host-visible memory only)

    // ---- ownership -------------------------------------------------------
    VulkanBufferHandle* parent = nullptr;     // non-null for in-place views
    bool owns_memory      = false;
    bool owns_buffer      = false;
    bool owns_image       = false;
    bool owns_image_view  = false;
    bool owns_buffer_view = false;
    bool exported         = false;    // memory allocated with an export handle type
};

using VulkanResource = VulkanBufferHandle;

// Rendering-device description handed from the display layer to the plugin.
struct HvmlVkDeviceInfo {
    void*    instance           = nullptr;  // VkInstance
    void*    physical_device    = nullptr;  // VkPhysicalDevice
    void*    device             = nullptr;  // VkDevice
    void*    queue              = nullptr;  // VkQueue (graphics)
    uint32_t queue_family       = 0;
    void*    command_pool       = nullptr;  // VkCommandPool (on queue_family)
    int      external_memory_fd = 0;        // VK_KHR_external_memory_fd enabled
};

// C ABI exported by the vulkan plugin (looked up with dlsym(RTLD_DEFAULT, ...)).
extern "C" {
    typedef VulkanResource* (*hvml_vk_find_resource_fn)(const void* ptr);
    typedef void            (*hvml_vk_set_rendering_device_fn)(const HvmlVkDeviceInfo* info);
    typedef int             (*hvml_vk_rendering_memory_type_fn)();
    typedef int             (*hvml_vk_rendering_device_index_fn)();
    typedef int             (*hvml_vk_upload_fn)(VulkanResource* r, const void* data, unsigned long bytes);
    typedef int             (*hvml_vk_download_fn)(VulkanResource* r, void* out, unsigned long bytes);
    typedef VulkanResource* (*hvml_vk_wrap_image_fn)(void* image, int format, uint32_t width, uint32_t height,
                                                     int resting_layout, uint32_t usage);
}

#endif // DEVICE_VULKAN_RESOURCE_HPP
