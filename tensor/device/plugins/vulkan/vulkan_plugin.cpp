// vulkan_plugin.cpp — Vulkan backend plugin
//
// Built with:  g++ -std=c++20 -fPIC -shared
//                  -I. -I../
//                  -o plugins/vulkan/libvulkan_plugin.so vulkan_plugin.cpp
//                  -lvulkan
//
// Priority 50 — loaded after CPU/Disk/CUDA/HIP so it can register converters
// on their AllocationMaps.
//
// ---------------------------------------------------------------------------
//  What this plugin allocates
// ---------------------------------------------------------------------------
//
//  Every kVULKAN / kVULKANTEXTURE allocation is a VulkanResource (see
//  vulkan_resource.hpp).  The AllocationFlags in the metadata decide which
//  Vulkan objects get created:
//
//    compute type     flags                  result
//    ------------     -----                  ------
//    kVULKAN          (any)                  VkBuffer (vertex/index/uniform/
//                                            storage/texel usage) over one
//                                            VkDeviceMemory
//    kVULKAN          + kTEXELBUFFER         ... plus a VkBufferView
//    kVULKAN          + kTEXTURE/kSURFACE/   ... plus a linear VkImage aliasing
//                       kSTORAGE             the same memory
//    kVULKANTEXTURE   (default)              buffer-backed linear VkImage —
//                                            viewable in place as anything
//    kVULKANTEXTURE   + kOPTIMAL, or depth   optimal-tiled VkImage + view
//                                            (falls back to this when the
//                                            device can't do linear)
//
//  Images get every usage their format supports, so a surface can always be
//  sampled and a texture can always be rendered into.  Swapchain images are
//  wrapped with hvml_vk_wrap_image() so they are tensors too.
//
//  In-place views (Tensor::to_compute(type, flags)):
//
//    kVULKAN        -> kVULKANTEXTURE    linear image alias (sampler2D, render
//                                        target, storage image)
//    kVULKAN        -> kVULKAN+TEXEL     VkBufferView (samplerBuffer)
//    kVULKANTEXTURE -> kVULKAN           the backing buffer (not kOPTIMAL)
//    kVULKANTEXTURE -> kHIP / kCUDA      imported device pointer (not kOPTIMAL)
//    kVULKAN*       -> kCPU              persistent host mapping (host-visible
//                                        memory, e.g. integrated GPUs)
//
//  Views share memory with the allocation, are cached on it, and are
//  destroyed with it.

#include "plugin.hpp"

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <map>
#include <unistd.h>

#define VK_CHECK(call)                                                         \
    do {                                                                       \
        VkResult result = call;                                                \
        if (result != VK_SUCCESS) {                                            \
            std::cerr << "[vulkan] Vulkan error at " << __FILE__ << ":"        \
                      << __LINE__ << " - Result: " << result << std::endl;     \
        }                                                                      \
    } while (0)

#define VK_THROW(call, what)                                                   \
    do {                                                                       \
        VkResult _r = call;                                                    \
        if (_r != VK_SUCCESS) {                                                \
            throw std::runtime_error(std::string("[vulkan] ") + (what) +       \
                                     " failed (VkResult " +                    \
                                     std::to_string((int)_r) + ")");           \
        }                                                                      \
    } while (0)

namespace {

// ===========================================================================
//  Devices
// ===========================================================================

struct VulkanDeviceState {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice         device          = VK_NULL_HANDLE;
    VkQueue          compute_queue   = VK_NULL_HANDLE;
    uint32_t         compute_queue_family = 0;
    VkCommandPool    command_pool    = VK_NULL_HANDLE;
};

VkInstance                     g_instance    = VK_NULL_HANDLE;
bool                           g_initialized = false;
std::vector<VulkanDeviceState> g_devices;

// The rendering device is shared by the display layer (VulkanContext) so that
// every image and buffer the tensor system creates lives on the same VkDevice
// as the swapchain, render passes and pipelines.
struct RenderingDevice {
    VkDevice         device                = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device       = VK_NULL_HANDLE;
    VkQueue          graphics_queue        = VK_NULL_HANDLE;
    uint32_t         graphics_queue_family = 0;
    VkCommandPool    command_pool          = VK_NULL_HANDLE;
    MemoryType       memory_type           = MemoryType::kUnknown_MEM;
    int              device_index          = -1;   // matching index into g_devices
    bool             external_memory_fd    = false;
    bool             active                = false;
};

RenderingDevice g_rendering_device;

constexpr int kRenderingDevice = -1;

// Everything needed to do work on one VkDevice.
struct Dev {
    VkDevice         device       = VK_NULL_HANDLE;
    VkPhysicalDevice phys         = VK_NULL_HANDLE;
    VkQueue          queue        = VK_NULL_HANDLE;
    VkCommandPool    pool         = VK_NULL_HANDLE;
    bool             external_fd  = false;
    int              id           = kRenderingDevice;
};

Dev rendering_dev() {
    Dev d;
    d.device      = g_rendering_device.device;
    d.phys        = g_rendering_device.physical_device;
    d.queue       = g_rendering_device.graphics_queue;
    d.pool        = g_rendering_device.command_pool;
    d.external_fd = g_rendering_device.external_memory_fd;
    d.id          = kRenderingDevice;
    return d;
}

// Resolve a plugin device index (or kRenderingDevice) to a Dev.  Once the
// display layer has shared its device, the matching plugin index resolves to
// the rendering device so allocations land next to the swapchain.
Dev dev_for(int device_id) {
    if (g_rendering_device.active &&
        (device_id == kRenderingDevice || device_id == g_rendering_device.device_index)) {
        return rendering_dev();
    }
    if (device_id == kRenderingDevice) device_id = 0;
    if (device_id >= 0 && device_id < (int)g_devices.size()) {
        Dev d;
        d.device = g_devices[device_id].device;
        d.phys   = g_devices[device_id].physical_device;
        d.queue  = g_devices[device_id].compute_queue;
        d.pool   = g_devices[device_id].command_pool;
        d.id     = device_id;
        return d;
    }
    throw std::runtime_error("[vulkan] no Vulkan device available for allocation");
}

// The Dev that created a resource (by VkDevice handle, not by index — the
// rendering device may have been set after the resource was created).
Dev dev_of(const VulkanResource* r) {
    if (g_rendering_device.active && r->device == (void*)g_rendering_device.device) {
        return rendering_dev();
    }
    for (size_t i = 0; i < g_devices.size(); i++) {
        if ((void*)g_devices[i].device == r->device) return dev_for((int)i);
    }
    throw std::runtime_error("[vulkan] resource belongs to an unknown VkDevice");
}

MemoryType memory_type_from_device_name(const char* name) {
    if (strstr(name, "NVIDIA") != nullptr) return MemoryType::kCUDA_VRAM;
    if (strstr(name, "AMD") != nullptr || strstr(name, "ATI") != nullptr) return MemoryType::kHIP_VRAM;
    // Intel, llvmpipe, SwiftShader, ... → host memory
    return MemoryType::kDDR;
}

// Compute type whose kernels can touch memory of a given MemoryType.
ComputeType interop_compute_type(MemoryType mem) {
    if (mem == MemoryType::kCUDA_VRAM) return ComputeType::kCUDA;
    if (mem == MemoryType::kHIP_VRAM)  return ComputeType::kHIP;
    return ComputeType::kCPU;
}

// ===========================================================================
//  Resource registry — lets any layer ask "is this pointer a VulkanResource?"
// ===========================================================================

std::mutex                      g_registry_mutex;
std::unordered_set<const void*> g_registry;

void registry_add(VulkanResource* r) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry.insert(r);
}

void registry_remove(VulkanResource* r) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry.erase(r);
}

VulkanResource* registry_find(const void* p) {
    if (!p) return nullptr;
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    return g_registry.count(p) ? (VulkanResource*)p : nullptr;
}

// ===========================================================================
//  Formats and shapes
// ===========================================================================

uint32_t format_texel_size(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:  case VK_FORMAT_R8_SRGB:
            return 1;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:   case VK_FORMAT_D16_UNORM:
            return 2;
        case VK_FORMAT_R8G8B8_UNORM: case VK_FORMAT_R8G8B8_SRGB:
            return 3;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R8G8B8A8_SINT:  case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R16G16_SFLOAT:  case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R32_SFLOAT:     case VK_FORMAT_R32_UINT:    case VK_FORMAT_R32_SINT:
        case VK_FORMAT_D32_SFLOAT:     case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            return 4;
        case VK_FORMAT_R16G16B16_SFLOAT:
            return 6;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_UINT:   case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return 8;
        case VK_FORMAT_R32G32B32_SFLOAT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT:
            return 12;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_R32G32B32A32_SINT:
            return 16;
        default:
            return 0;
    }
}

bool is_depth_format(VkFormat f) {
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           f == VK_FORMAT_X8_D24_UNORM_PACK32;
}

VkImageAspectFlags aspect_for(VkFormat f) {
    return is_depth_format(f) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Legacy inference from the element size (used when metadata.format == 0).
VkFormat format_from_type_size(size_t type_size) {
    switch (type_size) {
        case 1:  return VK_FORMAT_R8_UNORM;
        case 2:  return VK_FORMAT_R16_SFLOAT;
        case 3:  return VK_FORMAT_R8G8B8_UNORM;
        case 4:  return VK_FORMAT_R8G8B8A8_UNORM;
        case 6:  return VK_FORMAT_R16G16B16_SFLOAT;
        case 8:  return VK_FORMAT_R16G16B16A16_SFLOAT;
        case 12: return VK_FORMAT_R32G32B32_SFLOAT;
        case 16: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

// metadata.format:  0 = infer from type size (or reuse the base resource's
// format if the texel size matches), 1 = depth (legacy), >= 2 = a VkFormat.
VkFormat resolve_format(const AllocationMetadata& m, const VulkanResource* base) {
    if (m.format >= 2) return (VkFormat)m.format;
    if (m.format == 1 || has_flag(m.rwstatus, kDEPTH)) return VK_FORMAT_D32_SFLOAT;
    if (base && base->format &&
        format_texel_size((VkFormat)base->format) == m.type_size) {
        return (VkFormat)base->format;
    }
    return format_from_type_size(m.type_size);
}

// Image extent of a tensor: width = shape[0], height = the rest.  Pixel data
// is row-major with `width` texels per row (the convention load_texture and
// the display tensors use).
void dims_for(const AllocationMetadata& m, uint32_t& w, uint32_t& h) {
    size_t total = m.shape.total_size();
    size_t w0 = m.shape.ndim() > 0 ? (size_t)m.shape[0] : 1;
    if (w0 == 0) w0 = 1;
    w = (uint32_t)w0;
    h = (uint32_t)std::max<size_t>(1, total / w0);
}

bool wants_image(int flags) {
    return (flags & (kTEXTURE | kSURFACE | kSTORAGE)) != 0;
}

std::string flag_names(int flags) {
    std::string s;
    auto add = [&](int bit, const char* n) { if (flags & bit) { if (!s.empty()) s += "|"; s += n; } };
    add(kSURFACE, "kSURFACE"); add(kTEXTURE, "kTEXTURE"); add(kTEXELBUFFER, "kTEXELBUFFER");
    add(kSTORAGE, "kSTORAGE"); add(kDEPTH, "kDEPTH"); add(kLINEAR, "kLINEAR");
    return s.empty() ? "none" : s;
}

// ===========================================================================
//  Low-level helpers
// ===========================================================================

uint32_t pick_memory_type(VkPhysicalDevice phys, uint32_t type_bits, bool prefer_host,
                          bool* host_visible_out = nullptr) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys, &props);

    auto find = [&](VkMemoryPropertyFlags want) -> int {
        for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
            if ((type_bits & (1u << i)) &&
                (props.memoryTypes[i].propertyFlags & want) == want) return (int)i;
        }
        return -1;
    };

    int idx = -1;
    if (prefer_host) idx = find(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (idx < 0) idx = find(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (idx < 0) idx = find(0);
    if (idx < 0) throw std::runtime_error("[vulkan] no suitable memory type");

    if (host_visible_out) {
        *host_visible_out = (props.memoryTypes[idx].propertyFlags &
                             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    return (uint32_t)idx;
}

VkCommandBuffer begin_one_time_commands(const Dev& d) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = d.pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_THROW(vkAllocateCommandBuffers(d.device, &alloc_info, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);
    return cmd;
}

void end_one_time_commands(const Dev& d, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(d.queue, 1, &submit_info, VK_NULL_HANDLE));
    vkQueueWaitIdle(d.queue);
    vkFreeCommandBuffers(d.device, d.pool, 1, &cmd);
}

// Full barrier between two layouts.  One-shot work only, so ALL_COMMANDS is
// fine (and valid on compute-only queues too).
void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                   VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

// Host-visible scratch buffer for uploads/readbacks.
struct Staging {
    VkDevice       device = VK_NULL_HANDLE;
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void*          ptr    = nullptr;

    Staging(const Dev& d, VkDeviceSize size) : device(d.device) {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = size;
        ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_THROW(vkCreateBuffer(device, &ci, nullptr, &buffer), "vkCreateBuffer (staging)");

        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device, buffer, &reqs);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = reqs.size;
        ai.memoryTypeIndex = pick_memory_type(d.phys, reqs.memoryTypeBits, true);
        VK_THROW(vkAllocateMemory(device, &ai, nullptr, &memory), "vkAllocateMemory (staging)");
        vkBindBufferMemory(device, buffer, memory, 0);
        vkMapMemory(device, memory, 0, size, 0, &ptr);
    }

    ~Staging() {
        if (ptr) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};

VkBufferImageCopy full_image_copy(const VulkanResource* r) {
    VkBufferImageCopy region{};
    region.imageSubresource = {(VkImageAspectFlags)r->aspect, 0, 0, 1};
    region.imageExtent = {r->width, r->height, 1};
    return region;
}

// ===========================================================================
//  Resource creation
// ===========================================================================

constexpr VkBufferUsageFlags kBufferUsage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

// One VkDeviceMemory with a VkBuffer over all of it.  The memory is padded so
// a linear image of the same data can be bound to it later.
void create_backing_buffer(const Dev& d, VulkanResource* r, VkDeviceSize size,
                           bool prefer_host, bool want_export) {
    VkExternalMemoryBufferCreateInfo ext_ci{};
    ext_ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = kBufferUsage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (want_export) ci.pNext = &ext_ci;

    VkBuffer buffer;
    VK_THROW(vkCreateBuffer(d.device, &ci, nullptr, &buffer), "vkCreateBuffer");
    r->buffer = (void*)buffer;
    r->owns_buffer = true;

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(d.device, buffer, &reqs);

    bool host_visible = false;
    uint32_t mem_type = pick_memory_type(d.phys, reqs.memoryTypeBits, prefer_host, &host_visible);

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = (reqs.size + 4095) & ~(VkDeviceSize)4095;
    ai.memoryTypeIndex = mem_type;
    if (want_export) ai.pNext = &export_info;

    VkDeviceMemory memory;
    VK_THROW(vkAllocateMemory(d.device, &ai, nullptr, &memory), "vkAllocateMemory");
    r->memory = (void*)memory;
    r->owns_memory = true;
    r->alloc_size = ai.allocationSize;
    r->memory_type = mem_type;
    r->exported = want_export;
    VK_THROW(vkBindBufferMemory(d.device, buffer, memory, 0), "vkBindBufferMemory");

    if (want_export) {
        auto get_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(d.device, "vkGetMemoryFdKHR");
        VkMemoryGetFdInfoKHR fd_info{};
        fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fd_info.memory = memory;
        fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        if (!get_fd || get_fd(d.device, &fd_info, &fd) != VK_SUCCESS) {
            std::cerr << "[vulkan] could not export memory fd; HIP/CUDA interop unavailable" << std::endl;
            fd = -1;
        }
        r->fd = fd;
    }

    if (host_visible) {
        void* mapped = nullptr;
        if (vkMapMemory(d.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
            r->mapped = mapped;
        }
    }
}

// A VkBufferView over the buffer of `backing` (texel buffer / samplerBuffer).
void create_buffer_view(const Dev& d, VulkanResource* r, const VulkanResource* backing,
                        VkFormat format, int flags) {
    uint32_t texel = format_texel_size(format);
    if (format == VK_FORMAT_UNDEFINED || texel == 0) {
        throw std::runtime_error("[vulkan] kTEXELBUFFER needs a texel format; pass an explicit VkFormat "
                                 "in AllocationMetadata::format for this element type");
    }

    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(d.phys, format, &fp);
    if (!(fp.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT)) {
        throw std::runtime_error("[vulkan] format " + std::to_string((int)format) +
                                 " cannot be used as a uniform texel buffer on this device");
    }
    if ((flags & kSTORAGE) && !(fp.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT)) {
        throw std::runtime_error("[vulkan] format " + std::to_string((int)format) +
                                 " cannot be used as a storage texel buffer on this device");
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d.phys, &props);
    VkDeviceSize elements = backing->data_size / texel;
    if (elements == 0) elements = 1;
    if (elements > props.limits.maxTexelBufferElements) {
        throw std::runtime_error("[vulkan] texel buffer has " + std::to_string(elements) +
                                 " texels; this device allows at most " +
                                 std::to_string(props.limits.maxTexelBufferElements));
    }

    VkBufferViewCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    ci.buffer = (VkBuffer)backing->buffer;
    ci.format = format;
    ci.offset = 0;
    ci.range = elements * texel;

    VkBufferView view;
    VK_THROW(vkCreateBufferView(d.device, &ci, nullptr, &view), "vkCreateBufferView");
    r->buffer_view = (void*)view;
    r->owns_buffer_view = true;
    r->flags |= kTEXELBUFFER;
}

VkImageView create_image_view(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspect) {
    VkImageViewCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image = image;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format = format;
    ci.subresourceRange = {aspect, 0, 1, 0, 1};
    VkImageView view;
    VK_THROW(vkCreateImageView(device, &ci, nullptr, &view), "vkCreateImageView");
    return view;
}

// Image usage requested by the view flags, checked against the format
// features for the given tiling.  Throws with a readable message when the
// device cannot do what was asked.
VkImageUsageFlags image_usage_for(const Dev& d, VkFormat format, int flags, VkImageTiling tiling) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(d.phys, format, &fp);
    VkFormatFeatureFlags features = (tiling == VK_IMAGE_TILING_LINEAR) ? fp.linearTilingFeatures
                                                                       : fp.optimalTilingFeatures;
    const char* tiling_name = (tiling == VK_IMAGE_TILING_LINEAR) ? "linear" : "optimal";
    bool depth = is_depth_format(format);

    VkImageUsageFlags usage = 0;
    auto need = [&](VkFormatFeatureFlags feature, VkImageUsageFlags bit, const char* what) {
        if (!(features & feature)) {
            throw std::runtime_error(std::string("[vulkan] format ") + std::to_string((int)format) +
                                     " does not support " + what + " with " + tiling_name +
                                     " tiling on this device");
        }
        usage |= bit;
    };

    if (flags & kTEXTURE) need(VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT, VK_IMAGE_USAGE_SAMPLED_BIT, "sampling (kTEXTURE)");
    if (flags & kSURFACE) {
        if (depth) need(VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depth attachment (kSURFACE)");
        else       need(VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "colour attachment (kSURFACE)");
    }
    if (flags & kSTORAGE) need(VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, VK_IMAGE_USAGE_STORAGE_BIT, "storage (kSTORAGE)");

    // Everything else the format can do, so any image can later be viewed as
    // a texture or a render target without re-allocating.  (Storage stays
    // opt-in: it disables compression on some GPUs.)
    if (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (depth) {
        if (features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    } else {
        if (features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (features & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (features & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return usage;
}

// Where an image sits between operations (see vulkan_resource.hpp).
VkImageLayout resting_layout(VkImageUsageFlags usage, bool depth, bool linear) {
    if (linear || (usage & VK_IMAGE_USAGE_STORAGE_BIT)) return VK_IMAGE_LAYOUT_GENERAL;
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
        return depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    return VK_IMAGE_LAYOUT_GENERAL;
}

// The view flags an image with this usage can serve.
int flags_from_usage(VkImageUsageFlags usage) {
    int f = 0;
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT) f |= kTEXTURE;
    if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) f |= kSURFACE;
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT) f |= kSTORAGE;
    return f;
}

// A linear VkImage bound to `backing`'s memory at offset 0, so its pixels
// *are* the buffer's bytes.  Fills the image fields of `r`.
void create_linear_image(const Dev& d, VulkanResource* r, const VulkanResource* backing,
                         VkFormat format, uint32_t w, uint32_t h, int flags) {
    uint32_t texel = format_texel_size(format);
    if (format == VK_FORMAT_UNDEFINED || texel == 0) {
        throw std::runtime_error("[vulkan] cannot view this element type as an image; pass an explicit "
                                 "VkFormat in AllocationMetadata::format");
    }
    if (is_depth_format(format)) {
        throw std::runtime_error("[vulkan] depth images cannot be linear / buffer-backed");
    }
    if ((VkDeviceSize)w * h * texel > backing->alloc_size) {
        throw std::runtime_error("[vulkan] image view " + std::to_string(w) + "x" + std::to_string(h) +
                                 " is larger than the allocation");
    }
    if (!wants_image(flags)) flags |= kTEXTURE;

    VkImageUsageFlags usage = image_usage_for(d, format, flags, VK_IMAGE_TILING_LINEAR);

    VkExternalMemoryImageCreateInfo ext_ci{};
    ext_ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_LINEAR;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;   // keep the existing bytes
    if (backing->exported) ci.pNext = &ext_ci;

    VkImage image;
    VK_THROW(vkCreateImage(d.device, &ci, nullptr, &image), "vkCreateImage (linear)");

    auto fail = [&](const std::string& why) {
        vkDestroyImage(d.device, image, nullptr);
        throw std::runtime_error("[vulkan] cannot alias this allocation as a " + std::to_string(w) + "x" +
                                 std::to_string(h) + " image: " + why);
    };

    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(d.device, image, &reqs);
    if (!(reqs.memoryTypeBits & (1u << backing->memory_type))) fail("memory type not usable for linear images");
    if (reqs.size > backing->alloc_size) fail("image needs " + std::to_string(reqs.size) + " bytes");

    VkImageSubresource sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout sl;
    vkGetImageSubresourceLayout(d.device, image, &sub, &sl);
    if (sl.offset != 0 || sl.rowPitch != (VkDeviceSize)w * texel) {
        fail("the device pads linear rows to " + std::to_string(sl.rowPitch) + " bytes (need " +
             std::to_string((size_t)w * texel) + "); use a width whose row size is a multiple of the "
             "device's row alignment");
    }

    if (vkBindImageMemory(d.device, image, (VkDeviceMemory)backing->memory, 0) != VK_SUCCESS) {
        fail("vkBindImageMemory failed");
    }

    r->image = (void*)image;
    r->owns_image = true;
    r->image_view = (void*)create_image_view(d.device, image, format, VK_IMAGE_ASPECT_COLOR_BIT);
    r->owns_image_view = true;
    r->format = format;
    r->width = w;
    r->height = h;
    r->texel_size = texel;
    r->row_pitch = w * texel;
    r->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    r->image_usage = usage;
    r->layout = VK_IMAGE_LAYOUT_GENERAL;
    r->flags |= flags_from_usage(usage) | kLINEAR;

    VkCommandBuffer cmd = begin_one_time_commands(d);
    image_barrier(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_GENERAL);
    end_one_time_commands(d, cmd);
}

// Optimal-tiled image with its own memory.  Uploads `data` if given and
// leaves the image in its resting layout.
void create_optimal_image(const Dev& d, VulkanResource* r, VkFormat format, uint32_t w, uint32_t h,
                          int flags, const void* data, size_t bytes) {
    bool depth = is_depth_format(format);
    VkImageUsageFlags usage = image_usage_for(d, format, flags, VK_IMAGE_TILING_OPTIMAL);

    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    VK_THROW(vkCreateImage(d.device, &ci, nullptr, &image), "vkCreateImage");
    r->image = (void*)image;
    r->owns_image = true;

    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(d.device, image, &reqs);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = pick_memory_type(d.phys, reqs.memoryTypeBits, false);
    VkDeviceMemory memory;
    VK_THROW(vkAllocateMemory(d.device, &ai, nullptr, &memory), "vkAllocateMemory (image)");
    r->memory = (void*)memory;
    r->owns_memory = true;
    r->alloc_size = reqs.size;
    r->memory_type = ai.memoryTypeIndex;
    VK_THROW(vkBindImageMemory(d.device, image, memory, 0), "vkBindImageMemory");

    VkImageAspectFlags aspect = aspect_for(format);
    r->image_view = (void*)create_image_view(d.device, image, format, aspect);
    r->owns_image_view = true;
    r->format = format;
    r->width = w;
    r->height = h;
    r->texel_size = format_texel_size(format);
    r->row_pitch = w * r->texel_size;
    r->aspect = aspect;
    r->image_usage = usage;
    r->layout = resting_layout(usage, depth, false);
    r->flags |= flags_from_usage(usage) | (depth ? kDEPTH : 0);

    VkCommandBuffer cmd = begin_one_time_commands(d);
    if (data && (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        Staging staging(d, bytes);
        memcpy(staging.ptr, data, bytes);
        image_barrier(cmd, image, aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region = full_image_copy(r);
        vkCmdCopyBufferToImage(cmd, staging.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        image_barrier(cmd, image, aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (VkImageLayout)r->layout);
        end_one_time_commands(d, cmd);
    } else {
        image_barrier(cmd, image, aspect, VK_IMAGE_LAYOUT_UNDEFINED, (VkImageLayout)r->layout);
        end_one_time_commands(d, cmd);
    }
}

// ===========================================================================
//  Upload / download
// ===========================================================================

void upload_to_buffer(const Dev& d, VulkanResource* r, const void* data, size_t bytes) {
    if (!data || bytes == 0) return;
    bytes = std::min<size_t>(bytes, r->alloc_size);
    if (r->mapped) {
        vkQueueWaitIdle(d.queue);   // the GPU may still be reading it
        memcpy(r->mapped, data, bytes);
        return;
    }
    Staging staging(d, bytes);
    memcpy(staging.ptr, data, bytes);
    VkCommandBuffer cmd = begin_one_time_commands(d);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, staging.buffer, (VkBuffer)r->buffer, 1, &region);
    end_one_time_commands(d, cmd);
}

// Replace the pixels of an optimal-tiled image (keeps its resting layout).
void upload_to_image(const Dev& d, VulkanResource* r, const void* data, size_t bytes) {
    if (!(r->image_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        throw std::runtime_error("[vulkan] this image format cannot be written on this device");
    }
    size_t image_bytes = (size_t)r->width * r->height * r->texel_size;
    Staging staging(d, image_bytes);
    memcpy(staging.ptr, data, std::min(bytes, image_bytes));
    VkCommandBuffer cmd = begin_one_time_commands(d);
    image_barrier(cmd, (VkImage)r->image, r->aspect, (VkImageLayout)r->layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region = full_image_copy(r);
    vkCmdCopyBufferToImage(cmd, staging.buffer, (VkImage)r->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    image_barrier(cmd, (VkImage)r->image, r->aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (VkImageLayout)r->layout);
    end_one_time_commands(d, cmd);
}

void download(VulkanResource* r, void* out, size_t bytes) {
    Dev d = dev_of(r);
    vkQueueWaitIdle(d.queue);   // finish any rendering / compute into it first
    bytes = std::min<size_t>(bytes, r->data_size ? r->data_size : r->alloc_size);

    if (r->buffer) {
        if (r->mapped) {
            memcpy(out, r->mapped, bytes);
            return;
        }
        Staging staging(d, bytes);
        VkCommandBuffer cmd = begin_one_time_commands(d);
        VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, (VkBuffer)r->buffer, staging.buffer, 1, &region);
        end_one_time_commands(d, cmd);
        memcpy(out, staging.ptr, bytes);
        return;
    }

    if (r->image) {
        if (!(r->image_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            throw std::runtime_error("[vulkan] this image format cannot be read back on this device");
        }
        size_t image_bytes = (size_t)r->width * r->height * r->texel_size;
        Staging staging(d, image_bytes);
        VkCommandBuffer cmd = begin_one_time_commands(d);
        image_barrier(cmd, (VkImage)r->image, r->aspect, (VkImageLayout)r->layout,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region = full_image_copy(r);
        vkCmdCopyImageToBuffer(cmd, (VkImage)r->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.buffer, 1, &region);
        image_barrier(cmd, (VkImage)r->image, r->aspect, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      (VkImageLayout)r->layout);
        end_one_time_commands(d, cmd);
        memcpy(out, staging.ptr, std::min(bytes, image_bytes));
    }
}

// ===========================================================================
//  Allocation, views, destruction
// ===========================================================================

// Destroy the Vulkan objects `r` owns and clear its handles.
void release_contents(VulkanResource* r) {
    VkDevice device = (VkDevice)r->device;
    if (r->owns_buffer_view && r->buffer_view) vkDestroyBufferView(device, (VkBufferView)r->buffer_view, nullptr);
    if (r->owns_image_view && r->image_view)   vkDestroyImageView(device, (VkImageView)r->image_view, nullptr);
    if (r->owns_image && r->image)             vkDestroyImage(device, (VkImage)r->image, nullptr);
    if (r->owns_buffer && r->buffer)           vkDestroyBuffer(device, (VkBuffer)r->buffer, nullptr);
    if (r->owns_memory && r->memory) {
        if (r->mapped) vkUnmapMemory(device, (VkDeviceMemory)r->memory);
        vkFreeMemory(device, (VkDeviceMemory)r->memory, nullptr);
        if (r->fd >= 0) close(r->fd);
    }
    r->buffer_view = r->image_view = r->image = r->buffer = r->memory = r->mapped = nullptr;
    r->owns_buffer_view = r->owns_image_view = r->owns_image = r->owns_buffer = r->owns_memory = false;
    r->fd = -1;
    r->alloc_size = 0;
    r->flags = 0;
    r->exported = false;
}

void destroy_resource(VulkanResource* r) {
    if (!r) return;
    registry_remove(r);
    if (r->device) vkDeviceWaitIdle((VkDevice)r->device);   // it may still be in flight
    release_contents(r);
    delete r;
}

VulkanResource* allocate_resource(const Dev& d, const AllocationMetadata& meta,
                                  const void* data, ComputeType ct) {
    auto* r = new VulkanResource();
    r->device = (void*)d.device;
    r->device_id = d.id;

    int flags = (int)meta.rwstatus;
    VkFormat format = resolve_format(meta, nullptr);
    uint32_t w, h;
    dims_for(meta, w, h);

    r->format = format;
    r->width = w;
    r->height = h;
    r->texel_size = format_texel_size(format) ? format_texel_size(format) : (uint32_t)meta.type_size;
    r->row_pitch = w * r->texel_size;
    r->data_size = meta.byte_size;

    bool depth = is_depth_format(format);
    bool prefer_host = meta.storage_device == MemoryType::kDDR;
    bool want_export = d.external_fd && (meta.storage_device == MemoryType::kHIP_VRAM ||
                                         meta.storage_device == MemoryType::kCUDA_VRAM);
    VkDeviceSize bytes = std::max<VkDeviceSize>(meta.byte_size, 16);

    try {
        if (ct == ComputeType::kVULKAN) {
            // Buffer: always buffer-backed; image / texel views on request.
            create_backing_buffer(d, r, bytes, prefer_host, want_export);
            upload_to_buffer(d, r, data, meta.byte_size);
            if (flags & kTEXELBUFFER) create_buffer_view(d, r, r, format, flags);
            if (wants_image(flags)) create_linear_image(d, r, r, format, w, h, flags);
        } else if (ct == ComputeType::kVULKANTEXTURE) {
            if (format == VK_FORMAT_UNDEFINED) {
                throw std::runtime_error("[vulkan] no image format for element size " +
                                         std::to_string(meta.type_size) + "; pass a VkFormat in metadata.format");
            }
            // Colour images are buffer-backed (linear) unless kOPTIMAL is
            // asked for or the device can't do it — then every view (buffer,
            // texel buffer, HIP/CUDA/CPU pointer) works in place.
            bool linear = false;
            if (!depth && !(flags & kOPTIMAL)) {
                try {
                    create_backing_buffer(d, r, bytes, prefer_host, want_export);
                    upload_to_buffer(d, r, data, meta.byte_size);
                    create_linear_image(d, r, r, format, w, h, flags);
                    if (flags & kTEXELBUFFER) create_buffer_view(d, r, r, format, flags);
                    linear = true;
                } catch (const std::exception& e) {
                    if (flags & kLINEAR) throw;          // explicitly required
                    release_contents(r);                 // fall back to an optimal image
                }
            } else if ((flags & kLINEAR) && depth) {
                std::cerr << "[vulkan] kLINEAR ignored for depth format — using an optimal image" << std::endl;
            }
            if (!linear) create_optimal_image(d, r, format, w, h, flags, data, meta.byte_size);
        } else {
            throw std::runtime_error("[vulkan] allocate_resource: unsupported compute type");
        }
    } catch (...) {
        destroy_resource(r);
        throw;
    }

    registry_add(r);
    return r;
}

// In-place view of `base` as `target` with the view flags in `meta`.
VulkanResource* make_view(VulkanResource* base, const AllocationMetadata& meta, ComputeType target) {
    Dev d = dev_of(base);
    int want = (int)meta.rwstatus & kVIEW_FLAGS;
    VkFormat format = resolve_format(meta, base);
    uint32_t w, h;
    dims_for(meta, w, h);

    auto* v = new VulkanResource();
    v->parent      = base;
    v->device      = base->device;
    v->device_id   = base->device_id;
    v->buffer      = base->buffer;
    v->memory      = base->memory;
    v->fd          = base->fd;
    v->alloc_size  = base->alloc_size;
    v->data_size   = base->data_size;
    v->memory_type = base->memory_type;
    v->mapped      = base->mapped;
    v->exported    = base->exported;
    v->format      = format;
    v->width       = w;
    v->height      = h;
    v->texel_size  = format_texel_size(format) ? format_texel_size(format) : (uint32_t)meta.type_size;
    v->row_pitch   = w * v->texel_size;
    v->flags       = base->flags & kLINEAR;

    try {
        if (target == ComputeType::kVULKANTEXTURE) {
            int image_flags = want & (kTEXTURE | kSURFACE | kSTORAGE);
            if (!image_flags) image_flags = kTEXTURE;

            bool base_image_fits = base->image && base->format == (int32_t)format &&
                                   base->width == w && base->height == h &&
                                   (base->flags & image_flags) == image_flags;

            if (base_image_fits) {
                v->image = base->image;              // share, don't own
                v->image_view = base->image_view;
                v->layout = base->layout;
                v->aspect = base->aspect;
                v->image_usage = base->image_usage;
                v->flags |= base->flags & (kTEXTURE | kSURFACE | kSTORAGE | kDEPTH);
            } else if (base->buffer) {
                create_linear_image(d, v, base, format, w, h, image_flags);
            } else {
                throw std::runtime_error(
                    "[vulkan] this image is optimal-tiled and supports {" + flag_names(base->flags) +
                    "}; its format cannot be used as {" + flag_names(image_flags) + "} on this device");
            }
            if (want & kTEXELBUFFER) {
                if (!base->buffer) throw std::runtime_error("[vulkan] kTEXELBUFFER view needs a buffer-backed texture (not kOPTIMAL)");
                create_buffer_view(d, v, base, format, want);
            }
        } else if (target == ComputeType::kVULKAN) {
            if (!base->buffer) {
                throw std::runtime_error(
                    "[vulkan] this texture is optimal-tiled (kOPTIMAL, a depth format, or a format the "
                    "device can't use linearly) so it has no buffer to view in place");
            }
            if (want & (kTEXELBUFFER | kSTORAGE)) {
                if (base->buffer_view && base->format == (int32_t)format) {
                    v->buffer_view = base->buffer_view;
                    v->flags |= kTEXELBUFFER;
                } else {
                    create_buffer_view(d, v, base, format, want);
                }
            }
        } else {
            throw std::runtime_error("[vulkan] make_view: unsupported target");
        }
    } catch (...) {
        delete v;   // owns nothing that needs registry removal yet
        throw;
    }

    registry_add(v);
    return v;
}

// ===========================================================================
//  Registration on AllocationMaps
// ===========================================================================

// Remember the converter that was installed before ours (per map + key), so
// re-registering replaces our wrapper instead of stacking another one.
template <typename Map, typename Key>
typename Map::mapped_type original_of(Map& m, Key key) {
    static std::map<std::pair<const void*, int>, typename Map::mapped_type> originals;
    auto id = std::make_pair((const void*)&m, (int)key);
    auto it = originals.find(id);
    if (it != originals.end()) return it->second;
    typename Map::mapped_type prev;
    auto found = m.find(key);
    if (found != m.end()) prev = found->second;
    originals[id] = prev;
    return prev;
}

// Mapping deallocator for kVULKAN / kVULKANTEXTURE targets — releases views.
void install_view_releasers(AllocationMap& map) {
    for (ComputeType ct : {ComputeType::kVULKAN, ComputeType::kVULKANTEXTURE}) {
        auto prev = original_of(map.compute_mapping_deallocators, ct);
        map.compute_mapping_deallocators[ct] = [prev](void* ptr, BaseMemoryAllocation* original) {
            if (VulkanResource* v = registry_find(ptr)) {
                if (v->parent) destroy_resource(v);   // a view — owns only its extras
                return;
            }
            if (prev) prev(ptr, original);
        };
    }
}

// Allocators, deallocators and in-place view converters for Vulkan resources
// living on `map`.  `device_id` selects the VkDevice (kRenderingDevice for the
// display's device).  `interop` is the compute type kernels on this memory
// use (kCPU / kHIP / kCUDA).
void register_resource_functions(AllocationMap& map, int device_id, ComputeType interop) {
    map.supports_compute_device[ComputeType::kVULKAN] = true;
    map.supports_compute_device[ComputeType::kVULKANTEXTURE] = true;

    for (ComputeType ct : {ComputeType::kVULKAN, ComputeType::kVULKANTEXTURE}) {
        map.compute_device_allocators[ct] = [device_id, ct](AllocationMetadata meta, void* existing) {
            Dev d = dev_for(device_id);
            VulkanResource* r = allocate_resource(d, meta, existing, ct);
            return new BaseMemoryAllocation(meta, (void*)r);
        };
        map.compute_device_deallocators[ct] = [](void* ptr) {
            destroy_resource(registry_find(ptr));
        };
    }

    // In-place views between the two Vulkan compute types (including the
    // same type with extra view flags).
    for (ComputeType src : {ComputeType::kVULKAN, ComputeType::kVULKANTEXTURE}) {
        for (ComputeType dst : {ComputeType::kVULKAN, ComputeType::kVULKANTEXTURE}) {
            map.compute_type_converters[{src, dst}] =
                [dst](void* ptr, BaseMemoryAllocation*, AllocationMetadata meta) -> void* {
                    VulkanResource* base = registry_find(ptr);
                    if (!base) throw std::runtime_error("[vulkan] view requested on a non-Vulkan allocation");
                    return make_view(base, meta, dst);
                };
        }
    }
    install_view_releasers(map);

    if (interop == ComputeType::kCPU) {
        // Host-visible memory: CPU code reads/writes the Vulkan memory directly.
        for (ComputeType src : {ComputeType::kVULKAN, ComputeType::kVULKANTEXTURE}) {
            map.compute_type_converters[{src, ComputeType::kCPU}] =
                [](void* ptr, BaseMemoryAllocation*, AllocationMetadata) -> void* {
                    VulkanResource* r = registry_find(ptr);
                    if (r && r->buffer && r->mapped) return r->mapped;
                    return ptr;   // not host-mappable: the handle (bind-only, don't write through it)
                };
        }
        auto prev = original_of(map.compute_mapping_deallocators, ComputeType::kCPU);
        map.compute_mapping_deallocators[ComputeType::kCPU] = [prev](void* ptr, BaseMemoryAllocation* original) {
            if (registry_find(original->data)) return;   // persistent mapping, released with the resource
            if (prev) prev(ptr, original);
        };
    } else if (interop == ComputeType::kHIP || interop == ComputeType::kCUDA) {
        // kVULKAN → kHIP/kCUDA is registered by the HIP/CUDA plugin (fd import).
        // Buffer-backed textures reuse it, so kernels can write straight into
        // a kLINEAR texture.  Optimal textures only carry their handle.
        AllocationMap* mp = &map;
        map.compute_type_converters[{ComputeType::kVULKANTEXTURE, interop}] =
            [mp, interop](void* ptr, BaseMemoryAllocation* original, AllocationMetadata meta) -> void* {
                VulkanResource* r = registry_find(ptr);
                auto it = mp->compute_type_converters.find({ComputeType::kVULKAN, interop});
                if (r && r->buffer && r->fd >= 0 && it != mp->compute_type_converters.end()) {
                    void* p = it->second(ptr, original, meta);
                    if (p) return p;
                }
                return ptr;   // handle only — bind it, don't launch kernels on it
            };
        if (map.compute_mapping_deallocators.find(interop) == map.compute_mapping_deallocators.end()) {
            map.compute_mapping_deallocators[interop] = [](void*, BaseMemoryAllocation*) {};
        }
    }
}

BaseMemoryAllocation* download_to_host(VulkanResource* r, AllocationMetadata meta) {
    AllocationMap& host = global_device_manager.get_device(MemoryType::kDDR, 0);
    AllocationMetadata hm = meta;
    hm.storage_device = MemoryType::kDDR;
    hm.compute_device = ComputeType::kCPU;
    hm.rwstatus = AllocationFlags::kRW;
    hm.format = 0;
    BaseMemoryAllocation* out = host.allocate(hm);
    download(r, out->data, hm.byte_size);
    return out;
}

// Transfers between host memory and the Vulkan device's memory type:
//   host → `mem` with a Vulkan compute type  : allocate + upload
//   `mem` Vulkan resource → host             : readback
void install_transfer_converters(MemoryType mem, int device_id) {
    AllocationMap* host = nullptr;
    try { host = &global_device_manager.get_device(MemoryType::kDDR, 0); } catch (...) { return; }

    // host → mem
    {
        auto prev = original_of(host->memory_type_converters, mem);
        host->memory_type_converters[mem] = [prev, device_id](void* ptr, AllocationMetadata meta) -> BaseMemoryAllocation* {
            if (meta.compute_device == ComputeType::kVULKAN || meta.compute_device == ComputeType::kVULKANTEXTURE) {
                Dev d = dev_for(device_id);
                return new BaseMemoryAllocation(meta, (void*)allocate_resource(d, meta, ptr, meta.compute_device));
            }
            if (VulkanResource* r = registry_find(ptr)) return download_to_host(r, meta);
            if (prev) return prev(ptr, meta);
            throw std::runtime_error("[vulkan] no converter from host memory");
        };
    }

    // mem → host
    if (mem != MemoryType::kDDR) {
        try {
            AllocationMap& dev_map = global_device_manager.get_device(mem, 0);
            auto prev = original_of(dev_map.memory_type_converters, MemoryType::kDDR);
            dev_map.memory_type_converters[MemoryType::kDDR] = [prev](void* ptr, AllocationMetadata meta) -> BaseMemoryAllocation* {
                if (VulkanResource* r = registry_find(ptr)) return download_to_host(r, meta);
                if (prev) return prev(ptr, meta);
                throw std::runtime_error("[vulkan] no converter to host memory");
            };
        } catch (...) {}
    }
}

// ===========================================================================
//  Per-plugin-device AllocationMap (kUnknown_MEM) — compute-only use without
//  a display.
// ===========================================================================

AllocationMap* create_vulkan_mapper(int device_id) {
    AllocationMap* mapper = new AllocationMap();
    mapper->device_id = device_id;
    mapper->default_compute_type = ComputeType::kVULKAN;
    mapper->default_allocator_type = ComputeType::kVULKAN;
    mapper->this_device_type = MemoryType::kUnknown_MEM;

    register_resource_functions(*mapper, device_id, ComputeType::kUnknown);

    mapper->memory_type_converters[MemoryType::kDDR] = [](void* ptr, AllocationMetadata meta) {
        VulkanResource* r = registry_find(ptr);
        if (!r) throw std::runtime_error("[vulkan] readback of a non-Vulkan pointer");
        return download_to_host(r, meta);
    };

    mapper->synchronize_function = [device_id]() {
        vkQueueWaitIdle(dev_for(device_id).queue);
    };

    for (MemoryType src : {MemoryType::kDDR, MemoryType::kDISK}) {
        try {
            AllocationMap& m = global_device_manager.get_device(src, 0);
            m.memory_type_converters[MemoryType::kUnknown_MEM] = [mapper](void* ptr, AllocationMetadata meta) {
                return mapper->allocate(meta, ptr);
            };
        } catch (...) {}
    }
    return mapper;
}

// ===========================================================================
//  Vulkan ComputeDeviceBase
// ===========================================================================

ComputeDeviceBase* create_vulkan_compute_device(int device_id) {
    VkPhysicalDevice phys_dev = g_devices[device_id].physical_device;
    ComputeDeviceBase* device = new ComputeDeviceBase();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_dev, &props);
    std::cout << "[vulkan] Initializing device " << device_id << ": " << props.deviceName << std::endl;

    MemoryType mem = memory_type_from_device_name(props.deviceName);
    device->compute_units = props.limits.maxComputeWorkGroupCount[0];
    device->default_memory_type = mem;
    device->supports_memory_location[mem] = true;

    // Make Vulkan allocations possible on the memory type this GPU uses
    // (e.g. kHIP_VRAM for AMD) — they are re-pointed at the display's device
    // once it calls set_rendering_device().
    try {
        auto& mem_device = global_device_manager.get_device(mem, 0);
        register_resource_functions(mem_device, device_id, interop_compute_type(mem));
        if (device_id == 0) install_transfer_converters(mem, device_id);
    } catch (...) {
        std::cerr << "[vulkan] memory device for " << props.deviceName
                  << " not available (plugin for it not loaded)" << std::endl;
    }
    return device;
}

// ===========================================================================
//  Vulkan instance + device enumeration
// ===========================================================================

int count_vulkan_devices() {
    if (g_initialized) return static_cast<int>(g_devices.size());

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Tensor Compute";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "TensorEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&create_info, nullptr, &g_instance) != VK_SUCCESS) {
        std::cerr << "[vulkan] Failed to create Vulkan instance" << std::endl;
        return 0;
    }

    uint32_t device_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &device_count, nullptr));
    if (device_count == 0) {
        std::cerr << "[vulkan] No Vulkan devices found" << std::endl;
        return 0;
    }
    std::vector<VkPhysicalDevice> physical_devices(device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &device_count, physical_devices.data()));

    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_devices[i], &props);
        std::cout << "[vulkan]   Device " << i << ": " << props.deviceName << std::endl;

        VulkanDeviceState state;
        state.physical_device = physical_devices[i];

        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &qf_count, families.data());

        uint32_t family = UINT32_MAX;
        for (uint32_t q = 0; q < qf_count; q++) {
            if (families[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = q; break; }
        }
        if (family == UINT32_MAX) continue;
        state.compute_queue_family = family;

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        VkPhysicalDeviceFeatures features{};
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &features;
        if (vkCreateDevice(physical_devices[i], &dci, nullptr, &state.device) != VK_SUCCESS) continue;
        vkGetDeviceQueue(state.device, family, 0, &state.compute_queue);

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = family;
        VK_CHECK(vkCreateCommandPool(state.device, &pool_info, nullptr, &state.command_pool));

        g_devices.push_back(state);
    }

    g_initialized = true;
    std::cout << "[vulkan] Found " << g_devices.size() << " usable Vulkan devices" << std::endl;
    return static_cast<int>(g_devices.size());
}

// Index into g_devices of the same GPU as `phys` (from another VkInstance),
// matched by device UUID.
int match_plugin_device(VkPhysicalDevice phys, int fallback) {
    auto uuid_of = [](VkPhysicalDevice p, uint8_t out[VK_UUID_SIZE]) {
        VkPhysicalDeviceIDProperties id{};
        id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &id;
        vkGetPhysicalDeviceProperties2(p, &p2);
        memcpy(out, id.deviceUUID, VK_UUID_SIZE);
    };
    uint8_t want[VK_UUID_SIZE];
    uuid_of(phys, want);
    for (size_t i = 0; i < g_devices.size(); i++) {
        uint8_t have[VK_UUID_SIZE];
        uuid_of(g_devices[i].physical_device, have);
        if (memcmp(want, have, VK_UUID_SIZE) == 0) return (int)i;
    }
    return fallback;
}

} // anonymous namespace

// ===========================================================================
//  Plugin C ABI
// ===========================================================================

extern "C" const char* plugin_name() { return "vulkan"; }

extern "C" int plugin_priority() { return 50; }

extern "C" void plugin_register(DeviceManager* dm) {
    int count = count_vulkan_devices();
    if (count == 0) return;

    for (int i = 0; i < count; i++) {
        dm->register_memory_device(MemoryType::kUnknown_MEM, i, create_vulkan_mapper(i));
    }
    for (int i = 0; i < count; i++) {
        dm->register_compute_device(ComputeType::kVULKAN, i, create_vulkan_compute_device(i));
    }
}

// Devices are registered in plugin_register; nothing is deferred.
extern "C" void plugin_init(DeviceManager*) {}

extern "C" VulkanResource* hvml_vk_find_resource(const void* ptr) {
    return registry_find(ptr);
}

// Wrap an image created elsewhere (a swapchain image) as a resource so it can
// back a tensor.  The resource owns only the image view it creates.
extern "C" VulkanResource* hvml_vk_wrap_image(void* image, int format, uint32_t width, uint32_t height,
                                              int resting_layout, uint32_t usage) {
    try {
        Dev d = rendering_dev();
        auto* r = new VulkanResource();
        r->device = (void*)d.device;
        r->device_id = kRenderingDevice;
        r->image = image;
        r->format = format;
        r->width = width;
        r->height = height;
        r->texel_size = format_texel_size((VkFormat)format);
        r->row_pitch = width * r->texel_size;
        r->data_size = (unsigned long long)width * height * r->texel_size;
        r->aspect = aspect_for((VkFormat)format);
        r->image_usage = usage;
        r->layout = resting_layout;
        r->flags = flags_from_usage(usage);
        r->image_view = (void*)create_image_view(d.device, (VkImage)image, (VkFormat)format, r->aspect);
        r->owns_image_view = true;
        registry_add(r);
        return r;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return nullptr;
    }
}

// Copy host bytes into a resource (buffer-backed: straight into its memory;
// optimal image: staged copy).  Returns 0 on success.
extern "C" int hvml_vk_upload(VulkanResource* r, const void* data, size_t bytes) {
    try {
        r = registry_find(r);
        if (!r) return -1;
        Dev d = dev_of(r);
        if (r->buffer) upload_to_buffer(d, r, data, bytes);
        else if (r->image) upload_to_image(d, r, data, bytes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}

// Read a resource back into host memory.  Returns 0 on success.
extern "C" int hvml_vk_download(VulkanResource* r, void* out, size_t bytes) {
    try {
        r = registry_find(r);
        if (!r) return -1;
        download(r, out, bytes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}

extern "C" int get_rendering_device_index() {
    return g_rendering_device.active ? g_rendering_device.device_index : -1;
}

extern "C" int get_rendering_device_memory_type() {
    return (int)g_rendering_device.memory_type;
}

// Called by the display layer's VulkanContext after it created its device.
// From then on every kVULKAN / kVULKANTEXTURE allocation on the GPU's memory
// type (e.g. kHIP_VRAM) is created on that device.
extern "C" void hvml_vk_set_rendering_device(const HvmlVkDeviceInfo* info) {
    g_rendering_device.device                = (VkDevice)info->device;
    g_rendering_device.physical_device       = (VkPhysicalDevice)info->physical_device;
    g_rendering_device.graphics_queue        = (VkQueue)info->queue;
    g_rendering_device.graphics_queue_family = info->queue_family;
    g_rendering_device.command_pool          = (VkCommandPool)info->command_pool;
    g_rendering_device.external_memory_fd    = info->external_memory_fd != 0;
    g_rendering_device.device_index          = match_plugin_device(g_rendering_device.physical_device, 0);
    g_rendering_device.active                = true;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_rendering_device.physical_device, &props);
    MemoryType mem = memory_type_from_device_name(props.deviceName);
    g_rendering_device.memory_type = mem;

    std::cout << "[vulkan] Rendering device: " << props.deviceName
              << " (plugin device " << g_rendering_device.device_index
              << ", memory " << mem << ")" << std::endl;

    try {
        AllocationMap& render_mem = global_device_manager.get_device(mem, 0);
        register_resource_functions(render_mem, kRenderingDevice, interop_compute_type(mem));
        install_transfer_converters(mem, kRenderingDevice);
        // Host-visible Vulkan buffers (uniform blocks, staging-free uploads):
        // kVULKAN tensors on kDDR are mapped for the CPU.
        if (mem != MemoryType::kDDR) {
            register_resource_functions(global_device_manager.get_device(MemoryType::kDDR, 0),
                                        kRenderingDevice, ComputeType::kCPU);
        }
    } catch (const std::exception& e) {
        std::cerr << "[vulkan] could not register on rendering memory device: " << e.what() << std::endl;
    }
}

// Legacy entry point (kept for older display code).
extern "C" void set_rendering_device(VkDevice device, VkPhysicalDevice physical_device,
                                     VkQueue graphics_queue, uint32_t graphics_queue_family,
                                     VkCommandPool command_pool, int /*device_index*/) {
    HvmlVkDeviceInfo info;
    info.device = (void*)device;
    info.physical_device = (void*)physical_device;
    info.queue = (void*)graphics_queue;
    info.queue_family = graphics_queue_family;
    info.command_pool = (void*)command_pool;
    info.external_memory_fd = 1;
    hvml_vk_set_rendering_device(&info);
}
