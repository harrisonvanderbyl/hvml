// vulkan_plugin.cpp — Vulkan backend plugin
//
// Built with:  g++ -std=c++20 -fPIC -shared
//                  -I. -I../
//                  -o plugins/vulkan/libvulkan_plugin.so vulkan_plugin.cpp
//                  -lvulkan
//
// Priority 50 — loaded after CPU/Disk/CUDA/HIP so it can register converters
// on the CPU AllocationMap.

#include "plugin.hpp"

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <unistd.h>

#define VK_CHECK(call)                                                         \
    do {                                                                       \
        VkResult result = call;                                                \
        if (result != VK_SUCCESS) {                                            \
            std::cerr << "[vulkan] Vulkan error at " << __FILE__ << ":"        \
                      << __LINE__ << " - Result: " << result << std::endl;     \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  Per-plugin state (lives for the lifetime of the .so)
// ---------------------------------------------------------------------------

namespace {

struct VulkanDeviceState {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice         device          = VK_NULL_HANDLE;
    VkQueue          compute_queue   = VK_NULL_HANDLE;
    uint32_t         compute_queue_family = 0;
    VkCommandPool    command_pool    = VK_NULL_HANDLE;
};

VkInstance               g_instance          = VK_NULL_HANDLE;
bool                     g_initialized       = false;
std::vector<VulkanDeviceState> g_devices;

// ---------------------------------------------------------------------------
//  Global rendering device — set by plugin_init() from VulkanContext.
//  When set, all display-layer allocations (kVULKAN, kVULKANTEXTURE) use
//  this device instead of the plugin's own g_devices[].  This ensures
//  that VkImages/VkBuffers created through the tensor system are on the
//  same VkDevice as the render pass, swapchain, and pipelines.
//
//  `memory_type` is the MemoryType of the physical device (kCUDA_VRAM,
//  kHIP_VRAM, or kDDR) so allocations land on the correct AllocationMap.
//  `device_index` is the index into g_devices[] for the matching compute
//  device (used for fallback when rendering_device is not set).
// ---------------------------------------------------------------------------

struct RenderingDevice {
    VkDevice         device           = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device  = VK_NULL_HANDLE;
    VkQueue          graphics_queue   = VK_NULL_HANDLE;
    uint32_t         graphics_queue_family = 0;
    VkCommandPool    command_pool     = VK_NULL_HANDLE;
    MemoryType       memory_type      = MemoryType::kUnknown_MEM;
    int              device_index     = -1;  // index into g_devices[]
    bool             active           = false;
};

RenderingDevice g_rendering_device;

// (VulkanBufferHandle is defined in plugin.hpp — shared with the HIP plugin)

// Access the correct VkDevice for a given device_id:
//  - If g_rendering_device is active and device_id matches, use it
//  - Otherwise fall back to g_devices[device_id]
VkDevice get_vk_device(int device_id) {
    if (g_rendering_device.active && device_id == g_rendering_device.device_index) {
        return g_rendering_device.device;
    }
    if (device_id >= 0 && device_id < (int)g_devices.size()) {
        return g_devices[device_id].device;
    }
    return g_rendering_device.device;
}

VkPhysicalDevice get_vk_physical_device(int device_id) {
    if (g_rendering_device.active && device_id == g_rendering_device.device_index) {
        return g_rendering_device.physical_device;
    }
    if (device_id >= 0 && device_id < (int)g_devices.size()) {
        return g_devices[device_id].physical_device;
    }
    return g_rendering_device.physical_device;
}

VkQueue get_vk_queue(int device_id) {
    if (g_rendering_device.active && device_id == g_rendering_device.device_index) {
        return g_rendering_device.graphics_queue;
    }
    if (device_id >= 0 && device_id < (int)g_devices.size()) {
        return g_devices[device_id].compute_queue;
    }
    return g_rendering_device.graphics_queue;
}

VkCommandPool get_vk_command_pool(int device_id) {
    if (g_rendering_device.active && device_id == g_rendering_device.device_index) {
        return g_rendering_device.command_pool;
    }
    if (device_id >= 0 && device_id < (int)g_devices.size()) {
        return g_devices[device_id].command_pool;
    }
    return g_rendering_device.command_pool;
}

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    std::cerr << "[vulkan] Failed to find suitable memory type!" << std::endl;
    return 0;
}

// Determine the MemoryType from the device name so we can piggy-back on the
// existing CUDA_VRAM / HIP_VRAM / DDR AllocationMaps created by earlier plugins.
MemoryType memory_type_from_device_name(const char* name) {
    if (strstr(name, "NVIDIA") != nullptr) {
        return MemoryType::kCUDA_VRAM;
    } else if (strstr(name, "AMD") != nullptr || strstr(name, "ATI") != nullptr) {
        return MemoryType::kHIP_VRAM;
    }
    // Intel, llvmpipe, SwiftShader, etc. → host-visible DDR
    return MemoryType::kDDR;
}

// Create a one-shot command buffer for copy operations
VkCommandBuffer begin_one_time_commands(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd, &begin_info);
    return cmd;
}

void end_one_time_commands(VkDevice device, VkQueue queue, VkCommandPool pool,
                           VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

// ---------------------------------------------------------------------------
//  Vulkan AllocationMap
// ---------------------------------------------------------------------------

AllocationMap* create_vulkan_mapper(int device_id) {
    AllocationMap* mapper = new AllocationMap();
    mapper->device_id = device_id;
    mapper->default_compute_type = ComputeType::kVULKAN;
    mapper->default_allocator_type = ComputeType::kVULKAN;
    mapper->supports_compute_device[ComputeType::kVULKAN] = true;

    // Allocator — allocate device-local VkBuffer (vertex/index/storage)
    // Stores VkBuffer* in BaseMemoryAllocation::data
    mapper->compute_device_allocators[ComputeType::kVULKAN] = [device_id](AllocationMetadata metadata, void* existing_data) {
        VkDevice vk_device = get_vk_device(device_id);
        VkPhysicalDevice physical_device = get_vk_physical_device(device_id);
        VkQueue queue = get_vk_queue(device_id);
        VkCommandPool pool = get_vk_command_pool(device_id);

        // Create the device-local VkBuffer
        VkBufferCreateInfo bufferCI{};
        bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCI.size = metadata.byte_size;
        bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        VK_CHECK(vkCreateBuffer(vk_device, &bufferCI, nullptr, &buffer));

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(vk_device, buffer, &memReqs);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            physical_device, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkDeviceMemory bufferMemory;
        VK_CHECK(vkAllocateMemory(vk_device, &allocInfo, nullptr, &bufferMemory));
        vkBindBufferMemory(vk_device, buffer, bufferMemory, 0);

        // If existing data provided, stage-copy it into the buffer
        if (existing_data) {
            VkDeviceSize size = metadata.byte_size;

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;

            VkBufferCreateInfo stagingCI{};
            stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagingCI.size = size;
            stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            stagingCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            vkCreateBuffer(vk_device, &stagingCI, nullptr, &stagingBuffer);

            VkMemoryRequirements stagingReqs;
            vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &stagingReqs);

            VkMemoryAllocateInfo stagingAlloc{};
            stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            stagingAlloc.allocationSize = stagingReqs.size;
            stagingAlloc.memoryTypeIndex = find_memory_type(
                physical_device, stagingReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(vk_device, &stagingAlloc, nullptr, &stagingMemory);
            vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);

            void* mapped = nullptr;
            vkMapMemory(vk_device, stagingMemory, 0, size, 0, &mapped);
            memcpy(mapped, existing_data, (size_t)size);
            vkUnmapMemory(vk_device, stagingMemory);

            VkCommandBuffer cmd = begin_one_time_commands(vk_device, pool);
            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
            end_one_time_commands(vk_device, queue, pool, cmd);

            vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
            vkFreeMemory(vk_device, stagingMemory, nullptr);
        }

        // Store VkBuffer* as the allocation data
        VkBuffer* handle = new VkBuffer(buffer);
        return new BaseMemoryAllocation(metadata, handle);
    };

    // Deallocator — destroy VkBuffer + VkDeviceMemory
    mapper->compute_device_deallocators[ComputeType::kVULKAN] = [device_id](void* ptr) {
        VkBuffer buffer = *(VkBuffer*)ptr;
        // We don't track the VkDeviceMemory separately — it leaks.
        // In production, store a struct {VkBuffer, VkDeviceMemory} instead.
        vkDestroyBuffer(get_vk_device(device_id), buffer, nullptr);
        delete (VkBuffer*)ptr;
    };

    // Converter: Vulkan VkBuffer → host DDR
    mapper->memory_type_converters[MemoryType::kDDR] = [device_id](void* ptr, AllocationMetadata meta) {
        VkDevice vk_device = get_vk_device(device_id);
        VkPhysicalDevice physical_device = get_vk_physical_device(device_id);
        VkQueue queue = get_vk_queue(device_id);
        VkCommandPool pool = get_vk_command_pool(device_id);
        VkBuffer buffer = *(VkBuffer*)ptr;

        auto& host_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        auto host_ptr = host_device.allocate(meta);

        // Create a staging buffer to receive the data
        VkBuffer staging_buffer;
        VkDeviceMemory staging_memory;

        VkBufferCreateInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = meta.byte_size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(vk_device, &buffer_info, nullptr, &staging_buffer);

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(vk_device, staging_buffer, &mem_reqs);

        VkMemoryAllocateInfo staging_alloc = {};
        staging_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        staging_alloc.allocationSize = mem_reqs.size;
        staging_alloc.memoryTypeIndex = find_memory_type(
            physical_device, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkAllocateMemory(vk_device, &staging_alloc, nullptr, &staging_memory);
        vkBindBufferMemory(vk_device, staging_buffer, staging_memory, 0);

        // Copy buffer → staging (buffer is already a VkBuffer, no need for src_buffer)
        VkCommandBuffer cmd = begin_one_time_commands(vk_device, pool);

        VkBufferCopy copy_region = {};
        copy_region.size = meta.byte_size;
        vkCmdCopyBuffer(cmd, buffer, staging_buffer, 1, &copy_region);

        end_one_time_commands(vk_device, queue, pool, cmd);

        // Map staging and copy to host
        void* mapped = nullptr;
        vkMapMemory(vk_device, staging_memory, 0, meta.byte_size, 0, &mapped);
        memcpy(host_ptr->data, mapped, (size_t)meta.byte_size);
        vkUnmapMemory(vk_device, staging_memory);

        vkDestroyBuffer(vk_device, staging_buffer, nullptr);
        vkFreeMemory(vk_device, staging_memory, nullptr);

        return host_ptr;
    };

    // Synchronize
    mapper->synchronize_function = [device_id]() {
        vkQueueWaitIdle(get_vk_queue(device_id));
    };

    // Register converters on CPU + Disk devices (created earlier)
    try {
        AllocationMap& mem_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        mem_device.memory_type_converters[MemoryType::kUnknown_MEM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr);
        };
        std::cout << "[vulkan] Registered Vulkan converter for DDR memory" << std::endl;
    } catch (...) {
        std::cerr << "[vulkan] CPU device not available for converter registration" << std::endl;
    }

    try {
        AllocationMap& mem_device_disk = global_device_manager.get_device(MemoryType::kDISK, 0);
        mem_device_disk.memory_type_converters[MemoryType::kUnknown_MEM] = [mapper](void* ptr, AllocationMetadata meta) {
            return mapper->allocate(meta, ptr);
        };
        std::cout << "[vulkan] Registered Vulkan converter for DISK memory" << std::endl;
    } catch (...) {}

    // The Vulkan plugin uses kUnknown_MEM as its own memory type since there
    // is no dedicated VULKAN_VRAM enum.  Device-local allocations live in
    // VkDeviceMemory and are managed entirely through this AllocationMap.
    mapper->this_device_type = MemoryType::kUnknown_MEM;
    return mapper;
}

// ---------------------------------------------------------------------------
//  Vulkan ComputeDeviceBase
// ---------------------------------------------------------------------------

ComputeDeviceBase* create_vulkan_compute_device(int device_id) {
    VkPhysicalDevice phys_dev = get_vk_physical_device(device_id);

    ComputeDeviceBase* device = new ComputeDeviceBase();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_dev, &props);

    std::cout << "[vulkan] Initializing device " << device_id << ": "
              << props.deviceName << std::endl;

    MemoryType mem = memory_type_from_device_name(props.deviceName);

    device->compute_units = props.limits.maxComputeWorkGroupCount[0];
    device->default_memory_type = mem;
    device->supports_memory_location[mem] = true;

    // Register Vulkan compute support on the memory device
    try {
        auto& mem_device = global_device_manager.get_device(mem, 0);
        mem_device.supports_compute_device[ComputeType::kVULKAN] = true;

        // If the memory device is DDR (integrated GPU), register a host-visible
        // Vulkan allocator so we can allocate Vulkan-managed host memory.
        if (mem == MemoryType::kDDR) {
            mem_device.compute_device_allocators[ComputeType::kVULKAN] = [device_id](AllocationMetadata metadata, void* existing_data) {
                VkDevice vk_device = get_vk_device(device_id);
                VkPhysicalDevice physical_device = get_vk_physical_device(device_id);

                VkDeviceMemory* device_memory = new VkDeviceMemory();

                VkMemoryAllocateInfo alloc_info = {};
                alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                alloc_info.allocationSize = metadata.byte_size;
                alloc_info.memoryTypeIndex = find_memory_type(
                    physical_device, UINT32_MAX,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                VK_CHECK(vkAllocateMemory(vk_device, &alloc_info, nullptr, device_memory));

                if (existing_data) {
                    void* mapped = nullptr;
                    vkMapMemory(vk_device, *device_memory, 0, metadata.byte_size, 0, &mapped);
                    memcpy(mapped, existing_data, (size_t)metadata.byte_size);
                    vkUnmapMemory(vk_device, *device_memory);
                }

                return new BaseMemoryAllocation(metadata, device_memory);
            };

            mem_device.compute_device_deallocators[ComputeType::kVULKAN] = [device_id](void* ptr) {
                VkDeviceMemory device_memory = *(VkDeviceMemory*)ptr;
                vkFreeMemory(get_vk_device(device_id), device_memory, nullptr);
                delete (VkDeviceMemory*)ptr;
            };

            // Vulkan ↔ CPU conversion (host-visible memory is directly mappable)
            mem_device.compute_type_converters[{ComputeType::kVULKAN, ComputeType::kCPU}] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                VkDevice vk_device = get_vk_device(device_id);
                VkDeviceMemory device_memory = *(VkDeviceMemory*)ptr;
                void* mapped = nullptr;
                vkMapMemory(vk_device, device_memory, 0, metadata.byte_size, 0, &mapped);
                return mapped;
            };

            mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [device_id](void* ptr, BaseMemoryAllocation* original) {
                VkDevice vk_device = get_vk_device(device_id);
                VkDeviceMemory device_memory = *(VkDeviceMemory*)original->data;
                vkUnmapMemory(vk_device, device_memory);
            };

            mem_device.compute_type_converters[{ComputeType::kCPU, ComputeType::kVULKAN}] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                // CPU pointer is already a mapped view of the Vulkan memory
                return ptr;
            };

            mem_device.compute_mapping_deallocators[ComputeType::kVULKAN] = [device_id](void* ptr, BaseMemoryAllocation* original) {
                // Nothing to do — the mapping was created by the kVULKAN→kCPU converter
            };
        }

        std::cout << "[vulkan] Created Vulkan device " << device_id << std::endl;
    } catch (...) {
        std::cerr << "[vulkan] Failed to register Vulkan device " << device_id
                  << " with memory manager" << std::endl;
    }

    // Register kVULKANTEXTURE support on the memory device (for all device types)
    try {
        auto& mem_device = global_device_manager.get_device(mem, 0);
        mem_device.supports_compute_device[ComputeType::kVULKANTEXTURE] = true;

        mem_device.compute_device_allocators[ComputeType::kVULKANTEXTURE] = [device_id](AllocationMetadata metadata, void* existing_data) {
            VkDevice vk_device = get_vk_device(device_id);
            VkPhysicalDevice physical_device = get_vk_physical_device(device_id);

            VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
            if (metadata.type_size == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
            else if (metadata.type_size == 4) format = VK_FORMAT_R8G8B8A8_UNORM;
            else if (metadata.type_size == 6) format = VK_FORMAT_R16G16B16_SFLOAT;
            else if (metadata.type_size == 3) format = VK_FORMAT_R8G8B8_UNORM;
            if (metadata.format == 1) format = VK_FORMAT_D32_SFLOAT;

            uint32_t width = (uint32_t)metadata.shape.A;
            uint32_t height = (uint32_t)(metadata.shape.total_size() / std::max(1UL, (size_t)metadata.shape.A));

            VkImageCreateInfo imageCI{};
            imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.format = format;
            imageCI.extent = {width, height, 1};
            imageCI.mipLevels = 1;
            imageCI.arrayLayers = 1;
            imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (format == VK_FORMAT_D32_SFLOAT) {
                imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            }
            imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VkImage image;
            VK_CHECK(vkCreateImage(vk_device, &imageCI, nullptr, &image));

            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(vk_device, image, &memReqs);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memReqs.size;

            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(physical_device, &memProps);
            uint32_t memTypeIndex = 0;
            for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
                if ((memReqs.memoryTypeBits & (1 << i)) &&
                    (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    memTypeIndex = i;
                    break;
                }
            }
            allocInfo.memoryTypeIndex = memTypeIndex;

            VkDeviceMemory imageMemory;
            VK_CHECK(vkAllocateMemory(vk_device, &allocInfo, nullptr, &imageMemory));
            vkBindImageMemory(vk_device, image, imageMemory, 0);

            VkImageViewCreateInfo viewCI{};
            viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewCI.image = image;
            viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewCI.format = format;
            viewCI.subresourceRange.aspectMask = (format == VK_FORMAT_D32_SFLOAT) ?
                VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            viewCI.subresourceRange.baseMipLevel = 0;
            viewCI.subresourceRange.levelCount = 1;
            viewCI.subresourceRange.baseArrayLayer = 0;
            viewCI.subresourceRange.layerCount = 1;

            VkImageView view;
            VK_CHECK(vkCreateImageView(vk_device, &viewCI, nullptr, &view));

            return new BaseMemoryAllocation(metadata, reinterpret_cast<void*>(view));
        };

        mem_device.compute_device_deallocators[ComputeType::kVULKANTEXTURE] = [device_id](void* ptr) {
            VkImageView view = (VkImageView)ptr;
            vkDestroyImageView(get_vk_device(device_id), view, nullptr);
        };

        // kVULKANTEXTURE → kVULKAN: identity conversion (VkImageView is already usable)
        mem_device.compute_type_converters[{ComputeType::kVULKANTEXTURE, ComputeType::kVULKAN}] =
            [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                return ptr;  // VkImageView is already the correct handle
            };

        mem_device.compute_mapping_deallocators[ComputeType::kVULKAN] =
            [device_id](void* ptr, BaseMemoryAllocation* original) {
                // Nothing to do — no mapping was created
            };

        // kVULKANTEXTURE → kCPU: return the VkImageView as a pointer (for Material binding)
        mem_device.compute_type_converters[{ComputeType::kVULKANTEXTURE, ComputeType::kCPU}] =
            [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                return ptr;  // VkImageView cast to void*
            };

        mem_device.compute_mapping_deallocators[ComputeType::kCPU] =
            [device_id](void* ptr, BaseMemoryAllocation* original) {
                // Nothing to do
            };

        std::cout << "[vulkan] Registered kVULKANTEXTURE on device " << device_id << std::endl;
    } catch (...) {
        std::cerr << "[vulkan] Failed to register kVULKANTEXTURE for device " << device_id << std::endl;
    }

    return device;
}

// ---------------------------------------------------------------------------
//  Vulkan instance + device enumeration
// ---------------------------------------------------------------------------

int count_vulkan_devices() {
    if (g_initialized) {
        return static_cast<int>(g_devices.size());
    }

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Tensor Compute";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "TensorEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = 0;
    create_info.enabledExtensionCount = 0;

    VkResult result = vkCreateInstance(&create_info, nullptr, &g_instance);
    if (result != VK_SUCCESS) {
        std::cerr << "[vulkan] Failed to create Vulkan instance: " << result << std::endl;
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

    std::cout << "[vulkan] Device Count: " << device_count << std::endl;

    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_devices[i], &props);
        std::cout << "[vulkan]   Device " << i << ": " << props.deviceName << std::endl;

        VulkanDeviceState state;
        state.physical_device = physical_devices[i];

        // Find a compute queue family
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_count, queue_families.data());

        uint32_t compute_queue_family = UINT32_MAX;
        for (uint32_t q = 0; q < queue_family_count; q++) {
            if (queue_families[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_family = q;
                break;
            }
        }

        if (compute_queue_family == UINT32_MAX) {
            std::cerr << "[vulkan] No compute queue family found for device " << i << std::endl;
            continue;
        }

        state.compute_queue_family = compute_queue_family;

        // Create logical device
        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_info = {};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = compute_queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;

        VkPhysicalDeviceFeatures device_features = {};

        VkDeviceCreateInfo device_create_info = {};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.queueCreateInfoCount = 1;
        device_create_info.pQueueCreateInfos = &queue_create_info;
        device_create_info.enabledExtensionCount = 0;
        device_create_info.ppEnabledExtensionNames = nullptr;
        device_create_info.pEnabledFeatures = &device_features;

        VK_CHECK(vkCreateDevice(physical_devices[i], &device_create_info, nullptr, &state.device));

        vkGetDeviceQueue(state.device, compute_queue_family, 0, &state.compute_queue);

        // Create command pool for copy operations
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = compute_queue_family;

        VK_CHECK(vkCreateCommandPool(state.device, &pool_info, nullptr, &state.command_pool));

        g_devices.push_back(state);
    }

    g_initialized = true;
    std::cout << "[vulkan] Found " << g_devices.size() << " usable Vulkan devices" << std::endl;
    return static_cast<int>(g_devices.size());
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  Plugin C ABI
// ---------------------------------------------------------------------------

extern "C" const char* plugin_name() {
    return "vulkan";
}

extern "C" int plugin_priority() {
    return 50;
}

extern "C" void plugin_register(DeviceManager* dm) {
    int count = count_vulkan_devices();
    if (count == 0) return;

    for (int i = 0; i < count; i++) {
        AllocationMap* mapper = create_vulkan_mapper(i);
        dm->register_memory_device(MemoryType::kUnknown_MEM, i, mapper);
    }
    for (int i = 0; i < count; i++) {
        ComputeDeviceBase* dev = create_vulkan_compute_device(i);
        dm->register_compute_device(ComputeType::kVULKAN, i, dev);
    }
}

// Deferred init — the display layer calls dm->init_plugin("vulkan") after
// creating the Vulkan context.  Since the plugin already registered devices
// in plugin_register(), this is a no-op (devices are already available).
extern "C" void plugin_init(DeviceManager* dm) {
    // Already initialized in plugin_register — nothing to do.
}

// Called by display layer to get the device index of the rendering device.
// Returns -1 if not set.
extern "C" int get_rendering_device_index() {
    if (g_rendering_device.active) return g_rendering_device.device_index;
    return -1;
}

// Called by display layer to get the memory type of the rendering device.
extern "C" int get_rendering_device_memory_type() {
    return (int)g_rendering_device.memory_type;
}

// Called by VulkanContext (display layer) to share its VkDevice with the
// plugin.  After this call, all kVULKAN/kVULKANTEXTURE allocations go
// through the rendering device, ensuring they're on the same VkDevice as
// the swapchain, render pass, and pipelines.
//
// `device_index` is the index into g_devices[] that matches the physical
// device used by VulkanContext.  This lets us look up the MemoryType.
extern "C" void set_rendering_device(
    VkDevice         device,
    VkPhysicalDevice physical_device,
    VkQueue          graphics_queue,
    uint32_t         graphics_queue_family,
    VkCommandPool    command_pool,
    int              device_index)
{
    g_rendering_device.device           = device;
    g_rendering_device.physical_device  = physical_device;
    g_rendering_device.graphics_queue   = graphics_queue;
    g_rendering_device.graphics_queue_family = graphics_queue_family;
    g_rendering_device.command_pool     = command_pool;
    g_rendering_device.active           = true;

    // Find the correct plugin device index by matching the physical device handle
    g_rendering_device.device_index = -1;
    for (size_t i = 0; i < g_devices.size(); i++) {
        if (g_devices[i].physical_device == physical_device) {
            g_rendering_device.device_index = (int)i;
            break;
        }
    }
    if (g_rendering_device.device_index < 0) {
        // Fallback: use the passed-in index
        g_rendering_device.device_index = device_index;
    }

    // Determine memory type from physical device name
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);
    g_rendering_device.memory_type = memory_type_from_device_name(props.deviceName);

    std::cout << "[vulkan] Rendering device set to: " << props.deviceName
              << " (device_index=" << g_rendering_device.device_index
              << ", memory_type=" << (int)g_rendering_device.memory_type << ")"
              << std::endl;

    // Register a DDR → rendering_memory_type converter that creates a VkImage,
    // uploads the CPU data, and returns a BaseMemoryAllocation holding the VkImageView.
    // This allows tensor.to(memoryType, kVULKANTEXTURE) to work from CPU data.
    try {
        // Register DDR → renderMem memory type converter.
        // Only register for kVULKAN and kVULKANTEXTURE compute types.
        // For kHIP, the HIP plugin's own converter (hipMalloc) should be used.
        // We achieve this by wrapping: if the compute type is kHIP, delegate to
        // the previously-registered converter (from the HIP plugin).
        auto& ddr_device = global_device_manager.get_device(MemoryType::kDDR, 0);
        MemoryType renderMem = g_rendering_device.memory_type;

        // Save the existing converter (registered by the HIP plugin) so we can
        // delegate to it for non-Vulkan compute types.
        auto existingConverter = ddr_device.memory_type_converters[renderMem];

        // Register {kCPU, kVULKAN} compute type converter on the DDR device.
        // This is needed when to() takes the to_compute() path (non-standard strides).
        // It creates a VkBuffer, copies the CPU data via staging, and returns the VkBuffer*.
        ddr_device.compute_type_converters[{ComputeType::kCPU, ComputeType::kVULKAN}] =
            [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                VkDevice vk_device = g_rendering_device.device;
                VkPhysicalDevice physical_device = g_rendering_device.physical_device;
                VkQueue queue = g_rendering_device.graphics_queue;
                VkCommandPool pool = g_rendering_device.command_pool;

                VkBufferCreateInfo bufferCI{};
                bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferCI.size = metadata.byte_size;
                bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VkBuffer buffer;
                VK_CHECK(vkCreateBuffer(vk_device, &bufferCI, nullptr, &buffer));

                VkMemoryRequirements memReqs;
                vkGetBufferMemoryRequirements(vk_device, buffer, &memReqs);

                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memReqs.size;
                allocInfo.memoryTypeIndex = find_memory_type(
                    physical_device, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                VkDeviceMemory bufferMemory;
                VK_CHECK(vkAllocateMemory(vk_device, &allocInfo, nullptr, &bufferMemory));
                vkBindBufferMemory(vk_device, buffer, bufferMemory, 0);

                // Stage-copy CPU data into the buffer
                if (ptr) {
                    VkDeviceSize size = metadata.byte_size;
                    VkBuffer stagingBuffer;
                    VkDeviceMemory stagingMemory;

                    VkBufferCreateInfo stagingCI{};
                    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    stagingCI.size = size;
                    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    stagingCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    vkCreateBuffer(vk_device, &stagingCI, nullptr, &stagingBuffer);

                    VkMemoryRequirements stagingReqs;
                    vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &stagingReqs);

                    VkMemoryAllocateInfo stagingAlloc{};
                    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    stagingAlloc.allocationSize = stagingReqs.size;
                    stagingAlloc.memoryTypeIndex = find_memory_type(
                        physical_device, stagingReqs.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    vkAllocateMemory(vk_device, &stagingAlloc, nullptr, &stagingMemory);
                    vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);

                    void* mapped = nullptr;
                    vkMapMemory(vk_device, stagingMemory, 0, size, 0, &mapped);
                    memcpy(mapped, ptr, (size_t)size);
                    vkUnmapMemory(vk_device, stagingMemory);

                    VkCommandBuffer cmd = begin_one_time_commands(vk_device, pool);
                    VkBufferCopy copyRegion{};
                    copyRegion.size = size;
                    vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
                    end_one_time_commands(vk_device, queue, pool, cmd);

                    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
                    vkFreeMemory(vk_device, stagingMemory, nullptr);
                }

                VulkanBufferHandle* handle = new VulkanBufferHandle();
                handle->buffer = (void*)buffer;
                handle->memory = (void*)bufferMemory;
                handle->fd = -1;  // no fd export for to_compute path
                handle->alloc_size = memReqs.size;
                return reinterpret_cast<void*>(handle);
            };
        ddr_device.compute_mapping_deallocators[ComputeType::kVULKAN] =
            [](void* ptr, BaseMemoryAllocation* original) {
                VulkanBufferHandle* handle = (VulkanBufferHandle*)ptr;
                vkDestroyBuffer(g_rendering_device.device, (VkBuffer)handle->buffer, nullptr);
                vkFreeMemory(g_rendering_device.device, (VkDeviceMemory)handle->memory, nullptr);
                delete handle;
            };

        ddr_device.memory_type_converters[renderMem] = [existingConverter](void* ptr, AllocationMetadata meta) {
            // For non-Vulkan compute types, delegate to the existing converter
            // (e.g. the HIP plugin's hipMalloc-based converter)
            if (meta.compute_device != ComputeType::kVULKAN &&
                meta.compute_device != ComputeType::kVULKANTEXTURE) {
                if (existingConverter) return existingConverter(ptr, meta);
            }

            VkDevice vk_device = g_rendering_device.device;
            VkPhysicalDevice physical_device = g_rendering_device.physical_device;
            VkQueue queue = g_rendering_device.graphics_queue;
            VkCommandPool pool = g_rendering_device.command_pool;

            // For kVULKAN (buffers), create a VkBuffer with exported fd
            if (meta.compute_device == ComputeType::kVULKAN) {
                VkBufferCreateInfo bufferCI{};
                bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferCI.size = meta.byte_size;
                bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                 VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
                bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                VkExternalMemoryBufferCreateInfo externalBufferCI{};
                externalBufferCI.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
                externalBufferCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                bufferCI.pNext = &externalBufferCI;

                VkBuffer buffer;
                VK_CHECK(vkCreateBuffer(vk_device, &bufferCI, nullptr, &buffer));

                VkMemoryRequirements memReqs;
                vkGetBufferMemoryRequirements(vk_device, buffer, &memReqs);

                VkExportMemoryAllocateInfo exportInfo{};
                exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
                exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memReqs.size;
                allocInfo.memoryTypeIndex = find_memory_type(
                    physical_device, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                allocInfo.pNext = &exportInfo;

                VkDeviceMemory bufferMemory;
                VK_CHECK(vkAllocateMemory(vk_device, &allocInfo, nullptr, &bufferMemory));
                vkBindBufferMemory(vk_device, buffer, bufferMemory, 0);

                // Export fd for HIP interop
                int fd = -1;
                VkMemoryGetFdInfoKHR getFdInfo{};
                getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
                getFdInfo.memory = bufferMemory;
                getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)
                    vkGetDeviceProcAddr(vk_device, "vkGetMemoryFdKHR");
                if (vkGetMemoryFdKHR) {
                    if (vkGetMemoryFdKHR(vk_device, &getFdInfo, &fd) != VK_SUCCESS) fd = -1;
                }

                // Stage-copy CPU data
                if (ptr) {
                    VkDeviceSize size = meta.byte_size;
                    VkBuffer stagingBuffer;
                    VkDeviceMemory stagingMemory;
                    VkBufferCreateInfo stagingCI{};
                    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    stagingCI.size = size;
                    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    stagingCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    vkCreateBuffer(vk_device, &stagingCI, nullptr, &stagingBuffer);
                    VkMemoryRequirements stagingReqs;
                    vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &stagingReqs);
                    VkMemoryAllocateInfo stagingAlloc{};
                    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    stagingAlloc.allocationSize = stagingReqs.size;
                    stagingAlloc.memoryTypeIndex = find_memory_type(
                        physical_device, stagingReqs.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    vkAllocateMemory(vk_device, &stagingAlloc, nullptr, &stagingMemory);
                    vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);
                    void* mapped = nullptr;
                    vkMapMemory(vk_device, stagingMemory, 0, size, 0, &mapped);
                    memcpy(mapped, ptr, (size_t)size);
                    vkUnmapMemory(vk_device, stagingMemory);
                    VkCommandBuffer cmd = begin_one_time_commands(vk_device, pool);
                    VkBufferCopy copyRegion{};
                    copyRegion.size = size;
                    vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
                    end_one_time_commands(vk_device, queue, pool, cmd);
                    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
                    vkFreeMemory(vk_device, stagingMemory, nullptr);
                }

                VulkanBufferHandle* handle = new VulkanBufferHandle();
                handle->buffer = (void*)buffer;
                handle->memory = (void*)bufferMemory;
                handle->fd = fd;
                handle->alloc_size = memReqs.size;
                return new BaseMemoryAllocation(meta, reinterpret_cast<void*>(handle));
            }

            // For kVULKANTEXTURE (images), create a VkImage + VkImageView

            VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
            if (meta.type_size == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
            else if (meta.type_size == 4) format = VK_FORMAT_R8G8B8A8_UNORM;
            else if (meta.type_size == 6) format = VK_FORMAT_R16G16B16_SFLOAT;
            else if (meta.type_size == 3) format = VK_FORMAT_R8G8B8_UNORM;
            if (meta.format == 1) format = VK_FORMAT_D32_SFLOAT;

            uint32_t width = (uint32_t)meta.shape.A;
            uint32_t height = (uint32_t)(meta.shape.total_size() / std::max(1UL, (size_t)meta.shape.A));

            // Create VkImage
            VkImageCreateInfo imageCI{};
            imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.format = format;
            imageCI.extent = {width, height, 1};
            imageCI.mipLevels = 1;
            imageCI.arrayLayers = 1;
            imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (format == VK_FORMAT_D32_SFLOAT) {
                imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            }
            imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VkImage image;
            VK_CHECK(vkCreateImage(vk_device, &imageCI, nullptr, &image));

            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(vk_device, image, &memReqs);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memReqs.size;

            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(physical_device, &memProps);
            uint32_t memTypeIndex = 0;
            for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
                if ((memReqs.memoryTypeBits & (1 << i)) &&
                    (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    memTypeIndex = i;
                    break;
                }
            }
            allocInfo.memoryTypeIndex = memTypeIndex;

            VkDeviceMemory imageMemory;
            VK_CHECK(vkAllocateMemory(vk_device, &allocInfo, nullptr, &imageMemory));
            vkBindImageMemory(vk_device, image, imageMemory, 0);

            // Upload data via staging buffer
            if (ptr) {
                VkDeviceSize dataSize = meta.byte_size;

                VkBuffer stagingBuffer;
                VkDeviceMemory stagingMemory;

                VkBufferCreateInfo stagingCI{};
                stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                stagingCI.size = dataSize;
                stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                stagingCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                vkCreateBuffer(vk_device, &stagingCI, nullptr, &stagingBuffer);

                VkMemoryRequirements stagingReqs;
                vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &stagingReqs);

                VkMemoryAllocateInfo stagingAlloc{};
                stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                stagingAlloc.allocationSize = stagingReqs.size;
                stagingAlloc.memoryTypeIndex = find_memory_type(
                    physical_device, stagingReqs.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                vkAllocateMemory(vk_device, &stagingAlloc, nullptr, &stagingMemory);
                vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);

                void* mapped = nullptr;
                vkMapMemory(vk_device, stagingMemory, 0, dataSize, 0, &mapped);
                memcpy(mapped, ptr, (size_t)dataSize);
                vkUnmapMemory(vk_device, stagingMemory);

                // Transition to transfer dst
                VkCommandBuffer cmd = begin_one_time_commands(vk_device, pool);

                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image;
                barrier.subresourceRange.aspectMask = (format == VK_FORMAT_D32_SFLOAT) ?
                    VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.layerCount = 1;
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

                VkBufferImageCopy region{};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = (format == VK_FORMAT_D32_SFLOAT) ?
                    VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {width, height, 1};
                vkCmdCopyBufferToImage(cmd, stagingBuffer, image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                // Transition to shader read
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

                end_one_time_commands(vk_device, queue, pool, cmd);

                vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
                vkFreeMemory(vk_device, stagingMemory, nullptr);
            }

            // Create VkImageView
            VkImageViewCreateInfo viewCI{};
            viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewCI.image = image;
            viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewCI.format = format;
            viewCI.subresourceRange.aspectMask = (format == VK_FORMAT_D32_SFLOAT) ?
                VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            viewCI.subresourceRange.baseMipLevel = 0;
            viewCI.subresourceRange.levelCount = 1;
            viewCI.subresourceRange.baseArrayLayer = 0;
            viewCI.subresourceRange.layerCount = 1;

            VkImageView view;
            VK_CHECK(vkCreateImageView(vk_device, &viewCI, nullptr, &view));

            return new BaseMemoryAllocation(meta, reinterpret_cast<void*>(view));
        };

        std::cout << "[vulkan] Registered DDR → " << (int)renderMem
                  << " texture converter on rendering device" << std::endl;

        // Also register compute type converters and allocators on the rendering
        // device's memory type (e.g. kHIP_VRAM).  The to() function looks up
        // converters on the target memory device, not on the vulkan plugin's
        // kUnknown_MEM devices, so we must register them here too.
        auto& render_dev = global_device_manager.get_device(renderMem, 0);
        render_dev.supports_compute_device[ComputeType::kVULKAN] = true;
        render_dev.supports_compute_device[ComputeType::kVULKANTEXTURE] = true;
        render_dev.compute_type_converters[{ComputeType::kVULKANTEXTURE, ComputeType::kVULKAN}] =
            [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                return ptr;  // VkImageView is already the correct handle
            };
        render_dev.compute_mapping_deallocators[ComputeType::kVULKAN] =
            [](void* ptr, BaseMemoryAllocation* original) {
                // Nothing to do — no mapping was created
            };
        render_dev.compute_type_converters[{ComputeType::kVULKANTEXTURE, ComputeType::kCPU}] =
            [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                return ptr;  // VkImageView cast to void*
            };
        render_dev.compute_mapping_deallocators[ComputeType::kCPU] =
            [](void* ptr, BaseMemoryAllocation* original) {
                // Nothing to do
            };

        // Register kVULKANTEXTURE allocator on the rendering memory device
        // so direct allocation (without to()) also works.
        render_dev.compute_device_allocators[ComputeType::kVULKANTEXTURE] =
            [](AllocationMetadata meta, void* existing_data) -> BaseMemoryAllocation* {
                VkDevice vk_device = g_rendering_device.device;
                VkPhysicalDevice physical_device = g_rendering_device.physical_device;

                VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
                if (meta.type_size == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
                else if (meta.type_size == 4) format = VK_FORMAT_R8G8B8A8_UNORM;
                else if (meta.type_size == 6) format = VK_FORMAT_R16G16B16_SFLOAT;
                else if (meta.type_size == 3) format = VK_FORMAT_R8G8B8_UNORM;
                if (meta.format == 1) format = VK_FORMAT_D32_SFLOAT;

                uint32_t width = (uint32_t)meta.shape.A;
                uint32_t height = (uint32_t)(meta.shape.total_size() / std::max(1UL, (size_t)meta.shape.A));

                VkImageCreateInfo imageCI{};
                imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                imageCI.imageType = VK_IMAGE_TYPE_2D;
                imageCI.format = format;
                imageCI.extent = {width, height, 1};
                imageCI.mipLevels = 1;
                imageCI.arrayLayers = 1;
                imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
                imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
                imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

                VkImage image;
                VK_CHECK(vkCreateImage(vk_device, &imageCI, nullptr, &image));

                VkMemoryRequirements memReqs;
                vkGetImageMemoryRequirements(vk_device, image, &memReqs);
                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memReqs.size;
                allocInfo.memoryTypeIndex = find_memory_type(
                    physical_device, memReqs.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VkDeviceMemory imageMemory;
                VK_CHECK(vkAllocateMemory(vk_device, &allocInfo, nullptr, &imageMemory));
                VK_CHECK(vkBindImageMemory(vk_device, image, imageMemory, 0));

                VkImageViewCreateInfo viewCI{};
                viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewCI.image = image;
                viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewCI.format = format;
                viewCI.subresourceRange.aspectMask = (format == VK_FORMAT_D32_SFLOAT) ?
                    VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                viewCI.subresourceRange.baseMipLevel = 0;
                viewCI.subresourceRange.levelCount = 1;
                viewCI.subresourceRange.baseArrayLayer = 0;
                viewCI.subresourceRange.layerCount = 1;

                VkImageView view;
                VK_CHECK(vkCreateImageView(vk_device, &viewCI, nullptr, &view));

                return new BaseMemoryAllocation(meta, reinterpret_cast<void*>(view));
            };
        render_dev.compute_device_deallocators[ComputeType::kVULKANTEXTURE] =
            [](void* ptr) {
                VkImageView view = (VkImageView)ptr;
                vkDestroyImageView(g_rendering_device.device, view, nullptr);
            };

        // Register kVULKAN allocator on the rendering memory device
        // (creates VkBuffer for vertex/index/uniform buffers, exports fd for HIP interop)
        render_dev.compute_device_allocators[ComputeType::kVULKAN] =
            [](AllocationMetadata metadata, void* existing_data) -> BaseMemoryAllocation* {
                VkDevice vk_device = g_rendering_device.device;
                VkPhysicalDevice physical_device = g_rendering_device.physical_device;
                VkQueue queue = g_rendering_device.graphics_queue;
                VkCommandPool pool = g_rendering_device.command_pool;

                VkBufferCreateInfo bufferCI{};
                bufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bufferCI.size = metadata.byte_size;
                bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                 VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
                bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                // Declare external memory handle type on the buffer itself
                VkExternalMemoryBufferCreateInfo externalBufferCI{};
                externalBufferCI.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
                externalBufferCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                bufferCI.pNext = &externalBufferCI;

                VkBuffer buffer;
                VK_CHECK(vkCreateBuffer(vk_device, &bufferCI, nullptr, &buffer));

                VkMemoryRequirements memReqs;
                vkGetBufferMemoryRequirements(vk_device, buffer, &memReqs);

                // Allocate with exportable memory (for HIP interop via fd)
                VkExportMemoryAllocateInfo exportInfo{};
                exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
                exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize = memReqs.size;
                allocInfo.memoryTypeIndex = find_memory_type(
                    physical_device, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                allocInfo.pNext = &exportInfo;

                VkDeviceMemory bufferMemory;
                VK_CHECK(vkAllocateMemory(vk_device, &allocInfo, nullptr, &bufferMemory));
                vkBindBufferMemory(vk_device, buffer, bufferMemory, 0);

                // Export the memory as an fd for HIP interop
                int fd = -1;
                VkMemoryGetFdInfoKHR getFdInfo{};
                getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
                getFdInfo.memory = bufferMemory;
                getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

                auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)
                    vkGetDeviceProcAddr(vk_device, "vkGetMemoryFdKHR");
                if (vkGetMemoryFdKHR) {
                    VkResult fdRes = vkGetMemoryFdKHR(vk_device, &getFdInfo, &fd);
                    if (fdRes != VK_SUCCESS) {
                        std::cerr << "[vulkan] Failed to export memory fd: " << fdRes << std::endl;
                        fd = -1;
                    }
                } else {
                    std::cerr << "[vulkan] vkGetMemoryFdKHR not available" << std::endl;
                }

                if (existing_data) {
                    VkDeviceSize size = metadata.byte_size;
                    VkBuffer stagingBuffer;
                    VkDeviceMemory stagingMemory;

                    VkBufferCreateInfo stagingCI{};
                    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    stagingCI.size = size;
                    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    stagingCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    vkCreateBuffer(vk_device, &stagingCI, nullptr, &stagingBuffer);

                    VkMemoryRequirements stagingReqs;
                    vkGetBufferMemoryRequirements(vk_device, stagingBuffer, &stagingReqs);

                    VkMemoryAllocateInfo stagingAlloc{};
                    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    stagingAlloc.allocationSize = stagingReqs.size;
                    stagingAlloc.memoryTypeIndex = find_memory_type(
                        physical_device, stagingReqs.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    vkAllocateMemory(vk_device, &stagingAlloc, nullptr, &stagingMemory);
                    vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0);

                    void* mapped = nullptr;
                    vkMapMemory(vk_device, stagingMemory, 0, size, 0, &mapped);
                    memcpy(mapped, existing_data, (size_t)size);
                    vkUnmapMemory(vk_device, stagingMemory);

                    VkCommandBuffer cmd = begin_one_time_commands(vk_device, pool);
                    VkBufferCopy copyRegion{};
                    copyRegion.size = size;
                    vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
                    end_one_time_commands(vk_device, queue, pool, cmd);

                    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
                    vkFreeMemory(vk_device, stagingMemory, nullptr);
                }

                VulkanBufferHandle* handle = new VulkanBufferHandle();
                handle->buffer = (void*)buffer;
                handle->memory = (void*)bufferMemory;
                handle->fd     = fd;
                handle->alloc_size = memReqs.size;
                return new BaseMemoryAllocation(metadata, reinterpret_cast<void*>(handle));
            };
        render_dev.compute_device_deallocators[ComputeType::kVULKAN] =
            [](void* ptr) {
                VulkanBufferHandle* handle = (VulkanBufferHandle*)ptr;
                vkDestroyBuffer(g_rendering_device.device, (VkBuffer)handle->buffer, nullptr);
                vkFreeMemory(g_rendering_device.device, (VkDeviceMemory)handle->memory, nullptr);
                if (handle->fd >= 0) close(handle->fd);
                delete handle;
            };

        // Determine the compute type that matches the rendering device's memory type.
        // kHIP_VRAM → kHIP (HIP plugin registers {kVULKAN, kHIP} interop)
        // kCUDA_VRAM → kCUDA (CUDA plugin registers {kVULKAN, kCUDA} interop)
        ComputeType interop_compute_type = ComputeType::kHIP;
        if (renderMem == MemoryType::kCUDA_VRAM) interop_compute_type = ComputeType::kCUDA;

        // Allow compute on the rendering memory device
        // (the {kVULKAN, interop_compute_type} interop converter is registered
        //  by the HIP or CUDA plugin)
        render_dev.supports_compute_device[interop_compute_type] = true;

        // {kVULKANTEXTURE, interop_compute_type} identity converter — textures
        // don't need real compute access, but the Tensor constructor auto-calls
        // get_massaged_pointer(default_compute_type).  Return the VkImageView
        // pointer as-is; it's only used for descriptor binding.
        render_dev.compute_type_converters[{ComputeType::kVULKANTEXTURE, interop_compute_type}] =
            [](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                return ptr;  // VkImageView — not a real compute pointer
            };
        // Don't free the VkImageView
        render_dev.compute_mapping_deallocators[interop_compute_type] =
            [](void* ptr, BaseMemoryAllocation* original) {
                // No-op — the VkImageView is owned by the kVULKANTEXTURE deallocator
            };

        std::cout << "[vulkan] Registered converters+allocators on renderMem device ("
                  << (int)renderMem << ", interop=" << interop_compute_type << ")" << std::endl;
    } catch (...) {
        std::cerr << "[vulkan] Failed to register DDR texture converter" << std::endl;
    }
}
