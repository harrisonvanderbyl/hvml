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

    // Allocator — allocate device-local VkDeviceMemory
    mapper->compute_device_allocators[ComputeType::kVULKAN] = [device_id](AllocationMetadata metadata, void* existing_data) {
        VkDevice vk_device = g_devices[device_id].device;
        VkPhysicalDevice physical_device = g_devices[device_id].physical_device;

        VkDeviceMemory* device_memory = new VkDeviceMemory();

        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = metadata.byte_size;
        alloc_info.memoryTypeIndex = find_memory_type(
            physical_device, UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VK_CHECK(vkAllocateMemory(vk_device, &alloc_info, nullptr, device_memory));

        // If existing data is provided, copy it into the device-local allocation
        // via a staging buffer.
        if (existing_data) {
            VkDeviceSize size = metadata.byte_size;

            // Create staging buffer
            VkBuffer staging_buffer;
            VkDeviceMemory staging_memory;

            VkBufferCreateInfo buffer_info = {};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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

            void* mapped = nullptr;
            vkMapMemory(vk_device, staging_memory, 0, size, 0, &mapped);
            memcpy(mapped, existing_data, (size_t)size);
            vkUnmapMemory(vk_device, staging_memory);

            // Create destination buffer backed by the device-local allocation
            VkBuffer dst_buffer;
            VkBufferCreateInfo dst_info = {};
            dst_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            dst_info.size = size;
            dst_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            dst_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            vkCreateBuffer(vk_device, &dst_info, nullptr, &dst_buffer);
            vkBindBufferMemory(vk_device, dst_buffer, *device_memory, 0);

            // Record copy command
            VkCommandBuffer cmd = begin_one_time_commands(
                vk_device, g_devices[device_id].command_pool);

            VkBufferCopy copy_region = {};
            copy_region.size = size;
            vkCmdCopyBuffer(cmd, staging_buffer, dst_buffer, 1, &copy_region);

            end_one_time_commands(vk_device, g_devices[device_id].compute_queue,
                                  g_devices[device_id].command_pool, cmd);

            vkDestroyBuffer(vk_device, staging_buffer, nullptr);
            vkFreeMemory(vk_device, staging_memory, nullptr);
            vkDestroyBuffer(vk_device, dst_buffer, nullptr);
        }

        return new BaseMemoryAllocation(metadata, device_memory);
    };

    // Deallocator
    mapper->compute_device_deallocators[ComputeType::kVULKAN] = [device_id](void* ptr) {
        VkDeviceMemory device_memory = *(VkDeviceMemory*)ptr;
        vkFreeMemory(g_devices[device_id].device, device_memory, nullptr);
        delete (VkDeviceMemory*)ptr;
    };

    // Converter: Vulkan device-local → host DDR
    mapper->memory_type_converters[MemoryType::kDDR] = [device_id](void* ptr, AllocationMetadata meta) {
        VkDevice vk_device = g_devices[device_id].device;
        VkPhysicalDevice physical_device = g_devices[device_id].physical_device;
        VkDeviceMemory device_memory = *(VkDeviceMemory*)ptr;

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

        // Create source buffer backed by the device-local allocation
        VkBuffer src_buffer;
        VkBufferCreateInfo src_info = {};
        src_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        src_info.size = meta.byte_size;
        src_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        src_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(vk_device, &src_info, nullptr, &src_buffer);
        vkBindBufferMemory(vk_device, src_buffer, device_memory, 0);

        // Copy device → staging
        VkCommandBuffer cmd = begin_one_time_commands(
            vk_device, g_devices[device_id].command_pool);

        VkBufferCopy copy_region = {};
        copy_region.size = meta.byte_size;
        vkCmdCopyBuffer(cmd, src_buffer, staging_buffer, 1, &copy_region);

        end_one_time_commands(vk_device, g_devices[device_id].compute_queue,
                              g_devices[device_id].command_pool, cmd);

        // Map staging and copy to host
        void* mapped = nullptr;
        vkMapMemory(vk_device, staging_memory, 0, meta.byte_size, 0, &mapped);
        memcpy(host_ptr->data, mapped, (size_t)meta.byte_size);
        vkUnmapMemory(vk_device, staging_memory);

        vkDestroyBuffer(vk_device, staging_buffer, nullptr);
        vkFreeMemory(vk_device, staging_memory, nullptr);
        vkDestroyBuffer(vk_device, src_buffer, nullptr);

        return host_ptr;
    };

    // Synchronize
    mapper->synchronize_function = [device_id]() {
        vkQueueWaitIdle(g_devices[device_id].compute_queue);
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
    VulkanDeviceState& state = g_devices[device_id];

    ComputeDeviceBase* device = new ComputeDeviceBase();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(state.physical_device, &props);

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
                VkDevice vk_device = g_devices[device_id].device;
                VkPhysicalDevice physical_device = g_devices[device_id].physical_device;

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
                vkFreeMemory(g_devices[device_id].device, device_memory, nullptr);
                delete (VkDeviceMemory*)ptr;
            };

            // Vulkan ↔ CPU conversion (host-visible memory is directly mappable)
            mem_device.compute_type_converters[{ComputeType::kVULKAN, ComputeType::kCPU}] = [device_id](void* ptr, BaseMemoryAllocation* original, AllocationMetadata metadata) {
                VkDevice vk_device = g_devices[device_id].device;
                VkDeviceMemory device_memory = *(VkDeviceMemory*)ptr;
                void* mapped = nullptr;
                vkMapMemory(vk_device, device_memory, 0, metadata.byte_size, 0, &mapped);
                return mapped;
            };

            mem_device.compute_mapping_deallocators[ComputeType::kCPU] = [device_id](void* ptr, BaseMemoryAllocation* original) {
                VkDevice vk_device = g_devices[device_id].device;
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
