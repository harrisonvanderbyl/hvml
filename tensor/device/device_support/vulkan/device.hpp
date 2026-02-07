#ifndef TENSOR_ENUMS_DEVICE_SUPPORT_VULKAN_DEVICE_HPP
#define TENSOR_ENUMS_DEVICE_SUPPORT_VULKAN_DEVICE_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include <iostream>
#include "device/common.hpp"

#define VK_CHECK(call) \
    do { \
        VkResult result = call; \
        if (result != VK_SUCCESS) { \
            std::cerr << "Vulkan error at " << __FILE__ << ":" << __LINE__ \
                      << " - Result: " << result << std::endl; \
        } \
    } while(0)


__weak VkInstance vk_instance = VK_NULL_HANDLE;
__weak std::vector<VkPhysicalDevice> vk_physical_devices;
__weak std::vector<VkDevice> vk_devices;
__weak std::vector<VkQueue> vk_queues;
__weak std::vector<uint32_t> vk_compute_queue_families;
__weak bool vulkan_initialized = false;

// Helper function to find memory type index
__weak uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    std::cerr << "Failed to find suitable memory type!" << std::endl;
    return 0;
}

ComputeDeviceBase* create_vulkan_compute_device(int device_id){
    if (device_id < 0 || device_id >= static_cast<int>(vk_physical_devices.size())) {
        std::cerr << "Invalid Vulkan device_id: " << device_id << std::endl;
        return nullptr;
    }

    ComputeDeviceBase* device = new ComputeDeviceBase();
    
    VkPhysicalDevice physical_device = vk_physical_devices[device_id];
    
    // Get device properties
    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    
    
    // Determine memory type based on vendor
    MemoryType mem = MemoryType::kUnknown_MEM;
    if (strstr(device_properties.deviceName, "NVIDIA") != nullptr) {
        mem = MemoryType::kCUDA_VRAM;
    } else if (strstr(device_properties.deviceName, "AMD") != nullptr || 
               strstr(device_properties.deviceName, "ATI") != nullptr) {
        mem = MemoryType::kHIP_VRAM;
    } else if (strstr(device_properties.deviceName, "llvm") != nullptr || 
               strstr(device_properties.deviceName, "Intel") != nullptr) {
        mem = MemoryType::kDDR;
    } else {
        mem = MemoryType::kUnknown_MEM;
    }

    device->compute_units = device_properties.limits.maxComputeWorkGroupCount[0];
    device->default_memory_type = mem;
    device->supports_memory_location[mem] = true;

    // Find a compute queue family
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    
    uint32_t compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    
    if (compute_queue_family == UINT32_MAX) {
        std::cerr << "No compute queue family found for device " << device_id << std::endl;
        delete device;
        return nullptr;
    }
    
    vk_compute_queue_families.push_back(compute_queue_family);

    // Create logical device with compute queue
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

    VkDevice vk_device;
    VK_CHECK(vkCreateDevice(physical_device, &device_create_info, nullptr, &vk_device));
    vk_devices.push_back(vk_device);
    
    // Get the compute queue
    VkQueue compute_queue;
    vkGetDeviceQueue(vk_device, compute_queue_family, 0, &compute_queue);
    vk_queues.push_back(compute_queue);

    // Register with memory device
    try{
        auto& mem_device = global_device_manager.get_device(mem, 0);
        mem_device.supports_compute_device[ComputeType::kVULKAN] = true;

        // Setup allocator
        mem_device.compute_device_allocators[ComputeType::kVULKAN] = [device_id, physical_device](Shape<-1> size, size_t bitsize, void* existing_data) {
            VkDeviceMemory* device_memory = new VkDeviceMemory();
            
            VkMemoryAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = size.total_size() * bitsize;
            
            // Find device-local memory type
            alloc_info.memoryTypeIndex = find_memory_type(
                physical_device,
                UINT32_MAX,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            
            VK_CHECK(vkAllocateMemory(vk_devices[device_id], &alloc_info, nullptr, device_memory));
            return device_memory;
        };
        
        mem_device.compute_device_deallocators[ComputeType::kVULKAN] = [device_id](void* ptr) {
            VkDeviceMemory device_memory = *(VkDeviceMemory*)ptr;
            vkFreeMemory(vk_devices[device_id], device_memory, nullptr);
            delete (VkDeviceMemory*)ptr;
        };

        std::cout << "Created Vulkan device " << device_id  << std::endl;
    }catch(...){
        std::cerr << "Failed to register Vulkan device " << device_id << " with memory manager" << std::endl;
    }
    
    return device;
}

int count_vulkan_devices(){
    if (vulkan_initialized) {
        return static_cast<int>(vk_physical_devices.size());
    }
    
    // Application info
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Tensor Compute";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "TensorEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    // Instance create info
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    
    // No validation layers or extensions needed for compute-only
    create_info.enabledLayerCount = 0;
    create_info.enabledExtensionCount = 0;

    VkResult result = vkCreateInstance(&create_info, nullptr, &vk_instance);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance: " << result << std::endl;
        return 0;
    }

    // Enumerate physical devices
    uint32_t device_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(vk_instance, &device_count, nullptr));
    
    if (device_count == 0) {
        std::cerr << "No Vulkan devices found" << std::endl;
        return 0;
    }
    
    vk_physical_devices.resize(device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(vk_instance, &device_count, vk_physical_devices.data()));
    
    // Print device info
    std::cout << "Vulkan Device Count: " << device_count << std::endl;
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(vk_physical_devices[i], &props);
        std::cout << "  Device " << i << ": " << props.deviceName << std::endl;
    }
    
    vulkan_initialized = true;
    
    return device_count;
}


#endif // TENSOR_ENUMS_DEVICE_SUPPORT_VULKAN_DEVICE_HPP