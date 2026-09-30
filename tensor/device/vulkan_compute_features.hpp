#ifndef DEVICE_VULKAN_COMPUTE_FEATURES_HPP
#define DEVICE_VULKAN_COMPUTE_FEATURES_HPP

//
//  Device features that vulkcc kernels use, enabled wherever they are
//  supported.  Both the vulkan plugin's compute devices and the display's
//  VulkanContext create their VkDevice with these, so a kernel can run on
//  any tensor's device:
//
//    bufferDeviceAddress        pointers (a kVULKAN tensor's kernel view is
//                               its buffer's device address)
//    shaderInt64                pointer arithmetic, 64-bit integers
//    scalarBlockLayout          structs keep their C++ layout in memory
//    shaderInt8 / shaderInt16,  8/16-bit fields and values
//    8/16-bit storage
//    shaderFloat64              double
//    shaderBufferFloat32AtomicAdd (VK_EXT_shader_atomic_float) atomicAdd(float*)
//
//  Usage:  VulkanComputeFeatures f(physical, base_features);
//          ci.pNext = f.chain(); ci.pEnabledFeatures = nullptr;
//          extensions += f.extensions;
//

#include <vulkan/vulkan.h>
#include <cstring>
#include <string>
#include <vector>

struct VulkanComputeFeatures {
    VkPhysicalDeviceFeatures2 features2{};
    VkPhysicalDeviceVulkan11Features v11{};
    VkPhysicalDeviceVulkan12Features v12{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{};
    std::vector<const char*> extensions;
    bool device_address = false;
    bool api12 = false;

    VulkanComputeFeatures() = default;

    // `base`: the core features the caller wants on top of the compute ones.
    VulkanComputeFeatures(VkPhysicalDevice phys, const VkPhysicalDeviceFeatures& base = {}) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys, &props);
        api12 = props.apiVersion >= VK_API_VERSION_1_2;

        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
        auto has = [&](const char* name) {
            for (auto& e : exts) if (strcmp(e.extensionName, name) == 0) return true;
            return false;
        };
        bool has_atomic_float = has(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

        // what the device supports
        VkPhysicalDeviceFeatures2 s2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDeviceVulkan11Features s11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features s12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT saf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        if (api12) {
            s2.pNext = &s11;
            s11.pNext = &s12;
            if (has_atomic_float) s12.pNext = &saf;
        }
        vkGetPhysicalDeviceFeatures2(phys, &s2);

        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.features = base;
        features2.features.shaderInt64 = s2.features.shaderInt64;
        features2.features.shaderInt16 = s2.features.shaderInt16;
        features2.features.shaderFloat64 = s2.features.shaderFloat64;
        if (!api12) return;

        v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        v11.storageBuffer16BitAccess = s11.storageBuffer16BitAccess;
        v11.uniformAndStorageBuffer16BitAccess = s11.uniformAndStorageBuffer16BitAccess;

        v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12.bufferDeviceAddress = s12.bufferDeviceAddress;
        v12.scalarBlockLayout = s12.scalarBlockLayout;
        v12.shaderInt8 = s12.shaderInt8;
        v12.storageBuffer8BitAccess = s12.storageBuffer8BitAccess;
        v12.uniformAndStorageBuffer8BitAccess = s12.uniformAndStorageBuffer8BitAccess;
        v12.shaderFloat16 = s12.shaderFloat16;
        device_address = s12.bufferDeviceAddress == VK_TRUE;

        features2.pNext = &v11;
        v11.pNext = &v12;
        if (has_atomic_float && saf.shaderBufferFloat32AtomicAdd) {
            atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
            atomic_float.shaderBufferFloat32Atomics = saf.shaderBufferFloat32Atomics;
            atomic_float.shaderBufferFloat32AtomicAdd = saf.shaderBufferFloat32AtomicAdd;
            atomic_float.shaderSharedFloat32Atomics = saf.shaderSharedFloat32Atomics;
            atomic_float.shaderSharedFloat32AtomicAdd = saf.shaderSharedFloat32AtomicAdd;
            v12.pNext = &atomic_float;
            extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        }
    }

    // pNext chain for VkDeviceCreateInfo (set pEnabledFeatures to nullptr).
    // The object must not move after this is called.
    const void* chain() {
        if (api12) {
            features2.pNext = &v11;
            v11.pNext = &v12;
            v12.pNext = atomic_float.sType ? &atomic_float : nullptr;
        }
        return &features2;
    }
};

#endif // DEVICE_VULKAN_COMPUTE_FEATURES_HPP
