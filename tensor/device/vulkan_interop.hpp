#ifndef DEVICE_VULKAN_INTEROP_HPP
#define DEVICE_VULKAN_INTEROP_HPP

//
//  vulkan_interop.hpp — what the CUDA / HIP plugins need to import Vulkan
//  images (the Vulkan counterpart of their OpenGL texture interop).
//
//  An optimal-tiled VkImage has no linear layout, so CUDA / HIP map it as an
//  array (cudaExternalMemoryGetMappedMipmappedArray /
//  hipExternalMemoryGetMappedMipmappedArray) described by its format's
//  channels.  VkFormat values are the Vulkan spec's, so this header does not
//  need <vulkan/vulkan.h>.
//
//  Image usage bits used to pick array flags:
//    VK_IMAGE_USAGE_STORAGE_BIT          0x08 → surface load / store
//    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT 0x10 → colour attachment
//

#include <cstdint>

struct VulkanFormatChannels {
    int x = 0, y = 0, z = 0, w = 0;   // bits per channel
    enum Kind { kUnsigned, kSigned, kFloat } kind = kUnsigned;
    bool ok = false;                  // format has an array equivalent
};

inline VulkanFormatChannels vulkan_format_channels(int32_t vk_format) {
    VulkanFormatChannels c;
    auto set = [&](int x, int y, int z, int w, VulkanFormatChannels::Kind k) {
        c.x = x; c.y = y; c.z = z; c.w = w; c.kind = k; c.ok = true;
    };
    using K = VulkanFormatChannels;
    switch (vk_format) {
        case 9:   // R8_UNORM
        case 13:  // R8_UINT
            set(8, 0, 0, 0, K::kUnsigned); break;
        case 16:  // R8G8_UNORM
            set(8, 8, 0, 0, K::kUnsigned); break;
        case 37:  // R8G8B8A8_UNORM
        case 41:  // R8G8B8A8_UINT
        case 43:  // R8G8B8A8_SRGB
        case 44:  // B8G8R8A8_UNORM  (channels read in memory order: B, G, R, A)
        case 50:  // B8G8R8A8_SRGB
            set(8, 8, 8, 8, K::kUnsigned); break;
        case 38:  // R8G8B8A8_SNORM
            set(8, 8, 8, 8, K::kSigned); break;
        case 70:  // R16_UNORM
        case 74:  // R16_UINT
            set(16, 0, 0, 0, K::kUnsigned); break;
        case 91:  // R16G16B16A16_UNORM
            set(16, 16, 16, 16, K::kUnsigned); break;
        case 76:  // R16_SFLOAT
            set(16, 0, 0, 0, K::kFloat); break;
        case 83:  // R16G16_SFLOAT
            set(16, 16, 0, 0, K::kFloat); break;
        case 97:  // R16G16B16A16_SFLOAT
            set(16, 16, 16, 16, K::kFloat); break;
        case 98:  // R32_UINT
            set(32, 0, 0, 0, K::kUnsigned); break;
        case 99:  // R32_SINT
            set(32, 0, 0, 0, K::kSigned); break;
        case 100: // R32_SFLOAT
        case 126: // D32_SFLOAT
            set(32, 0, 0, 0, K::kFloat); break;
        case 103: // R32G32_SFLOAT
            set(32, 32, 0, 0, K::kFloat); break;
        case 107: // R32G32B32A32_UINT
            set(32, 32, 32, 32, K::kUnsigned); break;
        case 109: // R32G32B32A32_SFLOAT
            set(32, 32, 32, 32, K::kFloat); break;
        default: break;
    }
    return c;
}

constexpr uint32_t kVkImageUsageStorage = 0x08;
constexpr uint32_t kVkImageUsageColorAttachment = 0x10;

#endif // DEVICE_VULKAN_INTEROP_HPP
