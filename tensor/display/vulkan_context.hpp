#ifndef VULKAN_CONTEXT_HPP
#define VULKAN_CONTEXT_HPP

//
//  vulkan_context.hpp — the shared Vulkan device and the "render target"
//  machinery everything else in tensor/display builds on.
//
//  One VulkanContext exists per process (VulkanContext::get()).  It owns the
//  instance, the logical device, the graphics queue and command pool, and it
//  hands that device to the vulkan plugin so every tensor allocated as
//  kVULKAN / kVULKANTEXTURE lives on the same VkDevice as the windows and
//  pipelines.  It can run headless (no window) for offscreen rendering.
//
//  Render targets
//  --------------
//  A RenderTarget is a colour image (+ optional depth image) you can draw
//  into: a window's swapchain image, or any tensor allocated with kSURFACE.
//  While recording a command buffer, targets form a stack:
//
//      ctx.beginTarget(cmd, texture_target);   // ends the current pass,
//          ... draw ...                        // starts one on the texture
//      ctx.endTarget(cmd);                     // resumes the previous target
//                                              // (its contents are kept)
//
//  Every image returns to its resting layout at the end of a pass (see
//  device/vulkan_resource.hpp), so a texture rendered to can be sampled
//  straight away.
//

#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <unistd.h>

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <tuple>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <algorithm>
#include <array>
#include <atomic>

#include "tensor.hpp"
#include "device/vulkan_resource.hpp"
#include "device/vulkan_compute_features.hpp"

#define VK_CTX_CHECK(call)                                                     \
    do {                                                                       \
        VkResult _r = call;                                                    \
        if (_r != VK_SUCCESS) {                                                \
            std::cerr << "[vulkan-ctx] VK error " << _r << " at "              \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
        }                                                                      \
    } while (0)

struct VulkanContext;

// Global context pointer — set as soon as the context exists.
__weak VulkanContext* g_vk_ctx = nullptr;

// ---------------------------------------------------------------------------
//  Tensor ↔ VulkanResource
// ---------------------------------------------------------------------------

// Is `ptr` a VulkanResource the plugin created?  (Registry lookup — safe to
// call with any pointer.)
inline VulkanResource* vk_find_resource(const void* ptr) {
    static hvml_vk_find_resource_fn fn =
        (hvml_vk_find_resource_fn)dlsym(RTLD_DEFAULT, "hvml_vk_find_resource");
    return (fn && ptr) ? fn(ptr) : nullptr;
}

// The VulkanResource behind a tensor: the view it holds (after to_compute to
// a Vulkan type), otherwise the allocation itself.  nullptr if the tensor is
// not Vulkan memory.
template <typename T, int R>
inline VulkanResource* vk_resource(const Tensor<T, R>& t) {
    if (VulkanResource* r = vk_find_resource((const void*)t.data.data)) return r;
    return t.storage_pointer ? vk_find_resource(t.storage_pointer->data) : nullptr;
}

template <int R>
inline VulkanResource* vk_resource(const Tensor<void, R>& t) {
    if (VulkanResource* r = vk_find_resource(t.data)) return r;
    return t.storage_pointer ? vk_find_resource(t.storage_pointer->data) : nullptr;
}

// ---------------------------------------------------------------------------
//  RenderTarget
// ---------------------------------------------------------------------------

struct RenderTarget {
    VkImageView   color       = VK_NULL_HANDLE;
    VkFormat      colorFormat = VK_FORMAT_UNDEFINED;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_UNDEFINED;   // layout at rest / after the pass

    VkImageView   depth       = VK_NULL_HANDLE;              // optional
    VkFormat      depthFormat = VK_FORMAT_UNDEFINED;
    VkImageLayout depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    uint32_t width  = 0;
    uint32_t height = 0;

    VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};
    float             clearDepth = 1.0f;

    VkFramebuffer framebuffer = VK_NULL_HANDLE;              // built on first use

    // Fill colour (or depth) from a resource made with kSURFACE.
    void setColor(const VulkanResource* r) {
        color = (VkImageView)r->image_view;
        colorFormat = (VkFormat)r->format;
        colorLayout = (VkImageLayout)r->layout;
        width = r->width;
        height = r->height;
    }

    void setDepth(const VulkanResource* r) {
        depth = r ? (VkImageView)r->image_view : VK_NULL_HANDLE;
        depthFormat = r ? (VkFormat)r->format : VK_FORMAT_UNDEFINED;
        depthLayout = r ? (VkImageLayout)r->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void release(VkDevice device) {
        if (framebuffer) vkDestroyFramebuffer(device, framebuffer, nullptr);
        framebuffer = VK_NULL_HANDLE;
    }
};

// ---------------------------------------------------------------------------
//  VulkanContext
// ---------------------------------------------------------------------------

struct VulkanContext {

    // ---- core handles ------------------------------------------------------
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue  = VK_NULL_HANDLE;
    uint32_t         graphicsFamily = 0;
    VkCommandPool    commandPool    = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties physicalProperties{};
    VkPhysicalDeviceFeatures   enabledFeatures{};
    VkSampleCountFlagBits      msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    bool externalMemoryFd   = false;   // VK_KHR_external_memory_fd (HIP/CUDA interop)
    VulkanComputeFeatures computeFeatures;   // what vulkcc kernels need (device addresses, ...)
    bool swapchainSupported = false;   // VK_KHR_swapchain
    bool validation         = false;

    // Memory type tensors on this GPU use (kHIP_VRAM / kCUDA_VRAM / kDDR)
    MemoryType renderMemory = MemoryType::kDDR;
    int        renderDeviceIndex = 0;  // vulkan plugin's index for this GPU

    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    // ---- recording state (read by Material when building pipelines) ----------
    VkRenderPass currentRenderPass = VK_NULL_HANDLE;
    VkExtent2D   currentExtent     = {1, 1};

    // ================================================================
    //  Access
    // ================================================================

    // The process-wide context with a device (created headless if no window
    // has created it yet).
    static VulkanContext& get() {
        VulkanContext& ctx = getInstanceOnly();
        if (!ctx.device) ctx.createDevice(VK_NULL_HANDLE);
        return ctx;
    }

    // The context with only the instance created — windows use this to make
    // their surface before the device is chosen, so the device can present.
    static VulkanContext& getInstanceOnly() {
        if (!g_vk_ctx) {
            g_vk_ctx = new VulkanContext();
            g_vk_ctx->createInstance();
        }
        return *g_vk_ctx;
    }

    // Create the device if needed; `surface` (optional) must be presentable.
    void ensureDevice(VkSurfaceKHR surface) {
        if (!device) {
            createDevice(surface);
        } else if (surface && !canPresent(physicalDevice, graphicsFamily, surface)) {
            throw std::runtime_error("[vulkan-ctx] the selected GPU cannot present to this window");
        }
    }

    MemoryType getRenderingMemoryType() const { return renderMemory; }
    int getRenderingDeviceIndex() const { return renderDeviceIndex; }

    // Compute type whose kernels can read/write this GPU's tensors in place.
    ComputeType interopComputeType() const {
        if (renderMemory == MemoryType::kCUDA_VRAM) return ComputeType::kCUDA;
        if (renderMemory == MemoryType::kHIP_VRAM)  return ComputeType::kHIP;
        return ComputeType::kCPU;
    }

    // Destroy everything this context owns.  Optional — only needed for a
    // clean validation-layer report; all tensors must be gone first.
    void shutdown() {
        if (!device) return;
        vkDeviceWaitIdle(device);
        for (auto& [key, rp] : renderPassCache) vkDestroyRenderPass(device, rp, nullptr);
        renderPassCache.clear();
        for (auto& [key, s] : samplerCache) vkDestroySampler(device, s, nullptr);
        samplerCache.clear();
        if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
        if (debugMessenger) {
            auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
            if (fn) fn(instance, debugMessenger, nullptr);
        }
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }

    // ================================================================
    //  Instance
    // ================================================================

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void*)
    {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            std::cerr << "[vulkan-validation] " << data->pMessage << std::endl;
        return VK_FALSE;
    }

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "HVML";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "HVML";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        uint32_t count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());
        auto has = [&](const char* name) {
            for (auto& e : available) if (strcmp(e.extensionName, name) == 0) return true;
            return false;
        };

        // Every surface extension the loader offers, so windows can be
        // created whether or not one exists yet.
        std::vector<const char*> extensions;
        for (const char* name : {"VK_KHR_surface", "VK_KHR_xlib_surface", "VK_KHR_xcb_surface",
                                 "VK_KHR_wayland_surface", "VK_KHR_win32_surface",
                                 "VK_EXT_metal_surface", "VK_KHR_android_surface"}) {
            if (has(name)) extensions.push_back(name);
        }
        VkInstanceCreateFlags flags = 0;
        if (has("VK_KHR_portability_enumeration")) {
            extensions.push_back("VK_KHR_portability_enumeration");
            flags |= 0x00000001; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        }

        // Validation: on when the layer is installed, unless HVML_VK_VALIDATION=0.
        const char* env = std::getenv("HVML_VK_VALIDATION");
        bool wantValidation = !(env && std::string(env) == "0");
        std::vector<const char*> layers;
        if (wantValidation) {
            uint32_t layerCount = 0;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
            std::vector<VkLayerProperties> props(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, props.data());
            for (auto& l : props) {
                if (strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                    layers.push_back("VK_LAYER_KHRONOS_validation");
                    if (has(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
                        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                    validation = true;
                }
            }
        }

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.flags = flags;
        ci.pApplicationInfo = &appInfo;
        ci.enabledExtensionCount = (uint32_t)extensions.size();
        ci.ppEnabledExtensionNames = extensions.data();
        ci.enabledLayerCount = (uint32_t)layers.size();
        ci.ppEnabledLayerNames = layers.data();

        if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("[vulkan-ctx] vkCreateInstance failed");
        }

        if (validation) {
            VkDebugUtilsMessengerCreateInfoEXT dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debugCallback;
            auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
            if (fn) fn(instance, &dci, nullptr, &debugMessenger);
        }
    }

    // ================================================================
    //  Device
    // ================================================================

    static bool canPresent(VkPhysicalDevice d, uint32_t family, VkSurfaceKHR surface) {
        VkBool32 ok = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(d, family, surface, &ok);
        return ok == VK_TRUE;
    }

    static bool hasDeviceExtension(VkPhysicalDevice d, const char* name) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> exts(count);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &count, exts.data());
        for (auto& e : exts) if (strcmp(e.extensionName, name) == 0) return true;
        return false;
    }

    // Graphics queue family of `d` (that can present to `surface`, if given).
    static int graphicsFamilyOf(VkPhysicalDevice d, VkSurfaceKHR surface) {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &count, families.data());
        for (uint32_t i = 0; i < count; i++) {
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (surface && !canPresent(d, i, surface)) continue;
            return (int)i;
        }
        return -1;
    }

    void createDevice(VkSurfaceKHR surface) {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) throw std::runtime_error("[vulkan-ctx] no Vulkan GPUs found");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        // HVML_VK_DEVICE=<n> picks a GPU; otherwise prefer a discrete GPU
        // that has a graphics queue (and can present to `surface`).
        const char* forced = std::getenv("HVML_VK_DEVICE");
        int best = -1, bestScore = -1;
        for (uint32_t i = 0; i < count; i++) {
            if (forced && std::atoi(forced) != (int)i) continue;
            if (graphicsFamilyOf(devices[i], surface) < 0) continue;
            if (surface && !hasDeviceExtension(devices[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devices[i], &p);
            int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 :
                        p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
            if (score > bestScore) { best = (int)i; bestScore = score; }
        }
        if (best < 0) throw std::runtime_error("[vulkan-ctx] no suitable GPU found");

        physicalDevice = devices[best];
        graphicsFamily = (uint32_t)graphicsFamilyOf(physicalDevice, surface);
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalProperties);
        std::cout << "[vulkan-ctx] Selected GPU: " << physicalProperties.deviceName << std::endl;

        // Enable the optional features the materials use, where supported.
        VkPhysicalDeviceFeatures supported;
        vkGetPhysicalDeviceFeatures(physicalDevice, &supported);
        enabledFeatures = {};
        enabledFeatures.samplerAnisotropy = supported.samplerAnisotropy;
        enabledFeatures.fillModeNonSolid  = supported.fillModeNonSolid;
        enabledFeatures.geometryShader    = supported.geometryShader;
        enabledFeatures.largePoints       = supported.largePoints;
        enabledFeatures.depthClamp        = supported.depthClamp;
        enabledFeatures.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
        enabledFeatures.shaderStorageImageReadWithoutFormat  = supported.shaderStorageImageReadWithoutFormat;
        enabledFeatures.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics;
        enabledFeatures.vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics;

        std::vector<const char*> exts;
        swapchainSupported = hasDeviceExtension(physicalDevice, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        if (swapchainSupported) exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        externalMemoryFd = hasDeviceExtension(physicalDevice, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        if (externalMemoryFd) exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = graphicsFamily;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;

        // Compute-kernel features (vulkcc) on top of the material ones, so
        // kernels can run on tensors that live on this device.
        computeFeatures = VulkanComputeFeatures(physicalDevice, enabledFeatures);
        for (const char* e : computeFeatures.extensions) exts.push_back(e);

        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = &qi;
        ci.pNext = computeFeatures.chain();
        ci.pEnabledFeatures = nullptr;
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.ppEnabledExtensionNames = exts.data();
        if (vkCreateDevice(physicalDevice, &ci, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("[vulkan-ctx] vkCreateDevice failed");
        }
        vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);

        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = graphicsFamily;
        VK_CTX_CHECK(vkCreateCommandPool(device, &pci, nullptr, &commandPool));

        shareWithPlugin();
    }

    // Hand the device to the vulkan plugin and point the tensor system's
    // defaults at it.
    void shareWithPlugin() {
        auto setDevice = (hvml_vk_set_rendering_device_fn)dlsym(RTLD_DEFAULT, "hvml_vk_set_rendering_device");
        if (!setDevice) {
            std::cerr << "[vulkan-ctx] vulkan plugin not loaded — kVULKAN / kVULKANTEXTURE tensors "
                         "are unavailable (check DEVICE_PLUGIN_DIR)" << std::endl;
            return;
        }

        HvmlVkDeviceInfo info;
        info.instance = (void*)instance;
        info.physical_device = (void*)physicalDevice;
        info.device = (void*)device;
        info.queue = (void*)graphicsQueue;
        info.queue_family = graphicsFamily;
        info.command_pool = (void*)commandPool;
        info.external_memory_fd = externalMemoryFd ? 1 : 0;
        info.buffer_device_address = computeFeatures.device_address ? 1 : 0;
        setDevice(&info);

        auto memType = (hvml_vk_rendering_memory_type_fn)dlsym(RTLD_DEFAULT, "get_rendering_device_memory_type");
        auto devIndex = (hvml_vk_rendering_device_index_fn)dlsym(RTLD_DEFAULT, "get_rendering_device_index");
        if (memType) renderMemory = (MemoryType)memType();
        if (devIndex) renderDeviceIndex = std::max(0, devIndex());

        global_device_manager.init_plugin("vulkan");

        try {
            // Plain tensors on a dedicated GPU's memory become Vulkan buffers
            // that HIP/CUDA import in place — one allocation usable by both
            // kernels and the renderer.  Needs fd export.
            if (renderMemory != MemoryType::kDDR && externalMemoryFd) {
                AllocationMap& mem = global_device_manager.get_device(renderMemory, 0);
                mem.default_compute_type = interopComputeType();
                mem.default_allocator_type = ComputeType::kVULKAN;
            }
            ComputeDeviceBase& cd = global_device_manager.get_compute_device(ComputeType::kVULKAN, renderDeviceIndex);
            cd.default_memory_type = renderMemory;
            cd.supports_memory_location[renderMemory] = true;
        } catch (const std::exception& e) {
            std::cerr << "[vulkan-ctx] " << e.what() << std::endl;
        }
    }

    // ================================================================
    //  Render passes and targets
    // ================================================================

    // (colour format, depth format, colour final layout, depth final layout, load)
    using RenderPassKey = std::tuple<int, int, int, int, bool>;
    std::map<RenderPassKey, VkRenderPass> renderPassCache;

    // A render pass for `t`.  `load` keeps the existing contents (used when a
    // target is resumed); otherwise the attachments are cleared.  The two
    // variants are compatible, so pipelines and framebuffers work with both.
    VkRenderPass renderPass(const RenderTarget& t, bool load) {
        RenderPassKey key{(int)t.colorFormat, (int)t.depthFormat, (int)t.colorLayout, (int)t.depthLayout, load};
        auto it = renderPassCache.find(key);
        if (it != renderPassCache.end()) return it->second;

        std::vector<VkAttachmentDescription> atts;
        VkAttachmentDescription color{};
        color.format = t.colorFormat;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = load ? t.colorLayout : VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = t.colorLayout;
        atts.push_back(color);

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        bool hasDepth = t.depthFormat != VK_FORMAT_UNDEFINED;
        if (hasDepth) {
            VkAttachmentDescription depth{};
            depth.format = t.depthFormat;
            depth.samples = VK_SAMPLE_COUNT_1_BIT;
            depth.loadOp = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth.initialLayout = load ? t.depthLayout : VK_IMAGE_LAYOUT_UNDEFINED;
            depth.finalLayout = t.depthLayout;
            atts.push_back(depth);
        }

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

        // Order against whatever touched the images before/after the pass
        // (sampling in an earlier pass, transfers, compute).
        const VkPipelineStageFlags attachmentStages =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        const VkAccessFlags attachmentAccess =
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        const VkPipelineStageFlags useStages =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;

        std::array<VkSubpassDependency, 2> deps{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = attachmentStages | useStages;
        deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        deps[0].dstStageMask = attachmentStages;
        deps[0].dstAccessMask = attachmentAccess;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = attachmentStages;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = attachmentStages | useStages;
        deps[1].dstAccessMask = attachmentAccess | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

        VkRenderPassCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = (uint32_t)atts.size();
        ci.pAttachments = atts.data();
        ci.subpassCount = 1;
        ci.pSubpasses = &subpass;
        ci.dependencyCount = (uint32_t)deps.size();
        ci.pDependencies = deps.data();

        VkRenderPass rp;
        VK_CTX_CHECK(vkCreateRenderPass(device, &ci, nullptr, &rp));
        renderPassCache[key] = rp;
        return rp;
    }

    // Targets currently pushed while recording (innermost last).
    std::vector<RenderTarget*> targetStack;
    bool passOpen = false;

    bool insideRenderPass() const { return passOpen; }
    RenderTarget* currentTarget() const { return targetStack.empty() ? nullptr : targetStack.back(); }

    void setViewport(VkCommandBuffer cmd, uint32_t w, uint32_t h) {
        VkViewport viewport{0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{{0, 0}, {w, h}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }

    // Start drawing into `t` (ending the current pass, if any).
    void beginTarget(VkCommandBuffer cmd, RenderTarget& t, bool clear = true) {
        if (passOpen) vkCmdEndRenderPass(cmd);
        openPass(cmd, t, !clear);
        targetStack.push_back(&t);
    }

    // Finish drawing into the current target and resume the previous one.
    void endTarget(VkCommandBuffer cmd) {
        if (passOpen) vkCmdEndRenderPass(cmd);
        passOpen = false;
        currentRenderPass = VK_NULL_HANDLE;
        if (!targetStack.empty()) targetStack.pop_back();
        if (!targetStack.empty()) openPass(cmd, *targetStack.back(), /*load=*/true);
    }

    // Close every open target (end of a command buffer).
    void endAllTargets(VkCommandBuffer cmd) {
        if (passOpen) vkCmdEndRenderPass(cmd);
        passOpen = false;
        targetStack.clear();
        currentRenderPass = VK_NULL_HANDLE;
    }

    // ================================================================
    //  Samplers
    // ================================================================

    std::map<std::pair<bool, bool>, VkSampler> samplerCache;

    VkSampler sampler(bool nearest = false, bool repeat = true) {
        auto key = std::make_pair(nearest, repeat);
        auto it = samplerCache.find(key);
        if (it != samplerCache.end()) return it->second;

        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        sci.minFilter = nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        VkSamplerAddressMode mode = repeat ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = mode;
        sci.anisotropyEnable = (!nearest && enabledFeatures.samplerAnisotropy) ? VK_TRUE : VK_FALSE;
        sci.maxAnisotropy = sci.anisotropyEnable ? std::min(16.0f, physicalProperties.limits.maxSamplerAnisotropy) : 1.0f;
        sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sci.compareOp = VK_COMPARE_OP_ALWAYS;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VkSampler s;
        VK_CTX_CHECK(vkCreateSampler(device, &sci, nullptr, &s));
        samplerCache[key] = s;
        return s;
    }

    // ================================================================
    //  Small utilities
    // ================================================================

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        return cmd;
    }

    void endSingleTimeCommands(VkCommandBuffer cmd) {
        endAllTargets(cmd);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CTX_CHECK(vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE));
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    // Record with `fn`, submit, wait.  For offscreen work without a window:
    //   ctx.submit([&](VkCommandBuffer cmd){ tex.render(cmd, [&]{ ... }); });
    template <typename F>
    void submit(F&& fn) {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        fn(cmd);
        endSingleTimeCommands(cmd);
    }

    // ================================================================
    //  Shaders
    // ================================================================

    VkShaderModule createShaderModule(const std::vector<uint8_t>& spirv) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spirv.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(spirv.data());
        VkShaderModule mod;
        VK_CTX_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
        return mod;
    }

    // Compile GLSL → SPIR-V with glslangValidator (must be on PATH).
    std::vector<uint8_t> compileGLSL(const std::string& source, const std::string& stage) {
        static std::atomic<int> counter{0};
        std::string base = "/tmp/hvml_shader_" + std::to_string(getpid()) + "_" +
                           std::to_string(counter++) + "_" + stage;
        std::string srcFile = base + ".glsl";
        std::string spvFile = base + ".spv";

        { std::ofstream f(srcFile); f << source; }

        std::string cmd = "glslangValidator --auto-map-locations -V -S " + stage +
                          " -o " + spvFile + " " + srcFile + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "[vulkan-ctx] failed to run glslangValidator" << std::endl;
            return {};
        }
        char buf[256];
        std::string log;
        while (fgets(buf, sizeof(buf), pipe)) log += buf;
        int rc = pclose(pipe);
        std::remove(srcFile.c_str());
        if (rc != 0) {
            std::cerr << "[vulkan-ctx] shader compilation failed (" << stage << "):\n" << log
                      << "\n--- source ---\n" << source << std::endl;
            std::remove(spvFile.c_str());
            return {};
        }

        std::ifstream spv(spvFile, std::ios::binary | std::ios::ate);
        if (!spv) return {};
        size_t size = spv.tellg();
        spv.seekg(0);
        std::vector<uint8_t> data(size);
        spv.read(reinterpret_cast<char*>(data.data()), size);
        std::remove(spvFile.c_str());
        return data;
    }

private:
    void openPass(VkCommandBuffer cmd, RenderTarget& t, bool load) {
        VkRenderPass rp = renderPass(t, load);
        if (!t.framebuffer) {
            std::array<VkImageView, 2> views = {t.color, t.depth};
            VkFramebufferCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fci.renderPass = rp;
            fci.attachmentCount = t.depth ? 2 : 1;
            fci.pAttachments = views.data();
            fci.width = t.width;
            fci.height = t.height;
            fci.layers = 1;
            VK_CTX_CHECK(vkCreateFramebuffer(device, &fci, nullptr, &t.framebuffer));
        }

        std::array<VkClearValue, 2> clears{};
        clears[0].color = t.clearColor;
        clears[1].depthStencil = {t.clearDepth, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = rp;
        bi.framebuffer = t.framebuffer;
        bi.renderArea = {{0, 0}, {t.width, t.height}};
        bi.clearValueCount = load ? 0 : (t.depth ? 2u : 1u);
        bi.pClearValues = load ? nullptr : clears.data();
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);

        passOpen = true;
        currentRenderPass = rp;
        currentExtent = {t.width, t.height};
        setViewport(cmd, t.width, t.height);
    }
};

// Memory type display tensors are allocated on (creates the context if needed).
inline MemoryType vk_render_memory() {
    return VulkanContext::get().getRenderingMemoryType();
}

#endif // VULKAN_CONTEXT_HPP
