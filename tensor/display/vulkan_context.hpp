#ifndef VULKAN_CONTEXT_HPP
#define VULKAN_CONTEXT_HPP

#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cstring>
#include <fstream>
#include <optional>
#include <functional>
#include <algorithm>
#include <array>

#define VK_CTX_CHECK(call)                                                     \
    do {                                                                       \
        VkResult _r = call;                                                    \
        if (_r != VK_SUCCESS) {                                                \
            std::cerr << "[vulkan-ctx] VK error " << _r << " at "              \
                      << __FILE__ << ":" << __LINE__ << std::endl;             \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  Queue family helper
// ---------------------------------------------------------------------------

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    std::optional<uint32_t> compute;

    bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

// ---------------------------------------------------------------------------
//  Swapchain support details
// ---------------------------------------------------------------------------

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// ---------------------------------------------------------------------------
//  VulkanContext — owns instance, device, swapchain, render pass,
//  depth buffer, and per-frame command buffers / sync objects.
//
//  The display layer creates one of these, then the Material / RenderStruct
//  code uses ctx.device, ctx.render_pass, etc. to build pipelines.
// ---------------------------------------------------------------------------

struct VulkanContext {

    // Core handles
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue  = VK_NULL_HANDLE;
    VkQueue          presentQueue   = VK_NULL_HANDLE;
    VkQueue          computeQueue   = VK_NULL_HANDLE;
    VkSurfaceKHR     surface        = VK_NULL_HANDLE;

    // Queue families
    uint32_t graphicsFamily = 0;
    uint32_t presentFamily  = 0;
    uint32_t computeFamily  = 0;

    // Command pool (graphics)
    VkCommandPool commandPool = VK_NULL_HANDLE;

    // Swapchain
    VkSwapchainKHR   swapchain    = VK_NULL_HANDLE;
    VkFormat         swapchainFormat;
    VkExtent2D       swapchainExtent;
    std::vector<VkImage>       swapchainImages;
    std::vector<VkImageView>   swapchainViews;

    // Depth
    VkImage        depthImage       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView    depthView        = VK_NULL_HANDLE;
    VkFormat       depthFormat      = VK_FORMAT_D32_SFLOAT;

    // Render pass
    VkRenderPass renderPass = VK_NULL_HANDLE;

    // Framebuffers (one per swapchain image)
    std::vector<VkFramebuffer> framebuffers;

    // Per-frame resources (MAX_FRAMES in flight)
    static constexpr int MAX_FRAMES = 2;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence>     inFlightFences;
    uint32_t currentFrame = 0;

    // Offscreen render target support (for VectorDisplay)
    // When rendering to a texture instead of the swapchain:
    bool renderingOffscreen = false;
    VkFramebuffer activeOffscreenFramebuffer = VK_NULL_HANDLE;
    VkRenderPass  activeOffscreenRenderPass  = VK_NULL_HANDLE;

    // Properties
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPhysicalDeviceProperties physicalProperties;

    // --- Debug messenger (optional) ---
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* userData)
    {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            std::cerr << "[vulkan-validation] " << data->pMessage << std::endl;
        return VK_FALSE;
    }

    // ================================================================
    //  Initialization
    // ================================================================

    void init(SDL_Window* window, int width, int height) {
        createInstance();
        createSurface(window);
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        createSwapchain(window, width, height);
        createImageViews();
        createDepthResources();
        createRenderPass();
        createFramebuffers();
        createCommandBuffers();
        createSyncObjects();
    }

    // Share this VkDevice with the vulkan plugin so all tensor allocations
    // (kVULKAN, kVULKANTEXTURE) use the same device as the render pass.
    // Finds the matching device index by comparing VkPhysicalDevice handles.
    void shareWithPlugin();

    // Get the rendering device's index and memory type (for tensor allocations)
    int getRenderingDeviceIndex();
    MemoryType getRenderingMemoryType();

    void cleanup() {
        vkDeviceWaitIdle(device);

        for (auto& fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        if (renderPass) vkDestroyRenderPass(device, renderPass, nullptr);
        destroyDepthResources();
        for (auto& v : swapchainViews) vkDestroyImageView(device, v, nullptr);
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);

        for (auto& s : imageAvailableSemaphores) vkDestroySemaphore(device, s, nullptr);
        for (auto& s : renderFinishedSemaphores) vkDestroySemaphore(device, s, nullptr);
        for (auto& f : inFlightFences) vkDestroyFence(device, f, nullptr);

        if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (device)      vkDestroyDevice(device, nullptr);
        if (surface)     vkDestroySurfaceKHR(instance, surface, nullptr);
        if (debugMessenger) {
            auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
            if (fn) fn(instance, debugMessenger, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    // ================================================================
    //  Instance
    // ================================================================

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "HVML Vulkan";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "HVML";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        // SDL3 returns a NULL-terminated array of extension name strings
        Uint32 sdlExtCount = 0;
        char const * const * sdlExts = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
        std::vector<const char*> extensions;
        for (Uint32 i = 0; i < sdlExtCount; i++) {
            extensions.push_back(sdlExts[i]);
        }

        // Check for validation layer support
        bool enableValidation = false;
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
        for (auto& l : layers) {
            if (strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                enableValidation = true;
                break;
            }
        }

        std::vector<const char*> validationLayers;
        if (enableValidation) {
            validationLayers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &appInfo;
        ci.enabledExtensionCount = (uint32_t)extensions.size();
        ci.ppEnabledExtensionNames = extensions.data();
        ci.enabledLayerCount = (uint32_t)validationLayers.size();
        ci.ppEnabledLayerNames = validationLayers.data();

        VK_CTX_CHECK(vkCreateInstance(&ci, nullptr, &instance));

        if (enableValidation) {
            VkDebugUtilsMessengerCreateInfoEXT dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debugCallback;
            auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
            if (fn) fn(instance, &dci, nullptr, &debugMessenger);
        }

        std::cout << "[vulkan-ctx] Instance created" << std::endl;
    }

    // ================================================================
    //  Surface
    // ================================================================

    void createSurface(SDL_Window* window) {
        if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
            throw std::runtime_error("Failed to create Vulkan surface: " +
                std::string(SDL_GetError()));
        }
    }

    // ================================================================
    //  Physical device selection
    // ================================================================

    void pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) throw std::runtime_error("No Vulkan GPUs found");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        for (auto& d : devices) {
            QueueFamilyIndices qf = findQueueFamilies(d);
            if (!qf.complete()) continue;
            if (!checkDeviceExtensionSupport(d)) continue;
            SwapchainSupport sw = querySwapchainSupport(d);
            if (sw.formats.empty() || sw.presentModes.empty()) continue;

            physicalDevice = d;
            vkGetPhysicalDeviceProperties(d, &physicalProperties);
            std::cout << "[vulkan-ctx] Selected GPU: " << physicalProperties.deviceName << std::endl;

            // MSAA — keep at 1x for now (no resolve attachment configured)
            // VkSampleCountFlags counts = physicalProperties.limits.framebufferColorSampleCounts;
            // if (counts & VK_SAMPLE_COUNT_4_BIT) msaaSamples = VK_SAMPLE_COUNT_4_BIT;
            break;
        }
        if (physicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("No suitable GPU found");
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice d) {
        QueueFamilyIndices idx;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &count, families.data());

        for (uint32_t i = 0; i < count; i++) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                idx.graphics = i;
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                idx.compute = i;
            VkBool32 present = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &present);
            if (present) idx.present = i;
            if (idx.complete()) break;
        }
        return idx;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice d) {
        static const std::vector<const char*> required = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };
        uint32_t count;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> avail(count);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &count, avail.data());
        std::set<std::string> requiredSet(required.begin(), required.end());
        for (auto& a : avail) requiredSet.erase(a.extensionName);
        return requiredSet.empty();
    }

    SwapchainSupport querySwapchainSupport(VkPhysicalDevice d) {
        SwapchainSupport sw;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d, surface, &sw.capabilities);
        uint32_t fmtCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface, &fmtCount, nullptr);
        if (fmtCount) {
            sw.formats.resize(fmtCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(d, surface, &fmtCount, sw.formats.data());
        }
        uint32_t pmCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface, &pmCount, nullptr);
        if (pmCount) {
            sw.presentModes.resize(pmCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(d, surface, &pmCount, sw.presentModes.data());
        }
        return sw;
    }

    // ================================================================
    //  Logical device + queues
    // ================================================================

    void createLogicalDevice() {
        QueueFamilyIndices qf = findQueueFamilies(physicalDevice);
        graphicsFamily = qf.graphics.value();
        presentFamily  = qf.present.value();
        computeFamily  = qf.compute.value_or(qf.graphics.value());

        std::vector<VkDeviceQueueCreateInfo> queueCIs;
        std::set<uint32_t> uniqueFamilies = {graphicsFamily, presentFamily};
        float priority = 1.0f;
        for (uint32_t fam : uniqueFamilies) {
            VkDeviceQueueCreateInfo qi{};
            qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = fam;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            queueCIs.push_back(qi);
        }

        VkPhysicalDeviceFeatures features{};
        features.samplerAnisotropy = VK_TRUE;
        features.fillModeNonSolid = VK_TRUE;
        features.geometryShader = VK_TRUE;
        features.largePoints = VK_TRUE;
        features.depthClamp = VK_TRUE;

        static std::vector<const char*> exts = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        // Enable VK_KHR_external_memory_fd for Vulkan-HIP interop
        // (allows exporting VkBuffer memory as fd, imported by HIP)
        {
            uint32_t devExtCount = 0;
            vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, nullptr);
            std::vector<VkExtensionProperties> devExts(devExtCount);
            vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, devExts.data());
            for (const auto& e : devExts) {
                if (strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0) {
                    exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
                    std::cout << "[vulkan-ctx] Enabled " << VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME << std::endl;
                    break;
                }
            }
        }

        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = (uint32_t)queueCIs.size();
        ci.pQueueCreateInfos = queueCIs.data();
        ci.pEnabledFeatures = &features;
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.ppEnabledExtensionNames = exts.data();

        VK_CTX_CHECK(vkCreateDevice(physicalDevice, &ci, nullptr, &device));

        vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(device, presentFamily, 0, &presentQueue);
        vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);
    }

    // ================================================================
    //  Command pool
    // ================================================================

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = graphicsFamily;
        VK_CTX_CHECK(vkCreateCommandPool(device, &ci, nullptr, &commandPool));
    }

    // ================================================================
    //  Swapchain
    // ================================================================

    void createSwapchain(SDL_Window* window, int width, int height) {
        SwapchainSupport sw = querySwapchainSupport(physicalDevice);

        // Choose format
        VkSurfaceFormatKHR format = sw.formats[0];
        for (auto& f : sw.formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                format = f;
                break;
            }
        }
        swapchainFormat = format.format;

        // Choose extent
        VkExtent2D extent;
        if (sw.capabilities.currentExtent.width != UINT32_MAX) {
            extent = sw.capabilities.currentExtent;
        } else {
            extent.width = std::clamp((uint32_t)width,
                sw.capabilities.minImageExtent.width,
                sw.capabilities.maxImageExtent.width);
            extent.height = std::clamp((uint32_t)height,
                sw.capabilities.minImageExtent.height,
                sw.capabilities.maxImageExtent.height);
        }
        swapchainExtent = extent;

        uint32_t imageCount = sw.capabilities.minImageCount + 1;
        if (sw.capabilities.maxImageCount > 0 && imageCount > sw.capabilities.maxImageCount)
            imageCount = sw.capabilities.maxImageCount;

        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface;
        ci.minImageCount = imageCount;
        ci.imageFormat = swapchainFormat;
        ci.imageColorSpace = format.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = sw.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;

        VK_CTX_CHECK(vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain));

        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
    }

    void createImageViews() {
        swapchainViews.resize(swapchainImages.size());
        for (size_t i = 0; i < swapchainImages.size(); i++) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainFormat;
            ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ci.subresourceRange.baseMipLevel = 0;
            ci.subresourceRange.levelCount = 1;
            ci.subresourceRange.baseArrayLayer = 0;
            ci.subresourceRange.layerCount = 1;
            VK_CTX_CHECK(vkCreateImageView(device, &ci, nullptr, &swapchainViews[i]));
        }
    }

    // ================================================================
    //  Depth resources
    // ================================================================

    void createDepthResources() {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, depthFormat, &props);
        if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            depthFormat = VK_FORMAT_D16_UNORM;
        }

        createImage(swapchainExtent.width, swapchainExtent.height, depthFormat,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthImage, depthImageMemory);

        depthView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    void destroyDepthResources() {
        if (depthView)       vkDestroyImageView(device, depthView, nullptr);
        if (depthImage)      vkDestroyImage(device, depthImage, nullptr);
        if (depthImageMemory) vkFreeMemory(device, depthImageMemory, nullptr);
    }

    // ================================================================
    //  Render pass
    // ================================================================

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainFormat;
        colorAttachment.samples = msaaSamples;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthFormat;
        depthAttachment.samples = msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        VkRenderPassCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = (uint32_t)attachments.size();
        ci.pAttachments = attachments.data();
        ci.subpassCount = 1;
        ci.pSubpasses = &subpass;
        ci.dependencyCount = 1;
        ci.pDependencies = &dep;

        VK_CTX_CHECK(vkCreateRenderPass(device, &ci, nullptr, &renderPass));
    }

    // ================================================================
    //  Framebuffers
    // ================================================================

    void createFramebuffers() {
        framebuffers.resize(swapchainViews.size());
        for (size_t i = 0; i < swapchainViews.size(); i++) {
            std::array<VkImageView, 2> views = {
                swapchainViews[i],
                depthView
            };
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass;
            ci.attachmentCount = (uint32_t)views.size();
            ci.pAttachments = views.data();
            ci.width = swapchainExtent.width;
            ci.height = swapchainExtent.height;
            ci.layers = 1;
            VK_CTX_CHECK(vkCreateFramebuffer(device, &ci, nullptr, &framebuffers[i]));
        }
    }

    // ================================================================
    //  Command buffers + sync
    // ================================================================

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = (uint32_t)commandBuffers.size();
        VK_CTX_CHECK(vkAllocateCommandBuffers(device, &ai, commandBuffers.data()));
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES);
        renderFinishedSemaphores.resize(MAX_FRAMES);
        inFlightFences.resize(MAX_FRAMES);

        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES; i++) {
            VK_CTX_CHECK(vkCreateSemaphore(device, &si, nullptr, &imageAvailableSemaphores[i]));
            VK_CTX_CHECK(vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphores[i]));
            VK_CTX_CHECK(vkCreateFence(device, &fi, nullptr, &inFlightFences[i]));
        }
    }

    // ================================================================
    //  Drawing — begin/end frame
    // ================================================================

    VkCommandBuffer beginFrame() {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            return VK_NULL_HANDLE;  // caller should recreate swapchain
        }

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CTX_CHECK(vkBeginCommandBuffer(commandBuffers[currentFrame], &bi));

        // Choose target
        VkFramebuffer fb;
        VkRenderPass rp;
        VkExtent2D extent;

        if (renderingOffscreen && activeOffscreenFramebuffer != VK_NULL_HANDLE) {
            fb = activeOffscreenFramebuffer;
            rp = activeOffscreenRenderPass;
            extent = {offscreenWidth, offscreenHeight};
        } else {
            fb = framebuffers[imageIndex];
            rp = renderPass;
            extent = swapchainExtent;
        }

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rbi{};
        rbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rbi.renderPass = rp;
        rbi.framebuffer = fb;
        rbi.renderArea.offset = {0, 0};
        rbi.renderArea.extent = extent;
        rbi.clearValueCount = (uint32_t)clearValues.size();
        rbi.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[currentFrame], &rbi,
            VK_SUBPASS_CONTENTS_INLINE);

        currentImageIndex = imageIndex;
        return commandBuffers[currentFrame];
    }

    void endFrame() {
        vkCmdEndRenderPass(commandBuffers[currentFrame]);
        VK_CTX_CHECK(vkEndCommandBuffer(commandBuffers[currentFrame]));

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = waitSemaphores;
        si.pWaitDstStageMask = waitStages;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = signalSemaphores;

        VK_CTX_CHECK(vkQueueSubmit(graphicsQueue, 1, &si, inFlightFences[currentFrame]));

        if (!renderingOffscreen) {
            VkPresentInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = signalSemaphores;
            pi.swapchainCount = 1;
            pi.pSwapchains = &swapchain;
            pi.pImageIndices = &currentImageIndex;
            vkQueuePresentKHR(presentQueue, &pi);
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES;
    }

    uint32_t currentImageIndex = 0;
    uint32_t offscreenWidth = 0;
    uint32_t offscreenHeight = 0;

    // ================================================================
    //  Offscreen rendering helpers (for VectorDisplay)
    // ================================================================

    // Create a color image + view for offscreen rendering
    void createOffscreenImage(uint32_t w, uint32_t h, VkFormat format,
                              VkImage& image, VkDeviceMemory& memory,
                              VkImageView& view) {
        createImage(w, h, format, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, memory);
        view = createImageView(image, format, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // Create a render pass for offscreen rendering to a texture
    VkRenderPass createOffscreenRenderPass(VkFormat colorFormat) {
        VkAttachmentDescription colorAtt{};
        colorAtt.format = colorFormat;
        colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentDescription depthAtt{};
        depthAtt.format = depthFormat;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dep.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        std::array<VkAttachmentDescription, 2> atts = {colorAtt, depthAtt};
        VkRenderPassCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = (uint32_t)atts.size();
        ci.pAttachments = atts.data();
        ci.subpassCount = 1;
        ci.pSubpasses = &subpass;
        ci.dependencyCount = 1;
        ci.pDependencies = &dep;

        VkRenderPass rp;
        VK_CTX_CHECK(vkCreateRenderPass(device, &ci, nullptr, &rp));
        return rp;
    }

    VkFramebuffer createOffscreenFramebuffer(VkRenderPass rp, VkImageView colorView,
                                              VkImageView depthView, uint32_t w, uint32_t h) {
        std::array<VkImageView, 2> views = {colorView, depthView};
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = rp;
        ci.attachmentCount = (uint32_t)views.size();
        ci.pAttachments = views.data();
        ci.width = w;
        ci.height = h;
        ci.layers = 1;
        VkFramebuffer fb;
        VK_CTX_CHECK(vkCreateFramebuffer(device, &ci, nullptr, &fb));
        return fb;
    }

    // ================================================================
    //  Utility: image creation, view, memory, transition, copy
    // ================================================================

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }

    void createImage(uint32_t w, uint32_t h, VkFormat format,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties,
                     VkImage& image, VkDeviceMemory& memory)
    {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = tiling;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CTX_CHECK(vkCreateImage(device, &ci, nullptr, &image));

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, image, &memReqs);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = memReqs.size;
        ai.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
        VK_CTX_CHECK(vkAllocateMemory(device, &ai, nullptr, &memory));
        vkBindImageMemory(device, image, memory, 0);
    }

    VkImageView createImageView(VkImage image, VkFormat format,
                                VkImageAspectFlags aspect) {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = format;
        ci.subresourceRange.aspectMask = aspect;
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount = 1;
        VkImageView view;
        VK_CTX_CHECK(vkCreateImageView(device, &ci, nullptr, &view));
        return view;
    }

    // Upload CPU pixel data to a device-local VkImage, return the VkImageView
    VkImageView uploadTexture(const void* pixelData, uint32_t w, uint32_t h,
                              VkFormat format, size_t dataSize)
    {
        VkImage image;
        VkDeviceMemory imageMemory;
        createImage(w, h, format, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imageMemory);

        // Create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        createBuffer(dataSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingMemory);

        void* mapped;
        vkMapMemory(device, stagingMemory, 0, dataSize, 0, &mapped);
        memcpy(mapped, pixelData, dataSize);
        vkUnmapMemory(device, stagingMemory);

        // Transition to transfer dst
        transitionImageLayout(image, format,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // Copy buffer to image
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, stagingBuffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        endSingleTimeCommands(cmd);

        // Transition to shader read
        transitionImageLayout(image, format,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);

        VkImageView view = createImageView(image, format, VK_IMAGE_ASPECT_COLOR_BIT);
        // Note: image and imageMemory leak — they live for the lifetime of the app.
        // In a production system, track these for cleanup.
        return view;
    }

    // Upload CPU data to an existing VkImage (transitions to SHADER_READ_ONLY_OPTIMAL)
    void uploadTextureToImage(const void* pixelData, VkImage image,
                              VkFormat format, uint32_t w, uint32_t h, size_t dataSize)
    {
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        createBuffer(dataSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingMemory);

        void* mapped;
        vkMapMemory(device, stagingMemory, 0, dataSize, 0, &mapped);
        memcpy(mapped, pixelData, dataSize);
        vkUnmapMemory(device, stagingMemory);

        transitionImageLayout(image, format,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, stagingBuffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        endSingleTimeCommands(cmd);

        transitionImageLayout(image, format,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

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
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    void transitionImageLayout(VkImage image, VkFormat format,
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               uint32_t mipLevels = 1, uint32_t layerCount = 1) {
        VkCommandBuffer cmd = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = layerCount;

        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        } else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }

        VkPipelineStageFlags srcStage;
        VkPipelineStageFlags dstStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                   newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                   newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        } else {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = 0;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }

        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(cmd);
    }

    // Copy a buffer into an image
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t w, uint32_t h) {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, buffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        endSingleTimeCommands(cmd);
    }

    // Create a buffer
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer& buffer, VkDeviceMemory& memory) {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = size;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CTX_CHECK(vkCreateBuffer(device, &ci, nullptr, &buffer));

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, buffer, &memReqs);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = memReqs.size;
        ai.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
        VK_CTX_CHECK(vkAllocateMemory(device, &ai, nullptr, &memory));
        vkBindBufferMemory(device, buffer, memory, 0);
    }

    // ================================================================
    //  Shader module from SPIR-V
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

    // Compile GLSL → SPIR-V at runtime using glslangValidator
    std::vector<uint8_t> compileGLSL(const std::string& source,
                                     const std::string& stage) {
        std::string tempFile = "/tmp/hvml_shader_" + stage + ".glsl";
        std::string spvFile  = "/tmp/hvml_shader_" + stage + ".spv";

        std::ofstream f(tempFile);
        f << source;
        f.close();

        std::string cmd = "glslangValidator --auto-map-locations -V -S " + stage +
                          " -o " + spvFile + " " + tempFile + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "[vulkan-ctx] Failed to run glslangValidator" << std::endl;
            return {};
        }
        char buf[256];
        std::string result;
        while (fgets(buf, sizeof(buf), pipe)) result += buf;
        int rc = pclose(pipe);
        if (rc != 0) {
            // Check if temp file exists and has content
            std::ifstream checkFile(tempFile);
            if (checkFile) {
                std::string content((std::istreambuf_iterator<char>(checkFile)),
                                    std::istreambuf_iterator<char>());
                checkFile.close();
                std::cerr << "[vulkan-ctx] Shader compilation failed (" << stage << "), rc=" << rc << ":\n"
                          << result << "\n--- Temp file content (" << content.size() << " bytes) ---\n"
                          << content.substr(0, 500) << std::endl;
            } else {
                std::cerr << "[vulkan-ctx] Shader compilation failed (" << stage << "), rc=" << rc
                          << " — temp file missing: " << tempFile
                          << " (source size was " << source.size() << ")" << std::endl;
            }
            return {};
        }

        std::ifstream spv(spvFile, std::ios::binary | std::ios::ate);
        if (!spv) return {};
        size_t size = spv.tellg();
        spv.seekg(0);
        std::vector<uint8_t> data(size);
        spv.read(reinterpret_cast<char*>(data.data()), size);
        spv.close();
        std::remove(tempFile.c_str());
        std::remove(spvFile.c_str());
        return data;
    }

    // ================================================================
    //  Swapchain recreation (on resize)
    // ================================================================

    void recreateSwapchain(SDL_Window* window, int width, int height) {
        vkDeviceWaitIdle(device);

        for (auto& fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
        framebuffers.clear();
        for (auto& v : swapchainViews) vkDestroyImageView(device, v, nullptr);
        swapchainViews.clear();
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
        destroyDepthResources();
        if (renderPass) vkDestroyRenderPass(device, renderPass, nullptr);

        createSwapchain(window, width, height);
        createImageViews();
        createDepthResources();
        createRenderPass();
        createFramebuffers();
    }
};

// The plugin's set_rendering_device is an extern "C" function in the vulkan plugin .so
// We use dlsym to find it at runtime since the plugin is loaded via dlopen.
#include <dlfcn.h>

inline int VulkanContext::getRenderingDeviceIndex() {
    typedef int (*get_idx_fn)();
    static get_idx_fn fn = (get_idx_fn)dlsym(RTLD_DEFAULT, "get_rendering_device_index");
    if (!fn) return 0;
    int idx = fn();
    return idx >= 0 ? idx : 0;
}

inline MemoryType VulkanContext::getRenderingMemoryType() {
    typedef int (*get_mt_fn)();
    static get_mt_fn fn = (get_mt_fn)dlsym(RTLD_DEFAULT, "get_rendering_device_memory_type");
    if (!fn) return MemoryType::kDDR;
    return (MemoryType)fn();
}

inline void VulkanContext::shareWithPlugin() {
    typedef void (*set_rd_fn)(VkDevice, VkPhysicalDevice, VkQueue, uint32_t, VkCommandPool, int);
    static set_rd_fn set_rendering_device = (set_rd_fn)dlsym(RTLD_DEFAULT, "set_rendering_device");
    if (!set_rendering_device) {
        std::cerr << "[vulkan-ctx] set_rendering_device not found in plugin!" << std::endl;
        return;
    }

    // Find the device index by enumerating physical devices
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physDevs(count);
    vkEnumeratePhysicalDevices(instance, &count, physDevs.data());

    int deviceIndex = -1;
    for (uint32_t i = 0; i < count; i++) {
        if (physDevs[i] == physicalDevice) {
            deviceIndex = (int)i;
            break;
        }
    }

    set_rendering_device(device, physicalDevice, graphicsQueue,
        graphicsFamily, commandPool, deviceIndex);
}

#endif // VULKAN_CONTEXT_HPP
