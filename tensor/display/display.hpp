#ifndef VECTOR_DISPLAY_HPP
#define VECTOR_DISPLAY_HPP

//
//  display.hpp — Window: an SDL window with a Vulkan swapchain, input
//  handling and a frame loop.  Any number of windows can exist; they share
//  the one VulkanContext (and so every tensor, texture and pipeline).
//
//      Window window({1280, 720}, WP_RESIZABLE);
//      DisplayTensor<uint84> canvas({1280, 720});          // render texture
//      window.add_on_update([&](CurrentScreenInputInfo& in, VkCommandBuffer cmd) {
//          canvas.render(cmd, [&]{ mesh.bind(cmd); mesh.draw(cmd); });
//          canvas.draw(cmd);                                // onto the window
//      });
//      window.displayLoop();
//
//  Inside a frame callback the window is the current render target;
//  DisplayTensor::render()/begin_render() switch to a texture and switch back.
//

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <set>
#include <chrono>
#include <memory>
#include "tensor.hpp"
#include "device/device.hpp"
#include "vector/vectors.hpp"
#include "display/vulkan_context.hpp"
#include "display/displaytensor.hpp"



class CurrentScreenInputInfo {
private:
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    bool is_fullscreen = false;
    int mouse_x = 0;
    int mouse_y = 0;
    int mouse_move_x = 0;
    int mouse_move_y = 0;
    bool mouse_left_button = false;
    bool mouse_left_button_released = false;
    bool mouse_right_button = false;
    bool mouse_middle_button = false;
    bool mouse_wheel_up = false;
    bool mouse_wheel_down = false;
    bool mouse_wheel_left = false;
    bool mouse_wheel_right = false;
    float32x4 selectedarea = float32x4(0, 0, 0, 0);
    float32x2 lastClicked = float32x2(0, 0);
    
    std::map<SDL_Keycode, bool> key_states;
    std::map<SDL_Keycode, bool> key_pressed;  // Edge detection for key down
    std::map<SDL_Keycode, bool> key_released; // Edge detection for key up
    std::set<int> mouse_buttons_pressed;
    int accumulated_mouse_x = 0;
    int accumulated_mouse_y = 0;
    bool mouse_grabbed = false;
    bool mouse_visible = true;
    
public:
    int32x2 relativeWindowMove = int32x2(0, 0);
    int32x2 currentWindowPosition = int32x2(0, 0);
    bool just_selected_area = false;
    
    void updateMousePositionAbsolute(int new_x, int new_y) {
        mouse_move_x = new_x - mouse_x;
        mouse_move_y = new_y - mouse_y;
        mouse_x = new_x;
        mouse_y = new_y;
    }

    void updateMouseMotion(int dx, int dy) {
        mouse_move_x = dx;
        mouse_move_y = dy;
    }

    float32x4 getSelectedArea() const {
        return selectedarea;
    }

    float32x2 getGlobalMousePosition() const {
        float gx, gy;
        SDL_GetGlobalMouseState(&gx, &gy);
        return float32x2(gx, gy);
    }

    int32x2 getLocalMousePosition() const {
        return int32x2(mouse_x, mouse_y);
    }

    int32x4 getLocalSelectedArea() const {
        return int32x4(
            selectedarea[0] - currentWindowPosition[0],
            selectedarea[1] - currentWindowPosition[1],
            selectedarea[2],
            selectedarea[3]
        );
    }

    void updateMouseButtonState(int button_code, bool pressed) {
        switch (button_code) {
            case SDL_BUTTON_LEFT:
                mouse_left_button = pressed;
                if(pressed) {
                    lastClicked = getGlobalMousePosition();
                    mouse_buttons_pressed.insert(SDL_BUTTON_LEFT);
                } else {
                    mouse_buttons_pressed.erase(SDL_BUTTON_LEFT);
                    float32x2 mx = getGlobalMousePosition();
                    if (sqrt(pow(mx[0] - lastClicked[0], 2) + pow(mx[1] - lastClicked[1], 2)) > 5.0f) {
                        selectedarea = float32x4(lastClicked.x(), lastClicked[1], mx.x() - lastClicked.x(), mx[1] - lastClicked[1]);
                    }
                    just_selected_area = true;
                }
                break;
            case SDL_BUTTON_RIGHT:
                mouse_right_button = pressed;
                if(pressed) mouse_buttons_pressed.insert(SDL_BUTTON_RIGHT);
                else mouse_buttons_pressed.erase(SDL_BUTTON_RIGHT);
                break;
            case SDL_BUTTON_MIDDLE:
                mouse_middle_button = pressed;
                if(pressed) mouse_buttons_pressed.insert(SDL_BUTTON_MIDDLE);
                else 
                    mouse_buttons_pressed.erase(SDL_BUTTON_MIDDLE);
                break;
        }
    }

    void updateKeyState(SDL_Keycode key, bool pressed) {
        bool was_pressed = key_states[key];
        key_states[key] = pressed;
        
        if (pressed && !was_pressed) {
            key_pressed[key] = true;
        } else if (!pressed && was_pressed) {
            key_released[key] = true;
        }
    }

    bool isKeyPressed(SDL_Keycode key) const {
        auto it = key_states.find(key);
        return it != key_states.end() && it->second;
    }

    bool isKeyJustPressed(SDL_Keycode key) const {
        auto it = key_pressed.find(key);
        return it != key_pressed.end() && it->second;
    }

    bool isKeyJustReleased(SDL_Keycode key) const {
        auto it = key_released.find(key);
        return it != key_released.end() && it->second;
    }

    std::set<int> getMouseButtonsPressed() const {
        return mouse_buttons_pressed;
    }

    std::pair<int, int> getMouseRel() const {
        return {mouse_move_x, mouse_move_y};
    }
    
    void setScreenSize(int new_x, int new_y, int new_width, int new_height) {
        x = new_x;
        y = new_y;
        width = new_width;
        height = new_height;
    }
    
    void setFullscreen(bool fullscreen) {
        is_fullscreen = fullscreen;
    }

    void setMouseGrabbed(bool grabbed) {
        mouse_grabbed = grabbed;
    }

    void setMouseVisible(bool visible) {
        mouse_visible = visible;
    }

    bool isMouseGrabbed() const {
        return mouse_grabbed;
    }

    bool isMouseVisible() const {
        return mouse_visible;
    }
    
    int getX() const { return x; }
    int getY() const { return y; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    bool isFullbackbuffer() const { return is_fullscreen; }
    int getMouseX() const { return mouse_x; }
    int getMouseY() const { return mouse_y; }
    float32x2 getMousePosition() const { return float32x2(mouse_x, mouse_y); }
    float32x2 getMouseMove() const { return float32x2(mouse_move_x, mouse_move_y); }
    float32x4 getScreenSize() const { return float32x4(x, y, width, height); }
    int getMouseMoveX() const { return mouse_move_x; }
    int getMouseMoveY() const { return mouse_move_y; }
    bool isMouseLeftButtonPressed() const { return mouse_left_button; }
    bool isMouseRightButtonPressed() const { return mouse_right_button; }
    bool isMouseMiddleButtonPressed() const { return mouse_middle_button; }
    bool isMouseWheelUp() const { return mouse_wheel_up; }
    bool isMouseWheelDown() const { return mouse_wheel_down; }
    bool isMouseWheelLeft() const { return mouse_wheel_left; }
    bool isMouseWheelRight() const { return mouse_wheel_right; }

    void clearWheelStates() {
        mouse_wheel_up = false;
        mouse_wheel_down = false;
        mouse_wheel_left = false;
        mouse_wheel_right = false;
    }

    void clearKeyEdgeStates() {
        key_pressed.clear();
        key_released.clear();
    }
    
    void clear_mouse_states() {
        just_selected_area = false;
        mouse_move_x = 0;
        mouse_move_y = 0;
        clearWheelStates();
        clearKeyEdgeStates();
    }
};

enum WindowProperties {
    WP_BORDERLESS = SDL_WINDOW_BORDERLESS,
    WP_ALPHA_ENABLED = SDL_WINDOW_TRANSPARENT,
    WP_FULLSCREEN = SDL_WINDOW_FULLSCREEN,
    // WP_CLICKTHROUGH = 1 << 3, // SDL doesn't have a built-in click-through flag, so we define our own
    WP_ON_TOP = SDL_WINDOW_ALWAYS_ON_TOP,
    WP_RESIZABLE = SDL_WINDOW_RESIZABLE
};

struct WindowPropertiesFlags {
    bool borderless = false;
    bool alpha_enabled = false;
    bool fullscreen = false;
    // bool clickthrough = true;
    bool on_top = false;
    bool resizable = false;

    WindowPropertiesFlags(WindowProperties properties) {
        borderless = properties & WP_BORDERLESS;
        alpha_enabled = properties & WP_ALPHA_ENABLED;
        fullscreen = properties & WP_FULLSCREEN;
        // clickthrough = properties & WP_CLICKTHROUGH;
        on_top = properties & WP_ON_TOP;
        resizable = properties & WP_RESIZABLE;
    }

    WindowPropertiesFlags(int flags) {
        borderless = flags & WP_BORDERLESS;
        alpha_enabled = flags & WP_ALPHA_ENABLED;
        fullscreen = flags & WP_FULLSCREEN;
        // clickthrough = flags & WP_CLICKTHROUGH;
        on_top = flags & WP_ON_TOP;
        resizable = flags & WP_RESIZABLE;
    }

    operator WindowProperties() const {
        WindowProperties props = (WindowProperties)0;
        if (borderless) props = (WindowProperties)(props | WP_BORDERLESS);
        if (alpha_enabled) props = (WindowProperties)(props | WP_ALPHA_ENABLED);
        if (fullscreen) props = (WindowProperties)(props | WP_FULLSCREEN);
        // if (clickthrough) props = (WindowProperties)(props | WP_CLICKTHROUGH);
        if (on_top) props = (WindowProperties)(props | WP_ON_TOP);
        if (resizable) props = (WindowProperties)(props | WP_RESIZABLE);
        return props;
    }

        operator int() const {
            int flags = 0;
            if (borderless) flags |= WP_BORDERLESS;
            if (alpha_enabled) flags |= WP_ALPHA_ENABLED;
            if (fullscreen) flags |= WP_FULLSCREEN;
            // if (clickthrough) flags |= WP_CLICKTHROUGH;
            if (on_top) flags |= WP_ON_TOP;
            if (resizable) flags |= WP_RESIZABLE;
            return flags;
        }
    
};

// FPS Clock for frame rate limiting
class Clock {
private:
    std::chrono::steady_clock::time_point last_tick;
    
public:
    Clock() : last_tick(std::chrono::steady_clock::now()) {}
    
    void tick(int fps) {
        auto target_duration = std::chrono::microseconds(1000000 / fps);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_tick);
        
        if (elapsed < target_duration) {
            auto sleep_time = target_duration - elapsed;
            // std::this_thread::sleep_for(sleep_time);
            SDL_Delay(sleep_time.count() / 1000); // Convert microseconds to milliseconds
        }
        
        last_tick = std::chrono::steady_clock::now();
    }
    
    int get_fps() const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last_tick);
        if (elapsed.count() == 0) return 0;
        return 1000000 / elapsed.count();
    }
};

struct Window
{
public:
    // ---- SDL / input -------------------------------------------------------
    void* display = nullptr;               // X11 Display* (if on X11)
    SDL_Window* window = nullptr;
    void* root_window = nullptr;
    int screen = 0;
    int depth = 32;
    int height = 0;
    int width = 0;

    WindowPropertiesFlags properties;
    CurrentScreenInputInfo current_screen_input_info;
    ComputeDeviceBase* device = nullptr;   // the rendering GPU's compute device
    std::vector<std::function<void(CurrentScreenInputInfo&, VkCommandBuffer)>> display_loop_functions;
    Clock clock;

    VulkanContext& vk_ctx;

    // ---- swapchain ---------------------------------------------------------
    bool preferSRGB = true;                // B8G8R8A8_SRGB swapchain when available
    bool vsync = true;                     // FIFO; false → MAILBOX/IMMEDIATE if available
    VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkSurfaceKHR   surface   = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat       swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D     swapchainExtent = {0, 0};
    std::vector<VkSemaphore> renderFinished;        // one per swapchain image

    // The swapchain images, as tensors — the window is drawn to exactly like
    // any other render texture.  backbuffer() is the one being drawn this frame (backbuffer()).
    std::vector<DisplayTensor<uint84>> backbuffers;
    std::unique_ptr<Tensor<float, 2>> depthBuffer;  // shared depth attachment (a tensor)

    // ---- frames in flight ------------------------------------------------
    static constexpr int MAX_FRAMES = 2;
    VkCommandBuffer commandBuffers[MAX_FRAMES] = {};
    VkSemaphore     imageAvailable[MAX_FRAMES] = {};
    VkFence         inFlight[MAX_FRAMES] = {};
    uint32_t        currentFrame = 0;
    uint32_t        currentImage = 0;
    VkCommandBuffer currentCmd = VK_NULL_HANDLE;
    bool            swapchainDirty = false;

    Window(Shape<2> shape = 0, WindowPropertiesFlags properties = (WindowProperties)0,
           const char* title = "hvml")
        : height(shape[1]),
          width(shape[0]),
          properties(properties),
          vk_ctx(VulkanContext::getInstanceOnly())
    {
        if (!SDL_WasInit(SDL_INIT_VIDEO) && !SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error("SDL_Init failed: " + std::string(SDL_GetError()));
        }
        window_count()++;

        window = SDL_CreateWindow(title, width, height, SDL_WINDOW_VULKAN | int(properties));
        if (!window) {
            throw std::runtime_error("Failed to create SDL window: " + std::string(SDL_GetError()));
        }
        display = (void*)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        screen = SDL_GetDisplayForWindow(window);
        SDL_ShowWindow(window);

        if (!SDL_Vulkan_CreateSurface(window, vk_ctx.instance, nullptr, &surface)) {
            throw std::runtime_error("Failed to create Vulkan surface: " + std::string(SDL_GetError()));
        }
        vk_ctx.ensureDevice(surface);

        try {
            device = &global_device_manager.get_compute_device(kVULKAN, vk_ctx.getRenderingDeviceIndex());
        } catch (...) {
            device = nullptr;
        }

        createFrameResources();
        createSwapchain();
    }

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    ~Window() {
        if (vk_ctx.device) {
            vkDeviceWaitIdle(vk_ctx.device);
            destroySwapchain(true);
            for (int i = 0; i < MAX_FRAMES; i++) {
                vkDestroySemaphore(vk_ctx.device, imageAvailable[i], nullptr);
                vkDestroyFence(vk_ctx.device, inFlight[i], nullptr);
            }
            vkFreeCommandBuffers(vk_ctx.device, vk_ctx.commandPool, MAX_FRAMES, commandBuffers);
        }
        if (surface) vkDestroySurfaceKHR(vk_ctx.instance, surface, nullptr);
        if (window) SDL_DestroyWindow(window);
        if (--window_count() == 0) SDL_Quit();
    }

    // ================================================================
    //  Frames
    // ================================================================

    // Start a frame: returns a command buffer recording into this window
    // (cleared), or VK_NULL_HANDLE if there is nothing to draw into right
    // now (minimised / swapchain being recreated).
    VkCommandBuffer beginFrame() {
        if (currentCmd) return currentCmd;
        if (swapchainDirty || swapchain == VK_NULL_HANDLE) recreateSwapchain();
        if (swapchain == VK_NULL_HANDLE) return VK_NULL_HANDLE;

        vkWaitForFences(vk_ctx.device, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);

        VkResult res = vkAcquireNextImageKHR(vk_ctx.device, swapchain, UINT64_MAX,
                                             imageAvailable[currentFrame], VK_NULL_HANDLE, &currentImage);
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchainDirty = true;
            return VK_NULL_HANDLE;
        }
        if (res == VK_SUBOPTIMAL_KHR) swapchainDirty = true;
        vkResetFences(vk_ctx.device, 1, &inFlight[currentFrame]);

        VkCommandBuffer cmd = commandBuffers[currentFrame];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        backbuffer().set_clear_color(clearColor.float32[0], clearColor.float32[1],
                                 clearColor.float32[2], clearColor.float32[3]);
        backbuffer().begin_render(cmd, /*clear=*/true);
        currentCmd = cmd;
        return cmd;
    }

    // Submit and present the frame started by beginFrame().
    void endFrame() {
        if (!currentCmd) return;
        VkCommandBuffer cmd = currentCmd;
        currentCmd = VK_NULL_HANDLE;

        vk_ctx.endAllTargets(cmd);
        vkEndCommandBuffer(cmd);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &imageAvailable[currentFrame];
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &renderFinished[currentImage];
        VK_CTX_CHECK(vkQueueSubmit(vk_ctx.graphicsQueue, 1, &si, inFlight[currentFrame]));

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &renderFinished[currentImage];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &currentImage;
        VkResult res = vkQueuePresentKHR(vk_ctx.graphicsQueue, &pi);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) swapchainDirty = true;

        currentFrame = (currentFrame + 1) % MAX_FRAMES;
    }

    // The swapchain image being drawn this frame, as a tensor.
    DisplayTensor<uint84>& backbuffer() { return backbuffers[currentImage]; }

    // Make the window the current render target again (e.g. after drawing
    // into textures without ending them).  Keeps what was drawn so far.
    void activateBackBuffer(VkCommandBuffer cmd) {
        if (vk_ctx.currentTarget() != &backbuffer().render_target()) {
            backbuffer().begin_render(cmd, /*clear=*/false);
        } else {
            vk_ctx.setViewport(cmd, swapchainExtent.width, swapchainExtent.height);
        }
    }

    void displayLoop() {
        bool running = true;
        while (running) {
            resizeDisplay();
            VkCommandBuffer cmd = beginFrame();
            if (cmd != VK_NULL_HANDLE) {
                for (const auto& callback : display_loop_functions) {
                    callback(current_screen_input_info, cmd);
                }
                endFrame();
            } else {
                SDL_Delay(10);
            }
            running = processEvents();
            updateDisplay();
        }
        vkDeviceWaitIdle(vk_ctx.device);
    }

    void add_on_update(std::function<void(CurrentScreenInputInfo&, VkCommandBuffer)> func) {
        display_loop_functions.push_back(func);
    }

    // ================================================================
    //  Window management and input
    // ================================================================

    void setWindowCaption(const char* title) { SDL_SetWindowTitle(window, title); }

    void setMouseGrab(bool grab) {
        SDL_SetWindowMouseGrab(window, grab);
        SDL_SetWindowRelativeMouseMode(window, grab);
        current_screen_input_info.setMouseGrabbed(grab);
    }

    void setMouseVisible(bool visible) {
        if (visible) SDL_ShowCursor(); else SDL_HideCursor();
        current_screen_input_info.setMouseVisible(visible);
    }

    std::pair<int, int> getWindowSize() const {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        return {w, h};
    }

    void setWindowBorderless() { SDL_SetWindowBordered(window, false); }
    void enableAlphaBlending() { SDL_SetWindowOpacity(window, 1.0f); }

    void setWindowOpacity(float opacity) {
        if (!properties.alpha_enabled) return;
        SDL_SetWindowOpacity(window, opacity);
    }

    void updateDisplay() {
        auto oldWindowPosition = current_screen_input_info.currentWindowPosition;
        SDL_GetWindowPosition(window, &current_screen_input_info.currentWindowPosition[0],
                              &current_screen_input_info.currentWindowPosition[1]);
        current_screen_input_info.relativeWindowMove = int32x2(
            current_screen_input_info.currentWindowPosition.x() - oldWindowPosition.x(),
            current_screen_input_info.currentWindowPosition.y() - oldWindowPosition.y());
    }

    void resizeDisplay() {
        int w, h;
        SDL_GetWindowSizeInPixels(window, &w, &h);
        if (w != width || h != height) {
            width = w;
            height = h;
            swapchainDirty = true;
        }
    }

    bool processEvents() {
        SDL_Event e;
        current_screen_input_info.clear_mouse_states();
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_EVENT_QUIT:
                case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                    return false;
                case SDL_EVENT_KEY_DOWN:
                    current_screen_input_info.updateKeyState(e.key.key, true);
                    if (e.key.key == SDLK_ESCAPE) return false;
                    break;
                case SDL_EVENT_KEY_UP:
                    current_screen_input_info.updateKeyState(e.key.key, false);
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    current_screen_input_info.updateMouseButtonState(e.button.button, true);
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    current_screen_input_info.updateMouseButtonState(e.button.button, false);
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    if (current_screen_input_info.isMouseGrabbed())
                        current_screen_input_info.updateMouseMotion(e.motion.xrel, e.motion.yrel);
                    else
                        current_screen_input_info.updateMousePositionAbsolute(e.motion.x, e.motion.y);
                    break;
                case SDL_EVENT_MOUSE_WHEEL:
                    current_screen_input_info.clearWheelStates();
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                    resizeDisplay();
                    break;
                default:
                    break;
            }
        }
        return true;
    }

    void moveWindow(int x, int y) { SDL_SetWindowPosition(window, x, y); }
    void resizeWindow(int w, int h) { SDL_SetWindowSize(window, w, h); }

    void setFullscreen(bool fullscreen) {
        SDL_SetWindowFullscreen(window, fullscreen);
        properties.fullscreen = fullscreen;
        current_screen_input_info.setFullscreen(fullscreen);
    }

    static SDL_Keycode getKeyCode(const char* key_name) { return SDL_GetKeyFromName(key_name); }

    bool isKeyPressed(SDL_Keycode key) const { return current_screen_input_info.isKeyPressed(key); }

    // ================================================================
    //  Swapchain
    // ================================================================

    void recreateSwapchain() {
        vkDeviceWaitIdle(vk_ctx.device);
        destroySwapchain(false);
        createSwapchain();
        swapchainDirty = false;
    }

private:
    static int& window_count() {
        static int count = 0;
        return count;
    }

    void createFrameResources() {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = vk_ctx.commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES;
        VK_CTX_CHECK(vkAllocateCommandBuffers(vk_ctx.device, &ai, commandBuffers));

        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; i++) {
            VK_CTX_CHECK(vkCreateSemaphore(vk_ctx.device, &si, nullptr, &imageAvailable[i]));
            VK_CTX_CHECK(vkCreateFence(vk_ctx.device, &fi, nullptr, &inFlight[i]));
        }
    }

    void createSwapchain() {
        VkPhysicalDevice pd = vk_ctx.physicalDevice;
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps);

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX) {
            extent.width = std::clamp((uint32_t)std::max(width, 1), caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height = std::clamp((uint32_t)std::max(height, 1), caps.minImageExtent.height, caps.maxImageExtent.height);
        }
        if (extent.width == 0 || extent.height == 0) return;   // minimised
        swapchainExtent = extent;
        width = (int)extent.width;
        height = (int)extent.height;

        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, formats.data());
        VkSurfaceFormatKHR format = formats[0];
        VkFormat wanted = preferSRGB ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
        for (auto& f : formats) {
            if (f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { format = f; break; }
        }
        swapchainFormat = format.format;

        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        if (!vsync) {
            vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, nullptr);
            std::vector<VkPresentModeKHR> modes(count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, modes.data());
            for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
            if (mode == VK_PRESENT_MODE_FIFO_KHR)
                for (auto m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
        }

        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface;
        ci.minImageCount = imageCount;
        ci.imageFormat = swapchainFormat;
        ci.imageColorSpace = format.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        (caps.supportedUsageFlags & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                                ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                : (VkCompositeAlphaFlagBitsKHR)(caps.supportedCompositeAlpha & -caps.supportedCompositeAlpha);
        ci.presentMode = mode;
        ci.clipped = VK_TRUE;
        ci.oldSwapchain = swapchain;

        VkSwapchainKHR created;
        VK_CTX_CHECK(vkCreateSwapchainKHR(vk_ctx.device, &ci, nullptr, &created));
        if (swapchain) vkDestroySwapchainKHR(vk_ctx.device, swapchain, nullptr);
        swapchain = created;

        vkGetSwapchainImagesKHR(vk_ctx.device, swapchain, &imageCount, nullptr);
        std::vector<VkImage> images(imageCount);
        vkGetSwapchainImagesKHR(vk_ctx.device, swapchain, &imageCount, images.data());

        // Depth attachment — a tensor like everything else
        depthBuffer = std::make_unique<Tensor<float, 2>>(DisplayTensor<float>::metadata(
            Shape<2>{(long)extent.width, (long)extent.height}, ComputeType::kVULKANTEXTURE,
            AllocationFlags::kSURFACE | AllocationFlags::kDEPTH));

        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        renderFinished.resize(imageCount);
        backbuffers.clear();
        for (uint32_t i = 0; i < imageCount; i++) {
            VK_CTX_CHECK(vkCreateSemaphore(vk_ctx.device, &si, nullptr, &renderFinished[i]));
            backbuffers.push_back(DisplayTensor<uint84>::adopt(
                (void*)images[i], swapchainFormat, extent.width, extent.height,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, ci.imageUsage));
            backbuffers.back().attach_depth_buffer(*depthBuffer);
        }
    }

    void destroySwapchain(bool destroyHandle) {
        backbuffers.clear();          // releases their views and framebuffers
        depthBuffer.reset();
        for (auto s : renderFinished) vkDestroySemaphore(vk_ctx.device, s, nullptr);
        renderFinished.clear();
        if (destroyHandle && swapchain) {
            vkDestroySwapchainKHR(vk_ctx.device, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }
    }
};

// Old names
using VulkanDisplay = Window;
using BasicDisplay = Window;

#endif
