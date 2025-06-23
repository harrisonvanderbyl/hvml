
#include "display/display_manager.hpp"
#include "display/selected_text.hpp"
#ifndef VECTOR_DISPLAY_HPP
#define VECTOR_DISPLAY_HPP



class CurrentScreenInputInfo {
private:
    MultiDisplayManager* display_manager = nullptr;
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
    bool mouse_right_button = false;
    bool mouse_middle_button = false;
    bool mouse_wheel_up = false;
    bool mouse_wheel_down = false;
    bool mouse_wheel_left = false;
    bool mouse_wheel_right = false;
    bool key_pressed[KEY_MAX] = {false};
    float4 selectedarea = float4(0, 0, 0, 0);
    float2 lastClicked = float2(0, 0);
    
    // Raw input tracking
    std::map<int, bool> raw_key_states;
    int accumulated_mouse_x = 0;
    int accumulated_mouse_y = 0;
    
public:
    SelectedTextReader* selected_text_reader = nullptr;
    
    
    void updateMousePositionAbsolute(int new_x, int new_y) {
        mouse_move_x = (new_x-x) - mouse_x;
        mouse_move_y = (new_y-y) - mouse_y;
        mouse_x = (new_x-x);
        mouse_y = (new_y-y);
        accumulated_mouse_x = (new_x-x);
        accumulated_mouse_y = (new_y-y);
    }

    float4 getSelectedArea() const {
        return selectedarea;
    }
    
    void updateMouseButtonState(int button_code, bool pressed) {
        switch (button_code) {
            case BTN_LEFT:
                std::cout << "Mouse left button " << (pressed ? "pressed" : "released") << std::endl;
                mouse_left_button = pressed;
                if(pressed) {
                    lastClicked = getGlobalMousePosition();
                }else{
                    float2 mx = getGlobalMousePosition();
                    if (sqrt(pow(mx.x - lastClicked.x, 2) + pow(mx.y - lastClicked.y, 2)) > 5.0f) {
                        selectedarea = float4(lastClicked.x, lastClicked.y, mx.x - lastClicked.x, mx.y - lastClicked.y);
                    }
                    std::cout << "Selected area: " << selectedarea.x << ", " << selectedarea.y << ", " 
                              << selectedarea.z << ", " << selectedarea.w << std::endl;
                }
                break;
            case BTN_RIGHT:
                mouse_right_button = pressed;
                break;
            case BTN_MIDDLE:
                mouse_middle_button = pressed;
                break;
        }
    }
    
    void updateMouseWheel(int axis, int value) {
        if (axis == REL_WHEEL) {
            mouse_wheel_up = (value > 0);
            mouse_wheel_down = (value < 0);
        } else if (axis == REL_HWHEEL) {
            mouse_wheel_left = (value < 0);
            mouse_wheel_right = (value > 0);
        }
    }
    
    void updateKeyState(int keycode, bool pressed) {
        if (keycode >= 0 && keycode < KEY_MAX) {
            key_pressed[keycode] = pressed;
            raw_key_states[keycode] = pressed;
        }
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
    
    // Getters
    int getX() const { return x; }
    int getY() const { return y; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    bool isFullscreen() const { return is_fullscreen; }
    int getMouseX() const { return mouse_x; }
    int getMouseY() const { return mouse_y; }
    float2 getMousePosition() const { return float2(mouse_x, mouse_y); }
    float2 getMouseMove() const { return float2(mouse_move_x, mouse_move_y); }
    float4 getScreenSize() const { return float4(x, y, width, height); }
    int getMouseMoveX() const { return mouse_move_x; }
    int getMouseMoveY() const { return mouse_move_y; }
    bool isMouseLeftButtonPressed() const { return mouse_left_button; }
    bool isMouseRightButtonPressed() const { return mouse_right_button; }
    bool isMouseMiddleButtonPressed() const { return mouse_middle_button; }
    bool isMouseWheelUp() const { return mouse_wheel_up; }
    bool isMouseWheelDown() const { return mouse_wheel_down; }
    bool isMouseWheelLeft() const { return mouse_wheel_left; }
    bool isMouseWheelRight() const { return mouse_wheel_right; }
    bool isKeyPressed(int keycode) const {
        if (keycode >= 0 && keycode < KEY_MAX) {
            return key_pressed[keycode];
        }
        return false;
    }
    
    // Get current display info
    const DisplayInfo* getCurrentDisplayInfo() const {
        if (display_manager) {
            return display_manager->getCurrentDisplay();
        }
        return nullptr;
    }
    
    // Get all displays
    std::vector<DisplayInfo> getAllDisplays() const {
        if (display_manager) {
            return display_manager->getDisplays();
        }
        return {};
    }
    
    int getCurrentDisplayIndex() const {
        if (display_manager) {
            return display_manager->getCurrentDisplayIndex();
        }
        return 0;
    }

    void setDisplayManager(MultiDisplayManager* manager) {
        display_manager = manager;
    }
    MultiDisplayManager* getDisplayManager() const {
        return display_manager;
    }

    void clearWheelStates() {
        mouse_wheel_up = false;
        mouse_wheel_down = false;
        mouse_wheel_left = false;
        mouse_wheel_right = false;
    }

    // global coordinates for mouse position
    float2 getGlobalMousePosition() const {
        if (display_manager) {
            const DisplayInfo* display = display_manager->getCurrentDisplay();
            if (display) {
                auto local_pos = display->localToGlobal(mouse_x, mouse_y);
                return float2(local_pos.first, local_pos.second);
            }
        }
        return float2(mouse_x, mouse_y);
    }
    
};

class VectorDisplay : public Tensor<uint84, 2> {
public:
    Display* display = nullptr;
    Window window;
    Window* root_window = nullptr;
    GC gc;
    XImage* ximage = nullptr;
    Visual* visual = nullptr;
    Colormap colormap;
    int screen;
    int depth;
    bool borderless = false;
    bool alpha_enabled = false;
    bool is_fullscreen = false;
    CurrentScreenInputInfo current_screen_input_info;
    std::vector<std::function<void(CurrentScreenInputInfo&)>> display_loop_functions;
    
    // Direct input reading
    DirectInputReader input_reader;
    
    // Multi-display support
    MultiDisplayManager display_manager;
    
    VectorDisplay(Shape<2> shape = 0, bool borderless = true, bool enable_alpha = true, bool fullscreen = true)
        : Tensor<uint84, 2>(shape), borderless(borderless), alpha_enabled(enable_alpha), is_fullscreen(fullscreen), display_manager(nullptr) {
        
        // Initialize the display with a black background
        for (int y = 0; y < shape[0]; y++) {
            for (int x = 0; x < shape[1]; x++) {
                (*this)[y][x] = 0x00000000; // ARGB format: transparent black
            }
        }
        
        // Open connection to X server
        display = XOpenDisplay(nullptr);
        if (!display) {
            std::cerr << "Cannot open X display" << std::endl;
            return;
        }
        
        // Initialize display manager
        display_manager = MultiDisplayManager(display);
        current_screen_input_info.setDisplayManager(&display_manager);
        display_manager.detectDisplays();
        
        screen = DefaultScreen(display);
        
        // Check for composite extension if alpha is requested
        if (enable_alpha) {
            int composite_event_base, composite_error_base;
            if (!XCompositeQueryExtension(display, &composite_event_base, &composite_error_base)) {
                std::cerr << "Warning: Composite extension not available, alpha blending may not work" << std::endl;
            }
        }
        
        // Find appropriate visual for alpha support
        if (enable_alpha) {
            XVisualInfo vinfo;
            if (XMatchVisualInfo(display, screen, 32, TrueColor, &vinfo)) {
                visual = vinfo.visual;
                depth = vinfo.depth;
                colormap = XCreateColormap(display, RootWindow(display, screen), visual, AllocNone);
            } else {
                std::cerr << "Warning: 32-bit visual not found, falling back to default" << std::endl;
                visual = DefaultVisual(display, screen);
                depth = DefaultDepth(display, screen);
                colormap = DefaultColormap(display, screen);
            }
        } else {
            visual = DefaultVisual(display, screen);
            depth = DefaultDepth(display, screen);
            colormap = DefaultColormap(display, screen);
        }
        
        // Set window attributes
        // Replace the window creation and property setting section in your VectorDisplay constructor
// with this corrected version:

// Create window with proper input transparency
        XSetWindowAttributes attrs;
        attrs.colormap = colormap;
        attrs.border_pixel = 0;
        attrs.background_pixel = 0;
        attrs.event_mask = NoEventMask; // No events
        attrs.override_redirect = True; // Bypass window manager
        attrs.do_not_propagate_mask = KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask;

        unsigned long mask = CWColormap | CWBorderPixel | CWBackPixel | CWOverrideRedirect | CWDontPropagate;

        // Create window
        window = XCreateWindow(display, RootWindow(display, screen),
                            0, 0, shape[1], shape[0], 0, depth, InputOutput,
                            visual, mask, &attrs);

        // CRITICAL: Set input region to empty to make window completely click-through
        XserverRegion empty_region = XFixesCreateRegion(display, nullptr, 0);
        XFixesSetWindowShapeRegion(display, window, ShapeInput, 0, 0, empty_region);
        XFixesDestroyRegion(display, empty_region);

        // Set window properties for overlay behavior
        XStoreName(display, window, "Vector Display");

        // Make window borderless
        if (borderless) {
            Atom motifHints = XInternAtom(display, "_MOTIF_WM_HINTS", False);
            if (motifHints != None) {
                struct {
                    unsigned long flags;
                    unsigned long functions;
                    unsigned long decorations;
                    long input_mode;
                    unsigned long status;
                } hints = {0};
                
                hints.flags = 2; // MWM_HINTS_DECORATIONS
                hints.decorations = 0; // No decorations
                
                XChangeProperty(display, window, motifHints, motifHints, 32,
                            PropModeReplace, (unsigned char*)&hints, 5);
            }
        }

        // Set window to stay on top
        Atom netWmState = XInternAtom(display, "_NET_WM_STATE", False);
        Atom netWmStateAbove = XInternAtom(display, "_NET_WM_STATE_ABOVE", False);
        Atom netWmStateSkipTaskbar = XInternAtom(display, "_NET_WM_STATE_SKIP_TASKBAR", False);
        Atom netWmStateSkipPager = XInternAtom(display, "_NET_WM_STATE_SKIP_PAGER", False);

        Atom states[] = { netWmStateAbove, netWmStateSkipTaskbar, netWmStateSkipPager };
        XChangeProperty(display, window, netWmState, XA_ATOM, 32, 
                        PropModeReplace, (unsigned char*)states, 3);

        // Set window type to overlay/dock for proper stacking
        Atom windowType = XInternAtom(display, "_NET_WM_WINDOW_TYPE", False);
        Atom windowTypeDock = XInternAtom(display, "_NET_WM_WINDOW_TYPE_DOCK", False);
        XChangeProperty(display, window, windowType, XA_ATOM, 32,
                        PropModeReplace, (unsigned char*)&windowTypeDock, 1);

        // Ensure no input events are selected
        XSelectInput(display, window, NoEventMask);
                
        // Create graphics context
        gc = XCreateGC(display, window, 0, nullptr);
        
        // Create XImage for pixel buffer
        createXImage();
        
        // Map window
        XMapWindow(display, window);
        XFlush(display);

        // Set initial screen size based on current display
        const DisplayInfo* current_display = display_manager.getCurrentDisplay();
        if (current_display) {
            current_screen_input_info.setScreenSize(current_display->x, current_display->y, 
                                                   current_display->width, current_display->height);
            
            // Position window on current display
            XMoveResizeWindow(display, window, current_display->x, current_display->y, 
                            current_display->width, current_display->height);
        } else {
            // Fallback for single display
            XWindowAttributes attr;
            XGetWindowAttributes(display, window, &attr);
            int x, y;
            unsigned int width, height, border_width, depth;
            Window root;
            XGetGeometry(display, window, &root, &x, &y, &width, &height, &border_width, &depth);
            current_screen_input_info.setScreenSize(x, y, width, height);
        }

        current_screen_input_info.selected_text_reader = new SelectedTextReader(display, window);
        
        // Start direct input reading
        startDirectInputReading();
        
    }
    
private:
    void startDirectInputReading() {
        input_reader.start([this](const input_event& event, const std::string& device_name) {
            processInputEvent(event, device_name);
        });
    }

    void updateMousePositionFromRoot() {
        Window root = RootWindow(display, screen);
        Window child;
        int root_x, root_y, win_x, win_y;
        unsigned int mask;
        
        if (XQueryPointer(display, root, &root, &child, &root_x, &root_y, &win_x, &win_y, &mask)) {
            // root_x, root_y are the global screen coordinates
            current_screen_input_info.updateMousePositionAbsolute(root_x, root_y);
        }
    }

    
    void processInputEvent(const input_event& event, const std::string& device_name) {
        // Capture and process ALL events without filtering
        switch (event.type) {
            case EV_KEY:
                // Process ALL key events, including mouse buttons and keyboard keys
                if (event.code == BTN_LEFT || event.code == BTN_RIGHT || event.code == BTN_MIDDLE) {
                    current_screen_input_info.updateMouseButtonState(event.code, event.value);
                } else {
                    // For keyboard keys, update key state
                    current_screen_input_info.updateKeyState(event.code, event.value);
                }
                break;
                
            case EV_REL:
                // Process ALL relative events
                switch (event.code) {
                    case REL_X:
                        // current_screen_input_info.updateMousePositionRelative(event.value, 0);
                        break;
                    case REL_Y:
                        // current_screen_input_info.updateMousePositionRelative(0, event.value);
                        break;
                    case REL_WHEEL:
                    case REL_HWHEEL:
                        current_screen_input_info.updateMouseWheel(event.code, event.value);
                        break;
                }

                break;
                
            case EV_ABS:
                // Process ALL absolute positioning events
                switch (event.code) {
                    case ABS_X:
                        current_screen_input_info.updateMousePositionAbsolute(event.value, current_screen_input_info.getMouseY());
                        break;
                    case ABS_Y:
                        current_screen_input_info.updateMousePositionAbsolute(current_screen_input_info.getMouseX(), event.value);
                        break;
                }
                break;
                
            case EV_SYN:
                // Always process synchronization events
                if (event.code == SYN_REPORT) {
                    current_screen_input_info.clearWheelStates();
                }
                break;
        }

        
       
    }
    
public:
    void setWindowBorderless() {
        // Remove window decorations
        Atom motifHints = XInternAtom(display, "_MOTIF_WM_HINTS", False);
        if (motifHints != None) {
            struct {
                unsigned long flags;
                unsigned long functions;
                unsigned long decorations;
                long input_mode;
                unsigned long status;
            } hints = {0};
            
            hints.flags = 2; // MWM_HINTS_DECORATIONS
            hints.decorations = 0; // No decorations
            
            XChangeProperty(display, window, motifHints, motifHints, 32,
                           PropModeReplace, (unsigned char*)&hints, 5);
        }
    }
    
    void enableAlphaBlending() {
        // Set window opacity property for compositor
        Atom netWmWindowOpacity = XInternAtom(display, "_NET_WM_WINDOW_OPACITY", False);
        if (netWmWindowOpacity != None) {
            unsigned long opacity = 0xFFFFFFFF; // Fully opaque by default
            XChangeProperty(display, window, netWmWindowOpacity, XA_CARDINAL, 32,
                           PropModeReplace, (unsigned char*)&opacity, 1);
        }
        
        // Enable compositing for this window
        XCompositeRedirectWindow(display, window, CompositeRedirectAutomatic);
    }
    
    void setWindowOpacity(float opacity) {
        if (!alpha_enabled) return;
        
        Atom netWmWindowOpacity = XInternAtom(display, "_NET_WM_WINDOW_OPACITY", False);
        if (netWmWindowOpacity != None) {
            unsigned long opacityValue = (unsigned long)(opacity * 0xFFFFFFFF);
            XChangeProperty(display, window, netWmWindowOpacity, XA_CARDINAL, 32,
                           PropModeReplace, (unsigned char*)&opacityValue, 1);
            XFlush(display);
        }
    }
    
    void createXImage() {
        int bitmap_pad = (depth > 16) ? 32 : (depth > 8) ? 16 : 8;
        visual = DefaultVisual(display, screen);
        
        ximage = XCreateImage(display, visual, depth, ZPixmap, 0,
                             (char*)this->data, shape[1], shape[0],
                             bitmap_pad, shape[1] * sizeof(uint84));
        
        if (!ximage) {
            std::cerr << "Cannot create XImage" << std::endl;
            return;
        }
        
        // Set byte order
        ximage->byte_order = LSBFirst;
        ximage->bitmap_bit_order = LSBFirst;
    }
    
    void updateDisplay() {
        if (!ximage) return;
        
        // Update the image data pointer (in case tensor was reallocated)
        ximage->data = (char*)this->data;
        ximage->width = shape[1];
        ximage->height = shape[0];
        ximage->bytes_per_line = shape[1] * sizeof(uint84);
        
        if (shape[0] > 0 && shape[1] > 0) {
            XPutImage(display, window, gc, ximage, 0, 0, 0, 0, shape[1], shape[0]);
        }
        XFlush(display);
    }

    void resizeDisplay() {
        if (is_fullscreen) {
            XWindowAttributes attr;
            XGetWindowAttributes(display, window, &attr);
            if (attr.width != shape[1] || attr.height != shape[0]) {
                if (ximage) {
                    ximage->data = nullptr;
                    XDestroyImage(ximage);
                }

                int bitmap_pad = (depth > 16) ? 32 : (depth > 8) ? 16 : 8;
                this->data = (uint84*)realloc(data, attr.width * attr.height * sizeof(uint84));
                shape[0] = attr.height;
                shape[1] = attr.width;

                ximage = XCreateImage(display, visual, depth, ZPixmap, 0,
                                     (char*)this->data, shape[1], shape[0],
                                     bitmap_pad, shape[1] * sizeof(uint84));
                if (!ximage) {
                    std::cerr << "Cannot create resized XImage" << std::endl;
                    return;
                }
                
                ximage->byte_order = LSBFirst;
                ximage->bitmap_bit_order = LSBFirst;
                ximage->width = shape[1];
                ximage->height = shape[0];
                ximage->bytes_per_line = shape[1] * sizeof(uint84);
                
                calculate_metadata();
                XResizeWindow(display, window, ximage->width, ximage->height);
            } 
        } 
    }
    
    void displayLoop() {
        bool quit = false;
        XEvent event;
        
        while (!quit) {
            resizeDisplay();
            
            
            // Call display loop functions
            for (const auto& callback : display_loop_functions) {
                callback(current_screen_input_info);
            }

            // Handle minimal X11 events (mainly window management)
            while (XPending(display) > 0) {
                XNextEvent(display, &event);
                
                switch (event.type) {
                    case Expose:
                        updateDisplay();
                        break;
                    case ClientMessage:
                        if (event.xclient.data.l[0] == XInternAtom(display, "WM_DELETE_WINDOW", False)) {
                            quit = true;
                        }
                        break;
                    case ConfigureNotify:
                        // Window size changed
                        break;
                }
            }
            
            updateDisplay();
            usleep(11111); // ~90 FPS
            updateMousePositionFromRoot();
        }
        
        // Stop input reading when display loop ends
        input_reader.stop();
    }
    
    // Clear display with specified color
    void clear(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 0) {
        uint84 color;
        if (alpha_enabled && depth == 32) {
            color = (a << 24) | (r << 16) | (g << 8) | b;
        } else {
            color = (r << 16) | (g << 8) | b;
        }
        
        for (int y = 0; y < shape[0]; y++) {
            for (int x = 0; x < shape[1]; x++) {
                (*this)[y][x] = color;
            }
        }
    }
    
    ~VectorDisplay() {
        input_reader.stop();
        
        if (ximage) {
            ximage->data = nullptr;
            XDestroyImage(ximage);
        }
        if (gc) {
            XFreeGC(display, gc);
        }
        if (window) {
            XDestroyWindow(display, window);
        }
        if (display) {
            XCloseDisplay(display);
        }
    }
    
    void add_on_update(std::function<void(CurrentScreenInputInfo&)> func) {
        display_loop_functions.push_back(func);
    }
    
    // Utility functions for window management
    void moveWindow(int x, int y) {
        XMoveWindow(display, window, x, y);
        XFlush(display);
    }
    
    void resizeWindow(int width, int height) {
        XResizeWindow(display, window, width, height);
        XFlush(display);
    }
    
    // Multi-display functions
    void moveToDisplay(int display_index) {
        const DisplayInfo* target_display = display_manager.getDisplay(display_index);
        if (target_display) {
            display_manager.setCurrentDisplay(display_index);
            
            // Update window position and size to match display
            XMoveResizeWindow(display, window, target_display->x, target_display->y,
                            target_display->width, target_display->height);
            
            // Update screen info
            std::cout << "Moving to display: " << target_display->name << std::endl;
            std::cout << "Position: (" << target_display->x << ", " << target_display->y 
                      << "), Size: " << target_display->width << "x" << target_display->height << std::endl;
            current_screen_input_info.setScreenSize(target_display->x, target_display->y,
                                                   target_display->width, target_display->height);
            
            // Resize internal buffer if needed
            if (shape[0] != target_display->height || shape[1] != target_display->width) {
                // Resize tensor
                if (ximage) {
                    ximage->data = nullptr;
                    XDestroyImage(ximage);
                }
                
                this->data = (uint84*)realloc(data, target_display->width * target_display->height * sizeof(uint84));
                shape[0] = target_display->height;
                shape[1] = target_display->width;
                
                createXImage();
                calculate_metadata();
            }
            
            XFlush(display);
            std::cout << "Moved to display " << display_index << ": " << target_display->name << std::endl;
        }
    }
    
    void moveToNextDisplay() {
        int next_display = (display_manager.getCurrentDisplayIndex() + 1) % display_manager.getDisplayCount();
        moveToDisplay(next_display);
    }
    
    void moveToPreviousDisplay() {
        int prev_display = (display_manager.getCurrentDisplayIndex() - 1 + display_manager.getDisplayCount()) % display_manager.getDisplayCount();
        moveToDisplay(prev_display);
    }
    
    void moveToPrimaryDisplay() {
        const auto& displays = display_manager.getDisplays();
        for (size_t i = 0; i < displays.size(); i++) {
            if (displays[i].is_primary) {
                moveToDisplay(i);
                return;
            }
        }
    }
    
    void moveToDisplayContainingMouse() {
        auto [global_x, global_y] = current_screen_input_info.getGlobalMousePosition();
        const DisplayInfo* containing_display = display_manager.getDisplayContaining(global_x, global_y);
        if (containing_display && containing_display->index != display_manager.getCurrentDisplayIndex()) {
            moveToDisplay(containing_display->index);
        }
    }
    
    // Get display information
    std::vector<DisplayInfo> getAllDisplays() const {
        return display_manager.getDisplays();
    }
    
    const DisplayInfo* getCurrentDisplay() const {
        return display_manager.getCurrentDisplay();
    }
    
    int getCurrentDisplayIndex() const {
        return display_manager.getCurrentDisplayIndex();
    }
    
    int getDisplayCount() const {
        return display_manager.getDisplayCount();
    }
    
    void refreshDisplays() {
        display_manager.detectDisplays();
    }
};

#endif
