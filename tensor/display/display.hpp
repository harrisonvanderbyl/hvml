#include "tensor.hpp"
#include "device/device.hpp"
#include "vector/vectors.hpp"
#ifndef VECTOR_DISPLAY_HPP
#define VECTOR_DISPLAY_HPP



class VectorDisplay {

    public:
    SDL_GLContext glctx = nullptr;

    GLuint tbo = 0;  // Texture Buffer Object
    GLuint bufferTexture = 0;  // Buffer texture handle
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint program = 0;
    int width = 0;
    int height = 0;
    GLuint pbo = 0; // Pixel Buffer Object
   

    void preinit() {

    }

    VectorDisplay(Tensor<uint84,2>& buffer) {
        if(buffer.device->allocation_compute_types[buffer.storage_pointer] != ComputeType::kOPENGL){
            throw std::runtime_error("Buffer must be allocated with OpenGL compute type for VectorDisplay");
        }
        this->width = buffer.shape[0];
        this->height = buffer.shape[1];
        pbo = (GLuint)(unsigned long long)buffer.storage_pointer;
        // Create buffer texture that references the buffer
        glGenTextures(1, &bufferTexture);
        glBindTexture(GL_TEXTURE_BUFFER, bufferTexture);

        GLFuncs->glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, pbo);
        
        auto glErr = glGetError();
        if (glErr != GL_NO_ERROR) {
            throw std::runtime_error("OpenGL error creating buffer texture: " + std::to_string(glErr));
        }
        
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        GLFuncs->glBindBuffer(GL_TEXTURE_BUFFER, 0);
        glFinish();


        // Setup quad vertices (position and texture coordinates)
        float verts[] = {
            -1, -1, 0, 1,
             1, -1, 1, 1,
             1,  1, 1, 0,
            -1,  1, 0, 0
        };

        GLFuncs->glGenVertexArrays(1, &vao);
        GLFuncs->glBindVertexArray(vao);

        GLFuncs->glGenBuffers(1, &vbo);
        GLFuncs->glBindBuffer(GL_ARRAY_BUFFER, vbo);
        GLFuncs->glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

        GLFuncs->glEnableVertexAttribArray(0);
        GLFuncs->glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        GLFuncs->glEnableVertexAttribArray(1);
        GLFuncs->glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              (void*)(2 * sizeof(float)));

        GLFuncs->glBindVertexArray(0);

        program = createProgram(vertexSrc, fragmentSrc);
    }

    void present() {
        // Render using shader that samples from buffer texture
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        GLFuncs->glUseProgram(program);
        GLFuncs->glBindVertexArray(vao);

        // Bind buffer texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, bufferTexture);
        GLFuncs->glUniform1i(GLFuncs->glGetUniformLocation(program, "bufferTex"), 0);
        GLFuncs->glUniform2i(GLFuncs->glGetUniformLocation(program, "dimensions"), width, height);

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        glBindTexture(GL_TEXTURE_BUFFER, 0);
        GLFuncs->glBindVertexArray(0);

    }



private:
    static constexpr const char* vertexSrc = R"(
        #version 330 core
        layout(location = 0) in vec2 pos;
        layout(location = 1) in vec2 uv;
        out vec2 vUV;
        void main() {
            vUV = uv;
            gl_Position = vec4(pos, 0, 1);
        }
    )";

    static constexpr const char* fragmentSrc = R"(
        #version 330 core
        in vec2 vUV;
        out vec4 color;
        uniform samplerBuffer bufferTex;
        uniform ivec2 dimensions;
        
        void main() {
            // Convert UV coordinates to pixel coordinates
            ivec2 pixelCoord = ivec2(vUV * vec2(dimensions));
            
            // Calculate linear index into buffer (row-major order)
            int index = pixelCoord.y * dimensions.x + pixelCoord.x;
            
            // Fetch color directly from buffer
            color = texelFetch(bufferTex, index);
        }
    )";

    GLuint createProgram(const char* vs, const char* fs) {
        auto compile = [](GLenum type, const char* src) {
            GLuint s = GLFuncs->glCreateShader(type);
            GLFuncs->glShaderSource(s, 1, &src, nullptr);
            GLFuncs->glCompileShader(s);
            
            GLint success;
            GLFuncs->glGetShaderiv(s, GL_COMPILE_STATUS, &success);
            if (!success) {
                char infoLog[512];
                GLFuncs->glGetShaderInfoLog(s, 512, nullptr, infoLog);
                std::cerr << "Shader compilation failed: " << infoLog << std::endl;
            }
            return s;
        };

        GLuint v = compile(GL_VERTEX_SHADER, vs);
        GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        GLuint p = GLFuncs->glCreateProgram();
        GLFuncs->glAttachShader(p, v);
        GLFuncs->glAttachShader(p, f);
        GLFuncs->glLinkProgram(p);
        
        GLint success;
        GLFuncs->glGetProgramiv(p, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            GLFuncs->glGetProgramInfoLog(p, 512, nullptr, infoLog);
            std::cerr << "Program linking failed: " << infoLog << std::endl;
        }
        
        GLFuncs->glDeleteShader(v);
        GLFuncs->glDeleteShader(f);
        return p;
    }

   
};

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
    bool isFullscreen() const { return is_fullscreen; }
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
    WP_BORDERLESS = 1 << 0,
    WP_ALPHA_ENABLED = 1 << 1,
    WP_FULLSCREEN = 1 << 2,
    WP_CLICKTHROUGH = 1 << 3,
    WP_ON_TOP = 1 << 4,
    WP_RESIZABLE = 1 << 5
};

struct WindowPropertiesFlags {
    bool borderless = false;
    bool alpha_enabled = false;
    bool fullscreen = false;
    bool clickthrough = true;
    bool on_top = false;
    bool resizable = false;

    WindowPropertiesFlags(WindowProperties properties) {
        borderless = properties & WP_BORDERLESS;
        alpha_enabled = properties & WP_ALPHA_ENABLED;
        fullscreen = properties & WP_FULLSCREEN;
        clickthrough = properties & WP_CLICKTHROUGH;
        on_top = properties & WP_ON_TOP;
        resizable = properties & WP_RESIZABLE;
    }

    WindowPropertiesFlags(int flags) {
        borderless = flags & WP_BORDERLESS;
        alpha_enabled = flags & WP_ALPHA_ENABLED;
        fullscreen = flags & WP_FULLSCREEN;
        clickthrough = flags & WP_CLICKTHROUGH;
        on_top = flags & WP_ON_TOP;
        resizable = flags & WP_RESIZABLE;
    }

    operator WindowProperties() const {
        WindowProperties props = (WindowProperties)0;
        if (borderless) props = (WindowProperties)(props | WP_BORDERLESS);
        if (alpha_enabled) props = (WindowProperties)(props | WP_ALPHA_ENABLED);
        if (fullscreen) props = (WindowProperties)(props | WP_FULLSCREEN);
        if (clickthrough) props = (WindowProperties)(props | WP_CLICKTHROUGH);
        if (on_top) props = (WindowProperties)(props | WP_ON_TOP);
        if (resizable) props = (WindowProperties)(props | WP_RESIZABLE);
        return props;
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

struct OpenGLDisplay
{
public:
    void* display = nullptr;
    SDL_Window* window;
    void* root_window = nullptr;
    int screen;
    int depth = 32;
    int height = 0;
    int width = 0;
    bool borderless = false;
    bool alpha_enabled = false;
    bool is_fullscreen = false;
    bool clickthrough = true;
    bool on_top = false;
    bool resizable = false;
    CurrentScreenInputInfo current_screen_input_info;
    ComputeDeviceBase* device = nullptr;
    std::vector<std::function<void(CurrentScreenInputInfo&)>> display_loop_functions;
    
    Clock clock;

    void init_display(int width, int height) {
        if(GLFuncs)        {
            std::cerr << "OpenGL functions already loaded!" << std::endl;
            throw std::runtime_error("OpenGL functions already loaded!");
            return;
        }
        SDL_Init(SDL_INIT_VIDEO);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        std::cout << GLFuncs << std::endl;

        std::cout << "finished preinit" << std::endl;
      
        Uint32 window_flags = SDL_WINDOW_OPENGL;
        
        if (is_fullscreen) window_flags |= SDL_WINDOW_FULLSCREEN;
        if (borderless) window_flags |= SDL_WINDOW_BORDERLESS;
        if (alpha_enabled) window_flags |= SDL_WINDOW_TRANSPARENT;
        if (on_top) window_flags |= SDL_WINDOW_ALWAYS_ON_TOP;
        if (resizable) window_flags |= SDL_WINDOW_RESIZABLE;
        

        window = SDL_CreateWindow(
            "CUDA → OpenGL",
            // width and height
            width, height,
            window_flags
        );

        if (!window) {
            throw std::runtime_error("Failed to create SDL window: " + std::string(SDL_GetError()));
        }

        std::cout << "created window" << std::endl;

        display = (void *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);

        screen = SDL_GetDisplayForWindow(window);

        // GL enable blending
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        
        
        SDL_ShowWindow(window);

        std::cout << "loading GL functions" << std::endl;
        loadGLFunctions();

        auto glctx = SDL_GL_CreateContext(window);
        if (!glctx) {
            std::string error = SDL_GetError();
            throw std::runtime_error("Failed to create OpenGL context: " + error);
        }
        
        glEnable(GL_DEPTH_TEST);
        SDL_GL_SetSwapInterval(0);



        std::cout << "created GL context" << std::endl;
    }
    
    OpenGLDisplay(Shape<2> shape = 0, WindowPropertiesFlags properties = (WindowProperties)0)
        : borderless(properties.borderless), 
          alpha_enabled(properties.alpha_enabled), 
          is_fullscreen(properties.fullscreen),  
          clickthrough(properties.clickthrough), 
          on_top(properties.on_top),
          resizable(properties.resizable),
          width(shape[0]),
          height(shape[1])
    {

        
        

        init_display(shape[1], shape[0]);
        device = create_opengl_compute_device(0);




    }
    
private:


    void updateMousePositionFromRoot() {
        // Get global mouse state
        float gx, gy;
        SDL_GetGlobalMouseState(&gx, &gy);
        
        // Get window position
        int wx, wy;
        SDL_GetWindowPosition(window, &wx, &wy);
        
        // Calculate relative position
       
        current_screen_input_info.updateMousePositionAbsolute(gx - wx, gy - wy);
    }

public:
    void setWindowCaption(const char* title) {
        SDL_SetWindowTitle(window, title);
    }

    void setMouseGrab(bool grab) {
        SDL_SetWindowMouseGrab(window, grab ? true : false);
        if (grab) {
            SDL_SetWindowRelativeMouseMode(window, true);
        } else {
            SDL_SetWindowRelativeMouseMode(window, false);
        }
        current_screen_input_info.setMouseGrabbed(grab);
    }

    void setMouseVisible(bool visible) {
        if (visible) {
            SDL_ShowCursor();
        } else {
            SDL_HideCursor();
        }
        current_screen_input_info.setMouseVisible(visible);
    }

    std::pair<int, int> getWindowSize() const {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        return {w, h};
    }
    
    void setWindowBorderless() {
        SDL_SetWindowBordered(window, false);
    }
    
    void enableAlphaBlending() {
        // Set window opacity for compositor
        SDL_SetWindowOpacity(window, 1.0f);
    }
    
    void setWindowOpacity(float opacity) {
        if (!alpha_enabled) return;
        SDL_SetWindowOpacity(window, opacity);
    }
    
    void updateDisplay() {
        auto oldWindowPosition = current_screen_input_info.currentWindowPosition;
        SDL_GetWindowPosition(window, &current_screen_input_info.currentWindowPosition[0], &current_screen_input_info.currentWindowPosition[1]);
        current_screen_input_info.relativeWindowMove = int32x2(
            current_screen_input_info.currentWindowPosition.x() - oldWindowPosition.x(),
            current_screen_input_info.currentWindowPosition.y() - oldWindowPosition.y()
        );
        
        SDL_GL_SwapWindow(window);
    }

    void resizeDisplay() {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        
    }
    
    bool processEvents() {
        SDL_Event sdl_event;
        current_screen_input_info.clear_mouse_states();
        
        while (SDL_PollEvent(&sdl_event)) {
            if (sdl_event.type == SDL_EVENT_QUIT) {
                return false;
            }
            else if (sdl_event.type == SDL_EVENT_KEY_DOWN) {
                current_screen_input_info.updateKeyState(sdl_event.key.key, true);
                
                // Check for special keys (like ESC)
                if (sdl_event.key.key == SDLK_ESCAPE) {
                    return false;
                }
            }
            else if (sdl_event.type == SDL_EVENT_KEY_UP) {
                current_screen_input_info.updateKeyState(sdl_event.key.key, false);
            }
            else if (sdl_event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                current_screen_input_info.updateMouseButtonState(sdl_event.button.button, true);
            }
            else if (sdl_event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                current_screen_input_info.updateMouseButtonState(sdl_event.button.button, false);
            }
            else if (sdl_event.type == SDL_EVENT_MOUSE_MOTION) {
                if (current_screen_input_info.isMouseGrabbed()) {
                    current_screen_input_info.updateMouseMotion(sdl_event.motion.xrel, sdl_event.motion.yrel);
                } else {
                    current_screen_input_info.updateMousePositionAbsolute(sdl_event.motion.x, sdl_event.motion.y);
                }
            }
            else if (sdl_event.type == SDL_EVENT_MOUSE_WHEEL) {
                // Handle mouse wheel scrolling
                if (sdl_event.wheel.y > 0) {
                    current_screen_input_info.clearWheelStates();
                    // Wheel up
                } else if (sdl_event.wheel.y < 0) {
                    current_screen_input_info.clearWheelStates();
                    // Wheel down
                }
                if (sdl_event.wheel.x > 0) {
                    current_screen_input_info.clearWheelStates();
                    // Wheel right
                } else if (sdl_event.wheel.x < 0) {
                    current_screen_input_info.clearWheelStates();
                    // Wheel left
                }
            }
            else if (sdl_event.type == SDL_EVENT_WINDOW_RESIZED) {
                resizeDisplay();
            }
        }
        
        
        return true;
    }

    
    void displayLoop() {
        bool running = true;

        while (running) {
            resizeDisplay();

            glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

            
            // Call display loop functions
            for (const auto& callback : display_loop_functions) {
                callback(current_screen_input_info);
            }

            // Process events

            running = processEvents();
            
            updateDisplay();


            

        }
    }    
   
    
    ~OpenGLDisplay() {
        if (window) {
            SDL_DestroyWindow(window);
        }
        SDL_Quit();
    }
    
    void add_on_update(std::function<void(CurrentScreenInputInfo&)> func) {
        display_loop_functions.push_back(func);
    }
    
    // Utility functions for window management
    void moveWindow(int x, int y) {
        SDL_SetWindowPosition(window, x, y);
    }
    
    void resizeWindow(int width, int height) {
        SDL_SetWindowSize(window, width, height);
    }

    void setFullscreen(bool fullscreen) {
        if (fullscreen) {
            SDL_SetWindowFullscreen(window, true);
        } else {
            SDL_SetWindowFullscreen(window, false);
        }
        is_fullscreen = fullscreen;
        current_screen_input_info.setFullscreen(fullscreen);
    }

    // Get key code mapping (similar to pygame)
    static SDL_Keycode getKeyCode(const char* key_name) {
        return SDL_GetKeyFromName(key_name);
    }

    // Check if specific key is pressed
    bool isKeyPressed(SDL_Keycode key) const {
        return current_screen_input_info.isKeyPressed(key);
    }

    // Surface operations for compatibility
    struct Surface {
        int width;
        int height;
        std::vector<uint32_t> pixels;
        
        Surface(int w, int h, uint32_t * px) : width(w), height(h), pixels(px, px + (w * h)) {}
        Surface(int w, int h) : width(w), height(h), pixels(w * h, 0) {}
    };

    Surface createSurface(int w, int h) {
        return Surface(w, h);
    }

    void blitSurface(const Surface& surf, int x, int y) {
        // for (int sy = 0; sy < surf.height && (y + sy) < shape[0]; sy++) {
        //     for (int sx = 0; sx < surf.width && (x + sx) < shape[1]; sx++) {
        //         if (x + sx >= 0 && y + sy >= 0) {
        //             (*this)[y + sy][x + sx] = surf.pixels[sy * surf.width + sx];
        //         }
        //     }
        // }
    }

    // Font rendering placeholder (requires SDL_ttf)
    struct Font {
        // Placeholder for font data
        int size;
        Font(int s) : size(s) {}
    };

    Font loadFont(const char* name, int size) {
        return Font(size);
    }
};

#endif