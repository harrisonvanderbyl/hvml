// X11 headers

// OpenGL headers
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>

// SDL headers
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

// CUDA headers
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Standard headers
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <unistd.h>
#include <map>
#include <set>
#include <functional>
#include <vector>
#include <chrono>
#include "vector/float4.hpp"
#include "vector/int2.hpp"

// OpenGL function pointer typedefs
typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum type);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (APIENTRY *PFNGLDELETESHADERPROC)(GLuint shader);
typedef void (APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint program);
typedef void (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
typedef void (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint array);
typedef void (APIENTRY *PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint *arrays);
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
typedef void (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLVERTEXATTRIBIPOINTERPROC)(GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer);
typedef void (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
typedef GLint (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
typedef void (APIENTRY *PFNGLUNIFORMMATRIX4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (APIENTRY *PFNGLUNIFORM3FVPROC)(GLint location, GLsizei count, const GLfloat *value);
typedef void (APIENTRY *PFNGLUNIFORM1FPROC)(GLint location, GLfloat value);
typedef void (APIENTRY *PFNGLUNIFORM1IPROC)(GLint location, GLint value);
typedef GLboolean (APIENTRY *PFNGLUNMAPBUFFERPROC)(GLenum target);

// Global function pointers
PFNGLCREATESHADERPROC glCreateShader = nullptr;
PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
PFNGLATTACHSHADERPROC glAttachShader = nullptr;
PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
PFNGLDELETESHADERPROC glDeleteShader = nullptr;
PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray = nullptr;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays = nullptr;
PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
PFNGLBUFFERDATAPROC glBufferData = nullptr;
PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer = nullptr;
PFNGLVERTEXATTRIBIPOINTERPROC glVertexAttribIPointer = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = nullptr;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = nullptr;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv = nullptr;
PFNGLUNIFORM3FVPROC glUniform3fv = nullptr;
PFNGLUNIFORM1FPROC glUniform1f = nullptr;
PFNGLUNIFORM1IPROC glUniform1i = nullptr;
PFNGLMAPBUFFERPROC glMapBuffer = nullptr;
PFNGLUNMAPBUFFERPROC glUnmapBuffer = nullptr;

void loadGLFunctions() {
    glCreateShader = (PFNGLCREATESHADERPROC)SDL_GL_GetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)SDL_GL_GetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)SDL_GL_GetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)SDL_GL_GetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)SDL_GL_GetProcAddress("glGetShaderInfoLog");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)SDL_GL_GetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)SDL_GL_GetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)SDL_GL_GetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)SDL_GL_GetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)SDL_GL_GetProcAddress("glGetProgramInfoLog");
    glDeleteShader = (PFNGLDELETESHADERPROC)SDL_GL_GetProcAddress("glDeleteShader");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)SDL_GL_GetProcAddress("glDeleteProgram");
    glUseProgram = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)SDL_GL_GetProcAddress("glBindVertexArray");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glDeleteVertexArrays");
    glGenBuffers = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteBuffers");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribPointer");
    glVertexAttribIPointer = (PFNGLVERTEXATTRIBIPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribIPointer");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)SDL_GL_GetProcAddress("glEnableVertexAttribArray");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)SDL_GL_GetProcAddress("glGetUniformLocation");
    glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)SDL_GL_GetProcAddress("glUniformMatrix4fv");
    glUniform3fv = (PFNGLUNIFORM3FVPROC)SDL_GL_GetProcAddress("glUniform3fv");
    glUniform1f = (PFNGLUNIFORM1FPROC)SDL_GL_GetProcAddress("glUniform1f");
    glUniform1i = (PFNGLUNIFORM1IPROC)SDL_GL_GetProcAddress("glUniform1i");
    glMapBuffer = (PFNGLMAPBUFFERPROC)SDL_GL_GetProcAddress("glMapBuffer");
    glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)SDL_GL_GetProcAddress("glUnmapBuffer");
}

#ifndef VECTOR_DISPLAY_HPP
#define VECTOR_DISPLAY_HPP

template <DeviceType device>
class GLBackend {
public:
    SDL_Window* window = nullptr;
    SDL_GLContext glctx = nullptr;

    GLuint tex = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint program = 0;

    int width = 0;
    int height = 0;

    cudaGraphicsResource* cudaTex = nullptr;
    cudaArray_t cudaArray = nullptr;
    void* cudaDevPtr = nullptr;
    bool resourcesMapped = false;

    void preinit() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error("Failed to initialize SDL: " + std::string(SDL_GetError()));
        }

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
    }

    void initialize(void* ptr, SDL_Window* win, int w, int h, Visual* visual = nullptr, int depth = 24, Colormap colormap = 0) {
        cudaDevPtr = ptr;
        width = w;
        height = h;
        window = win;

        if (!window) {
            throw std::runtime_error("Failed to create SDL window");
        }

        glctx = SDL_GL_CreateContext(window);
        if (!glctx) {
            std::string error = SDL_GetError();
            throw std::runtime_error("Failed to create OpenGL context: " + error);
        }
        
        SDL_GL_SetSwapInterval(0);

        loadGLFunctions();

        if (!glGenTextures || !glBindTexture || !glTexImage2D) {
            throw std::runtime_error("Failed to load required OpenGL functions");
        }

        glGenTextures(1, &tex);
        if (tex == 0) {
            throw std::runtime_error("Failed to generate OpenGL texture");
        }

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        
        GLenum glErr = glGetError();
        if (glErr != GL_NO_ERROR) {
            throw std::runtime_error("OpenGL error creating texture: " + std::to_string(glErr));
        }
        
        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();

        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA devices found: " + std::string(cudaGetErrorString(err)));
        }

        int cudaDevice = -1;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (prop.major >= 3) {
                cudaDevice = i;
                break;
            }
        }

        if (cudaDevice == -1) {
            throw std::runtime_error("No suitable CUDA device found for GL interop");
        }

        err = cudaSetDevice(cudaDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA GL device: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaGraphicsGLRegisterImage(&cudaTex, tex, GL_TEXTURE_2D,
                                    cudaGraphicsRegisterFlagsSurfaceLoadStore);
        if (err != cudaSuccess) {
            if(err == 999){
                std::cout << "CUDA–GL interop registration failed with error code: " << err << " (USING wrong gpu for openGL)" << std::endl;
            }
            std::cout << "CUDA–GL interop registration failed with error code: " << err << std::endl;
            std::string errorMsg = "Failed to register CUDA-GL interop: " + 
                                  std::string(cudaGetErrorString(err));
            glDeleteTextures(1, &tex);
            throw std::runtime_error(errorMsg);
        }

        float verts[] = {
            -1, -1, 0, 1,
             1, -1, 1, 1,
             1,  1, 1, 0,
            -1,  1, 0, 0
        };

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              (void*)(2 * sizeof(float)));

        glBindVertexArray(0);

        program = createProgram(vertexSrc, fragmentSrc);
    }

    void present() {
        auto errMap = cudaGraphicsMapResources(1, &cudaTex);
        if (errMap != cudaSuccess) {
            throw std::runtime_error("Failed to map CUDA-GL resources: " + std::string(cudaGetErrorString(errMap)));
        }
        cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaTex, 0, 0);

        cudaError_t erra = cudaMemcpy2DToArray(
            cudaArray,
            0, 0,
            cudaDevPtr,
            width * sizeof(uint32_t),
            width * sizeof(uint32_t),
            height,
            cudaMemcpyDeviceToDevice
        );
        if (erra != cudaSuccess) {
            throw std::runtime_error("Failed to copy data to CUDA array: " + std::string(cudaGetErrorString(erra)));
        }

        auto err = cudaGraphicsUnmapResources(1, &cudaTex);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to unmap CUDA-GL resources: " + std::string(cudaGetErrorString(err)));
        }
        resourcesMapped = false;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindVertexArray(vao);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(program, "tex"), 0);

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        SDL_GL_SwapWindow(window);
    }

    void resize(int w, int h) {
        width = w;
        height = h;
        resourcesMapped = true;
    }

    ~GLBackend() {
        if (resourcesMapped && cudaTex) {
            cudaGraphicsUnmapResources(1, &cudaTex);
        }
        if (cudaTex) cudaGraphicsUnregisterResource(cudaTex);
        if (tex) glDeleteTextures(1, &tex);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        if (program) glDeleteProgram(program);
        if (glctx) SDL_GL_DestroyContext(glctx);
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
        uniform sampler2D tex;
        void main() {
            color = texture(tex, vUV);
        }
    )";

    GLuint createProgram(const char* vs, const char* fs) {
        auto compile = [](GLenum type, const char* src) {
            GLuint s = glCreateShader(type);
            glShaderSource(s, 1, &src, nullptr);
            glCompileShader(s);
            
            GLint success;
            glGetShaderiv(s, GL_COMPILE_STATUS, &success);
            if (!success) {
                char infoLog[512];
                glGetShaderInfoLog(s, 512, nullptr, infoLog);
                std::cerr << "Shader compilation failed: " << infoLog << std::endl;
            }
            return s;
        };

        GLuint v = compile(GL_VERTEX_SHADER, vs);
        GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        GLuint p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        
        GLint success;
        glGetProgramiv(p, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(p, 512, nullptr, infoLog);
            std::cerr << "Program linking failed: " << infoLog << std::endl;
        }
        
        glDeleteShader(v);
        glDeleteShader(f);
        return p;
    }

    DeviceType getDeviceType() const {
        return DeviceType::kCUDA;
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
            selectedarea.x - currentWindowPosition.x,
            selectedarea.y - currentWindowPosition.y,
            selectedarea.z,
            selectedarea.w
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
                    if (sqrt(pow(mx.x - lastClicked.x, 2) + pow(mx.y - lastClicked.y, 2)) > 5.0f) {
                        selectedarea = float32x4(lastClicked.x, lastClicked.y, mx.x - lastClicked.x, mx.y - lastClicked.y);
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

template <DeviceType Device = DeviceType::kCPU>
class VectorDisplay : public Tensor<uint84, 2> {
public:
    Display* display = nullptr;
    SDL_Window* window;
    Window* root_window = nullptr;
    Visual* visual = nullptr;
    Colormap colormap;
    int screen;
    int depth;
    bool borderless = false;
    bool alpha_enabled = false;
    bool is_fullscreen = false;
    bool clickthrough = true;
    bool on_top = false;
    bool resizable = false;
    CurrentScreenInputInfo current_screen_input_info;
    std::vector<std::function<void(CurrentScreenInputInfo&)>> display_loop_functions;
    
    GLBackend<Device> backend = GLBackend<Device>();
    Clock clock;
    
    VectorDisplay(Shape<2> shape = 0, WindowPropertiesFlags properties = (WindowProperties)0)
        : Tensor<uint84, 2>(shape, Device), 
          borderless(properties.borderless), 
          alpha_enabled(properties.alpha_enabled), 
          is_fullscreen(properties.fullscreen),  
          clickthrough(properties.clickthrough), 
          on_top(properties.on_top),
          resizable(properties.resizable) {

        backend.preinit();

        std::cout << "finished preinit" << std::endl;
      
        Uint32 window_flags = SDL_WINDOW_OPENGL;
        
        if (is_fullscreen) window_flags |= SDL_WINDOW_FULLSCREEN;
        if (borderless) window_flags |= SDL_WINDOW_BORDERLESS;
        if (alpha_enabled) window_flags |= SDL_WINDOW_TRANSPARENT;
        if (on_top) window_flags |= SDL_WINDOW_ALWAYS_ON_TOP;
        if (resizable) window_flags |= SDL_WINDOW_RESIZABLE;
        
        window = SDL_CreateWindow(
            "CUDA → OpenGL",
            shape[1], shape[0],
            window_flags
        );

        if (!window) {
            throw std::runtime_error("Failed to create SDL window: " + std::string(SDL_GetError()));
        }

        std::cout << "created window" << std::endl;

        display = (Display *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);

        screen = SDL_GetDisplayForWindow(window);
        
        SDL_ShowWindow(window);

        backend.initialize(data, window, shape[1], shape[0], visual, depth, colormap);
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
        SDL_GetWindowPosition(window, &current_screen_input_info.currentWindowPosition.x, &current_screen_input_info.currentWindowPosition.y);
        current_screen_input_info.relativeWindowMove = int32x2(
            current_screen_input_info.currentWindowPosition.x - oldWindowPosition.x,
            current_screen_input_info.currentWindowPosition.y - oldWindowPosition.y
        );
        
        if (shape[0] > 0 && shape[1] > 0) {
            backend.present();
        }
    }

    void resizeDisplay() {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        if (w != shape[1] || h != shape[0]) {
            shape[0] = h;
            shape[1] = w;
            backend.resize(w, h);
            current_screen_input_info.setScreenSize(0, 0, w, h);
        }
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
            
            // Call display loop functions
            for (const auto& callback : display_loop_functions) {
                callback(current_screen_input_info);
            }

            // Process events
            running = processEvents();
            
            updateDisplay();
            updateMousePositionFromRoot();
        }
    }

    void displayLoopWithFPS(int target_fps) {
        bool running = true;

        while (running) {
            resizeDisplay();
            
            // Call display loop functions
            for (const auto& callback : display_loop_functions) {
                callback(current_screen_input_info);
            }

            // Process events
            running = processEvents();
            
            updateDisplay();
            updateMousePositionFromRoot();
            
            clock.tick(target_fps);
        }
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
        for (int sy = 0; sy < surf.height && (y + sy) < shape[0]; sy++) {
            for (int sx = 0; sx < surf.width && (x + sx) < shape[1]; sx++) {
                if (x + sx >= 0 && y + sy >= 0) {
                    (*this)[y + sy][x + sx] = surf.pixels[sy * surf.width + sx];
                }
            }
        }
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